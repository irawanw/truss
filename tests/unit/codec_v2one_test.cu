// v2one codec: encoder -> pack -> kernel decode, bit-exact.
//   1. pack: random rings -> v2one_pack_tile -> v2one_state gives every state back;
//   2. decode: V2One<S>::tile on packed words == the host codebook of each state, in the mma fragment order;
//   3. encode: Gaussian tiles (codebook rms) -> v2one_encode: the states form a ring, q == codebook(states), the
//      kernel decode of the packed ring == q bit for bit, and the MSE (printed with the time per tile). MSE gate:
//      the iid lab value (tools/codec-lab/viterbi_mse.cu: plain Viterbi at the best scale) + 10% for tail-biting and
//      the unsearched scale (data rms = codebook rms). K1 (S = 2) has no lab value: structural checks only.
// usage: codec_v2one_test [n_tiles]
#include "core/cuda_check.h"
#include "encode/v2one_encode.h"
#include "kernels/trellis/codec_v2one.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace truss;

namespace {

// one warp per VEC: decode every tile of it; out[tile][j] as raw fp16 bits, j = 8 * lane + i
template <int S>
__global__ void decode_kernel(const uint32_t * words, int n_vec, uint16_t * out)
{
    using C = V2One<S>;
    const int lane = threadIdx.x & 31, vec = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (vec >= n_vec) return;
    const C codec(lane);
    const uint32_t w = codec.loads(lane) ? words[(size_t) vec * C::VEC_WORDS + lane] : 0u;
    for (int sub = 0; sub < C::TILES_PER_VEC; ++sub) {
        FragB f0, f1;
        codec.tile(w, sub, f0, f1);
        const half2 h[4] = { f0[0], f0[1], f1[0], f1[1] };
        uint16_t * o = out + ((size_t) vec * C::TILES_PER_VEC + sub) * 256 + 8 * lane;
        for (int k = 0; k < 4; ++k) {
            o[2 * k] = __half_as_ushort(__low2half(h[k]));
            o[2 * k + 1] = __half_as_ushort(__high2half(h[k]));
        }
    }
}

// a random valid ring of 128 states at rate S
template <int S>
void random_ring(std::mt19937 & rng, uint16_t * st)
{
    // new bits per step, circular: the state of step t is the 16 bits ending at bit (t + 1) S
    constexpr int BITS = 128 * S;
    std::vector<int> bit(BITS);
    for (auto & b : bit) b = rng() & 1;
    for (int t = 0; t < 128; ++t) {
        uint32_t s = 0;
        for (int i = 0; i < 16; ++i) s = (s << 1) | bit[((t + 1) * S - 16 + i + BITS) % BITS];
        st[t] = (uint16_t) s;
    }
}

template <int S>
bool is_ring(const uint16_t * st)
{
    for (int t = 0; t < 128; ++t) {
        const uint32_t prev = st[(t + 127) & 127];
        if ((((prev << S) & 0xffffu) >> S) != ((uint32_t) st[t] >> S)) return false;
    }
    return true;
}

// pack n tiles of states and run the kernel decode; returns fp16 bits per weight
template <int S>
std::vector<uint16_t> pack_and_decode(const std::vector<uint16_t> & states, int n)
{
    using C = V2One<S>;
    const int n_vec = (n + C::TILES_PER_VEC - 1) / C::TILES_PER_VEC;
    std::vector<uint32_t> words((size_t) n_vec * C::VEC_WORDS, 0);
    for (int t = 0; t < n; ++t) v2one_pack_tile<S>(&states[(size_t) t * 128], &words[(size_t) t * C::TILE_WORDS]);
    uint32_t * d_w;
    uint16_t * d_o;
    const size_t out_n = (size_t) n_vec * C::TILES_PER_VEC * 256;
    TRUSS_CUDA(cudaMalloc(&d_w, words.size() * 4));
    TRUSS_CUDA(cudaMalloc(&d_o, out_n * 2));
    TRUSS_CUDA(cudaMemcpy(d_w, words.data(), words.size() * 4, cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemset(d_o, 0xff, out_n * 2));
    decode_kernel<S><<<(n_vec + 3) / 4, 128>>>(d_w, n_vec, d_o);
    TRUSS_CUDA(cudaGetLastError());
    std::vector<uint16_t> out(out_n);
    TRUSS_CUDA(cudaMemcpy(out.data(), d_o, out_n * 2, cudaMemcpyDeviceToHost));
    cudaFree(d_w);
    cudaFree(d_o);
    out.resize((size_t) n * 256);
    return out;
}

template <int S>
int run(int n, std::mt19937 & rng)
{
    int fails = 0;
    // 1 + 2: random rings
    {
        std::vector<uint16_t> st((size_t) n * 128);
        for (int t = 0; t < n; ++t) random_ring<S>(rng, &st[(size_t) t * 128]);
        uint32_t words[4 * S];
        int bad_pack = 0;
        for (int t = 0; t < n; ++t) {
            v2one_pack_tile<S>(&st[(size_t) t * 128], words);
            for (int k = 0; k < 128; ++k) bad_pack += v2one_state<S>(words, k) != st[(size_t) t * 128 + k];
        }
        const auto dec = pack_and_decode<S>(st, n);
        int bad_dec = 0;
        for (size_t j = 0; j < dec.size(); ++j) {
            const uint32_t h = v2one_codebook_bits(st[j / 2]);
            bad_dec += dec[j] != (uint16_t) (j & 1 ? h >> 16 : h & 0xffffu);
        }
        const bool ok = !bad_pack && !bad_dec;
        fails += !ok;
        std::printf("S=%d random rings  %d tiles: pack mismatches %d, decode mismatches %d  %s\n", S, n, bad_pack,
                    bad_dec, ok ? "PASS" : "FAIL");
    }
    // 3: encoder
    {
        std::normal_distribution<float> nd(0.f, V2One<S>::RMS);
        std::vector<float> x((size_t) n * 256);
        for (auto & v : x) v = nd(rng);
        const size_t ws_bytes = (size_t) std::min(n, 256) * encode::v2one_workspace_bytes_per_tile(S);
        float * d_x, * d_q;
        uint16_t * d_s;
        void * ws;
        TRUSS_CUDA(cudaMalloc(&d_x, x.size() * 4));
        TRUSS_CUDA(cudaMalloc(&d_q, x.size() * 4));
        TRUSS_CUDA(cudaMalloc(&d_s, (size_t) n * 128 * 2));
        TRUSS_CUDA(cudaMalloc(&ws, ws_bytes));
        TRUSS_CUDA(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        TRUSS_CUDA(cudaMemset(d_q, 0xff, x.size() * 4));
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0);
        encode::v2one_encode(S, d_x, n, d_q, d_s, ws, ws_bytes, nullptr);
        cudaEventRecord(e1);
        TRUSS_CUDA(cudaDeviceSynchronize());
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        std::vector<float> q(x.size());
        std::vector<uint16_t> st((size_t) n * 128);
        TRUSS_CUDA(cudaMemcpy(q.data(), d_q, q.size() * 4, cudaMemcpyDeviceToHost));
        TRUSS_CUDA(cudaMemcpy(st.data(), d_s, st.size() * 2, cudaMemcpyDeviceToHost));
        cudaFree(d_x); cudaFree(d_q); cudaFree(d_s); cudaFree(ws);

        int not_ring = 0, bad_q = 0;
        double se = 0, sx = 0;
        for (int t = 0; t < n; ++t) not_ring += !is_ring<S>(&st[(size_t) t * 128]);
        const auto dec = pack_and_decode<S>(st, n);
        for (size_t j = 0; j < x.size(); ++j) {
            const uint32_t h = v2one_codebook_bits(st[j / 2]);
            const uint16_t hb = (uint16_t) (j & 1 ? h >> 16 : h & 0xffffu);
            __half hh;
            std::memcpy(&hh, &hb, 2);
            bad_q += q[j] != __half2float(hh) || dec[j] != hb;
            se += (q[j] - x[j]) * (double) (q[j] - x[j]);
            sx += (double) x[j] * x[j];
        }
        const double mse = se / sx;
        // tk-codec-lab 512, row "V2 one M8fff X3b60" (2026-09-30), K1.5 / K2 / K2.5
        constexpr double LAB[6] = { 0, 0, 0, 0.13083, 0.07100, 0.03759 };
        const bool ok = !not_ring && !bad_q && (LAB[S] == 0 || mse <= LAB[S] * 1.10);
        fails += !ok;
        std::printf("S=%d encoder       %d tiles: not a ring %d, q/decode mismatches %d, MSE %.5f (lab %.4f)  "
                    "%.1f us/tile  %s\n",
                    S, n, not_ring, bad_q, mse, LAB[S], ms * 1e3 / n, ok ? "PASS" : "FAIL");
    }
    return fails;
}

}  // namespace

int main(int argc, char ** argv)
{
    const int n = argc > 1 ? std::atoi(argv[1]) : 512;
    std::mt19937 rng(1234);
    try {
        int fails = run<2>(n, rng) + run<3>(n, rng) + run<4>(n, rng) + run<5>(n, rng);
        std::printf("%s\n", fails ? "FAIL" : "all PASS");
        return fails ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
