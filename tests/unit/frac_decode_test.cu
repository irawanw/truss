// Half-integer trellis rates (K2.5, K3.5; TRACKER #118, E3) against exllamav3 v1.5.1's own decode.
// Fixtures (tests/data/frac, README there): real X3.1 tiles of Flash-Next L24 expert 0 and exllamav3's
// ext.reconstruct of them (the codebook value of every weight, no Hadamard or scales). Each decoder here must give
// the same fp16 values bit for bit:
//   ref::trellis_dequant (the bitstream definition, formats/trellis_k.h)
//   Mul1Frac<KA>::tile (the GPU codec of moe_window / moe_prefill), fragment values mapped back to (n, k)
// usage: frac_decode_test [fixture dir] (default tests/data/frac, run from the repo root)
#include "core/cuda_check.h"
#include "formats/trellis_k.h"
#include "kernels/reference/ref.cuh"
#include "kernels/trellis/codec_mul1.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace truss;

namespace {

template <class T> std::vector<T> load(const std::string & path, size_t n)
{
    std::vector<T> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char *>(v.data()), (std::streamsize) (n * sizeof(T))))
        throw std::runtime_error("cannot read " + path);
    return v;
}

constexpr int IN = 128, OUT = 640;   // the fixture slice: 8 k-tiles x 40 n-tiles

// one warp per tile: the codec's fragments, written to W[o][i] by the fragment order (lane = 4 (n % 8) + (k % 8) / 2,
// value m = (k & 1) + 2 (k >= 8) + 4 (n >= 8))
template <class Codec>
__global__ void codec_kernel(const uint32_t * tiles, int in, int out, float * W)
{
    const int tile = blockIdx.x, lane = threadIdx.x, kt = tile / (out / 16), nt = tile % (out / 16);
    const Codec codec(lane);
    const uint32_t w = codec.loads(lane) ? tiles[(size_t) tile * Codec::TILE_WORDS + lane] : 0u;
    FragB f0, f1;
    codec.tile(w, 0, f0, f1);
    const half2 v[4] = { f0[0], f0[1], f1[0], f1[1] };
    for (int m = 0; m < 8; ++m) {
        const float x = (m & 1) ? __high2float(v[m / 2]) : __low2float(v[m / 2]);
        const int n = lane / 4 + ((m & 4) ? 8 : 0), k = 2 * (lane % 4) + (m & 1) + ((m & 2) ? 8 : 0);
        W[(size_t) (nt * 16 + n) * in + kt * 16 + k] = x;
    }
}

}  // namespace

int main(int argc, char ** argv)
{
    const std::string dir = argc > 1 ? argv[1] : "tests/data/frac";
    int fails = 0;
    for (int code : { 25, 35 })
        for (const char * proj : { "gate", "down" }) {
            const std::string base = dir + "/k" + std::to_string(code) + "_" + proj;
            const size_t words = (size_t) IN / 16 * OUT / 16 * formats::k_tile_u16(code);
            const auto tiles = load<uint16_t>(base + ".trellis.u16", words);
            const auto recon = load<uint16_t>(base + ".recon.f16", (size_t) IN * OUT);   // [in][out] fp16 bits

            uint16_t * d_t;
            float * d_w;
            TRUSS_CUDA(cudaMalloc(&d_t, words * 2));
            TRUSS_CUDA(cudaMalloc(&d_w, sizeof(float) * IN * OUT));
            TRUSS_CUDA(cudaMemcpy(d_t, tiles.data(), words * 2, cudaMemcpyHostToDevice));
            auto check = [&](const char * what) {
                std::vector<float> w((size_t) IN * OUT);
                TRUSS_CUDA(cudaDeviceSynchronize());
                TRUSS_CUDA(cudaMemcpy(w.data(), d_w, w.size() * 4, cudaMemcpyDeviceToHost));
                long bad = 0;
                for (int o = 0; o < OUT; ++o)
                    for (int i = 0; i < IN; ++i) {
                        const float want = __half2float(__ushort_as_half(recon[(size_t) i * OUT + o]));
                        bad += w[(size_t) o * IN + i] != want;
                    }
                std::printf("K%.1f %-4s %-22s vs exllamav3 reconstruct: %ld of %d differ  %s\n", formats::k_bits(code),
                            proj, what, bad, IN * OUT, bad ? "FAIL" : "ok");
                fails += bad != 0;
                TRUSS_CUDA(cudaMemset(d_w, 0xff, sizeof(float) * IN * OUT));
            };
            ref::trellis_dequant(d_t, code, IN, OUT, d_w, nullptr);
            check("ref::trellis_dequant");
            const unsigned n_tiles = IN / 16 * (OUT / 16);
            if (code == 25) codec_kernel<Mul1Frac<2>><<<n_tiles, 32>>>((const uint32_t *) d_t, IN, OUT, d_w);
            else codec_kernel<Mul1Frac<3>><<<n_tiles, 32>>>((const uint32_t *) d_t, IN, OUT, d_w);
            TRUSS_CUDA(cudaGetLastError());
            check("Mul1Frac (GPU codec)");
            cudaFree(d_t);
            cudaFree(d_w);
        }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
