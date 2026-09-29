// Codec lab: trellis quantization MSE of candidate codebooks on iid N(0,1) data, against the Gaussian bound 2^-2K.
//
// A codebook is any map from a 16-bit trellis state to V weights (V = 1 or 2), given here as a host lookup table
// that must match the device decode bit for bit. The trellis is the bitshift trellis of exllamav3/QTIP: each step
// emits V weights and shifts V*K new bits into the state, next = ((s << VK) | b) & 0xffff. The Viterbi here is
// plain (free start state, no tail biting), the same for every codebook, so results compare codebooks, and
// sit slightly below exllamav3's tail-biting numbers.
//
// usage: tk-codec-lab [n_tiles]              all codebooks, all rates (paired: every codebook sees the same data)
//        tk-codec-lab search [n_tiles]       v2one family h = ((s*A+B) & M) ^ X over M, X, A at K1.5 / K2 / K2.5
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

constexpr int L = 16, NSTATE = 1 << L, TILE = 256;

// bits shifted in at a step, from s2 = 2 x (bits per step): integer rates are constant, half rates alternate
// floor/ceil (exllamav3's K+0.5 pattern for V = 1; V = 2 at a half rate is an integer 2K)
__host__ __device__ inline int step_shift(int s2, int step) { return ((step + 1) * s2) / 2 - (step * s2) / 2; }

// ---------------------------------------------------------------------------------------------------------
// Viterbi, one step: grid (NSTATE/256, n_tiles), one thread per new state

template <int V>
__global__ void vit_step(const float * __restrict__ cost_in, float * __restrict__ cost_out, uint8_t * __restrict__ bp,
                         const float * __restrict__ lut, const float * __restrict__ x, int s2, int step, float alpha)
{
    const int shift = step_shift(s2, step);
    const int n = blockIdx.x * blockDim.x + threadIdx.x, tile = blockIdx.y;
    const float * ci = cost_in + (size_t) tile * NSTATE;
    float e = 0.f;
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        const float d = x[(size_t) tile * TILE + step * V + v] - alpha * lut[(size_t) n * V + v];
        e += d * d;
    }
    const int base = n >> shift, hs = L - shift;
    float best = INFINITY;
    int arg = 0;
    for (int h = 0; h < (1 << shift); ++h) {
        const float c = ci[base | (h << hs)];
        if (c < best) { best = c; arg = h; }
    }
    cost_out[(size_t) tile * NSTATE + n] = best + e;
    bp[((size_t) tile * (TILE / V) + step) * NSTATE + n] = (uint8_t) arg;
}

// per tile: pick the best end state, walk back, return the tile's squared error
template <int V>
__global__ void vit_back(const float * __restrict__ cost, const uint8_t * __restrict__ bp, const float * __restrict__ lut,
                         const float * __restrict__ x, int s2, float alpha, float * __restrict__ sse)
{
    const int tile = blockIdx.x;
    __shared__ float s_best[256];
    __shared__ int s_arg[256];
    const float * c = cost + (size_t) tile * NSTATE;
    float b = INFINITY;
    int a = 0;
    for (int n = threadIdx.x; n < NSTATE; n += blockDim.x) if (c[n] < b) { b = c[n]; a = n; }
    s_best[threadIdx.x] = b; s_arg[threadIdx.x] = a;
    __syncthreads();
    if (threadIdx.x) return;
    for (int i = 1; i < blockDim.x; ++i) if (s_best[i] < b) { b = s_best[i]; a = s_arg[i]; }
    int n = a;
    float err = 0.f;
    for (int step = TILE / V - 1; step >= 0; --step) {
        for (int v = 0; v < V; ++v) {
            const float d = x[(size_t) tile * TILE + step * V + v] - alpha * lut[(size_t) n * V + v];
            err += d * d;
        }
        const int h = bp[((size_t) tile * (TILE / V) + step) * NSTATE + n];
        const int shift = step_shift(s2, step);
        n = (n >> shift) | (h << (L - shift));
    }
    sse[tile] = err;
}

// ---------------------------------------------------------------------------------------------------------
// codebooks (host tables; each mirrors a device decode)

static float h2f(uint16_t b) { __half_raw r; r.x = b; return __half2float(__half(r)); }
static float rh(float f) { return __half2float(__float2half_rn(f)); }   // round to fp16

struct Codebook {
    std::string name;
    int V;
    std::function<void(uint32_t, float *)> f;   // state -> V weights
    std::string cost;                           // device instructions per V weights, excluding state extraction
};

static std::vector<Codebook> codebooks()
{
    std::vector<Codebook> cbs;
    // mul1 (PAW X3 / exllamav3): bytesum(s * 0x83DCD12D) + 0x6400 as fp16, then hfma2(1/147.7, -10.39)
    cbs.push_back({ "mul1 (X3)", 1, [] (uint32_t s, float * o) {
        const uint32_t x = s * 0x83DCD12Du;
        const uint32_t sum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24) + 0x6400u;
        o[0] = rh(h2f((uint16_t) sum) * h2f(0x1eee) + h2f(0xc931));
    }, "IMAD IDP + 1/2 PRMT + 1/2 HFMA2" });
    // ideal V=1: iid Gaussian table (what a free codebook could reach with this trellis)
    {
        auto g = std::make_shared<std::vector<float>>(NSTATE);
        std::mt19937 r(7); std::normal_distribution<float> nd;
        for (auto & v : *g) v = nd(r);
        cbs.push_back({ "ideal V1 (random gauss LUT)", 1, [g] (uint32_t s, float * o) { o[0] = (*g)[s]; }, "-" });
    }
    {
        auto g = std::make_shared<std::vector<float>>(2 * NSTATE);
        std::mt19937 r(8); std::normal_distribution<float> nd;
        for (auto & v : *g) v = nd(r);
        cbs.push_back({ "ideal V2 (random gauss LUT)", 2, [g] (uint32_t s, float * o) { o[0] = (*g)[2 * s]; o[1] = (*g)[2 * s + 1]; }, "-" });
    }
    // V2 "pair": two affine hashes, each masked into two fp16 halves, summed lane-wise:
    //   x1 = a1*s+b1, x2 = a2*s+b2 (IMAD each); h = lop3(x) = (x & M) ^ X; w = hadd2(h1, h2)
    auto pair = [] (uint32_t a1, uint32_t b1, uint32_t a2, uint32_t b2, uint32_t M, uint32_t X) {
        return [=] (uint32_t s, float * o) {
            const uint32_t h1 = ((a1 * s + b1) & M) ^ X, h2 = ((a2 * s + b2) & M) ^ X;
            o[0] = rh(h2f((uint16_t) h1) + h2f((uint16_t) h2));
            o[1] = rh(h2f((uint16_t) (h1 >> 16)) + h2f((uint16_t) (h2 >> 16)));
        };
    };
    cbs.push_back({ "V2 pair  M8fff X3b60", 2, pair(0x83DCD12Du, 0x6A09E667u, 0xCBAC1FEDu, 0xBB67AE85u, 0x8fff8fffu, 0x3b603b60u), "2 IMAD 2 LOP3 1 HADD2" });
    cbs.push_back({ "V2 pair  M8bff X3960", 2, pair(0x83DCD12Du, 0x6A09E667u, 0xCBAC1FEDu, 0xBB67AE85u, 0x8bff8bffu, 0x39603960u), "2 IMAD 2 LOP3 1 HADD2" });
    // V2 "swap": one IMAD, second operand = the same word with halves swapped under a second mask
    auto swap = [] (uint32_t a, uint32_t b, uint32_t M1, uint32_t X1, uint32_t M2, uint32_t X2) {
        return [=] (uint32_t s, float * o) {
            const uint32_t x = a * s + b;
            const uint32_t y = (x >> 16) | (x << 16);
            const uint32_t h1 = (x & M1) ^ X1, h2 = (y & M2) ^ X2;
            o[0] = rh(h2f((uint16_t) h1) + h2f((uint16_t) h2));
            o[1] = rh(h2f((uint16_t) (h1 >> 16)) + h2f((uint16_t) (h2 >> 16)));
        };
    };
    cbs.push_back({ "V2 swap  (1 IMAD)", 2, swap(0x83DCD12Du, 0x6A09E667u, 0x8fff8fffu, 0x3b603b60u, 0x83ff83ffu, 0x34003400u), "1 IMAD 1 PRMT 2 LOP3 1 HADD2" });
    // V2 "one": one IMAD, one lop3, the two halves are the two weights (no sum)
    auto one = [] (uint32_t a, uint32_t b, uint32_t M, uint32_t X) {
        return [=] (uint32_t s, float * o) {
            const uint32_t h = ((a * s + b) & M) ^ X;
            o[0] = h2f((uint16_t) h); o[1] = h2f((uint16_t) (h >> 16));
        };
    };
    cbs.push_back({ "V2 one   M8fff X3b60", 2, one(0x83DCD12Du, 0x6A09E667u, 0x8fff8fffu, 0x3b603b60u), "1 IMAD 1 LOP3" });
    cbs.push_back({ "V2 one   M83ff X3800", 2, one(0x83DCD12Du, 0x6A09E667u, 0x83ff83ffu, 0x38003800u), "1 IMAD 1 LOP3" });
    return cbs;
}

// ---------------------------------------------------------------------------------------------------------

static std::vector<Codebook> v2one_family()
{
    auto one = [] (uint32_t a, uint32_t b, uint32_t M, uint32_t X) {
        return [=] (uint32_t s, float * o) {
            const uint32_t h = ((a * s + b) & M) ^ X;
            o[0] = h2f((uint16_t) h); o[1] = h2f((uint16_t) (h >> 16));
        };
    };
    std::vector<Codebook> cbs;
    // M keeps the sign, the mantissa and a subset R of the 5 exponent bits (random per state); X sets the other
    // exponent bits. Any R and fixed pattern: octaves need not be contiguous (e.g. tiny values near zero).
    for (uint32_t R = 1; R < 32; ++R) {
        const int nr = __builtin_popcount(R);
        if (nr > 3) continue;
        const uint32_t fixed_bits = 31u & ~R;
        for (uint32_t F = 0; F < 32; ++F) {
            if (F & ~fixed_bits) continue;
            const uint32_t m16 = 0x8000u | (R << 10) | 0x3ffu, x16 = F << 10;
            // skip patterns that can make inf/nan (exponent 31) or only subnormals
            bool bad = false;
            for (uint32_t r = 0; r < 32; ++r) if (!(r & ~R) && ((r | F) == 31)) bad = true;
            if (bad) continue;
            char name[64];
            snprintf(name, sizeof name, "v2one R%02x F%02x M%04x X%04x", R, F, m16, x16);
            cbs.push_back({ name, 2, one(0x83DCD12Du, 0x6A09E667u, m16 * 0x10001u, x16 * 0x10001u), "1 IMAD 1 LOP3" });
        }
    }
    return cbs;
}

int main(int argc, char ** argv)
{
    const bool search = argc > 1 && std::string(argv[1]) == "search";
    if (search) { --argc; ++argv; }
    const int n_tiles = argc > 1 ? atoi(argv[1]) : 64;
    std::vector<float> hx((size_t) n_tiles * TILE);
    { std::mt19937 r(1234); std::normal_distribution<float> nd; for (auto & v : hx) v = nd(r); }
    float *x, *lut, *c0, *c1, *sse;
    uint8_t * bp;
    CK(cudaMalloc(&x, hx.size() * 4));
    CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&lut, (size_t) 2 * NSTATE * 4));
    CK(cudaMalloc(&c0, (size_t) n_tiles * NSTATE * 4));
    CK(cudaMalloc(&c1, (size_t) n_tiles * NSTATE * 4));
    CK(cudaMalloc(&bp, (size_t) n_tiles * TILE * NSTATE));
    CK(cudaMalloc(&sse, n_tiles * 4));

    auto run = [&] (int V, int s2, float alpha) -> double {
        CK(cudaMemset(c0, 0, (size_t) n_tiles * NSTATE * 4));
        float *ci = c0, *co = c1;
        const dim3 grid(NSTATE / 256, n_tiles);
        for (int step = 0; step < TILE / V; ++step) {
            if (V == 1) vit_step<1><<<grid, 256>>>(ci, co, bp, lut, x, s2, step, alpha);
            else        vit_step<2><<<grid, 256>>>(ci, co, bp, lut, x, s2, step, alpha);
            std::swap(ci, co);
        }
        if (V == 1) vit_back<1><<<n_tiles, 256>>>(ci, bp, lut, x, s2, alpha, sse);
        else        vit_back<2><<<n_tiles, 256>>>(ci, bp, lut, x, s2, alpha, sse);
        std::vector<float> h(n_tiles);
        CK(cudaMemcpy(h.data(), sse, n_tiles * 4, cudaMemcpyDeviceToHost));
        double s = 0; for (float v : h) s += v;
        return s / ((double) n_tiles * TILE);
    };

    std::vector<double> rates = { 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };
    if (search) rates = { 1.5, 2.0, 2.5 };
    printf("%d tiles x %d iid N(0,1); MSE at best scale; bound = 2^-2K\n", n_tiles, TILE);
    printf("%-30s %-34s", "codebook", "decode cost per V weights");
    for (double K : rates) printf("   K%.1f     ", K);
    printf("\n%-30s %-34s", "Gaussian bound", "");
    for (double K : rates) printf("  %.5f   ", std::pow(2.0, -2 * K));
    printf("\n");
    std::vector<Codebook> list = codebooks();
    if (search) {
        list.resize(1);                          // mul1 as the reference row
        for (auto & c : v2one_family()) list.push_back(c);
    }
    for (const Codebook & cb : list) {
        std::vector<float> h((size_t) NSTATE * cb.V);
        for (uint32_t s = 0; s < NSTATE; ++s) cb.f(s, &h[(size_t) s * cb.V]);
        double ms = 0, m = 0;
        for (float v : h) { ms += v * v; m += v; }
        const double rms = std::sqrt(ms / h.size());
        CK(cudaMemcpy(lut, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        printf("%-30s %-34s", cb.name.c_str(), cb.cost.c_str());
        for (double K : rates) {
            const int s2 = (int) std::lround(2 * K * cb.V);
            // scale search: coarse log grid around the rms-matched scale, then a finer one around the best
            double best = 1e9, best_a = 1.0 / rms;
            for (int pass = 0; pass < 2; ++pass) {
                const double c = best_a, span = pass == 0 ? 0.5 : 0.08;
                for (int i = -4; i <= 4; ++i) {
                    const double a = c * std::exp(span * i / 4.0);
                    const double e = run(cb.V, s2, (float) a);
                    if (e < best) { best = e; best_a = a; }
                }
            }
            printf("  %.5f%s ", best, best <= std::pow(2.0, -2 * K) * 1.0 ? "*" : " ");
        }
        printf("   (rms %.3f mean %+.3f)\n", rms, m / h.size());
        fflush(stdout);
    }
    return 0;
}
