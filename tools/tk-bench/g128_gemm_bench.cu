// Speed probe (lead 10-07): dense int8 GEMM with scales per 128 k (W8A8 g128) vs today's per-32 fold (Q8_0).
// g128: the 4 int8 mma of a 128-deep group accumulate in int32 (exact), one fp32 fold per group instead of per mma.
// Data use one scale per 4 blocks, so both kernels compute the same products; outputs are compared (rel diff).
// usage: g128_gemm_bench [iters]
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
using half = __half;
using int8 = signed char;
struct Q8Matrix { int8 * q; half * d; int in, out; };   // d: one scale per 32 k (g32) or per 128 k (g128)

constexpr int THREADS = 256;
constexpr int BM = 128, BN = 128, KC = 64;
constexpr int WM = 64, WN = 32;
constexpr int SK = KC + 16;
constexpr int STAGE = (BM + BN) * SK;

__device__ __forceinline__ void cp16(void * dst, const void * src, bool valid)
{
    const unsigned s = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void mma_s8(const uint32_t (&a)[4], const uint32_t (&b)[2], int (&c)[4])
{
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0), "r"(0), "r"(0), "r"(0));
}
__device__ __forceinline__ void mma_s8_acc(const uint32_t (&a)[4], const uint32_t (&b)[2], int (&c)[4])
{
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// G = blocks of 32 per scale group: 1 = today's Q8_0 fold, 4 = g128
template <int G, int SW, int MINB>
__global__ __launch_bounds__(THREADS, MINB) void gemm(Q8Matrix W, const int8 * __restrict__ xq, const half * __restrict__ xd,
                                                int rows, float * __restrict__ y)
{
    __shared__ __align__(16) int8 sq[2][STAGE];
    __shared__ float sd[2][BM + BN][KC / 32];   // the scale of each 32-block (group scale repeated)
    int bx = blockIdx.x, by = blockIdx.y;
    if (SW > 0) {   // grouped raster: SW token tiles share each weight tile while it is in L2
        const int pid = blockIdx.y * gridDim.x + blockIdx.x, per = SW * gridDim.x;
        const int first = pid / per * SW, gs = min((int) gridDim.y - first, SW);
        by = first + (pid % per) % gs, bx = (pid % per) / gs;
    }
    const int o0 = bx * BM, t0 = by * BN;
    const int K = W.in, NG = K / (32 * G);
    const int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = (wid / (BN / WN)) * WM, wn = (wid % (BN / WN)) * WN;
    const int g = lane >> 2, q = lane & 3;
    auto issue = [&](int kc, int s) {
        for (int i = threadIdx.x; i < (BM + BN) * (KC / 16); i += THREADS) {
            const int r = i / (KC / 16), v = i % (KC / 16);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            cp16(&sq[s][r * SK + v * 16], (is_w ? W.q : xq) + (size_t) (ok ? row : 0) * K + kc * KC + v * 16, ok);
        }
        for (int i = threadIdx.x; i < (BM + BN) * (KC / 32); i += THREADS) {
            const int r = i / (KC / 32), b = i % (KC / 32);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            const int blk = kc * (KC / 32) + b;
            sd[s][r][b] = ok ? __half2float((is_w ? W.d : xd)[(size_t) row * NG + blk / G]) : 0.f;
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[WM / 16][WN / 8][4] = {};
    int ci[WM / 16][WN / 8][4] = {};
    const int NK = K / KC;
    issue(0, 0);
    for (int kc = 0; kc < NK; ++kc) {
        const int s = kc & 1;
        if (kc + 1 < NK) {
            issue(kc + 1, s ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const int8 * A = sq[s] + wm * SK;
        const int8 * B = sq[s] + (BM + wn) * SK;
#pragma unroll
        for (int kb = 0; kb < KC / 32; ++kb) {
            uint32_t a[WM / 16][4], b[WN / 8][2];
#pragma unroll
            for (int m = 0; m < WM / 16; ++m) {
                const int8 * p = A + (m * 16 + g) * SK + kb * 32 + 4 * q;
                a[m][0] = *(const uint32_t *) p;
                a[m][1] = *(const uint32_t *) (p + 8 * SK);
                a[m][2] = *(const uint32_t *) (p + 16);
                a[m][3] = *(const uint32_t *) (p + 8 * SK + 16);
            }
#pragma unroll
            for (int n = 0; n < WN / 8; ++n) {
                const int8 * p = B + (n * 8 + g) * SK + kb * 32 + 4 * q;
                b[n][0] = *(const uint32_t *) p;
                b[n][1] = *(const uint32_t *) (p + 16);
            }
            const int blk = kc * (KC / 32) + kb;
            const bool fold = (blk % G) == G - 1;
            float dw[WM / 16][2], dx[WN / 8][2];
            if (fold) {
#pragma unroll
                for (int m = 0; m < WM / 16; ++m) dw[m][0] = sd[s][wm + m * 16 + g][kb], dw[m][1] = sd[s][wm + m * 16 + g + 8][kb];
#pragma unroll
                for (int n = 0; n < WN / 8; ++n)
                    dx[n][0] = sd[s][BM + wn + n * 8 + 2 * q][kb], dx[n][1] = sd[s][BM + wn + n * 8 + 2 * q + 1][kb];
            }
#pragma unroll
            for (int m = 0; m < WM / 16; ++m)
#pragma unroll
                for (int n = 0; n < WN / 8; ++n) {
                    if (G == 1) mma_s8(a[m], b[n], ci[m][n]);
                    else mma_s8_acc(a[m], b[n], ci[m][n]);
                    if (fold) {
                        acc[m][n][0] += (float) ci[m][n][0] * (dw[m][0] * dx[n][0]);
                        acc[m][n][1] += (float) ci[m][n][1] * (dw[m][0] * dx[n][1]);
                        acc[m][n][2] += (float) ci[m][n][2] * (dw[m][1] * dx[n][0]);
                        acc[m][n][3] += (float) ci[m][n][3] * (dw[m][1] * dx[n][1]);
                        if (G > 1) ci[m][n][0] = ci[m][n][1] = ci[m][n][2] = ci[m][n][3] = 0;
                    }
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < WM / 16; ++m)
#pragma unroll
        for (int n = 0; n < WN / 8; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int o = o0 + wm + m * 16 + g + (i >> 1) * 8, t = t0 + wn + n * 8 + 2 * q + (i & 1);
                if (o < W.out && t < rows) y[(size_t) t * W.out + o] = acc[m][n][i];
            }
}

static uint32_t xs = 12345u;
static uint32_t rnd() { xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5; return xs; }

int main(int argc, char ** argv)
{
    const int iters = argc > 1 ? atoi(argv[1]) : 8;
    struct Shape { int M, N, K; const char * name; };
    const Shape shapes[] = { {8192, 10240, 2560, "hc_up   10240x2560"}, {8192, 6144, 2560, "qkv      6144x2560"},
                             {8192, 2560, 6144, "outproj  2560x6144"}, {8192, 12288, 2560, "shared  12288x2560"},
                             {8192, 640, 2560, "ple_mixer 640x2560"} };
    for (const auto & S : shapes) {
        const size_t wq = (size_t) S.N * S.K, xq = (size_t) S.M * S.K;
        int8 * Wq, * Xq; half * Wd32, * Xd32, * Wd128, * Xd128; float * y1, * y4;
        CK(cudaMalloc(&Wq, wq)); CK(cudaMalloc(&Xq, xq));
        CK(cudaMalloc(&Wd32, wq / 32 * 2)); CK(cudaMalloc(&Xd32, xq / 32 * 2));
        CK(cudaMalloc(&Wd128, wq / 128 * 2)); CK(cudaMalloc(&Xd128, xq / 128 * 2));
        CK(cudaMalloc(&y1, (size_t) S.M * S.N * 4)); CK(cudaMalloc(&y4, (size_t) S.M * S.N * 4));
        {
            std::vector<int8> hw(wq), hx(xq);
            for (auto & v : hw) v = (int8) (rnd() % 255 - 127);
            for (auto & v : hx) v = (int8) (rnd() % 255 - 127);
            CK(cudaMemcpy(Wq, hw.data(), wq, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(Xq, hx.data(), xq, cudaMemcpyHostToDevice));
            auto scales = [](size_t n128, half * d32, half * d128) {
                std::vector<half> a(n128), b(n128 * 4);
                for (size_t i = 0; i < n128; ++i) {
                    a[i] = __float2half(0.001f + 0.004f * (rnd() % 1000) / 1000.f);
                    for (int j = 0; j < 4; ++j) b[4 * i + j] = a[i];
                }
                CK(cudaMemcpy(d128, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d32, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
            };
            scales(wq / 128, Wd32, Wd128);
            scales(xq / 128, Xd32, Xd128);
        }
        const dim3 grid((S.N + BM - 1) / BM, (S.M + BN - 1) / BN);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        const double flops = 2.0 * S.M * S.N * S.K;
        auto time = [&](auto launch) {
            launch(); launch(); CK(cudaDeviceSynchronize());
            cudaEventRecord(e0);
            for (int i = 0; i < iters; ++i) launch();
            cudaEventRecord(e1); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            float m; cudaEventElapsedTime(&m, e0, e1); return m / iters;
        };
        const Q8Matrix W32{Wq, Wd32, S.K, S.N}, W128{Wq, Wd128, S.K, S.N};
        std::vector<float> ref((size_t) S.M * S.N), out(ref.size());
        const float base = time([&] { gemm<1, 0, 1><<<grid, THREADS>>>(W32, Xq, Xd32, S.M, y1); });
        CK(cudaMemcpy(ref.data(), y1, ref.size() * 4, cudaMemcpyDeviceToHost));
        printf("%-22s g32 base %6.2f ms %6.1f TOPS", S.name, base, flops / (base * 1e-3) / 1e12);
        auto report = [&](const char * nm, float m, float * y, bool exact_expected) {
            CK(cudaMemcpy(out.data(), y, out.size() * 4, cudaMemcpyDeviceToHost));
            size_t mism = 0; double num = 0, den = 0;
            for (size_t i = 0; i < out.size(); ++i) {
                mism += memcmp(&out[i], &ref[i], 4) != 0;
                num += (out[i] - ref[i]) * (double) (out[i] - ref[i]), den += (double) ref[i] * ref[i];
            }
            printf(" | %s %6.2f ms x%.2f %s", nm, m, base / m,
                   exact_expected ? (mism ? "DIFF!" : "bit-exact") : (std::sqrt(num / den) < 1e-5 ? "~same" : "DIFF!"));
        };
        float m;
        m = time([&] { gemm<1, 8, 1><<<grid, THREADS>>>(W32, Xq, Xd32, S.M, y1); }); report("g32 sw8", m, y1, true);
        m = time([&] { gemm<1, 16, 1><<<grid, THREADS>>>(W32, Xq, Xd32, S.M, y1); }); report("g32 sw16", m, y1, true);
        m = time([&] { gemm<4, 8, 1><<<grid, THREADS>>>(W128, Xq, Xd128, S.M, y4); }); report("g128 sw8", m, y4, false);
        m = time([&] { gemm<4, 8, 2><<<grid, THREADS>>>(W128, Xq, Xd128, S.M, y4); }); report("g128 sw8 2blk", m, y4, false);
        printf("\n");
        cudaFree(Wq), cudaFree(Xq), cudaFree(Wd32), cudaFree(Xd32), cudaFree(Wd128), cudaFree(Xd128), cudaFree(y1), cudaFree(y4);
    }
    return 0;
}
