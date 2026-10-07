// P1 microbench (PLAN-20261006 §1 P1): dense Q8 GEMM variants on the real mixer shapes.
// Baseline = gemm_kernel from src/kernels/dense/q8_gemm.cu copied verbatim (anonymous namespace there).
// Variants must be BIT-IDENTICAL to baseline output (same fold `acc += float(c) * (dw * dx)`, same order).
// Usage: q8_gemm_bench [iters]   (prints TOPS table + bit-identity per variant)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

#include <cstdint>
#include <cuda_fp16.h>
using half = __half;
using int8 = signed char;
struct Q8Matrix { int8 * q; half * d; int in, out; };

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

// ---------------------------------------------------------------- baseline (verbatim)
__global__ __launch_bounds__(THREADS) void gemm_base(Q8Matrix W, const int8 * __restrict__ xq,
                                                     const half * __restrict__ xd, int rows, float * __restrict__ y)
{
    __shared__ __align__(16) int8 sq[2][STAGE];
    __shared__ float sd[2][BM + BN][KC / 32];
    const int o0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    const int K = W.in, KB = K / 32;
    const int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = (wid / (BN / WN)) * WM, wn = (wid % (BN / WN)) * WN;
    const int g = lane >> 2, q = lane & 3;
    auto issue = [&](int kc, int s) {
        for (int i = threadIdx.x; i < (BM + BN) * (KC / 16); i += THREADS) {
            const int r = i / (KC / 16), v = i % (KC / 16);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            const int8 * src = (is_w ? W.q : xq) + (size_t) (ok ? row : 0) * K + kc * KC + v * 16;
            cp16(&sq[s][r * SK + v * 16], src, ok);
        }
        for (int i = threadIdx.x; i < (BM + BN) * (KC / 32); i += THREADS) {
            const int r = i / (KC / 32), b = i % (KC / 32);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            sd[s][r][b] = ok ? __half2float((is_w ? W.d : xd)[(size_t) row * KB + kc * (KC / 32) + b]) : 0.f;
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[WM / 16][WN / 8][4] = {};
    const int NK = K / KC;
    issue(0, 0);
    for (int kc = 0; kc < NK; ++kc) {
        const int s = kc & 1;
        if (kc + 1 < NK) { issue(kc + 1, s ^ 1); asm volatile("cp.async.wait_group 1;\n" ::); }
        else asm volatile("cp.async.wait_group 0;\n" ::);
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
            float dw[WM / 16][2], dx[WN / 8][2];
#pragma unroll
            for (int m = 0; m < WM / 16; ++m) {
                dw[m][0] = sd[s][wm + m * 16 + g][kb];
                dw[m][1] = sd[s][wm + m * 16 + g + 8][kb];
            }
#pragma unroll
            for (int n = 0; n < WN / 8; ++n) {
                dx[n][0] = sd[s][BM + wn + n * 8 + 2 * q][kb];
                dx[n][1] = sd[s][BM + wn + n * 8 + 2 * q + 1][kb];
            }
#pragma unroll
            for (int m = 0; m < WM / 16; ++m)
#pragma unroll
                for (int n = 0; n < WN / 8; ++n) {
                    int c[4];
                    mma_s8(a[m], b[n], c);
                    acc[m][n][0] += (float) c[0] * (dw[m][0] * dx[n][0]);
                    acc[m][n][1] += (float) c[1] * (dw[m][0] * dx[n][1]);
                    acc[m][n][2] += (float) c[2] * (dw[m][1] * dx[n][0]);
                    acc[m][n][3] += (float) c[3] * (dw[m][1] * dx[n][1]);
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

// ---------------------------------------------------------------- V1: KC=128, 2-stage (dynamic smem)
constexpr int KC1 = 128, SK1 = KC1 + 16, STAGE1 = (BM + BN) * SK1;
__global__ __launch_bounds__(THREADS) void gemm_v1(Q8Matrix W, const int8 * __restrict__ xq,
                                                   const half * __restrict__ xd, int rows, float * __restrict__ y)
{
    extern __shared__ __align__(16) char smem[];
    int8 * sq = (int8 *) smem;                                   // [2][STAGE1]
    float * sd = (float *) (smem + 2 * STAGE1);                  // [2][BM+BN][KC1/32]
    const int o0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    const int K = W.in, KB = K / 32;
    const int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = (wid / (BN / WN)) * WM, wn = (wid % (BN / WN)) * WN;
    const int g = lane >> 2, q = lane & 3;
    auto srow = [&](int s, int r) { return sq + (size_t) s * STAGE1 + r * SK1; };
    auto issue = [&](int kc, int s) {
        for (int i = threadIdx.x; i < (BM + BN) * (KC1 / 16); i += THREADS) {
            const int r = i / (KC1 / 16), v = i % (KC1 / 16);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            const int8 * src = (is_w ? W.q : xq) + (size_t) (ok ? row : 0) * K + kc * KC1 + v * 16;
            cp16(srow(s, r) + v * 16, src, ok);
        }
        for (int i = threadIdx.x; i < (BM + BN) * (KC1 / 32); i += THREADS) {
            const int r = i / (KC1 / 32), b = i % (KC1 / 32);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            sd[(s * (BM + BN) + r) * (KC1 / 32) + b] =
                ok ? __half2float((is_w ? W.d : xd)[(size_t) row * KB + kc * (KC1 / 32) + b]) : 0.f;
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[WM / 16][WN / 8][4] = {};
    const int NK = K / KC1;
    issue(0, 0);
    for (int kc = 0; kc < NK; ++kc) {
        const int s = kc & 1;
        if (kc + 1 < NK) { issue(kc + 1, s ^ 1); asm volatile("cp.async.wait_group 1;\n" ::); }
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        const int8 * A = srow(s, wm);
        const int8 * B = srow(s, BM + wn);
#pragma unroll
        for (int kb = 0; kb < KC1 / 32; ++kb) {
            uint32_t a[WM / 16][4], b[WN / 8][2];
#pragma unroll
            for (int m = 0; m < WM / 16; ++m) {
                const int8 * p = A + (m * 16 + g) * SK1 + kb * 32 + 4 * q;
                a[m][0] = *(const uint32_t *) p;
                a[m][1] = *(const uint32_t *) (p + 8 * SK1);
                a[m][2] = *(const uint32_t *) (p + 16);
                a[m][3] = *(const uint32_t *) (p + 8 * SK1 + 16);
            }
#pragma unroll
            for (int n = 0; n < WN / 8; ++n) {
                const int8 * p = B + (n * 8 + g) * SK1 + kb * 32 + 4 * q;
                b[n][0] = *(const uint32_t *) p;
                b[n][1] = *(const uint32_t *) (p + 16);
            }
            float dw[WM / 16][2], dx[WN / 8][2];
#pragma unroll
            for (int m = 0; m < WM / 16; ++m) {
                dw[m][0] = sd[(s * (BM + BN) + wm + m * 16 + g) * (KC1 / 32) + kb];
                dw[m][1] = sd[(s * (BM + BN) + wm + m * 16 + g + 8) * (KC1 / 32) + kb];
            }
#pragma unroll
            for (int n = 0; n < WN / 8; ++n) {
                dx[n][0] = sd[(s * (BM + BN) + BM + wn + n * 8 + 2 * q) * (KC1 / 32) + kb];
                dx[n][1] = sd[(s * (BM + BN) + BM + wn + n * 8 + 2 * q + 1) * (KC1 / 32) + kb];
            }
#pragma unroll
            for (int m = 0; m < WM / 16; ++m)
#pragma unroll
                for (int n = 0; n < WN / 8; ++n) {
                    int c[4];
                    mma_s8(a[m], b[n], c);
                    acc[m][n][0] += (float) c[0] * (dw[m][0] * dx[n][0]);
                    acc[m][n][1] += (float) c[1] * (dw[m][0] * dx[n][1]);
                    acc[m][n][2] += (float) c[2] * (dw[m][1] * dx[n][0]);
                    acc[m][n][3] += (float) c[3] * (dw[m][1] * dx[n][1]);
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

// ---------------------------------------------------------------- V2: scale hoist (both kb's scales to regs before mma)
__global__ __launch_bounds__(THREADS) void gemm_v2(Q8Matrix W, const int8 * __restrict__ xq,
                                                   const half * __restrict__ xd, int rows, float * __restrict__ y)
{
    __shared__ __align__(16) int8 sq[2][STAGE];
    __shared__ float sd[2][BM + BN][KC / 32];
    const int o0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    const int K = W.in, KB = K / 32;
    const int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = (wid / (BN / WN)) * WM, wn = (wid % (BN / WN)) * WN;
    const int g = lane >> 2, q = lane & 3;
    auto issue = [&](int kc, int s) {
        for (int i = threadIdx.x; i < (BM + BN) * (KC / 16); i += THREADS) {
            const int r = i / (KC / 16), v = i % (KC / 16);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            const int8 * src = (is_w ? W.q : xq) + (size_t) (ok ? row : 0) * K + kc * KC + v * 16;
            cp16(&sq[s][r * SK + v * 16], src, ok);
        }
        for (int i = threadIdx.x; i < (BM + BN) * (KC / 32); i += THREADS) {
            const int r = i / (KC / 32), b = i % (KC / 32);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            sd[s][r][b] = ok ? __half2float((is_w ? W.d : xd)[(size_t) row * KB + kc * (KC / 32) + b]) : 0.f;
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[WM / 16][WN / 8][4] = {};
    const int NK = K / KC;
    issue(0, 0);
    for (int kc = 0; kc < NK; ++kc) {
        const int s = kc & 1;
        if (kc + 1 < NK) { issue(kc + 1, s ^ 1); asm volatile("cp.async.wait_group 1;\n" ::); }
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        const int8 * A = sq[s] + wm * SK;
        const int8 * B = sq[s] + (BM + wn) * SK;
        float dw[KC / 32][WM / 16][2], dx[KC / 32][WN / 8][2];
#pragma unroll
        for (int kb = 0; kb < KC / 32; ++kb) {
#pragma unroll
            for (int m = 0; m < WM / 16; ++m) {
                dw[kb][m][0] = sd[s][wm + m * 16 + g][kb];
                dw[kb][m][1] = sd[s][wm + m * 16 + g + 8][kb];
            }
#pragma unroll
            for (int n = 0; n < WN / 8; ++n) {
                dx[kb][n][0] = sd[s][BM + wn + n * 8 + 2 * q][kb];
                dx[kb][n][1] = sd[s][BM + wn + n * 8 + 2 * q + 1][kb];
            }
        }
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
#pragma unroll
            for (int m = 0; m < WM / 16; ++m)
#pragma unroll
                for (int n = 0; n < WN / 8; ++n) {
                    int c[4];
                    mma_s8(a[m], b[n], c);
                    acc[m][n][0] += (float) c[0] * (dw[kb][m][0] * dx[kb][n][0]);
                    acc[m][n][1] += (float) c[1] * (dw[kb][m][0] * dx[kb][n][1]);
                    acc[m][n][2] += (float) c[2] * (dw[kb][m][1] * dx[kb][n][0]);
                    acc[m][n][3] += (float) c[3] * (dw[kb][m][1] * dx[kb][n][1]);
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

// magic int->float: exact for |c| < 2^22 (|c| <= 32*127*127 = 516096 here)
__device__ __forceinline__ float magic_f32(int c) { return __int_as_float(0x4B400000 + c) - 12582912.f; }

__global__ void magic_proof(int * bad)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = i <= 516096 ? i : -(i - 516096);            // sweeps [-516096, +516096]
    if (i <= 2 * 516096 && __float_as_uint(magic_f32(c)) != __float_as_uint((float) c)) atomicAdd(bad, 1);
}

// ---------------------------------------------------------------- V3: baseline + magic conversion
__global__ __launch_bounds__(THREADS) void gemm_v3(Q8Matrix W, const int8 * __restrict__ xq,
                                                   const half * __restrict__ xd, int rows, float * __restrict__ y)
{
    __shared__ __align__(16) int8 sq[2][STAGE];
    __shared__ float sd[2][BM + BN][KC / 32];
    const int o0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    const int K = W.in, KB = K / 32;
    const int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = (wid / (BN / WN)) * WM, wn = (wid % (BN / WN)) * WN;
    const int g = lane >> 2, q = lane & 3;
    auto issue = [&](int kc, int s) {
        for (int i = threadIdx.x; i < (BM + BN) * (KC / 16); i += THREADS) {
            const int r = i / (KC / 16), v = i % (KC / 16);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            const int8 * src = (is_w ? W.q : xq) + (size_t) (ok ? row : 0) * K + kc * KC + v * 16;
            cp16(&sq[s][r * SK + v * 16], src, ok);
        }
        for (int i = threadIdx.x; i < (BM + BN) * (KC / 32); i += THREADS) {
            const int r = i / (KC / 32), b = i % (KC / 32);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            sd[s][r][b] = ok ? __half2float((is_w ? W.d : xd)[(size_t) row * KB + kc * (KC / 32) + b]) : 0.f;
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[WM / 16][WN / 8][4] = {};
    const int NK = K / KC;
    issue(0, 0);
    for (int kc = 0; kc < NK; ++kc) {
        const int s = kc & 1;
        if (kc + 1 < NK) { issue(kc + 1, s ^ 1); asm volatile("cp.async.wait_group 1;\n" ::); }
        else asm volatile("cp.async.wait_group 0;\n" ::);
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
            float dw[WM / 16][2], dx[WN / 8][2];
#pragma unroll
            for (int m = 0; m < WM / 16; ++m) {
                dw[m][0] = sd[s][wm + m * 16 + g][kb];
                dw[m][1] = sd[s][wm + m * 16 + g + 8][kb];
            }
#pragma unroll
            for (int n = 0; n < WN / 8; ++n) {
                dx[n][0] = sd[s][BM + wn + n * 8 + 2 * q][kb];
                dx[n][1] = sd[s][BM + wn + n * 8 + 2 * q + 1][kb];
            }
#pragma unroll
            for (int m = 0; m < WM / 16; ++m)
#pragma unroll
                for (int n = 0; n < WN / 8; ++n) {
                    int c[4];
                    mma_s8(a[m], b[n], c);
                    acc[m][n][0] += magic_f32(c[0]) * (dw[m][0] * dx[n][0]);
                    acc[m][n][1] += magic_f32(c[1]) * (dw[m][0] * dx[n][1]);
                    acc[m][n][2] += magic_f32(c[2]) * (dw[m][1] * dx[n][0]);
                    acc[m][n][3] += magic_f32(c[3]) * (dw[m][1] * dx[n][1]);
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

// ---------------------------------------------------------------- V4: BN=256 (512 threads, 16 warps 2x8, WN=32)
constexpr int BN2 = 256, THREADS2 = 512, SK2 = KC + 16, STAGE2 = (BM + BN2) * SK2;
__global__ __launch_bounds__(THREADS2) void gemm_v4(Q8Matrix W, const int8 * __restrict__ xq,
                                                    const half * __restrict__ xd, int rows, float * __restrict__ y)
{
    __shared__ __align__(16) int8 sq[2][STAGE2];
    __shared__ float sd[2][BM + BN2][KC / 32];
    const int o0 = blockIdx.x * BM, t0 = blockIdx.y * BN2;
    const int K = W.in, KB = K / 32;
    const int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = (wid / (BN2 / WN)) * WM, wn = (wid % (BN2 / WN)) * WN;
    const int g = lane >> 2, q = lane & 3;
    auto issue = [&](int kc, int s) {
        for (int i = threadIdx.x; i < (BM + BN2) * (KC / 16); i += THREADS2) {
            const int r = i / (KC / 16), v = i % (KC / 16);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            const int8 * src = (is_w ? W.q : xq) + (size_t) (ok ? row : 0) * K + kc * KC + v * 16;
            cp16(&sq[s][r * SK2 + v * 16], src, ok);
        }
        for (int i = threadIdx.x; i < (BM + BN2) * (KC / 32); i += THREADS2) {
            const int r = i / (KC / 32), b = i % (KC / 32);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            sd[s][r][b] = ok ? __half2float((is_w ? W.d : xd)[(size_t) row * KB + kc * (KC / 32) + b]) : 0.f;
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[WM / 16][WN / 8][4] = {};
    const int NK = K / KC;
    issue(0, 0);
    for (int kc = 0; kc < NK; ++kc) {
        const int s = kc & 1;
        if (kc + 1 < NK) { issue(kc + 1, s ^ 1); asm volatile("cp.async.wait_group 1;\n" ::); }
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        const int8 * A = sq[s] + wm * SK2;
        const int8 * B = sq[s] + (BM + wn) * SK2;
#pragma unroll
        for (int kb = 0; kb < KC / 32; ++kb) {
            uint32_t a[WM / 16][4], b[WN / 8][2];
#pragma unroll
            for (int m = 0; m < WM / 16; ++m) {
                const int8 * p = A + (m * 16 + g) * SK2 + kb * 32 + 4 * q;
                a[m][0] = *(const uint32_t *) p;
                a[m][1] = *(const uint32_t *) (p + 8 * SK2);
                a[m][2] = *(const uint32_t *) (p + 16);
                a[m][3] = *(const uint32_t *) (p + 8 * SK2 + 16);
            }
#pragma unroll
            for (int n = 0; n < WN / 8; ++n) {
                const int8 * p = B + (n * 8 + g) * SK2 + kb * 32 + 4 * q;
                b[n][0] = *(const uint32_t *) p;
                b[n][1] = *(const uint32_t *) (p + 16);
            }
            float dw[WM / 16][2], dx[WN / 8][2];
#pragma unroll
            for (int m = 0; m < WM / 16; ++m) {
                dw[m][0] = sd[s][wm + m * 16 + g][kb];
                dw[m][1] = sd[s][wm + m * 16 + g + 8][kb];
            }
#pragma unroll
            for (int n = 0; n < WN / 8; ++n) {
                dx[n][0] = sd[s][BM + wn + n * 8 + 2 * q][kb];
                dx[n][1] = sd[s][BM + wn + n * 8 + 2 * q + 1][kb];
            }
#pragma unroll
            for (int m = 0; m < WM / 16; ++m)
#pragma unroll
                for (int n = 0; n < WN / 8; ++n) {
                    int c[4];
                    mma_s8(a[m], b[n], c);
                    acc[m][n][0] += (float) c[0] * (dw[m][0] * dx[n][0]);
                    acc[m][n][1] += (float) c[1] * (dw[m][0] * dx[n][1]);
                    acc[m][n][2] += (float) c[2] * (dw[m][1] * dx[n][0]);
                    acc[m][n][3] += (float) c[3] * (dw[m][1] * dx[n][1]);
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

// ---------------------------------------------------------------- V5: BN=256 + magic
__global__ __launch_bounds__(THREADS2) void gemm_v5(Q8Matrix W, const int8 * __restrict__ xq,
                                                    const half * __restrict__ xd, int rows, float * __restrict__ y)
{
    __shared__ __align__(16) int8 sq[2][STAGE2];
    __shared__ float sd[2][BM + BN2][KC / 32];
    const int o0 = blockIdx.x * BM, t0 = blockIdx.y * BN2;
    const int K = W.in, KB = K / 32;
    const int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = (wid / (BN2 / WN)) * WM, wn = (wid % (BN2 / WN)) * WN;
    const int g = lane >> 2, q = lane & 3;
    auto issue = [&](int kc, int s) {
        for (int i = threadIdx.x; i < (BM + BN2) * (KC / 16); i += THREADS2) {
            const int r = i / (KC / 16), v = i % (KC / 16);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            const int8 * src = (is_w ? W.q : xq) + (size_t) (ok ? row : 0) * K + kc * KC + v * 16;
            cp16(&sq[s][r * SK2 + v * 16], src, ok);
        }
        for (int i = threadIdx.x; i < (BM + BN2) * (KC / 32); i += THREADS2) {
            const int r = i / (KC / 32), b = i % (KC / 32);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            sd[s][r][b] = ok ? __half2float((is_w ? W.d : xd)[(size_t) row * KB + kc * (KC / 32) + b]) : 0.f;
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[WM / 16][WN / 8][4] = {};
    const int NK = K / KC;
    issue(0, 0);
    for (int kc = 0; kc < NK; ++kc) {
        const int s = kc & 1;
        if (kc + 1 < NK) { issue(kc + 1, s ^ 1); asm volatile("cp.async.wait_group 1;\n" ::); }
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        const int8 * A = sq[s] + wm * SK2;
        const int8 * B = sq[s] + (BM + wn) * SK2;
#pragma unroll
        for (int kb = 0; kb < KC / 32; ++kb) {
            uint32_t a[WM / 16][4], b[WN / 8][2];
#pragma unroll
            for (int m = 0; m < WM / 16; ++m) {
                const int8 * p = A + (m * 16 + g) * SK2 + kb * 32 + 4 * q;
                a[m][0] = *(const uint32_t *) p;
                a[m][1] = *(const uint32_t *) (p + 8 * SK2);
                a[m][2] = *(const uint32_t *) (p + 16);
                a[m][3] = *(const uint32_t *) (p + 8 * SK2 + 16);
            }
#pragma unroll
            for (int n = 0; n < WN / 8; ++n) {
                const int8 * p = B + (n * 8 + g) * SK2 + kb * 32 + 4 * q;
                b[n][0] = *(const uint32_t *) p;
                b[n][1] = *(const uint32_t *) (p + 16);
            }
            float dw[WM / 16][2], dx[WN / 8][2];
#pragma unroll
            for (int m = 0; m < WM / 16; ++m) {
                dw[m][0] = sd[s][wm + m * 16 + g][kb];
                dw[m][1] = sd[s][wm + m * 16 + g + 8][kb];
            }
#pragma unroll
            for (int n = 0; n < WN / 8; ++n) {
                dx[n][0] = sd[s][BM + wn + n * 8 + 2 * q][kb];
                dx[n][1] = sd[s][BM + wn + n * 8 + 2 * q + 1][kb];
            }
#pragma unroll
            for (int m = 0; m < WM / 16; ++m)
#pragma unroll
                for (int n = 0; n < WN / 8; ++n) {
                    int c[4];
                    mma_s8(a[m], b[n], c);
                    acc[m][n][0] += magic_f32(c[0]) * (dw[m][0] * dx[n][0]);
                    acc[m][n][1] += magic_f32(c[1]) * (dw[m][0] * dx[n][1]);
                    acc[m][n][2] += magic_f32(c[2]) * (dw[m][1] * dx[n][0]);
                    acc[m][n][3] += magic_f32(c[3]) * (dw[m][1] * dx[n][1]);
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

// ---------------------------------------------------------------- driver
static uint32_t xorshift = 12345u;
static uint32_t rnd() { xorshift ^= xorshift << 13; xorshift ^= xorshift >> 17; xorshift ^= xorshift << 5; return xorshift; }

int main(int argc, char ** argv)
{
    const int iters = argc > 1 ? atoi(argv[1]) : 8;
    // exhaustive magic proof first (no shape work if it fails)
    int * bad; CK(cudaMalloc(&bad, 4)); CK(cudaMemset(bad, 0, 4));
    const int span = 2 * 516096 + 1;
    magic_proof<<<(span + 255) / 256, 256>>>(bad);
    int hbad = 0; CK(cudaMemcpy(&hbad, bad, 4, cudaMemcpyDeviceToHost)); CK(cudaGetLastError());
    printf("magic_proof: |c|<=516096 exhaustive mismatches = %d\n", hbad);
    if (hbad) { printf("MAGIC UNSAFE - abort\n"); return 1; }

    struct Shape { int M, N, K; const char * name; };
    const Shape shapes[] = {
        {8192, 10240, 2560, "hc_up   10240x2560"},
        {8192,  6144, 2560, "qkv      6144x2560"},
        {8192,  2560, 6144, "outproj  2560x6144"},
        {8192, 12288, 2560, "shared  12288x2560"},
        {8192,   640, 2560, "ple_mixer 640x2560"},
    };
    printf("%-22s %8s  %s\n", "shape", "ms", "TOPS (variant: bit-identical?)");
    for (const auto & S : shapes) {
        const size_t wq = (size_t) S.N * S.K, xq = (size_t) S.M * S.K;
        int8 * Wq, * Xq; half * Wd, * Xd; float * yref, * yvar;
        CK(cudaMalloc(&Wq, wq)); CK(cudaMalloc(&Xq, xq));
        CK(cudaMalloc(&Wd, (size_t) S.N * S.K / 32 * 2)); CK(cudaMalloc(&Xd, (size_t) S.M * S.K / 32 * 2));
        CK(cudaMalloc(&yref, (size_t) S.M * S.N * 4)); CK(cudaMalloc(&yvar, (size_t) S.M * S.N * 4));
        {
            std::vector<int8> hw(wq), hx(xq);
            for (size_t i = 0; i < wq; ++i) hw[i] = (int8) (rnd() % 255 - 127);
            for (size_t i = 0; i < xq; ++i) hx[i] = (int8) (rnd() % 255 - 127);
            CK(cudaMemcpy(Wq, hw.data(), wq, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(Xq, hx.data(), xq, cudaMemcpyHostToDevice));
            std::vector<half> hd(S.N * S.K / 32), hxd(S.M * S.K / 32);
            for (auto & v : hd) v = __float2half(0.001f + 0.004f * (rnd() % 1000) / 1000.f);
            for (auto & v : hxd) v = __float2half(0.001f + 0.004f * (rnd() % 1000) / 1000.f);
            CK(cudaMemcpy(Wd, hd.data(), hd.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(Xd, hxd.data(), hxd.size() * 2, cudaMemcpyHostToDevice));
        }
        Q8Matrix W{Wq, Wd, S.K, S.N};
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        double flops = 2.0 * S.M * S.N * S.K;
        auto run = [&](int which) {
            const dim3 g((S.N + BM - 1) / BM, (S.M + ((which >= 4) ? BN2 : BN) - 1) / ((which >= 4) ? BN2 : BN));
            switch (which) {
            case 0: gemm_base<<<g, THREADS>>>(W, Xq, Xd, S.M, yvar); break;
            case 1: gemm_v1<<<g, THREADS, 2 * STAGE1 + 2 * (BM + BN) * (KC1 / 32) * 4>>>(W, Xq, Xd, S.M, yvar); break;
            case 2: gemm_v2<<<g, THREADS>>>(W, Xq, Xd, S.M, yvar); break;
            case 3: gemm_v3<<<g, THREADS>>>(W, Xq, Xd, S.M, yvar); break;
            case 4: gemm_v4<<<g, THREADS2>>>(W, Xq, Xd, S.M, yvar); break;
            case 5: gemm_v5<<<g, THREADS2>>>(W, Xq, Xd, S.M, yvar); break;
            }
        };
        CK(cudaFuncSetAttribute(gemm_v1, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * STAGE1 + 2 * (BM + BN) * (KC1 / 32) * 4));
        // reference output (baseline is deterministic)
        CK(cudaMemset(yref, 0, (size_t) S.M * S.N * 4));
        run(0); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(yref, yvar, (size_t) S.M * S.N * 4, cudaMemcpyDeviceToDevice));
        float base_ms = 0;
        for (int v = 0; v < 6; ++v) {
            for (int w = 0; w < 2; ++w) run(v);
            CK(cudaDeviceSynchronize());
            cudaEventRecord(e0);
            for (int i = 0; i < iters; ++i) run(v);
            cudaEventRecord(e1); CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());
            float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= iters;
            bool ident = true;
            CK(cudaMemset(yvar + 0, 0, 0));  // no-op touch
            run(v); CK(cudaDeviceSynchronize());
            std::vector<float> a((size_t) S.M * S.N), b((size_t) S.M * S.N);
            CK(cudaMemcpy(a.data(), yvar, a.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), yref, b.size() * 4, cudaMemcpyDeviceToHost));
            size_t mism = 0;
            for (size_t i = 0; i < a.size(); ++i) if (memcmp(&a[i], &b[i], 4)) ++mism;
            ident = mism == 0;
            if (v == 0) base_ms = ms;
            printf("%-22s v%d %7.2f ms %6.1f TOPS  %s%s\n", S.name, v, ms, flops / (ms * 1e-3) / 1e12,
                   ident ? "BIT-IDENT" : "DIFFERS!", v ? (ms < 0.9 * base_ms ? " -gain" : (ms > 1.1 * base_ms ? " LOSS" : "")) : "");
        }
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaFree(Wq); cudaFree(Xq); cudaFree(Wd); cudaFree(Xd); cudaFree(yref); cudaFree(yvar);
    }
    return 0;
}
