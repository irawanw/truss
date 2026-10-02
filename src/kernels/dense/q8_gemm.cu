// Dense Q8 GEMM (see q8_gemm.cuh).
//
// Block tile: 128 weight rows (outputs) x 128 activation rows (tokens), k chunks of 64 (two Q8 blocks), double
// buffered with cp.async. 8 warps as 2 (outputs) x 4 (tokens): a warp computes 64 outputs x 32 tokens = 4 x 4 mma
// tiles, weights as the A operand (m16 x k32), activations as B (k32 x n8). Per Q8 block the int32 tile results are
// scaled and folded into fp32 accumulators: acc += float(c) * d_w * d_x.
#include "q8_gemm.cuh"

#include "core/cublas_check.h"
#include "core/cuda_check.h"

#include <cublas_v2.h>

#include <stdexcept>
#include <string>

namespace truss::dense {
namespace {

constexpr int THREADS = 256;
constexpr int BM = 128, BN = 128, KC = 64;     // outputs, tokens, k per stage
constexpr int WM = 64, WN = 32;                // warp tile
constexpr int SK = KC + 16;                    // shared row stride (bytes): rows g = 0..7 x k words q hit 32 banks
constexpr int STAGE = (BM + BN) * SK;          // bytes of int8 per stage
static_assert((BM / WM) * (BN / WN) * 32 == THREADS, "8 warps");

struct BlockQ8_0 {
    half d;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34);

__global__ void repack_kernel(const BlockQ8_0 * __restrict__ b, int nb, int8_t * __restrict__ q, half * __restrict__ d)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // block index (row-major over [out][in / 32])
    if (i >= nb) return;
    d[i] = b[i].d;
    for (int j = 0; j < 32; ++j) q[(size_t) i * 32 + j] = b[i].qs[j];
}

__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(half v) { return __half2float(v); }

// one warp per 32-block of a row
template <class X>
__global__ void quantize_kernel(const X * __restrict__ x, int n_blocks, int8_t * __restrict__ q, half * __restrict__ d)
{
    const int blk = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (blk >= n_blocks) return;
    const float v = to_f32(x[(size_t) blk * 32 + lane]);
    float amax = fabsf(v);
    for (int m = 16; m; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
    const float s = amax / 127.f;
    q[(size_t) blk * 32 + lane] = (int8_t) (amax == 0.f ? 0 : (int) roundf(v / s));
    if (lane == 0) d[blk] = __float2half(s);
}

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

__global__ __launch_bounds__(THREADS) void gemm_kernel(Q8Matrix W, const int8_t * __restrict__ xq,
                                                       const half * __restrict__ xd, int rows, float * __restrict__ y)
{
    __shared__ __align__(16) int8_t sq[2][STAGE];            // [stage][W rows | x rows][SK]
    __shared__ float sd[2][BM + BN][KC / 32];                 // scales of the stage's two blocks

    const int o0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    const int K = W.in, KB = K / 32;
    const int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = (wid / (BN / WN)) * WM, wn = (wid % (BN / WN)) * WN;
    const int g = lane >> 2, q = lane & 3;

    auto issue = [&](int kc, int s) {
        // 16-byte vectors: (BM + BN) rows x KC / 16
        for (int i = threadIdx.x; i < (BM + BN) * (KC / 16); i += THREADS) {
            const int r = i / (KC / 16), v = i % (KC / 16);
            const bool is_w = r < BM;
            const int row = is_w ? o0 + r : t0 + r - BM;
            const bool ok = row < (is_w ? W.out : rows);
            const int8_t * src = (is_w ? W.q : xq) + (size_t) (ok ? row : 0) * K + kc * KC + v * 16;
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
        if (kc + 1 < NK) {
            issue(kc + 1, s ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const int8_t * A = sq[s] + wm * SK;          // this warp's weight rows
        const int8_t * B = sq[s] + (BM + wn) * SK;   // this warp's token rows
#pragma unroll
        for (int kb = 0; kb < KC / 32; ++kb) {
            uint32_t a[WM / 16][4], b[WN / 8][2];
#pragma unroll
            for (int m = 0; m < WM / 16; ++m) {
                const int8_t * p = A + (m * 16 + g) * SK + kb * 32 + 4 * q;
                a[m][0] = *(const uint32_t *) p;
                a[m][1] = *(const uint32_t *) (p + 8 * SK);
                a[m][2] = *(const uint32_t *) (p + 16);
                a[m][3] = *(const uint32_t *) (p + 8 * SK + 16);
            }
#pragma unroll
            for (int n = 0; n < WN / 8; ++n) {
                const int8_t * p = B + (n * 8 + g) * SK + kb * 32 + 4 * q;
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
                    // c: (output g, tokens 2q, 2q+1), (output g + 8, same tokens)
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

// ---------------------------------------------------------------------------------------------------------
// W8A16: dequantize to fp16 scratch, then cuBLAS (fp16 in, fp32 accumulate). An own fused kernel (int8 staged,
// converted in shared memory, per-block fp32 fold) reached 38-42 TFLOPS vs cuBLAS 57-66 (TRACKER #44); the dequant
// pass is ~1.5% of a layer's dense time at 8K tokens.

__global__ void dequant_kernel(const int8_t * __restrict__ q, const half * __restrict__ d, size_t n, half * __restrict__ w)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;   // 4 weights per thread
    if (i * 4 >= n) return;
    const char4 v = *(const char4 *) (q + i * 4);
    const float s = __half2float(d[i * 4 / 32]);
    half2 * o = (half2 *) (w + i * 4);
    o[0] = __floats2half2_rn(v.x * s, v.y * s);
    o[1] = __floats2half2_rn(v.z * s, v.w * s);
}

// one block per row
__global__ void rows_kernel(const int8_t * __restrict__ q, const half * __restrict__ d, const int * ids, int in,
                            float * out)
{
    const size_t r = (size_t) ids[blockIdx.x];
    for (int i = threadIdx.x; i < in; i += blockDim.x)
        out[(size_t) blockIdx.x * in + i] = q[r * in + i] * __half2float(d[r * (in / 32) + i / 32]);
}

// ---- multi-matrix gemv (decode): several matrices that read the same activations, one launch (TRACKER #83) ----
// D0 measured 2,672 launches per pass and small projections far below bandwidth: hc inject (10240 -> 4) one block
// of 4 warps, 9.4 us; hc up (320 inputs = 10 blocks) 22 of 32 lanes idle; hc down (324 outputs) too few warps.
// Mapping, from the input width only (never the row count or the other matrices of the launch, so a row's result
// does not depend on the window size or on what it is fused with):
//   LPO lanes per output (in / 32 blocks < 32: 4, 8 or 16; several outputs per warp), lane sl takes blocks
//   sl, sl + LPO, ...; reduction by shuffles inside the lane group;
//   WPO warps per output (few outputs, long rows: 2 or 4), warp wi takes blocks lane + 32 wi, step 32 WPO;
//   partial sums reduced in shared memory in warp order. Fixed orders throughout.
struct MultiQ8 {
    const int8_t * q[MULTI_MAX];
    const half * d[MULTI_MAX];
    float * y[MULTI_MAX];
    int start[MULTI_MAX + 1];   // output prefix sums
    int n;
};

template <int ROWS, int LPO, int WPO>
__global__ void __launch_bounds__(128) gemv_multi_kernel(MultiQ8 m, int in, const int8_t * __restrict__ xq,
                                                         const half * __restrict__ xd)
{
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nb = in / 32;
    int o, sl, step;
    if (WPO == 1) o = blockIdx.x * (4 * 32 / LPO) + warp * (32 / LPO) + lane / LPO, sl = lane % LPO, step = LPO;
    else o = blockIdx.x * (4 / WPO) + warp / WPO, sl = lane + 32 * (warp % WPO), step = 32 * WPO;
    const bool valid = o < m.start[m.n];
    int seg = 0;
    if (valid)
        while (o >= m.start[seg + 1]) ++seg;
    const int oo = valid ? o - m.start[seg] : 0;
    float acc[ROWS] = {};
    if (valid) {
        const int8_t * q = m.q[seg] + (size_t) oo * in;
        const half * d = m.d[seg] + (size_t) oo * nb;
        for (int b = sl; b < nb; b += step) {
            const int4 * wp = reinterpret_cast<const int4 *>(q + b * 32);
            const int4 w0 = __ldg(wp), w1 = __ldg(wp + 1);
            const float dw = __half2float(d[b]);
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int4 * xp = reinterpret_cast<const int4 *>(xq + (size_t) r * in + b * 32);
                const int4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
                int s = 0;
                s = __dp4a(w0.x, x0.x, s), s = __dp4a(w0.y, x0.y, s), s = __dp4a(w0.z, x0.z, s), s = __dp4a(w0.w, x0.w, s);
                s = __dp4a(w1.x, x1.x, s), s = __dp4a(w1.y, x1.y, s), s = __dp4a(w1.z, x1.z, s), s = __dp4a(w1.w, x1.w, s);
                acc[r] += (float) s * (dw * __half2float(xd[(size_t) r * nb + b]));
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
        for (int k = (WPO == 1 ? LPO : 32) / 2; k; k >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], k);
    if (WPO == 1) {
        if (valid && sl == 0)
#pragma unroll
            for (int r = 0; r < ROWS; ++r) m.y[seg][(size_t) r * (m.start[seg + 1] - m.start[seg]) + oo] = acc[r];
        return;
    }
    __shared__ float part[4][ROWS];
    if (lane == 0)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) part[warp][r] = acc[r];
    __syncthreads();
    if (valid && lane == 0 && warp % WPO == 0)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float v = part[warp][r];
            for (int k = 1; k < WPO; ++k) v += part[warp + k][r];
            m.y[seg][(size_t) r * (m.start[seg + 1] - m.start[seg]) + oo] = v;
        }
}

template <int ROWS>
void gemv_multi_rows(const MultiQ8 & m, int in, const int8_t * xq, const half * xd, cudaStream_t stream)
{
    const int nb = in / 32, outs = m.start[m.n];
    // long rows get 2 or 4 warps per output: the long-row matrices of this model have few outputs (hc down 10240 ->
    // 324: 1,296 warps at 4; out-proj 6144 -> 2560), and one warp per output leaves the 82 SMs short of warps
    const int wpo = nb >= 256 ? 4 : nb >= 128 ? 2 : 1;
    const int lpo = nb >= 32 ? 32 : nb > 8 ? 16 : nb > 4 ? 8 : 4;
    if (lpo < 32) {
        const int per = 4 * 32 / lpo, grid = (outs + per - 1) / per;
        if (lpo == 16) gemv_multi_kernel<ROWS, 16, 1><<<grid, 128, 0, stream>>>(m, in, xq, xd);
        else if (lpo == 8) gemv_multi_kernel<ROWS, 8, 1><<<grid, 128, 0, stream>>>(m, in, xq, xd);
        else gemv_multi_kernel<ROWS, 4, 1><<<grid, 128, 0, stream>>>(m, in, xq, xd);
    } else if (wpo == 1) gemv_multi_kernel<ROWS, 32, 1><<<(outs + 3) / 4, 128, 0, stream>>>(m, in, xq, xd);
    else if (wpo == 2) gemv_multi_kernel<ROWS, 32, 2><<<(outs + 1) / 2, 128, 0, stream>>>(m, in, xq, xd);
    else gemv_multi_kernel<ROWS, 32, 4><<<outs, 128, 0, stream>>>(m, in, xq, xd);
}

// fp32 weights: lane l reads float4 l, l + 32, ... (+ 32 * 32 wi with WPO warps per output); rows in groups of 8
struct MultiF32 {
    const float * W[MULTI_MAX];
    float * y[MULTI_MAX];
    int start[MULTI_MAX + 1];
    int n;
};

template <int WPO>
__global__ void __launch_bounds__(128) f32_gemv_multi_kernel(MultiF32 m, int in, const float * __restrict__ x, int rows)
{
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * (4 / WPO) + warp / WPO, wi = warp % WPO;
    const bool valid = o < m.start[m.n];
    int seg = 0;
    if (valid)
        while (o >= m.start[seg + 1]) ++seg;
    const int oo = valid ? o - m.start[seg] : 0, out = valid ? m.start[seg + 1] - m.start[seg] : 0;
    const float4 * w = reinterpret_cast<const float4 *>(m.W[seg] + (size_t) oo * in);
    __shared__ float part[4][8];
    for (int r0 = 0; r0 < rows; r0 += 8) {
        const int R = min(8, rows - r0);
        float acc[8] = {};
        if (valid)
            for (int i = lane + 32 * wi; i < in / 4; i += 32 * WPO) {
                const float4 a = __ldg(w + i);
#pragma unroll
                for (int r = 0; r < 8; ++r)
                    if (r < R) {
                        const float4 b = reinterpret_cast<const float4 *>(x + (size_t) (r0 + r) * in)[i];
                        acc[r] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                    }
            }
#pragma unroll
        for (int r = 0; r < 8; ++r)
            for (int k = 16; k; k >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], k);
        if (WPO == 1) {
#pragma unroll
            for (int r = 0; r < 8; ++r)
                if (valid && r < R && lane == 0) m.y[seg][(size_t) (r0 + r) * out + oo] = acc[r];
            continue;
        }
        if (lane == 0)
#pragma unroll
            for (int r = 0; r < 8; ++r) part[warp][r] = acc[r];
        __syncthreads();
        if (valid && lane == 0 && wi == 0)
#pragma unroll
            for (int r = 0; r < 8; ++r)
                if (r < R) {
                    float v = part[warp][r];
                    for (int k = 1; k < WPO; ++k) v += part[warp + k][r];
                    m.y[seg][(size_t) (r0 + r) * out + oo] = v;
                }
        __syncthreads();
    }
}

__global__ void gather_kernel(const int8_t * __restrict__ q, const half * __restrict__ d, const int * ids, int in,
                              int8_t * __restrict__ oq, half * __restrict__ od)
{
    const size_t r = (size_t) ids[blockIdx.x], o = blockIdx.x;
    for (int i = threadIdx.x; i < in; i += blockDim.x) oq[o * in + i] = q[r * in + i];
    for (int i = threadIdx.x; i < in / 32; i += blockDim.x) od[o * (in / 32) + i] = d[r * (in / 32) + i];
}

}  // namespace

void q8_gather(const Q8Matrix & W, const int * ids, int n, int8_t * q, half * d, cudaStream_t stream)
{
    gather_kernel<<<n, 256, 0, stream>>>(W.q, W.d, ids, W.in, q, d);
    TRUSS_CUDA(cudaGetLastError());
}

void f32_gemv(const float * W, int in, int out, const float * x, int rows, float * y, cudaStream_t stream)
{
    f32_gemv_multi(&W, &out, &y, 1, in, x, rows, stream);   // one kernel for the router, fused or alone
}

void q8_gemv(const Q8Matrix & W, const int8_t * xq, const half * xd, int rows, float * y, cudaStream_t stream)
{
    if (W.in % 32) throw std::runtime_error("q8_gemv: in must be a multiple of 32");
    const Q8Matrix * w = &W;
    q8_gemv_multi(&w, &y, 1, xq, xd, rows, stream);   // its lane/warp mapping (short rows, few outputs)
}

void q8_rows(const Q8Matrix & W, const int * ids, int n, float * out, cudaStream_t stream)
{
    rows_kernel<<<n, 256, 0, stream>>>(W.q, W.d, ids, W.in, out);
    TRUSS_CUDA(cudaGetLastError());
}

void q8_gemm_a16(const Q8Matrix & W, const half * x, int rows, float * y, half * w16, cublasHandle_t cublas,
                 cudaStream_t stream)
{
    if (W.in % 64) throw std::runtime_error("q8_gemm_a16: in must be a multiple of 64, got " + std::to_string(W.in));
    const size_t n = (size_t) W.in * W.out;
    dequant_kernel<<<(unsigned) ((n / 4 + 255) / 256), 256, 0, stream>>>(W.q, W.d, n, w16);
    TRUSS_CUDA(cudaGetLastError());
    // column-major view: W^T is (in x out, ld in), x^T (in x rows, ld in), y^T (out x rows, ld out)
    const float one = 1.f, zero = 0.f;
    TRUSS_CUBLAS(cublasSetStream(cublas, stream));
    TRUSS_CUBLAS(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, W.out, rows, W.in, &one, w16, CUDA_R_16F, W.in, x,
                              CUDA_R_16F, W.in, &zero, y, CUDA_R_32F, W.out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void q8_repack(const void * blocks, int in, int out, int8_t * q, half * d, cudaStream_t stream)
{
    if (in % 64) throw std::runtime_error("q8_repack: in must be a multiple of 64, got " + std::to_string(in));
    const int nb = in / 32 * out;
    repack_kernel<<<(nb + 255) / 256, 256, 0, stream>>>((const BlockQ8_0 *) blocks, nb, q, d);
    TRUSS_CUDA(cudaGetLastError());
}

template <class X> static void quantize_act(const X * x, int rows, int in, int8_t * xq, half * xd, cudaStream_t stream)
{
    if (in % 32) throw std::runtime_error("q8_quantize_act: in must be a multiple of 32");
    const int nb = rows * (in / 32);
    quantize_kernel<X><<<(nb * 32 + 255) / 256, 256, 0, stream>>>(x, nb, xq, xd);
    TRUSS_CUDA(cudaGetLastError());
}

void q8_quantize_act(const float * x, int rows, int in, int8_t * xq, half * xd, cudaStream_t stream)
{
    quantize_act(x, rows, in, xq, xd, stream);
}

void q8_quantize_act(const half * x, int rows, int in, int8_t * xq, half * xd, cudaStream_t stream)
{
    quantize_act(x, rows, in, xq, xd, stream);
}

void q8_gemm(const Q8Matrix & W, const int8_t * xq, const half * xd, int rows, float * y, cudaStream_t stream)
{
    if (W.in % KC) throw std::runtime_error("q8_gemm: in must be a multiple of 64, got " + std::to_string(W.in));
    const dim3 grid((W.out + BM - 1) / BM, (rows + BN - 1) / BN);
    gemm_kernel<<<grid, THREADS, 0, stream>>>(W, xq, xd, rows, y);
    TRUSS_CUDA(cudaGetLastError());
}

void q8_gemv_multi(const Q8Matrix * const * W, float * const * y, int n, const int8_t * xq, const half * xd, int rows,
                   cudaStream_t stream)
{
    if (n < 1 || n > MULTI_MAX) throw std::runtime_error("q8_gemv_multi: 1.." + std::to_string(MULTI_MAX) + " matrices");
    MultiQ8 m{};
    m.n = n, m.start[0] = 0;
    for (int i = 0; i < n; ++i) {
        if (W[i]->in != W[0]->in || W[i]->in % 32) throw std::runtime_error("q8_gemv_multi: same in, multiple of 32");
        m.q[i] = W[i]->q, m.d[i] = W[i]->d, m.y[i] = y[i], m.start[i + 1] = m.start[i] + W[i]->out;
    }
    const int in = W[0]->in;
    switch (rows) {
    case 1: gemv_multi_rows<1>(m, in, xq, xd, stream); break;
    case 2: gemv_multi_rows<2>(m, in, xq, xd, stream); break;
    case 3: gemv_multi_rows<3>(m, in, xq, xd, stream); break;
    case 4: gemv_multi_rows<4>(m, in, xq, xd, stream); break;
    case 5: gemv_multi_rows<5>(m, in, xq, xd, stream); break;
    case 6: gemv_multi_rows<6>(m, in, xq, xd, stream); break;
    case 7: gemv_multi_rows<7>(m, in, xq, xd, stream); break;
    case 8: gemv_multi_rows<8>(m, in, xq, xd, stream); break;
    default: throw std::runtime_error("q8_gemv_multi: rows must be 1.." + std::to_string(GEMV_ROWS));
    }
    TRUSS_CUDA(cudaGetLastError());
}

void f32_gemv_multi(const float * const * W, const int * out, float * const * y, int n, int in, const float * x,
                    int rows, cudaStream_t stream)
{
    if (n < 1 || n > MULTI_MAX) throw std::runtime_error("f32_gemv_multi: 1.." + std::to_string(MULTI_MAX) + " matrices");
    if (in % 128) throw std::runtime_error("f32_gemv_multi: in must be a multiple of 128");
    MultiF32 m{};
    m.n = n, m.start[0] = 0;
    for (int i = 0; i < n; ++i) m.W[i] = W[i], m.y[i] = y[i], m.start[i + 1] = m.start[i] + out[i];
    const int outs = m.start[n];   // warps per output from `in` only (router 2560 -> 512: 2), as q8_gemv_multi
    if (in < 2048) f32_gemv_multi_kernel<1><<<(outs + 3) / 4, 128, 0, stream>>>(m, in, x, rows);
    else f32_gemv_multi_kernel<2><<<(outs + 1) / 2, 128, 0, stream>>>(m, in, x, rows);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::dense
