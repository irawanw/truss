// Dense Q8 decode GEMVs (see q8_gemm.cuh): several matrices that read the same activations in one launch (int8 or
// fp16 activations), the W8A16 tensor-core gemv with its split-k rule and fused epilogues (silu, hc up + collapse),
// and the row-invariant fp32 gemv the router uses. Prompt-sized GEMMs: q8_gemm.cu.
#include <cstdlib>
#include "q8_gemm.cuh"

#include "core/cublas_check.h"
#include "core/cuda_check.h"

#include <cublas_v2.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>


namespace truss::dense {


// warps per tile S and blocks per tile KB of the W8A16 gemv, from the shape only (never from `rows`, so a row's
// result does not depend on the window size). P = S * KB warps share a tile's k range: enough for ~1280 warps in
// all (the small matrices fill the 82 SMs: hc down 10240 -> 324 has 21 tiles) and at most ~12 Q8 blocks per warp, at
// least 2; S up to 16, the rest across KB blocks. Measured on the 3090 (tk-bench-gemv-split, 10-12): hc down 13.4 ->
// 8.6 us, shexp down 10.7 -> 5.0, shexp gate+up 12.0 -> 7.2, dsa in 57.9 -> 49.2. TRUSS_GEMV_SPLIT=0: the earlier
// rule (S from `in`, KB 1).
MmaSplit mma_split(int in, int outs)
{
    const int nb = in / 32, tiles = (outs + 15) / 16;
    static const bool on = [] { const char * e = std::getenv("TRUSS_GEMV_SPLIT"); return !e || std::atoi(e); }();
    if (!on) {
        const int want = nb / 16;
        return { want >= 16 ? 16 : want >= 8 ? 8 : want >= 4 ? 4 : want >= 2 ? 2 : 1, 1 };
    }
    auto pow2_at_least = [](int v) { int p = 1; while (p < v) p *= 2; return p; };
    int P = std::max(pow2_at_least((1280 + tiles - 1) / tiles), pow2_at_least((nb + 11) / 12));
    while (P > 1 && P * 2 > nb) P /= 2;   // at least 2 blocks per warp
    const int S = std::min(P, 16);
    return { S, P / S };
}

namespace {

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
    half * y16[MULTI_MAX];   // non-null: this matrix's row goes out as silu(acc * y16s) fp16 here, y[i] unused
    float y16s[MULTI_MAX];
    int start[MULTI_MAX + 1];   // output prefix sums
    int n;
};

// A16: the activations are fp16 (xq is the half array [rows][in], xd unused): W8A16 without dequantizing the weights,
// each int8 weight is converted in registers and multiplied in fp32, one scale multiply per 32-block.
__device__ __forceinline__ float dot32_a16(const int4 w0, const int4 w1, const half * __restrict__ x)
{
    const int4 * xp = reinterpret_cast<const int4 *>(x);
    const int4 xv[4] = { __ldg(xp), __ldg(xp + 1), __ldg(xp + 2), __ldg(xp + 3) };
    const int wv[8] = { w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w };
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const half2 * h = reinterpret_cast<const half2 *>(&xv[i >> 1]) + 2 * (i & 1);
        const float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
        s = fmaf((float) (int8_t) (wv[i]), a.x, s);
        s = fmaf((float) (int8_t) (wv[i] >> 8), a.y, s);
        s = fmaf((float) (int8_t) (wv[i] >> 16), b.x, s);
        s = fmaf((float) (int8_t) (wv[i] >> 24), b.y, s);
    }
    return s;
}

template <int ROWS, int LPO, int WPO, bool A16 = false>
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
                if (A16) {
                    acc[r] += dot32_a16(w0, w1, reinterpret_cast<const half *>(xq) + (size_t) r * in + b * 32) * dw;
                    continue;
                }
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

// ---- W8A16 decode gemv on the tensor cores (mma.m16n8k16, fp16 x fp16 -> fp32) ----
// The scalar A16 path above converts every int8 weight with a quarter-rate I2F per row and multiplies in fp32: ALU
// bound at ~470-570 GB/s on the 3090 (2-GPU split trace 10-12). Here a warp computes a tile of 16 outputs x 8 rows
// (rows >= the window's are zero): per 32-weight Q8 block two mmas, then the block's scale in fp32, as the scalar
// path does. int8 -> fp16 is exact (byte_perm into 0x64xx = 1024 + (w + 128), minus 1152). Each thread loads 8
// contiguous weights of its two output rows per block; the mma's logical k order is permuted identically for weights
// and activations (thread q's k pairs {2q, 2q+1}, {2q+8, 2q+9} <-> physical 8q..8q+3, then 8q+4..8q+7), which leaves
// every dot product's terms the same. A row's result does not depend on the row count (each C element is its own
// dot product of fixed order) nor on the other matrices of the launch: the k split S comes from `in` only, partial
// sums are added in warp order.
__device__ __forceinline__ void i8x4_to_h2(uint32_t w, uint32_t & lo, uint32_t & hi)   // bytes 0,1 / 2,3, exact
{
    const uint32_t u = w ^ 0x80808080u;
    lo = __byte_perm(u, 0x64646464u, 0x5140);
    hi = __byte_perm(u, 0x64646464u, 0x7362);
    const half2 bias = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));   // 1152
    half2 l = *reinterpret_cast<half2 *>(&lo), h = *reinterpret_cast<half2 *>(&hi);
    l = __hsub2(l, bias), h = __hsub2(h, bias);
    lo = *reinterpret_cast<uint32_t *>(&l), hi = *reinterpret_cast<uint32_t *>(&h);
}

__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], const uint32_t b[2])
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// sigmoid exactly as kernels/hc/hc_prefill.cu computes it, so an epilogue that does the work of that file's silu
// or collapse kernel is bit-identical to it.
__device__ __forceinline__ float hc_sigmoid(float v) { return 1.f / (1.f + __expf(-v)); }

template <int S>   // warps per tile (k split); a CTA has max(4, S) warps = max(4, S) / S tiles
__global__ void __launch_bounds__(512) gemv_mma_a16_kernel(MultiQ8 m, int in, const half * __restrict__ x, int rows,
                                                           float * __restrict__ ws, int * __restrict__ ctr)
{
    // gridDim.y = KB blocks split the k range of each tile group too (warp p = kb * S + ks of P = S * KB takes Q8
    // blocks p, p + P, ...); each block sums its S warps, then the group's last block to arrive adds the KB partials
    // in kb order (fixed order, no float atomics)
    constexpr int WARPS = S > 4 ? S : 4, TILES = WARPS / S;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, q = lane & 3, nb = in / 32;
    const int tile = blockIdx.x * TILES + warp / S, ks = warp % S, KB = gridDim.y, kb = blockIdx.y, P = S * KB;
    const int outs = m.start[m.n];
    // this thread's two output rows (g, g + 8 of the tile) and their matrices
    const int o[2] = { tile * 16 + g, tile * 16 + g + 8 };
    const int8_t * wq[2] = { nullptr, nullptr };
    const half * wd[2] = { nullptr, nullptr };
    float * wy[2] = { nullptr, nullptr };
    half * w16[2] = { nullptr, nullptr };
    float s16[2] = { 0.f, 0.f };
    int wout[2] = { 0, 0 }, woo[2] = { 0, 0 };
#pragma unroll
    for (int j = 0; j < 2; ++j)
        if (o[j] < outs) {
            int seg = 0;
            while (o[j] >= m.start[seg + 1]) ++seg;
            woo[j] = o[j] - m.start[seg], wout[j] = m.start[seg + 1] - m.start[seg];
            wq[j] = m.q[seg] + (size_t) woo[j] * in + 8 * q, wd[j] = m.d[seg] + (size_t) woo[j] * nb, wy[j] = m.y[seg];
            w16[j] = m.y16[seg], s16[j] = m.y16s[seg];
        }
    const bool xrow = g < rows;   // B column g = window row g
    const half * xr = x + (size_t) (xrow ? g : 0) * in + 8 * q;
    float acc[4] = { 0.f, 0.f, 0.f, 0.f };
#pragma unroll 4
    for (int b = kb * S + ks; b < nb; b += P) {
        const uint2 w0 = wq[0] ? __ldg(reinterpret_cast<const uint2 *>(wq[0] + b * 32)) : make_uint2(0, 0);
        const uint2 w1 = wq[1] ? __ldg(reinterpret_cast<const uint2 *>(wq[1] + b * 32)) : make_uint2(0, 0);
        const uint4 xv = xrow ? __ldg(reinterpret_cast<const uint4 *>(xr + b * 32)) : make_uint4(0, 0, 0, 0);
        const float d0 = wq[0] ? __half2float(wd[0][b]) : 0.f, d1 = wq[1] ? __half2float(wd[1][b]) : 0.f;
        uint32_t a[4], bb[2];
        float c[4] = { 0.f, 0.f, 0.f, 0.f };
        i8x4_to_h2(w0.x, a[0], a[2]);   // row g, k 8q..8q+3
        i8x4_to_h2(w1.x, a[1], a[3]);   // row g + 8
        bb[0] = xv.x, bb[1] = xv.y;
        mma16816(c, a, bb);
        i8x4_to_h2(w0.y, a[0], a[2]);   // k 8q+4..8q+7
        i8x4_to_h2(w1.y, a[1], a[3]);
        bb[0] = xv.z, bb[1] = xv.w;
        mma16816(c, a, bb);
        acc[0] += c[0] * d0, acc[1] += c[1] * d0, acc[2] += c[2] * d1, acc[3] += c[3] * d1;
    }
    if (S > 1) {   // k split: the S warps' partials added in warp order
        __shared__ float part[WARPS][32][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) part[warp][lane][i] = acc[i];
        __syncthreads();
        if (ks) {
            if (KB == 1) return;
        } else {
#pragma unroll
            for (int k = 1; k < S; ++k)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[i] += part[warp + k][lane][i];
        }
    }
    if (KB > 1) {   // block partials to ws[tile][kb]; the last block of the group adds them in kb order
        __shared__ int s_last;
        float4 * wp = reinterpret_cast<float4 *>(ws) + ((size_t) tile * KB) * 32 + lane;
        if (ks == 0) wp[(size_t) kb * 32] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) s_last = atomicAdd(&ctr[blockIdx.x], 1) == KB - 1;
        __syncthreads();
        if (!s_last || ks) return;
        __threadfence();
        float4 v = __ldcg(wp);
        acc[0] = v.x, acc[1] = v.y, acc[2] = v.z, acc[3] = v.w;
        for (int k = 1; k < KB; ++k) {
            v = __ldcg(wp + (size_t) k * 32);
            acc[0] += v.x, acc[1] += v.y, acc[2] += v.z, acc[3] += v.w;
        }
        if (warp == 0 && lane == 0) ctr[blockIdx.x] = 0;   // the next launch's
    }
    // C: (row g, cols 2q, 2q + 1), (row g + 8, same cols); col = window row
#pragma unroll
    for (int j = 0; j < 2; ++j)
        if (w16[j] || wy[j])
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int r = 2 * q + t;
                if (r >= rows) continue;
                if (w16[j]) {   // the hc silu: hc::silu_kernel's expression and rounding, on this thread's value
                    const float a = acc[2 * j + t] * s16[j];
                    w16[j][(size_t) r * wout[j] + woo[j]] = __float2half(a * hc_sigmoid(a));
                } else wy[j][(size_t) r * wout[j] + woo[j]] = acc[2 * j + t];
            }
}

// The up projection of a hyper-connection mix with the collapse of the streams folded into the same launch
// (TRUSS_HC_COLLAPSE). gate = W . x runs on exactly the arithmetic of gemv_mma_a16_kernel<S> with KB = 1: same tile
// of 16 outputs, same k slice per warp (warp ks takes Q8 blocks ks, ks + S, ...), same warp-order k combine, so
// every gate value is bit-identical to the separate launch's. What changes is where the value goes. A CTA owns 16
// consecutive output columns of *every* stream - warp c * S + ks is stream c, k slice ks - so the hc gate values of
// one element are all in the CTA that has to combine them: they go to shared instead of to memory, and the CTA
// finishes what collapse_kernel would have done for those elements, in the same order:
//   mixed[t][e] = (1 / hc) * sum_c res[(t*hc + c)*d + e] * rstd[t*hc + c] * gamma[c*d + e] * sigmoid(gate[t][c][e])
// (out == hc * d, d % 16 == 0, rows <= 8; the separate path pays a launch plus the gate round trip instead).
template <int S, int HC>
__global__ void __launch_bounds__(32 * HC * S) up_collapse_kernel(const int8_t * __restrict__ wq,
                                                                  const half * __restrict__ wd, int in,
                                                                  const half * __restrict__ x, int rows, int d,
                                                                  const float * __restrict__ res,
                                                                  const float * __restrict__ rstd,
                                                                  const float * __restrict__ gamma,
                                                                  float * __restrict__ mixed, half * __restrict__ mixed16)
{
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nb = in / 32;
    const int cs = warp / S, ks = warp % S;   // this warp's stream and its slice of the k range
    const int g = lane >> 2, q = lane & 3;
    const int e = blockIdx.x * 16 + g;   // this thread's two output rows (e, e + 8) within every stream
    const int8_t * wr = wq + ((size_t) cs * d + e) * in + 8 * q;
    const half * wds = wd + ((size_t) cs * d + e) * nb;
    const bool xrow = g < rows;   // B column g = window row g
    const half * xr = x + (size_t) (xrow ? g : 0) * in + 8 * q;
    float acc[4] = { 0.f, 0.f, 0.f, 0.f };
#pragma unroll 4
    for (int b = ks; b < nb; b += S) {
        const uint2 w0 = __ldg(reinterpret_cast<const uint2 *>(wr + b * 32));
        const uint2 w1 = __ldg(reinterpret_cast<const uint2 *>(wr + (size_t) 8 * in + b * 32));
        const uint4 xv = xrow ? __ldg(reinterpret_cast<const uint4 *>(xr + b * 32)) : make_uint4(0, 0, 0, 0);
        const float d0 = __half2float(wds[b]), d1 = __half2float(wds[(size_t) 8 * nb + b]);
        uint32_t a[4], bb[2];
        float c[4] = { 0.f, 0.f, 0.f, 0.f };
        i8x4_to_h2(w0.x, a[0], a[2]);   // row e, k 8q..8q+3
        i8x4_to_h2(w1.x, a[1], a[3]);   // row e + 8
        bb[0] = xv.x, bb[1] = xv.y;
        mma16816(c, a, bb);
        i8x4_to_h2(w0.y, a[0], a[2]);   // k 8q+4..8q+7
        i8x4_to_h2(w1.y, a[1], a[3]);
        bb[0] = xv.z, bb[1] = xv.w;
        mma16816(c, a, bb);
        acc[0] += c[0] * d0, acc[1] += c[1] * d0, acc[2] += c[2] * d1, acc[3] += c[3] * d1;
    }
    __shared__ float part[HC * S][32][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) part[warp][lane][i] = acc[i];
    __shared__ float g4[HC][16][8];   // gate[stream][column][window row]
    __syncthreads();
    if (ks == 0) {   // the S warps' partials added in warp order, as the separate launch does
#pragma unroll
        for (int k = 1; k < S; ++k)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[i] += part[warp + k][lane][i];
        // C: (row e, cols 2q, 2q + 1), (row e + 8, the same cols); col = window row
        g4[cs][g][2 * q] = acc[0], g4[cs][g][2 * q + 1] = acc[1];
        g4[cs][g + 8][2 * q] = acc[2], g4[cs][g + 8][2 * q + 1] = acc[3];
    }
    __syncthreads();
    const int el = threadIdx.x % 16, t = threadIdx.x / 16;
    if (t >= rows) return;
    const int ee = blockIdx.x * 16 + el;
    float s = 0.f;
#pragma unroll
    for (int cc = 0; cc < HC; ++cc)
        s += res[((size_t) t * HC + cc) * d + ee] * rstd[(size_t) t * HC + cc] * gamma[(size_t) cc * d + ee] *
             hc_sigmoid(g4[cc][el][t]);
    s *= 1.f / HC;
    mixed[(size_t) t * d + ee] = s;
    if (mixed16) mixed16[(size_t) t * d + ee] = __float2half(s);
}

// cross-block split workspace, one per (device, stream): launches on one stream run in order, so each finds the
// counters zero (the last block of a group resets its own)
constexpr size_t SPLIT_WS_FLOATS = (size_t) 1 << 20, SPLIT_CTRS = 8192;
struct SplitWs { float * part = nullptr; int * ctr = nullptr; };
SplitWs split_ws(cudaStream_t stream)
{
    static std::mutex mu;
    static std::map<std::pair<int, cudaStream_t>, SplitWs> all;
    int dev = 0;
    TRUSS_CUDA(cudaGetDevice(&dev));
    std::lock_guard<std::mutex> g(mu);
    SplitWs & w = all[{ dev, stream }];
    if (!w.part) {
        TRUSS_CUDA(cudaMalloc(&w.part, sizeof(float) * SPLIT_WS_FLOATS));
        TRUSS_CUDA(cudaMalloc(&w.ctr, sizeof(int) * SPLIT_CTRS));
        TRUSS_CUDA(cudaMemset(w.ctr, 0, sizeof(int) * SPLIT_CTRS));
    }
    return w;
}

void gemv_mma_a16_cfg(const MultiQ8 & m, int in, const half * x, int rows, MmaSplit c, cudaStream_t stream)
{
    const int S = c.S, warps = S > 4 ? S : 4, tiles = (m.start[m.n] + 15) / 16, per = warps / S;
    const int grid = (tiles + per - 1) / per;
    int KB = c.KB < 1 ? 1 : c.KB;
    SplitWs w;
    if (KB > 1) {
        if ((size_t) grid * per * KB * 128 > SPLIT_WS_FLOATS || grid > (int) SPLIT_CTRS) KB = 1;
        else w = split_ws(stream);
    }
    const dim3 gd(grid, KB);
    switch (S) {
    case 1: gemv_mma_a16_kernel<1><<<gd, 32 * warps, 0, stream>>>(m, in, x, rows, w.part, w.ctr); break;
    case 2: gemv_mma_a16_kernel<2><<<gd, 32 * warps, 0, stream>>>(m, in, x, rows, w.part, w.ctr); break;
    case 4: gemv_mma_a16_kernel<4><<<gd, 32 * warps, 0, stream>>>(m, in, x, rows, w.part, w.ctr); break;
    case 8: gemv_mma_a16_kernel<8><<<gd, 32 * warps, 0, stream>>>(m, in, x, rows, w.part, w.ctr); break;
    default: gemv_mma_a16_kernel<16><<<gd, 32 * warps, 0, stream>>>(m, in, x, rows, w.part, w.ctr); break;
    }
}

void gemv_mma_a16(const MultiQ8 & m, int in, const half * x, int rows, cudaStream_t stream)
{
    gemv_mma_a16_cfg(m, in, x, rows, mma_split(in, m.start[m.n]), stream);
}

template <int ROWS, bool A16 = false>
void gemv_multi_rows(const MultiQ8 & m, int in, const int8_t * xq, const half * xd, cudaStream_t stream)
{
    const int nb = in / 32, outs = m.start[m.n];
    // long rows get 2 or 4 warps per output: the long-row matrices of this model have few outputs (hc down 10240 ->
    // 324: 1,296 warps at 4; out-proj 6144 -> 2560), and one warp per output leaves the 82 SMs short of warps
    const int wpo = nb >= 256 ? 4 : nb >= 128 ? 2 : 1;
    const int lpo = nb >= 32 ? 32 : nb > 8 ? 16 : nb > 4 ? 8 : 4;
    if (lpo < 32) {
        const int per = 4 * 32 / lpo, grid = (outs + per - 1) / per;
        if (lpo == 16) gemv_multi_kernel<ROWS, 16, 1, A16><<<grid, 128, 0, stream>>>(m, in, xq, xd);
        else if (lpo == 8) gemv_multi_kernel<ROWS, 8, 1, A16><<<grid, 128, 0, stream>>>(m, in, xq, xd);
        else gemv_multi_kernel<ROWS, 4, 1, A16><<<grid, 128, 0, stream>>>(m, in, xq, xd);
    } else if (wpo == 1) gemv_multi_kernel<ROWS, 32, 1, A16><<<(outs + 3) / 4, 128, 0, stream>>>(m, in, xq, xd);
    else if (wpo == 2) gemv_multi_kernel<ROWS, 32, 2, A16><<<(outs + 1) / 2, 128, 0, stream>>>(m, in, xq, xd);
    else gemv_multi_kernel<ROWS, 32, 4, A16><<<outs, 128, 0, stream>>>(m, in, xq, xd);
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


}  // namespace

bool mma_on()   // TRUSS_GEMV_MMA=0: the scalar A16 kernel (A/B switch, read once); the fused epilogues live only
{               // in the tensor-core one, so the engine asks this before asking for one
    static const bool v = [] { const char * e = std::getenv("TRUSS_GEMV_MMA"); return !e || std::atoi(e); }();
    return v;
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

void q8_gemv_multi_a16(const Q8Matrix * const * W, float * const * y, int n, const half * x, int rows, cudaStream_t stream)
{
    if (n < 1 || n > MULTI_MAX) throw std::runtime_error("q8_gemv_multi_a16: 1.." + std::to_string(MULTI_MAX) + " matrices");
    MultiQ8 m{};
    m.n = n, m.start[0] = 0;
    for (int i = 0; i < n; ++i) {
        if (W[i]->in != W[0]->in || W[i]->in % 32) throw std::runtime_error("q8_gemv_multi_a16: same in, multiple of 32");
        m.q[i] = W[i]->q, m.d[i] = W[i]->d, m.y[i] = y[i], m.start[i + 1] = m.start[i] + W[i]->out;
    }
    const int in = W[0]->in;
    if (rows < 1 || rows > GEMV_ROWS) throw std::runtime_error("q8_gemv_multi_a16: rows must be 1.." + std::to_string(GEMV_ROWS));
    if (mma_on()) {
        gemv_mma_a16(m, in, x, rows, stream);
        TRUSS_CUDA(cudaGetLastError());
        return;
    }
    const int8_t * xq = reinterpret_cast<const int8_t *>(x);
    switch (rows) {
    case 1: gemv_multi_rows<1, true>(m, in, xq, nullptr, stream); break;
    case 2: gemv_multi_rows<2, true>(m, in, xq, nullptr, stream); break;
    case 3: gemv_multi_rows<3, true>(m, in, xq, nullptr, stream); break;
    case 4: gemv_multi_rows<4, true>(m, in, xq, nullptr, stream); break;
    case 5: gemv_multi_rows<5, true>(m, in, xq, nullptr, stream); break;
    case 6: gemv_multi_rows<6, true>(m, in, xq, nullptr, stream); break;
    case 7: gemv_multi_rows<7, true>(m, in, xq, nullptr, stream); break;
    case 8: gemv_multi_rows<8, true>(m, in, xq, nullptr, stream); break;
    default: throw std::runtime_error("q8_gemv_multi_a16: rows must be 1.." + std::to_string(GEMV_ROWS));
    }
    TRUSS_CUDA(cudaGetLastError());
}

void q8_gemv_multi_a16_cfg(const Q8Matrix * const * W, float * const * y, int n, const half * x, int rows, MmaSplit c,
                           cudaStream_t stream)
{
    MultiQ8 m{};
    m.n = n, m.start[0] = 0;
    for (int i = 0; i < n; ++i) m.q[i] = W[i]->q, m.d[i] = W[i]->d, m.y[i] = y[i], m.start[i + 1] = m.start[i] + W[i]->out;
    gemv_mma_a16_cfg(m, W[0]->in, x, rows, c, stream);
    TRUSS_CUDA(cudaGetLastError());
}

void q8_gemv_a16(const Q8Matrix & W, const half * x, int rows, float * y, cudaStream_t stream)
{
    const Q8Matrix * w = &W;
    q8_gemv_multi_a16(&w, &y, 1, x, rows, stream);
}

// The hc down projection's silu, folded into the gemv that produces its input (TRUSS_HC_SILU): a matrix whose
// y16[i] is non-null has its row written as silu(acc * y16s[i]) in fp16 by the gemv epilogue instead of as fp32
// y[i]. The value is the one the separate hc::silu kernel would have read and computed, in the same expression.
void q8_gemv_multi_a16_silu(const Q8Matrix * const * W, float * const * y, half * const * y16,
                            const float * y16s, int n, const half * x, int rows, cudaStream_t stream)
{
    if (n < 1 || n > MULTI_MAX) throw std::runtime_error("q8_gemv_multi_a16_silu: 1.." + std::to_string(MULTI_MAX) +
                                                         " matrices");
    if (!mma_on()) throw std::runtime_error("q8_gemv_multi_a16_silu: needs the mma gemv (TRUSS_GEMV_MMA)");
    if (rows < 1 || rows > GEMV_ROWS)
        throw std::runtime_error("q8_gemv_multi_a16_silu: rows must be 1.." + std::to_string(GEMV_ROWS));
    MultiQ8 m{};
    m.n = n, m.start[0] = 0;
    for (int i = 0; i < n; ++i) {
        if (W[i]->in != W[0]->in || W[i]->in % 32)
            throw std::runtime_error("q8_gemv_multi_a16_silu: same in for all, a multiple of 32");
        m.q[i] = W[i]->q, m.d[i] = W[i]->d, m.y[i] = y[i], m.start[i + 1] = m.start[i] + W[i]->out;
        m.y16[i] = y16 ? y16[i] : nullptr, m.y16s[i] = y16s ? y16s[i] : 0.f;
    }
    gemv_mma_a16(m, W[0]->in, x, rows, stream);
    TRUSS_CUDA(cudaGetLastError());
}

// The hc up projection with the collapse folded in (TRUSS_HC_COLLAPSE). The shape must be the engine's own up
// projection: out == hc * d with d a multiple of 16, hc 4 streams, and the shape's k split must stay inside one
// block (KB 1, S <= 4), because a CTA has to see every stream's gate value of its own 16 columns.
bool up_collapse_ok(int in, int out, int hc, int d)
{
    const MmaSplit sp = mma_split(in, out);
    return hc == 4 && out == hc * d && d % 16 == 0 && sp.KB == 1 && sp.S >= 1 && sp.S <= 4;
}

void q8_gemv_a16_up_collapse(const Q8Matrix & W, const half * x, int rows, const float * res, const float * rstd,
                             const float * gamma, float * mixed, half * mixed16, int hc, int d, cudaStream_t stream)
{
    if (!up_collapse_ok(W.in, W.out, hc, d))
        throw std::runtime_error("q8_gemv_a16_up_collapse: hc " + std::to_string(hc) + ", out " + std::to_string(W.out) +
                                 " != hc * d, or the shape's k split leaves the block");
    if (rows < 1 || rows > GEMV_ROWS)
        throw std::runtime_error("q8_gemv_a16_up_collapse: rows must be 1.." + std::to_string(GEMV_ROWS));
    const MmaSplit sp = mma_split(W.in, W.out);
    const dim3 gd(d / 16);
    // one warp per (stream, k slice), so the block is 32 * hc * S
    void (*k)(const int8_t *, const half *, int, const half *, int, int, const float *, const float *, const float *,
              float *, half *) = sp.S == 1 ? up_collapse_kernel<1, 4>
                            : sp.S == 2 ? up_collapse_kernel<2, 4> : up_collapse_kernel<4, 4>;
    k<<<gd, 32 * 4 * sp.S, 0, stream>>>(W.q, W.d, W.in, x, rows, d, res, rstd, gamma, mixed, mixed16);
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
