// gemv_window.cu: the trellis-dense decode GEMV for n = 1..16 columns, built the TRUSS way (src/kernels/moe/
// moe_window.cu: decode in registers, one mma per tile with the weights as the A operand, no grid barrier) instead
// of the exllamav3 cooperative kernel.
//
// One matvec = three launches over ALL row parts of the matrix (e.g. ffn_gate + ffn_up):
//   had_in  A_p[t]           = H128(x[t] * suh_p)                     one warp per (part, t, 128 block)
//   gemv    P_p[kp][t][cols] = A_p[t][k range kp] . W_p               persistent grid over the items
//   out     y[t][off_p + c]  = svh_p * H128(sum_kp P_p[kp][t])        one warp per (part, t, 128 block)
// An item is (part, 64-column group, k part). The k split depends on the part's shape only, never on n, so column t
// is bit-identical for every n (the drafter's verify batch equals decode). n > 8 adds a second mma per decoded tile
// (batch half NB = 2): a 12-column tree verify decodes the weights once instead of twice. Deterministic: fixed
// reduction orders, no atomics.
//
// The codec is TRUSS's own (src/kernels/trellis/codec_mul1.cuh): the same mul1 windows and codebook as tqLLM's
// Mul1Codec (bit-identical states, TRACKER #208's codec diff), so this kernel and tqLLM's gemv_window.cu compute the
// same product. The plan (plan.hpp) is tqLLM's window.hpp, the layout of its kernels.
//
// Derived from tqLLM (Apache-2.0) kernels/formats/x3/gemv_window.cu, itself ported from llama-paw-k25
// ggml/src/ggml-cuda/paw-x3.cu (MIT, exllamav3 lineage). The MIT notice of the original applies to this file.
#include "kernels/trellis/dense/plan.hpp"

#include "kernels/trellis/codec_mul1.cuh"
#include "kernels/trellis/hadamard.cuh"
#include "kernels/trellis/mma.cuh"

#include <cuda_runtime.h>

#include <algorithm>

namespace truss::trellis_dense {

namespace {

// ---------------------------------------------------------------------------------------------------------
// had_in / out: one warp per (part, t, 128-element block); the warp-index walk is the same for both.

struct WarpTask {
    int part, t, c0;     // c0 = this lane's first element (4 per lane)
};

// The calling warp's task: (part, t, 128-element block) over x (width = cols) or over y (width < 0: rows).
__device__ __forceinline__ bool warp_task(const List & L, int width, WarpTask & w)
{
    const int lane = threadIdx.x & 31;
    int g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    for (int i = 0; i < L.count; ++i) {
        const int blocks = (width < 0 ? L.p[i].rows : width) / 128;
        const int tasks = L.n * blocks;
        if (g < tasks) {
            w.part = i;
            w.t = g / blocks;
            w.c0 = (g % blocks) * 128 + lane * 4;
            return true;
        }
        g -= tasks;
    }
    return false;
}

__global__ void td_had_in(List L, const half * __restrict__ x)
{
    WarpTask w;
    if (!warp_task(L, L.cols, w)) return;
    const Part & P = L.p[w.part];
    const half2 * xv = reinterpret_cast<const half2 *>(x + (size_t) w.t * L.cols + w.c0);
    const half2 * sv = reinterpret_cast<const half2 *>(P.suh + w.c0);
    const half2 a = __hmul2(xv[0], sv[0]), b = __hmul2(xv[1], sv[1]);
    const float v0 = __low2float(a), v1 = __high2float(a), v2 = __low2float(b), v3 = __high2float(b);
    float h0 = v0, h1 = v1, h2 = v2, h3 = v3;
    had4x32(h0, h1, h2, h3, threadIdx.x & 31);   // the 4-value butterfly, the cross-lane stages and 1/sqrt(128)
    half2 * o = reinterpret_cast<half2 *>(P.A + (size_t) w.t * L.cols + w.c0);
    o[0] = __floats2half2_rn(h0, h1);
    o[1] = __floats2half2_rn(h2, h3);
}

__global__ void td_out(List L)
{
    WarpTask w;
    if (!warp_task(L, -1, w)) return;
    const Part & P = L.p[w.part];
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int kp = 0; kp < P.kparts; ++kp) {
        const float4 q = *reinterpret_cast<const float4 *>(P.P + ((size_t) kp * L.n + w.t) * P.rows + w.c0);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
    }
    had4x32(v.x, v.y, v.z, v.w, threadIdx.x & 31);
    const half2 * sv = reinterpret_cast<const half2 *>(P.svh + w.c0);
    const half2 sa = sv[0], sb = sv[1];
    float * o = P.y + (size_t) w.t * P.ldy + w.c0;
    o[0] = v.x * __low2float(sa);
    o[1] = v.y * __high2float(sa);
    o[2] = v.z * __low2float(sb);
    o[3] = v.w * __high2float(sb);
}

// ---------------------------------------------------------------------------------------------------------
// gemv: a persistent grid walks the items (Geom: WK k-split warps x WNT tiles, PF slices in flight).
// NB = batch halves of 8 columns (1: n <= 8, 2: n <= 16); every half gets its own mma on the same decoded tile.

// Shared memory of one gemv block: the A stage (n rows, padded stride) or the k-split reduction, whichever is larger.
__host__ __device__ constexpr int a_stride(int k_slices) { return k_slices * 8 + 4; }
inline size_t gemv_smem(int n, int k_slices)
{
    const size_t a = (size_t) n * a_stride(k_slices) * 4;
    const size_t red = (size_t) Geom::WK * TD_MAX_N * Geom::COLS * 4;
    return a > red ? a : red;
}

// One item (part, column group, k part): P_part[kp][t][group columns].
// E18 gate 1 (brief "Past failures to avoid"): compile-time bisect of the item's inner work, built in separate build
// dirs so the bottleneck is proven before arithmetic changes. 0 = production; 1 = loads only (the decode replaced by
// a cheap use of the words); 2 = decode from registers, no weight loads; 3 = no mma/fold.
#ifndef TD_BISECT
#define TD_BISECT 0
#endif

template <class Codec, int NB>
__device__ __forceinline__ void td_item(const List & L, int item)
{
    using G = Geom;
    constexpr int TWORDS = Codec::TILE_WORDS;
    constexpr int LOADS = G::WNT / Codec::TILES_PER_VEC;
    static_assert(G::WNT % Codec::TILES_PER_VEC == 0, "tiles per warp must fill whole loads");

    int pi = 0;
    while (pi + 1 < L.count && item >= L.p[pi + 1].item0) ++pi;
    const Part & P = L.p[pi];
    item -= P.item0;

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int ntiles = P.rows / 16;
    const int groups = P.rows / G::COLS;
    const int group = item % groups, kp = item / groups;   // group-minor: neighbour blocks read neighbour tiles
    const int k_slices = L.cols / 16 / P.kparts;
    const int chunk = ceil_div(k_slices, G::WK);
    const int ks0 = kp * k_slices + warp * chunk;
    const int myn = max(0, min(chunk, k_slices - warp * chunk));

    const size_t slice_stride = (size_t) ntiles * TWORDS;
    const uint32_t * bp = P.B + tile_offset(ks0, group * G::WNT, ntiles, TWORDS) + lane;
    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (Codec::VEC_WORDS < 32)
            return lane < Codec::VEC_WORDS ? __ldcs(bp + (size_t) i * slice_stride + l * Codec::VEC_WORDS) : 0u;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * Codec::VEC_WORDS);
    };

    uint32_t pf[G::PF][LOADS];
#pragma unroll
    for (int d = 0; d < G::PF; ++d)
        if (d < myn) {
#pragma unroll
#if TD_BISECT == 2
            // registers: words from a cheap register pattern, no global weight loads
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = (uint32_t) (d * 7919u + l * 104729u + (uint32_t) lane * 31u) | 1u;
#else
            for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);
#endif
        }
    // The item's activations (n rows x its k range) are staged in shared memory once, behind the first weight
    // loads: per-slice L2 reads of A were a stall of the loop. Row stride padded by 4 words so the 8 batch rows of a
    // warp read hit 32 distinct banks.
    extern __shared__ __align__(16) uint32_t td_smem[];
    const int a_words = k_slices * 8, as = a_stride(k_slices);
    {
        const uint4 * src = reinterpret_cast<const uint4 *>(P.A) + (size_t) kp * a_words / 4;
        for (int idx = threadIdx.x; idx < L.n * a_words / 4; idx += G::THREADS) {
            const int r = idx / (a_words / 4), c = idx % (a_words / 4);
            reinterpret_cast<uint4 *>(td_smem + r * as)[c] = src[(size_t) r * (L.cols / 8) + c];
        }
        __syncthreads();
    }
    const int r0 = lane >> 2;
    bool r_ok[NB];
    const uint32_t * As[NB];
#pragma unroll
    for (int h = 0; h < NB; ++h) {
        r_ok[h] = r0 + 8 * h < L.n;
        As[h] = td_smem + (r0 + 8 * h) * as + warp * chunk * 8 + (lane & 3);
    }

    const Codec codec(lane);
    const half2 hzero = __half2half2(__ushort_as_half(0));
    FragCh ch[G::WNT][NB];
    float2 acc[G::WNT][NB][2];   // [t][h][f]: column t * 16 + f * 8 + g, batch rows 8h + 2q, 8h + 2q + 1
#pragma unroll
    for (int t = 0; t < G::WNT; ++t)
#pragma unroll
        for (int h = 0; h < NB; ++h) {
            ch[t][h][0] = ch[t][h][1] = hzero;
            acc[t][h][0] = acc[t][h][1] = make_float2(0.f, 0.f);
        }

    for (int ib = 0; ib < myn; ib += G::PF) {
#pragma unroll
        for (int d = 0; d < G::PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;
            uint32_t bw[LOADS];
#pragma unroll
            for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];
            uint32_t alo[NB], ahi[NB];
#pragma unroll
            for (int h = 0; h < NB; ++h) {
                alo[h] = r_ok[h] ? As[h][i * 8] : 0u;
                ahi[h] = r_ok[h] ? As[h][i * 8 + 4] : 0u;
            }
            if (i + G::PF < myn) {
#pragma unroll
#if TD_BISECT == 2
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = (uint32_t) ((i + G::PF) * 7919u + l * 104729u + (uint32_t) lane * 31u) | 1u;
#else
                for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + G::PF, l);
#endif
            }
#pragma unroll
            for (int t = 0; t < G::WNT; ++t) {
                FragB f0, f1;
#if TD_BISECT == 1
                // loads only: the decode replaced by a cheap use of the words
                const uint32_t wc = bw[t / Codec::TILES_PER_VEC] + (uint32_t) t;
                f0[0] = f0[1] = f1[0] = f1[1] = __half2half2(__ushort_as_half((uint16_t) wc));
#else
                codec.tile(bw[t / Codec::TILES_PER_VEC], t % Codec::TILES_PER_VEC, f0, f1);
#endif
#if TD_BISECT == 3
                // no mma/fold: consume the decode cheaply into the accumulator
                for (int h = 0; h < NB; ++h) {
                    acc[t][h][0].x += __low2float(f0[0]) + __low2float(f1[0]);
                    acc[t][h][1].y += __high2float(f0[1]) + __high2float(f1[1]);
                }
#else
                for (int h = 0; h < NB; ++h)
#if TRUSS_MMA_F32ACC
                    mma_w_f32(f0, f1, *(const half2 *) &alo[h], *(const half2 *) &ahi[h], acc[t][h]);
#else
                    mma_w(f0, f1, *(const half2 *) &alo[h], *(const half2 *) &ahi[h], ch[t][h]);
#endif
#endif
            }
#if TD_BISECT != 3
            if (!TRUSS_MMA_F32ACC && ((d + 1) % G::FOLD == 0 || i + 1 == myn)) {
#pragma unroll
                for (int t = 0; t < G::WNT; ++t)
#pragma unroll
                    for (int h = 0; h < NB; ++h)
#pragma unroll
                        for (int f = 0; f < 2; ++f) {
                            acc[t][h][f].x += __low2float(ch[t][h][f]);
                            acc[t][h][f].y += __high2float(ch[t][h][f]);
                            ch[t][h][f] = hzero;
                        }
            }
#endif
        }
    }

    // k-split warps summed in fixed order through shared memory (reusing the A stage): red[warp][row][column]
    __syncthreads();
    auto red = reinterpret_cast<float (*)[TD_MAX_N][G::COLS]>(td_smem);
    {
        const int g = lane >> 2, q2 = 2 * (lane & 3);
#pragma unroll
        for (int t = 0; t < G::WNT; ++t)
#pragma unroll
            for (int h = 0; h < NB; ++h)
#pragma unroll
                for (int f = 0; f < 2; ++f) {
                    const int col = t * 16 + f * 8 + g;
                    red[warp][8 * h + q2 + 0][col] = acc[t][h][f].x;
                    red[warp][8 * h + q2 + 1][col] = acc[t][h][f].y;
                }
    }
    __syncthreads();
    float * out = P.P + (size_t) kp * L.n * P.rows + group * G::COLS;
    for (int idx = threadIdx.x; idx < G::COLS * L.n; idx += G::THREADS) {
        const int r = idx / G::COLS, c = idx % G::COLS;
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < G::WK; ++j) sum += red[j][r][c];
        out[(size_t) r * P.rows + c] = sum;
    }

    __syncthreads();   // the next item restages A over red
}

// Persistent: a resident grid walks the items, so block start-up and drain are paid once per launch.
template <class Codec, int NB>
__global__ __launch_bounds__(Geom::THREADS) void td_gemv(List L)
{
    for (int item = blockIdx.x; item < L.items; item += gridDim.x) td_item<Codec, NB>(L, item);
}

// One gemv launch with NB batch halves; the resident grid is sized once per device (the occupancy and the shared
// memory opt-in are per device: a stage may run on either GPU).
template <class Codec, int NB>
void gemv_launch(const List & L, size_t smem, cudaStream_t s)
{
    constexpr int MAX_DEV = 16;
    static int resident[MAX_DEV] = { 0 };
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= MAX_DEV) dev = 0;
    if (!resident[dev]) {
        const size_t most = gemv_smem(8 * NB, TD_MAX_ITEM_SLICES);   // NB = 2: above the 48 KB default
        cudaFuncSetAttribute(td_gemv<Codec, NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(most));
        int b = 1, sms = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, td_gemv<Codec, NB>, Geom::THREADS, most);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        resident[dev] = max(1, b) * sms;
    }
    td_gemv<Codec, NB><<<min(L.items, resident[dev]), Geom::THREADS, smem, s>>>(L);
}

inline int warp_blocks(const List & L, bool in)
{
    int warps = 0;
    for (int i = 0; i < L.count; ++i) warps += L.n * (in ? L.cols : L.p[i].rows) / 128;
    return ceil_div(warps, 8);
}

// The three launches. Every part of L must use the codec (the dispatcher splits mixed lists).
template <class Codec>
void matvec_codec(const List & L, const half * x, cudaStream_t s)
{
    if (L.count == 0 || L.items == 0) return;
    td_had_in<<<warp_blocks(L, true), 256, 0, s>>>(L, x);
    int ks_max = 0;
    for (int i = 0; i < L.count; ++i) ks_max = max(ks_max, L.cols / 16 / L.p[i].kparts);
    const size_t smem = gemv_smem(L.n, ks_max);
    if (L.n > 8) gemv_launch<Codec, 2>(L, smem, s);
    else gemv_launch<Codec, 1>(L, smem, s);
    td_out<<<warp_blocks(L, false), 256, 0, s>>>(L);
}

}  // namespace

// One matvec over a list of parts of one rate: each part writes y[t][c] = W x[t] into its own buffer.
void matvec(const List & L, int k, const half * x, cudaStream_t s)
{
    switch (k) {
        case 1: matvec_codec<Mul1<1>>(L, x, s); break;
        case 2: matvec_codec<Mul1<2>>(L, x, s); break;
        case 3: matvec_codec<Mul1<3>>(L, x, s); break;
        case 4: matvec_codec<Mul1<4>>(L, x, s); break;
        case 15: matvec_codec<Mul1Frac<1>>(L, x, s); break;
        case 25: matvec_codec<Mul1Frac<2>>(L, x, s); break;
        case 35: matvec_codec<Mul1Frac<3>>(L, x, s); break;
        default: break;   // unsupported rate: the caller checks before calling
    }
}

}  // namespace truss::trellis_dense
