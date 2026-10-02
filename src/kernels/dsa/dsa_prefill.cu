// DSA prefill (see dsa_prefill.cuh).
//   score_kernel      64 queries x 64 blocks per CTA, 8 warps of 16 queries x 32 blocks; each warp keeps all IH heads'
//                     accumulators so relu and the head sum happen in registers. Fragments come straight from
//                     global memory (rows are 256 B; the op is ~2% of prefill FLOPs, TRACKER #49).
//   select_kernel     one CTA per query: 4 radix passes of 8 bits find the TOP_BLOCKS-th largest score key (scores are
//                     >= 0, so the fp32 bits order like the values), then one pass from the last block down keeps the
//                     keys above it and the latest keys equal to it, writing ids from the end so they come out
//                     ascending.
//   attn_kernel       one CTA per (query, KV head), 2 warps. The H / HKV query heads of a KV head share the
//                     selection and fill one m16 tile. Cells are gathered TILE at a time into shared memory (K and V
//                     rows are 512 B contiguous); each warp owns half of a tile's cells and its own softmax state,
//                     and the two states merge at the end.
//                     Split form (T <= SPLIT_ROWS: decode steps, verify windows): a third grid dimension cuts each
//                     query's cells into contiguous ranges; each CTA writes its unnormalized state (o, m, l) and
//                     combine_kernel merges the ranges in order and applies the gate. One query has only HKV = 2
//                     CTAs otherwise: 227 us per decode layer at 3.4K context (TRACKER #57).
#include "dsa_prefill.cuh"

#include "core/cuda_check.h"
#include "kernels/trellis/mma.cuh"

#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace truss::dsa {
namespace {

// ---- indexer scores

constexpr int SQ = 64, SB = 64;           // queries, blocks per score CTA
constexpr int SELECT_THREADS = 256;

__device__ __forceinline__ uint32_t ld32(const half * p) { return __ldg(reinterpret_cast<const unsigned *>(p)); }

// scores [n_q][ld] for queries q0 .. q0 + n_q - 1 of the chunk (positions pos0 + q0 + i); only blocks a query sees
// are written
template <class Shape>
__global__ void __launch_bounds__(256) score_kernel(const half * idx_q, const half * idx_k, int pos0, int q0, int n_q,
                                                    float * scores, int ld)
{
    constexpr int IH = Shape::IH, ID = Shape::ID, R = Shape::RATIO;
    static_assert(ID % 16 == 0, "indexer head dim in k16 steps");
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4;
    const int tq0 = blockIdx.x * SQ + (warp % 4) * 16;         // local query of fragment row 0
    const int b0 = blockIdx.y * SB + (warp / 4) * 32;          // block of fragment column 0
    const int last_pos = pos0 + q0 + min(n_q, tq0 + 16) - 1;   // warp's last query
    if (tq0 >= n_q || R * b0 + R - 1 > last_pos) return;       // no visible block in this warp's tile
    const int nb_seen = (last_pos + 1) / R;
    const half * qr[2];
    for (int i = 0; i < 2; ++i) qr[i] = idx_q + (size_t) (q0 + min(tq0 + g + 8 * i, n_q - 1)) * IH * ID + 2 * qd;
    const half * kr[4];
    for (int j = 0; j < 4; ++j) kr[j] = idx_k + (size_t) min(b0 + 8 * j + g, nb_seen - 1) * ID + 2 * qd;

    float acc[IH][4][4] = {};
#pragma unroll
    for (int kk = 0; kk < ID / 16; ++kk) {
        uint32_t b[4][2];
#pragma unroll
        for (int j = 0; j < 4; ++j) b[j][0] = ld32(kr[j] + 16 * kk), b[j][1] = ld32(kr[j] + 16 * kk + 8);
#pragma unroll
        for (int h = 0; h < IH; ++h) {
            const uint32_t a[4] = { ld32(qr[0] + h * ID + 16 * kk), ld32(qr[1] + h * ID + 16 * kk),
                                    ld32(qr[0] + h * ID + 16 * kk + 8), ld32(qr[1] + h * ID + 16 * kk + 8) };
#pragma unroll
            for (int j = 0; j < 4; ++j) mma_f32(a, b[j][0], b[j][1], acc[h][j]);
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int tq = tq0 + g + 8 * i;
        if (tq >= n_q) continue;
        const int seen = (pos0 + q0 + tq + 1) / R;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int b = b0 + 8 * j + 2 * qd + e;
                if (b >= seen) continue;
                float s = 0.f;
#pragma unroll
                for (int h = 0; h < IH; ++h) s += fmaxf(acc[h][j][2 * i + e], 0.f);
                scores[(size_t) tq * ld + b] = s;
            }
    }
}

// ---- top TOP_BLOCKS selection

template <class Shape>
__global__ void __launch_bounds__(SELECT_THREADS) select_kernel(const float * scores, int ld, int pos0, int q0,
                                                                int * blocks, int * n_blocks)
{
    constexpr int R = Shape::RATIO, TOP = Shape::TOP_BLOCKS;
    using Scan = cub::BlockScan<int, SELECT_THREADS>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ unsigned hist[256];
    __shared__ unsigned s_prefix, s_mask, s_need;
    const int tq = blockIdx.x, t = q0 + tq, tid = threadIdx.x;
    const int seen = (pos0 + t + 1) / R;
    int * out = blocks + (size_t) t * TOP;
    if (seen <= TOP) {
        for (int b = tid; b < seen; b += SELECT_THREADS) out[b] = b;
        if (tid == 0) n_blocks[t] = seen;
        return;
    }
    const float * row = scores + (size_t) tq * ld;
    auto key = [&](int b) { return __float_as_uint(row[b]); };

    // the TOP-th largest key, and how many keys equal to it are kept
    if (tid == 0) s_prefix = 0, s_mask = 0, s_need = TOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += SELECT_THREADS) hist[i] = 0;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask;
        for (int b = tid; b < seen; b += SELECT_THREADS) {
            const unsigned k = key(b);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            unsigned above = 0;
            for (int bin = 255; bin >= 0; --bin) {
                if (above + hist[bin] >= s_need) {
                    s_prefix |= (unsigned) bin << shift;
                    s_mask |= 255u << shift;
                    s_need -= above;
                    break;
                }
                above += hist[bin];
            }
        }
        __syncthreads();
    }
    const unsigned kth = s_prefix, need_eq = s_need;

    // from the last block down: thread i holds block base + (SELECT_THREADS - 1 - i), so exclusive scans count later
    // blocks
    int taken = 0, taken_eq = 0;
    for (int base = (seen - 1) / SELECT_THREADS * SELECT_THREADS; base >= 0 && taken < TOP; base -= SELECT_THREADS) {
        const int b = base + SELECT_THREADS - 1 - tid;
        const bool valid = b < seen;
        const unsigned k = valid ? key(b) : 0u;
        const bool eq = valid && k == kth;
        int eq_before, n_eq, take_before, n_take;
        Scan(scan).ExclusiveSum((int) eq, eq_before, n_eq);
        __syncthreads();
        const bool take = valid && (k > kth || (eq && taken_eq + eq_before < (int) need_eq));
        Scan(scan).ExclusiveSum((int) take, take_before, n_take);
        __syncthreads();
        if (take) out[TOP - 1 - (taken + take_before)] = b;
        taken += n_take;
        taken_eq += n_eq;   // equal keys seen so far, kept or not
    }
    if (tid == 0) n_blocks[t] = TOP;
}

// ---- sparse attention

constexpr int ATTN_WARPS = 2, TILE = 16 * ATTN_WARPS;   // cells per gathered tile, 16 per warp

template <class Shape> struct AttnSmem {
    static constexpr int STRIDE = Shape::D + 8;         // halfs per row: 528 B, ldmatrix rows hit distinct banks
    half q[16][STRIDE];
    half k[TILE][STRIDE];
    half v[TILE][STRIDE];
};

__device__ __forceinline__ void cp16(void * dst, const void * src)
{
    const unsigned d = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}

template <class Shape> struct Partial {             // one split's state per (query, KV head, split)
    static constexpr int G = Shape::H / Shape::HKV;
    static constexpr size_t FLOATS = (size_t) G * Shape::D + 2 * G;   // o [G][D], m [G], l [G]
};

// SPLIT: cells [blockIdx.z * cells_per_split, ...) of each query, unnormalized state to `part`; else all cells, final
// gated output to `out`.
// int8 cache: 8 codes at dims c .. c + 7 of row `off / D` times its group's scale, as 8 fp16 into the tile
__device__ __forceinline__ uint4 dequant8(const int8_t * code, const half * scale, size_t off, int D)
{
    const uint2 b = *reinterpret_cast<const uint2 *>(code + off);
    const float sc = __half2float(scale[off / KV_GROUP]);   // rows are D = a multiple of KV_GROUP values
    const int8_t * c8 = reinterpret_cast<const int8_t *>(&b);
    uint4 r;
    half2 * h = reinterpret_cast<half2 *>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2half2_rn(c8[2 * i] * sc, c8[2 * i + 1] * sc);
    (void) D;
    return r;
}

template <class Shape, bool SPLIT>
__global__ void __launch_bounds__(32 * ATTN_WARPS) attn_kernel(const half * q, const float * gate, KvCache kc,
                                                               const int * blocks,
                                                               const int * n_blocks, int pos0, half * out,
                                                               int cells_per_split, float * part)
{
    const half * k = kc.k16, * v = kc.v16;
    constexpr int H = Shape::H, HKV = Shape::HKV, D = Shape::D, R = Shape::RATIO, G = H / HKV;
    constexpr int NT = D / 8;   // output n8 tiles
    static_assert(G <= 16, "the query heads of one KV head fill one m16 tile");
    static_assert(D % 64 == 0, "row staging in 16 B chunks by 64 threads");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto & sm = *reinterpret_cast<AttnSmem<Shape> *>(smem_raw);
    const int t = blockIdx.x, kv = blockIdx.y, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int g = lane / 4, qd = lane % 4;
    const int pos = pos0 + t, seen = (pos + 1) / R, nsel = n_blocks[t];
    const int all_cells = R * nsel + (pos + 1 - R * seen);
    const int begin = SPLIT ? blockIdx.z * cells_per_split : 0;
    const int n_cells = SPLIT ? min(all_cells, begin + cells_per_split) : all_cells;   // end of this CTA's range
    const int * sel = blocks + (size_t) t * Shape::TOP_BLOCKS;
    const float scale_log2 = 1.44269504f / sqrtf((float) D);

    // q rows G..15 are zero
    for (int i = tid; i < 16 * D / 8; i += 32 * ATTN_WARPS) {
        const int r = i / (D / 8), c = 8 * (i % (D / 8));
        uint4 val = make_uint4(0, 0, 0, 0);
        if (r < G) val = *reinterpret_cast<const uint4 *>(q + ((size_t) t * H + kv * G + r) * D + c);
        *reinterpret_cast<uint4 *>(&sm.q[r][c]) = val;
    }

    float o[NT][4] = {};
    float m[2] = { -INFINITY, -INFINITY }, l[2] = { 0.f, 0.f };   // rows g, g + 8
    for (int c0 = begin; c0 < n_cells; c0 += TILE) {
        __syncthreads();   // previous tile consumed (and q stored, first time)
        for (int i = tid; i < TILE * D / 8; i += 32 * ATTN_WARPS) {
            const int r = i / (D / 8), c = 8 * (i % (D / 8)), cell = c0 + r;
            if (cell < n_cells) {
                const int p = cell < R * nsel ? R * sel[cell / R] + cell % R : R * seen + (cell - R * nsel);
                const size_t off = ((size_t) p * HKV + kv) * D + c;
                if (kc.kq) {
                    *reinterpret_cast<uint4 *>(&sm.k[r][c]) = dequant8(kc.kq, kc.ks, off, D);
                    *reinterpret_cast<uint4 *>(&sm.v[r][c]) = dequant8(kc.vq, kc.vs, off, D);
                } else {
                    cp16(&sm.k[r][c], k + off);
                    cp16(&sm.v[r][c], v + off);
                }
            } else {
                *reinterpret_cast<uint4 *>(&sm.k[r][c]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4 *>(&sm.v[r][c]) = make_uint4(0, 0, 0, 0);
            }
        }
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        // s = q k^T over this warp's 16 cells: two n8 tiles
        float s[2][4] = {};
        const int r0 = 16 * warp;
#pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            uint32_t a[4], b[4];
            ldsm_x4(a, &sm.q[lane % 8 + 8 * ((lane / 8) % 2)][16 * kk + 8 * (lane / 16)]);
            ldsm_x4(b, &sm.k[r0 + lane % 8 + 8 * (lane / 16)][16 * kk + 8 * ((lane / 8) % 2)]);
            mma_f32(a, b[0], b[1], s[0]);
            mma_f32(a, b[2], b[3], s[1]);
        }
        // online softmax (base 2), rows g and g + 8
        float mx[2] = { -INFINITY, -INFINITY };
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const bool ok = c0 + r0 + 8 * j + 2 * qd + (e & 1) < n_cells;
                s[j][e] = ok ? s[j][e] * scale_log2 : -INFINITY;
                mx[e / 2] = fmaxf(mx[e / 2], s[j][e]);
            }
        float corr[2], base[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
            const float mn = fmaxf(m[i], mx[i]);
            base[i] = mn == -INFINITY ? 0.f : mn;   // a row with no cell yet
            corr[i] = exp2f(m[i] - base[i]);
            m[i] = mn;
            l[i] *= corr[i];
        }
        uint32_t pa[4];   // P as the A fragment (k = cells): {row g k 0-7, row g+8 k 0-7, row g k 8-15, row g+8 k 8-15}
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const half2 p = __floats2half2_rn(exp2f(s[j][2 * i] - base[i]), exp2f(s[j][2 * i + 1] - base[i]));
                const float2 pf = __half22float2(p);   // the rowsum uses the rounded weights P.V sees
                l[i] += pf.x + pf.y;
                pa[2 * j + i] = *reinterpret_cast<const uint32_t *>(&p);
            }
#pragma unroll
        for (int n = 0; n < NT; ++n) o[n][0] *= corr[0], o[n][1] *= corr[0], o[n][2] *= corr[1], o[n][3] *= corr[1];
#pragma unroll
        for (int n2 = 0; n2 < NT / 2; ++n2) {
            uint32_t b[4];
            ldsm_x4_t(b, &sm.v[r0 + lane % 8 + 8 * ((lane / 8) % 2)][16 * n2 + 8 * (lane / 16)]);
            mma_f32(pa, b[0], b[1], o[2 * n2]);
            mma_f32(pa, b[2], b[3], o[2 * n2 + 1]);
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
    }

    // merge warp 1 into warp 0 through the (consumed) K/V tiles
    static_assert(sizeof(float) * (16 * D + 32) <= sizeof(AttnSmem<Shape>::k) + sizeof(AttnSmem<Shape>::v),
                  "merge buffer fits in the K/V tiles");
    static_assert(ATTN_WARPS == 2, "two-way merge");
    __syncthreads();
    float * mo = reinterpret_cast<float *>(&sm.k[0][0]);   // [16][D]
    float * ml = mo + 16 * D;                               // m [16], l [16]
    if (warp == 1) {
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) mo[(g + 8 * (e / 2)) * D + 8 * n + 2 * qd + (e & 1)] = o[n][e];
        if (qd == 0)
            for (int i = 0; i < 2; ++i) ml[g + 8 * i] = m[i], ml[16 + g + 8 * i] = l[i];
    }
    __syncthreads();
    if (warp == 1) return;
    if constexpr (SPLIT) {
        float * P = part + (((size_t) t * HKV + kv) * gridDim.z + blockIdx.z) * Partial<Shape>::FLOATS;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int r = g + 8 * i;
            if (r >= G) continue;
            const float m1 = ml[r], l1 = ml[16 + r], mm = fmaxf(m[i], m1);
            const float a = m[i] == -INFINITY ? 0.f : exp2f(m[i] - mm), b = m1 == -INFINITY ? 0.f : exp2f(m1 - mm);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int d = 8 * n + 2 * qd;
                *reinterpret_cast<float2 *>(P + r * D + d) =
                    make_float2(o[n][2 * i] * a + mo[r * D + d] * b, o[n][2 * i + 1] * a + mo[r * D + d + 1] * b);
            }
            if (qd == 0) P[G * D + r] = mm, P[G * D + G + r] = l[i] * a + l1 * b;
        }
        return;
    }
    float f0[2], f1[2], inv[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float m1 = ml[g + 8 * i], l1 = ml[16 + g + 8 * i], mm = fmaxf(m[i], m1);
        f0[i] = m[i] == -INFINITY ? 0.f : exp2f(m[i] - mm);
        f1[i] = m1 == -INFINITY ? 0.f : exp2f(m1 - mm);
        inv[i] = 1.f / (l[i] * f0[i] + l1 * f1[i]);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = g + 8 * i;
        if (r >= G) continue;
        const size_t row = (size_t) t * H + kv * G + r;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int d = 8 * n + 2 * qd;
            const float2 gt = *reinterpret_cast<const float2 *>(gate + row * D + d);
            const float a0 = (o[n][2 * i] * f0[i] + mo[r * D + d] * f1[i]) * inv[i];
            const float a1 = (o[n][2 * i + 1] * f0[i] + mo[r * D + d + 1] * f1[i]) * inv[i];
            *reinterpret_cast<half2 *>(out + row * D + d) =
                __floats2half2_rn(a0 / (1.f + __expf(-gt.x)), a1 / (1.f + __expf(-gt.y)));
        }
    }
}

// out [t][kv G + r][d] = gated merge of the splits' states, in split order
template <class Shape>
__global__ void __launch_bounds__(256) combine_kernel(const float * part, int splits, const float * gate, half * out)
{
    constexpr int H = Shape::H, HKV = Shape::HKV, D = Shape::D, G = H / HKV;
    const int t = blockIdx.x, kv = blockIdx.y;
    const float * P0 = part + ((size_t) t * HKV + kv) * splits * Partial<Shape>::FLOATS;
    for (int i = threadIdx.x; i < G * D; i += blockDim.x) {
        const int r = i / D, d = i % D;
        float mm = -INFINITY;
        for (int z = 0; z < splits; ++z) mm = fmaxf(mm, P0[z * Partial<Shape>::FLOATS + G * D + r]);
        float num = 0.f, den = 0.f;
        for (int z = 0; z < splits; ++z) {
            const float * P = P0 + z * Partial<Shape>::FLOATS;
            const float m = P[G * D + r];
            if (m == -INFINITY) continue;
            const float w = exp2f(m - mm);
            num += P[r * D + d] * w;
            den += P[G * D + G + r] * w;
        }
        const size_t row = (size_t) t * H + kv * G + r;
        const float gt = gate[row * D + d];
        out[row * D + d] = __float2half(num / den / (1.f + __expf(-gt)));
    }
}

// splits per query: ranges of >= 2 tiles, at most MAX_SPLITS
constexpr int MAX_SPLITS = 32;

template <class Shape> int max_query_cells(int pos0, int T)   // the most cells any query of the chunk attends to
{
    return std::min(pos0 + T, Shape::RATIO * Shape::TOP_BLOCKS + Shape::RATIO - 1);
}

int split_count(int max_cells) { return std::max(1, std::min(MAX_SPLITS, (max_cells + 2 * TILE - 1) / (2 * TILE))); }

}  // namespace

template <class Shape> size_t attention_workspace_bytes(int max_queries)
{
    if (max_queries > SPLIT_ROWS) return 0;
    return (size_t) max_queries * Shape::HKV * MAX_SPLITS * Partial<Shape>::FLOATS * sizeof(float);
}

template <class Shape> size_t select_workspace_bytes(int max_queries, int n_ctx)
{
    const int rows = std::min(max_queries, 1024);
    return (size_t) rows * (n_ctx / Shape::RATIO) * sizeof(float);
}

template <class Shape>
void select(const half * idx_q, const half * idx_k, int pos0, int T, int * blocks, int * n_blocks, void * ws,
            size_t ws_bytes, cudaStream_t stream)
{
    const int ld = (pos0 + T) / Shape::RATIO;   // blocks the last query can see
    if (T <= 0 || pos0 < 0) throw std::invalid_argument("dsa::select: bad chunk");
    const int rows = ld > 0 ? (int) std::min<size_t>(T, ws_bytes / ((size_t) ld * sizeof(float))) : T;
    if (rows < std::min(T, SQ))
        throw std::invalid_argument("dsa::select: workspace holds " + std::to_string(rows) + " score rows, need " +
                                    std::to_string(std::min(T, SQ)));
    float * scores = static_cast<float *>(ws);
    for (int q0 = 0; q0 < T; q0 += rows) {
        const int n_q = std::min(rows, T - q0);
        const int last_seen = (pos0 + q0 + n_q) / Shape::RATIO;
        if (last_seen > Shape::TOP_BLOCKS)   // some query needs scores
            score_kernel<Shape><<<dim3((n_q + SQ - 1) / SQ, (last_seen + SB - 1) / SB), 256, 0, stream>>>(
                idx_q, idx_k, pos0, q0, n_q, scores, ld);
        select_kernel<Shape><<<n_q, SELECT_THREADS, 0, stream>>>(scores, ld, pos0, q0, blocks, n_blocks);
    }
    TRUSS_CUDA(cudaGetLastError());
}

template <class Shape>
void attention(const half * q, const float * gate, const half * k, const half * v, const int * blocks,
               const int * n_blocks, int pos0, int T, half * out, void * ws, size_t ws_bytes, cudaStream_t stream)
{
    KvCache kc;
    kc.k16 = const_cast<half *>(k), kc.v16 = const_cast<half *>(v);
    attention<Shape>(q, gate, kc, blocks, n_blocks, pos0, T, out, ws, ws_bytes, stream);
}

template <class Shape>
void attention(const half * q, const float * gate, const KvCache & kc, const int * blocks, const int * n_blocks,
               int pos0, int T, half * out, void * ws, size_t ws_bytes, cudaStream_t stream)
{
    if (T <= 0 || pos0 < 0) throw std::invalid_argument("dsa::attention: bad chunk");
    const size_t smem = sizeof(AttnSmem<Shape>);
    static bool attr = [&] {
        TRUSS_CUDA(cudaFuncSetAttribute(attn_kernel<Shape, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        TRUSS_CUDA(cudaFuncSetAttribute(attn_kernel<Shape, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        return true;
    }();
    (void) attr;
    if (T <= SPLIT_ROWS) {
        if (ws_bytes < attention_workspace_bytes<Shape>(T))
            throw std::invalid_argument("dsa::attention: split workspace too small");
        const int max_cells = max_query_cells<Shape>(pos0, T), S = split_count(max_cells);
        const int per = ((max_cells + S - 1) / S + TILE - 1) / TILE * TILE;   // whole tiles per split
        auto * part = static_cast<float *>(ws);
        attn_kernel<Shape, true><<<dim3(T, Shape::HKV, S), 32 * ATTN_WARPS, smem, stream>>>(
            q, gate, kc, blocks, n_blocks, pos0, out, per, part);
        combine_kernel<Shape><<<dim3(T, Shape::HKV), 256, 0, stream>>>(part, S, gate, out);
    } else {
        attn_kernel<Shape, false><<<dim3(T, Shape::HKV), 32 * ATTN_WARPS, smem, stream>>>(
            q, gate, kc, blocks, n_blocks, pos0, out, 0, nullptr);
    }
    TRUSS_CUDA(cudaGetLastError());
}

template size_t select_workspace_bytes<FlashNext>(int, int);
template void select<FlashNext>(const half *, const half *, int, int, int *, int *, void *, size_t, cudaStream_t);
template size_t attention_workspace_bytes<FlashNext>(int);
template void attention<FlashNext>(const half *, const float *, const half *, const half *, const int *, const int *,
                                   int, int, half *, void *, size_t, cudaStream_t);
template void attention<FlashNext>(const half *, const float *, const KvCache &, const int *, const int *, int, int,
                                   half *, void *, size_t, cudaStream_t);

}  // namespace truss::dsa
