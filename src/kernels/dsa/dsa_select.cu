// DSA block selection (see dsa_prefill.cuh): the indexer's scores and the top TOP_BLOCKS blocks per query.
//   score_kernel      64 queries x 64 blocks per CTA, 8 warps of 16 queries x 32 blocks; each warp keeps all IH heads'
//                     accumulators so relu and the head sum happen in registers. Fragments come straight from
//                     global memory (rows are 256 B; the op is ~2% of prefill FLOPs, TRACKER #49).
//   select_kernel     one CTA per query: 4 radix passes of 8 bits find the TOP_BLOCKS-th largest score key (scores are
//                     >= 0, so the fp32 bits order like the values), then one pass from the last block down keeps the
//                     keys above it and the latest keys equal to it, writing ids from the end so they come out
//                     ascending.
//   select_few_kernel / select_multi_kernel  the same for a few rows (decode, verify windows): one 1024-thread CTA
//                     per query, or SEL_NB CTAs per query that cooperate (all resident at once).
#include <atomic>
#include "dsa_prefill.cuh"

#include "core/cuda_check.h"
#include "kernels/trellis/mma.cuh"

#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <cstdlib>
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
// FEW (<= 16 queries: decode steps, verify windows): every warp takes the same 16 queries and its own 32 blocks, so a
// CTA covers 8 x 32 blocks with all 8 warps busy (the 64-query tiling leaves 6 of them idle); same per-(query, block)
// arithmetic, so the same scores
template <class Shape, bool FEW = false>
__global__ void __launch_bounds__(256) score_kernel(const half * idx_q, const half * idx_k, int pos0, int q0, int n_q,
                                                    float * scores, int ld)
{
    constexpr int IH = Shape::IH, ID = Shape::ID, R = Shape::RATIO;
    static_assert(ID % 16 == 0, "indexer head dim in k16 steps");
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4;
    const int tq0 = FEW ? 0 : blockIdx.x * SQ + (warp % 4) * 16;              // local query of fragment row 0
    const int b0 = FEW ? blockIdx.y * 256 + warp * 32 : blockIdx.y * SB + (warp / 4) * 32;   // block of column 0
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

// Few queries (decode steps, verify windows): the same selection with 1024 threads per query, and the final pass
// over contiguous per-thread ranges of blocks (two block scans) instead of SELECT_THREADS blocks per round with two
// scans each (148 us per call at 150K context, TRACKER #111). Same keys, same tie rule, same output.
constexpr int FEW_THREADS = 1024;

template <class Shape>
__global__ void __launch_bounds__(FEW_THREADS) select_few_kernel(const float * scores, int ld, int pos0, int q0,
                                                                 int * blocks, int * n_blocks)
{
    constexpr int R = Shape::RATIO, TOP = Shape::TOP_BLOCKS;
    using Scan = cub::BlockScan<int, FEW_THREADS>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ unsigned hist[256];
    __shared__ int after[FEW_THREADS];
    __shared__ unsigned s_prefix, s_mask, s_need;
    const int tq = blockIdx.x, t = q0 + tq, tid = threadIdx.x;
    const int seen = (pos0 + t + 1) / R;
    int * out = blocks + (size_t) t * TOP;
    if (seen <= TOP) {
        for (int b = tid; b < seen; b += FEW_THREADS) out[b] = b;
        if (tid == 0) n_blocks[t] = seen;
        return;
    }
    const float * row = scores + (size_t) tq * ld;
    auto key = [&](int b) { return __float_as_uint(row[b]); };

    if (tid == 0) s_prefix = 0, s_mask = 0, s_need = TOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += FEW_THREADS) hist[i] = 0;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask;
        for (int b = tid; b < seen; b += FEW_THREADS) {
            const unsigned k = key(b);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255], 1u);
        }
        __syncthreads();
        if (tid < 32) {   // the bin holding the need-th largest key: warp 0, 8 bins per lane from the top
            const int lane = tid;
            unsigned c[8], sum = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) c[i] = hist[255 - 8 * lane - i], sum += c[i];
            unsigned incl = sum;   // keys in the bins at or above this lane's (inclusive scan from the top)
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned v = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += v;
            }
            const unsigned need = s_need, above0 = incl - sum;
            const bool mine = above0 < need && incl >= need;
            if (mine) {
                unsigned above = above0;
                for (int i = 0; i < 8; ++i) {
                    if (above + c[i] >= need) {
                        const unsigned bin = 255 - 8 * lane - i;
                        s_prefix = prefix | bin << shift;
                        s_mask = mask | 255u << shift;
                        s_need = need - above;
                        break;
                    }
                    above += c[i];
                }
            }
        }
        __syncthreads();
    }
    const unsigned kth = s_prefix;
    const int need_eq = (int) s_need;

    // thread i owns blocks [lo, hi); counts of later threads come from scans over the reversed thread order
    const int per = (seen + FEW_THREADS - 1) / FEW_THREADS;
    const int lo = min(seen, tid * per), hi = min(seen, lo + per);
    int gt = 0, eq = 0;
    for (int b = lo; b < hi; ++b) {
        const unsigned k = key(b);
        gt += k > kth, eq += k == kth;
    }
    auto later = [&](int v) {   // sum of v over the threads after this one
        after[FEW_THREADS - 1 - tid] = v;
        __syncthreads();
        int x = after[tid], ex;
        __syncthreads();
        Scan(scan).ExclusiveSum(x, ex);
        __syncthreads();
        after[FEW_THREADS - 1 - tid] = ex;
        __syncthreads();
        const int r = after[tid];
        __syncthreads();
        return r;
    };
    const int eq_after = later(eq);
    const int eq_keep = max(0, min(eq, need_eq - eq_after));   // this range's latest equal keys
    int pos = later(gt + eq_keep);                              // kept blocks after this range
    int eq_seen = 0;
    for (int b = hi - 1; b >= lo; --b) {
        const unsigned k = key(b);
        bool take = k > kth;
        if (k == kth) take = eq_seen++ < eq_keep;
        if (take) out[TOP - 1 - pos++] = b;
    }
    if (tid == 0) n_blocks[t] = TOP;
}

// Multi-CTA form of select_few (decode, verify windows at long context): SEL_NB CTAs per query each histogram a
// strided share of the keys into a global histogram per radix pass, wait at a grid barrier, and all derive the same
// prefix from it; the final pass gives each CTA a contiguous range and the counts of the later CTAs. Same kth, same
// tie rule (the latest equal keys), same ascending output as select_few. The CTAs of a launch must be co-resident:
// n_q * SEL_NB <= the SM count (one 1024-thread CTA per SM). The sync state is per device and is left zero by the last
// CTA out.
constexpr int SEL_NB = 8, SEL_MAXQ = 8;
struct SelSync {
    unsigned hist[4][256];
    int gt[SEL_NB], eq[SEL_NB];
    unsigned bar, out;
};
__device__ SelSync g_sel[SEL_MAXQ];

__device__ __forceinline__ void sel_barrier(unsigned * bar, unsigned target)
{
    __threadfence();   // every thread's global atomics and stores before the arrival
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(bar, 1u);
        while (*(volatile unsigned *) bar < target) __nanosleep(32);
        __threadfence();
    }
    __syncthreads();
}

template <class Shape>
__global__ void __launch_bounds__(FEW_THREADS) select_multi_kernel(const float * scores, int ld, int pos0, int q0,
                                                                   int * blocks, int * n_blocks)
{
    constexpr int R = Shape::RATIO, TOP = Shape::TOP_BLOCKS;
    using Scan = cub::BlockScan<int, FEW_THREADS>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ unsigned hist[256];
    __shared__ int after[FEW_THREADS];
    __shared__ unsigned s_prefix, s_mask, s_need;
    __shared__ int s_gt, s_eq;
    const int tq = blockIdx.y, t = q0 + tq, tid = threadIdx.x, cb = blockIdx.x;
    SelSync & G = g_sel[tq];
    const int seen = (pos0 + t + 1) / R;
    int * out = blocks + (size_t) t * TOP;
    if (seen <= TOP) {   // the launcher only sends long rows here, but stay correct
        if (cb == 0) {
            for (int b = tid; b < seen; b += FEW_THREADS) out[b] = b;
            if (tid == 0) n_blocks[t] = seen;
        }
        return;
    }
    const float * row = scores + (size_t) tq * ld;
    auto key = [&](int b) { return __float_as_uint(row[b]); };
    unsigned prefix = 0, mask = 0, need = TOP, phase = 0;
    for (int p = 0, shift = 24; shift >= 0; ++p, shift -= 8) {
        for (int i = tid; i < 256; i += FEW_THREADS) hist[i] = 0;
        __syncthreads();
        for (int b = cb * FEW_THREADS + tid; b < seen; b += SEL_NB * FEW_THREADS) {
            const unsigned k = key(b);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255], 1u);
        }
        __syncthreads();
        for (int i = tid; i < 256; i += FEW_THREADS)
            if (hist[i]) atomicAdd(&G.hist[p][i], hist[i]);
        sel_barrier(&G.bar, SEL_NB * ++phase);
        if (tid < 32) {   // as select_few, on the global histogram (every CTA derives the same bin)
            const int lane = tid;
            unsigned c[8], sum = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) c[i] = __ldcg(&G.hist[p][255 - 8 * lane - i]), sum += c[i];
            unsigned incl = sum;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned v = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += v;
            }
            const unsigned above0 = incl - sum;
            if (above0 < need && incl >= need) {
                unsigned above = above0;
                for (int i = 0; i < 8; ++i) {
                    if (above + c[i] >= need) {
                        const unsigned bin = 255 - 8 * lane - i;
                        s_prefix = prefix | bin << shift;
                        s_mask = mask | 255u << shift;
                        s_need = need - above;
                        break;
                    }
                    above += c[i];
                }
            }
        }
        __syncthreads();
        prefix = s_prefix, mask = s_mask, need = s_need;
    }
    const unsigned kth = prefix;
    const int need_eq = (int) need;

    // final pass: CTA cb owns blocks [clo, chi), thread i a contiguous piece of it
    const int cper = (seen + SEL_NB - 1) / SEL_NB, clo = min(seen, cb * cper), chi = min(seen, clo + cper);
    const int per = (chi - clo + FEW_THREADS - 1) / FEW_THREADS;
    const int lo = min(chi, clo + tid * per), hi = min(chi, lo + per);
    int gt = 0, eq = 0;
    for (int b = lo; b < hi; ++b) {
        const unsigned k = key(b);
        gt += k > kth, eq += k == kth;
    }
    // later threads' counts within the CTA (reversed exclusive scans, as select_few)
    auto later = [&](int v, int & total) {
        after[FEW_THREADS - 1 - tid] = v;
        __syncthreads();
        int x = after[tid], ex;
        __syncthreads();
        Scan(scan).ExclusiveSum(x, ex, total);
        __syncthreads();
        after[FEW_THREADS - 1 - tid] = ex;
        __syncthreads();
        const int r = after[tid];
        __syncthreads();
        return r;
    };
    int gt_cta, eq_cta;
    const int gt_after = later(gt, gt_cta), eq_after = later(eq, eq_cta);
    if (tid == 0) G.gt[cb] = gt_cta, G.eq[cb] = eq_cta;
    sel_barrier(&G.bar, SEL_NB * ++phase);
    if (tid == 0) {   // the later CTAs' counts
        int g2 = 0, e2 = 0;
        for (int j = cb + 1; j < SEL_NB; ++j) g2 += __ldcg(&G.gt[j]), e2 += __ldcg(&G.eq[j]);
        s_gt = g2, s_eq = e2;
    }
    __syncthreads();
    const int eq_later = eq_after + s_eq;                          // equal keys after this thread's range
    const int eq_keep = max(0, min(eq, need_eq - eq_later));       // this range's latest equal keys
    const int kept_cta_after = s_gt + min(s_eq, need_eq);          // kept in the later CTAs
    int dummy;
    int pos = kept_cta_after + later(gt + eq_keep, dummy);         // kept blocks after this range
    int eq_seen = 0;
    for (int b = hi - 1; b >= lo; --b) {
        const unsigned k = key(b);
        bool take = k > kth;
        if (k == kth) take = eq_seen++ < eq_keep;
        if (take) out[TOP - 1 - pos++] = b;
    }
    if (cb == 0 && tid == 0) n_blocks[t] = TOP;
    // the last CTA out leaves the sync state zero for the next launch
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(&G.out, 1u) == SEL_NB - 1) {
            for (int p = 0; p < 4; ++p)
                for (int i = 0; i < 256; ++i) G.hist[p][i] = 0;
            for (int j = 0; j < SEL_NB; ++j) G.gt[j] = 0, G.eq[j] = 0;
            G.bar = 0;
            __threadfence();
            G.out = 0;
        }
    }
}

// select_multi_kernel's CTAs must all be resident at once (they wait for each other): one 1024-thread CTA per SM
bool sel_multi_ok(int n_q)
{
    static const bool off = std::getenv("TRUSS_DSA_SELECT_MULTI") && !std::atoi(std::getenv("TRUSS_DSA_SELECT_MULTI"));
    if (off) return false;
    int dev = 0, sms = 0;
    TRUSS_CUDA(cudaGetDevice(&dev));
    TRUSS_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    return n_q * SEL_NB <= sms;
}


}  // namespace

template <class Shape> size_t select_workspace_bytes(int max_queries, int n_ctx)
{
    const int rows = std::min(max_queries, 1024);
    return (size_t) rows * (n_ctx / Shape::RATIO) * sizeof(float);
}

template <class Shape>
void select(const half * idx_q, const half * idx_k, int pos0, int T, int * blocks, int * n_blocks, void * ws,
            size_t ws_bytes, cudaStream_t stream, bool rowwise)
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
        if (last_seen > Shape::TOP_BLOCKS && n_q <= 16)   // some query needs scores
            score_kernel<Shape, true><<<dim3(1, (last_seen + 255) / 256), 256, 0, stream>>>(idx_q, idx_k, pos0, q0, n_q,
                                                                                          scores, ld);
        else if (last_seen > Shape::TOP_BLOCKS)
            score_kernel<Shape><<<dim3((n_q + SQ - 1) / SQ, (last_seen + SB - 1) / SB), 256, 0, stream>>>(
                idx_q, idx_k, pos0, q0, n_q, scores, ld);
        if (n_q <= SEL_MAXQ && !rowwise && sel_multi_ok(n_q) && (pos0 + q0 + 1) / Shape::RATIO > 8 * Shape::TOP_BLOCKS)
            select_multi_kernel<Shape><<<dim3(SEL_NB, n_q), FEW_THREADS, 0, stream>>>(scores, ld, pos0, q0, blocks, n_blocks);
        else if (n_q <= SELECT_FEW_ROWS && !rowwise)
            select_few_kernel<Shape><<<n_q, FEW_THREADS, 0, stream>>>(scores, ld, pos0, q0, blocks, n_blocks);
        else
            select_kernel<Shape><<<n_q, SELECT_THREADS, 0, stream>>>(scores, ld, pos0, q0, blocks, n_blocks);
    }
    TRUSS_CUDA(cudaGetLastError());
}


template size_t select_workspace_bytes<FlashNext>(int, int);
template void select<FlashNext>(const half *, const half *, int, int, int *, int *, void *, size_t, cudaStream_t, bool);

}  // namespace truss::dsa
