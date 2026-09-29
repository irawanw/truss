// MoE window: one persistent dataflow kernel (see moe_window.cuh).
//
// Why not llama-paw's x3m_moe_kernel: it hands one expert at a time to a group of blocks, pads each expert's rows to
// a 16-row GEMM tile and crosses four group barriers per expert, so a 4-row window over ~35 experts reads at 12% of
// the 3090's bandwidth (CP0a). Why not a chain of small kernels (v6): launch gaps, ramp-up and drain of each GEMV
// cost ~16 us of a 117 us window (TRACKER #19).
//
// One launch. Every block builds the routing table in shared memory (same result everywhere, no global table),
// then pulls work items from a queue in this order:
//   H   (slot, proj)          A_p[slot][i] = H128(x[row_i] * suh_p)                             p = gate, up
//   GU  (slot, proj, cols)    C_p[slot][i][cols] = A_p . W_p        waits: H(slot, p) done
//   D   (slot, cols)          C_d[slot][i][cols] = A_d . W_down     waits: mid(slot) done
// Dependencies are counters, not grid barriers:
//   - the block that finishes a slot's last GU item runs mid for it:
//         A_d[slot][i] = H128(silu(H128(C_g) * svh_g) * H128(C_u) * svh_u * suh_d)
//   - the block that finishes a column group's last D item runs combine for it:
//         out[row][cols] = sum over the row's routed experts, in routing order, of w * H128(C_d) * svh_d
// Waiting warps first issue their weight loads (they do not depend on the producer), then spin.
// Deadlock-free without co-residency: an item waits only on items earlier in the queue, which were already taken
// by running blocks. The last block to leave resets the counters for the next launch.
// Deterministic: fixed reduction orders, no float atomics.
//
// Weight layout: PAW X3 expert tensors (ProjView); tiles k-slice-major (all n tiles of a k slice contiguous).
#include "moe_window.cuh"

#include "../trellis/codec_mul1.cuh"
#include "../trellis/hadamard.cuh"
#include "../trellis/mma.cuh"

#include <cstdio>
#include <cstdlib>

namespace truss::moe {
namespace {

constexpr int THREADS = 256, WARPS = THREADS / 32;
constexpr int MAX_EXPERTS = 1024;          // host check only

__device__ __forceinline__ unsigned long long now_ns()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

// GEMV geometry: a block is WK x WG warps; WK warps split the k slices, WG column groups of WNT 16-column tiles;
// PF = k slices in flight per warp (loads ring).
template <int WK_, int WG_, int PF_, int WNT_>
struct Geom {
    static constexpr int WK = WK_, WG = WG_, PF = PF_, WNT = WNT_;
    static constexpr int COLS = WNT * 16;                 // columns per warp
    static constexpr int BLOCK_COLS = WG * COLS;          // columns per work item
    static_assert(WK * WG == WARPS, "block is 8 warps");
};

// Per-shape GEMV geometry, best measured on the 3090 (cp3_moe_tune.log v6).
template <class Shape> struct Tune;
template <> struct Tune<FlashNext> {
    using GateUp = Geom<8, 1, 2, 8>;       // 8 k-split warps x 8 tiles
    using Down = Geom<4, 2, 2, 8>;         // 4 k-split warps x 2 groups x 8 tiles
};

// Compile-time sizes of one shape
template <class Shape>
struct Plan {
    static constexpr int D = Shape::D_MODEL, F = Shape::D_FF, TOPK = Shape::TOPK;
    static constexpr int SLOTS = MAX_ROWS * TOPK;          // distinct experts a window can route to
    using GU = typename Tune<Shape>::GateUp;
    using DN = typename Tune<Shape>::Down;
    static constexpr int GU_KSPLIT = 1;                     // K parts per gate/up item; 2 measured 5% slower (TRACKER #24)
    static constexpr int GU_GROUPS = F / GU::BLOCK_COLS;   // column groups per (slot, proj)
    static constexpr int GU_ITEMS = 2 * GU_GROUPS * GU_KSPLIT;   // GU items per slot
    static constexpr int GU_KSLICES = D / 16 / GU_KSPLIT;   // k slices per GU item
    static constexpr int DN_GROUPS = D / DN::BLOCK_COLS;   // D items per slot
    static constexpr int RED_FLOATS = (GU::WK * GU::BLOCK_COLS > DN::WK * DN::BLOCK_COLS ? GU::WK * GU::BLOCK_COLS
                                                                                            : DN::WK * DN::BLOCK_COLS)
                                      * MAX_ROWS;
    static_assert(D % 128 == 0 && F % 128 == 0, "Hadamard-128 blocks");
    static_assert(F % GU::BLOCK_COLS == 0, "gate/up column groups must tile D_FF");
    static_assert(D % DN::BLOCK_COLS == 0, "down column groups must tile D_MODEL");
    static_assert(DN::BLOCK_COLS % 128 == 0, "combine works on whole Hadamard blocks of a down group");
    static_assert(SLOTS <= THREADS, "routing: one thread per (row, choice) pair");
    static_assert((D / 16) % GU_KSPLIT == 0, "gate/up K split must divide the k slices");
};

// Queue and dependency counters. Zero at the start of every launch (workspace_init, then the kernel's own reset).
template <class Shape>
struct Counters {
    int next_item;
    int exited;
    int h_done[Plan<Shape>::SLOTS];        // H items done per slot (ready at 2)
    int gu_done[Plan<Shape>::SLOTS];       // GU items done per slot
    int mid_done[Plan<Shape>::SLOTS];      // 1 when A_d of the slot is written
    int d_done[Plan<Shape>::DN_GROUPS];    // D items done per column group
};

// Workspace layout, shared by workspace_bytes() and window()
template <class Shape>
struct Workspace {
    using P = Plan<Shape>;
    Counters<Shape> * ctr;
    half * A_gu;     // [2][SLOTS][MAX_ROWS][D]
    float * C_gu;    // [GU_KSPLIT][2][SLOTS][MAX_ROWS][F], partial sums over each K part
    half * A_d;      // [SLOTS][MAX_ROWS][F]
    float * C_d;     // [SLOTS][MAX_ROWS][D]
    size_t bytes;

    explicit Workspace(void * base)
    {
        char * p = (char *) base;
        auto take = [&] (size_t n) { char * r = p; p += (n + 255) & ~(size_t) 255; return r; };
        ctr = (Counters<Shape> *) take(sizeof(Counters<Shape>));
        A_gu = (half *) take(sizeof(half) * 2 * P::SLOTS * MAX_ROWS * P::D);
        C_gu = (float *) take(sizeof(float) * P::GU_KSPLIT * 2 * P::SLOTS * MAX_ROWS * P::F);
        A_d = (half *) take(sizeof(half) * P::SLOTS * MAX_ROWS * P::F);
        C_d = (float *) take(sizeof(float) * P::SLOTS * MAX_ROWS * P::D);
        bytes = (size_t) (p - (char *) base);
    }
};

// ---------------------------------------------------------------------------------------------------------
// routing, built by every block in shared memory

struct Slot {
    int expert;
    int count;
    int rows[MAX_ROWS];
    float w[MAX_ROWS];
};

template <class Shape>
struct Route {
    int n;                                  // active slots
    Slot slot[Plan<Shape>::SLOTS];
    int2 meta[Plan<Shape>::SLOTS][3];       // (K, word offset) per projection
    unsigned char pair_slot[Plan<Shape>::SLOTS], pair_idx[Plan<Shape>::SLOTS];   // (row, choice) -> slot, row index
};

// Pair j = (row j / TOPK, choice j % TOPK). A pair opens a slot iff no earlier pair routes to its expert; slots are
// numbered in scan order (prefix count of openers) and hold their rows in scan order.
template <class Shape>
__device__ __noinline__ void build_route(const int * __restrict__ ids, const float * __restrict__ wts, int n_rows,
                            const Weights & W, Route<Shape> & R)
{
    using P = Plan<Shape>;
    __shared__ int s_ids[P::SLOTS], s_warp_open[WARPS];
    const int j = threadIdx.x, lane = j % 32, warp = j / 32, n_pairs = n_rows * P::TOPK;
    const int e = j < n_pairs ? ids[j] : -1;
    if (j < P::SLOTS) s_ids[j] = e;
    __syncthreads();
    int first = j, earlier = 0;
    if (j < n_pairs)
        for (int k = 0; k < j; ++k)
            if (s_ids[k] == e) { if (earlier == 0) first = k; ++earlier; }
    const bool opens = j < n_pairs && first == j;
    const unsigned ballot = __ballot_sync(0xffffffffu, opens);
    if (lane == 0) s_warp_open[warp] = __popc(ballot);
    __syncthreads();
    int my_slot = __popc(ballot & ((1u << lane) - 1u));
    for (int w = 0; w < warp; ++w) my_slot += s_warp_open[w];
    if (j == 0) {
        int n = 0;
        for (int w = 0; w < WARPS; ++w) n += s_warp_open[w];
        R.n = n;
    }
    if (opens) {
        int count = 0;
        for (int k = j; k < n_pairs; ++k) count += s_ids[k] == e;
        R.slot[my_slot].expert = e;
        R.slot[my_slot].count = count;
        R.pair_slot[j] = (unsigned char) my_slot;
    }
    __syncthreads();
    if (j < n_pairs) {
        const int sl = R.pair_slot[first];
        R.pair_slot[j] = (unsigned char) sl;
        R.pair_idx[j] = (unsigned char) earlier;
        R.slot[sl].rows[earlier] = j / P::TOPK;
        R.slot[sl].w[earlier] = wts[j];
    }
    __syncthreads();
    for (int t = threadIdx.x; t < R.n * 3; t += THREADS) {
        const int sl = t / 3, p = t % 3, ex = R.slot[sl].expert;
        R.meta[sl][p] = make_int2(W.proj[p].meta[2 * ex], W.proj[p].meta[2 * ex + 1]);
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------------------
// cross-block signalling

__device__ __forceinline__ int ld_acquire(const int * p)
{
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// every warp waits on its own (lane 0 spins, the warp follows)
__device__ __forceinline__ void wait_at_least(const int * flag, int target)
{
    if ((threadIdx.x & 31) == 0)
        while (ld_acquire(flag) < target) __nanosleep(64);
    __syncwarp();
}

// called by all threads after the block's writes: returns the counter value before this block's increment
__device__ __forceinline__ int block_signal(int * counter)
{
    __shared__ int s_old;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_old = atomicAdd(counter, 1);
    __syncthreads();
    __threadfence();
    return s_old;
}

// ---------------------------------------------------------------------------------------------------------
// GEMV over one work item = (slot, projection, BLOCK_COLS columns), all of K. The expert's rate picks the codec.
// `ready` runs after the first weight loads are issued and before any activation is read.

// K range of the item: k slices [k_off, k_off + k_slices) of 16; A rows are a_stride halfs apart.
template <class Codec, class G, class Ready>
__device__ __forceinline__ void gemv_item(const half * __restrict__ A, int a_stride, int rows,
                                          const uint32_t * __restrict__ B32, float * __restrict__ C, int k_off,
                                          int k_slices, int size_n, int item_group, float * __restrict__ red,
                                          Ready ready)
{
    constexpr int WK = G::WK, PF = G::PF, WNT = G::WNT;
    constexpr int FOLD = 2;                                // k slices per fp16 accumulation before the fp32 fold
    constexpr int LOADS = WNT / Codec::TILES_PER_VEC;      // warp-wide word loads per k slice
    static_assert(WNT % Codec::TILES_PER_VEC == 0, "tiles per warp must fill whole loads");
    const int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int warp = wid % WK;                 // k part
    const int wg = wid / WK;                   // column group inside the block
    const int group = item_group * G::WG + wg;
    const int ntiles = size_n / 16, kslices = k_slices;
    const int chunk = (kslices + WK - 1) / WK;
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * Codec::TILE_WORDS;
    const half2 * A2 = (const half2 *) A;
    const half2 hzero = __half2half2(__ushort_as_half(0));
    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (a_stride / 2);
    const bool r0_ok = r0 < rows;
    const Codec codec(lane);
    const bool lane_loads = codec.loads(lane);

    const uint32_t * bp = B32 + (size_t) (k_off + ks0) * slice_stride + group * WNT * Codec::TILE_WORDS + lane;
    auto ld_b = [&] (int i, int l) -> uint32_t {
        return lane_loads ? __ldcs(bp + (size_t) i * slice_stride + l * Codec::VEC_WORDS) : 0u;
    };
    // activations were written by other blocks in this launch: read through L2 (L1 is not coherent). They ride in
    // the same ring as the trellis words.
    auto ld_a = [&] (int i, half2 & lo, half2 & hi) {
        const size_t a_col = (size_t) (k_off + ks0 + i) * 8 + (lane & 3);
        const unsigned * a = (const unsigned *) (A2 + a_row0 + a_col);
        const unsigned vlo = r0_ok ? __ldcg(a) : 0u, vhi = r0_ok ? __ldcg(a + 4) : 0u;
        lo = *(const half2 *) &vlo;
        hi = *(const half2 *) &vhi;
    };
    uint32_t pf[PF][LOADS];
    half2 pa[PF][2];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn) {
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);
        }
    ready();
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn) ld_a(d, pa[d][0], pa[d][1]);

    FragCh ch[WNT];
    float2 acc[WNT][2];     // [t][0] = column g, rows (2q, 2q+1); [t][1] = column g + 8
    #pragma unroll
    for (int t = 0; t < WNT; ++t) {
        ch[t][0] = ch[t][1] = hzero;
        acc[t][0] = acc[t][1] = make_float2(0.f, 0.f);
    }

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;
            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];
            const half2 act_lo = pa[d][0], act_hi = pa[d][1];
            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
                ld_a(i + PF, pa[d][0], pa[d][1]);
            }
            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                codec.tile(bw[t / Codec::TILES_PER_VEC], t % Codec::TILES_PER_VEC, f0, f1);
                mma_w(f0, f1, act_lo, act_hi, ch[t]);
            }
            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc[t][f].x += __low2float(ch[t][f]);
                        acc[t][f].y += __high2float(ch[t][f]);
                        ch[t][f] = hzero;
                    }
            }
        }
    }

    // lane holds columns g = lane/4 and g + 8 of each tile, rows 2q and 2q + 1 (q = lane % 4); k parts are summed
    // in fixed order through shared memory: red[k part][row][block column]
    const int g = lane >> 2, q2 = 2 * (lane & 3);
    auto red_at = [&] (int k, int r, int c) -> float & { return red[(k * MAX_ROWS + r) * G::BLOCK_COLS + c]; };
    #pragma unroll
    for (int t = 0; t < WNT; ++t)
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            const int col = wg * G::COLS + t * 16 + f * 8 + g;
            red_at(warp, q2, col) = acc[t][f].x;
            red_at(warp, q2 + 1, col) = acc[t][f].y;
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < G::BLOCK_COLS * rows; idx += THREADS) {
        const int r = idx / G::BLOCK_COLS, c = idx % G::BLOCK_COLS;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < WK; ++k) sum += red_at(k, r, c);
        C[(size_t) r * size_n + item_group * G::BLOCK_COLS + c] = sum;
    }
    __syncthreads();
}

template <class G, class Ready>
__device__ __forceinline__ void gemv_dispatch(int2 meta, const ProjView & P, const half * A, int a_stride, int rows,
                                              float * C, int k_off, int k_slices, int size_n, int grp, float * red,
                                              Ready ready)
{
    const uint32_t * B32 = (const uint32_t *) (P.trellis + meta.y);
    switch (meta.x) {
        case 2: gemv_item<Mul1<2>, G>(A, a_stride, rows, B32, C, k_off, k_slices, size_n, grp, red, ready); break;
        case 3: gemv_item<Mul1<3>, G>(A, a_stride, rows, B32, C, k_off, k_slices, size_n, grp, red, ready); break;
        case 4: gemv_item<Mul1<4>, G>(A, a_stride, rows, B32, C, k_off, k_slices, size_n, grp, red, ready); break;
        default: break;   // unsupported rate: the pack loader rejects it before a kernel runs
    }
}

// ---------------------------------------------------------------------------------------------------------
// the small steps, run by one block each. Not inlined: they are off the hot path and would share the GEMV's
// register budget.

__device__ __forceinline__ float4 h4_to_f4(uint2 v)
{
    const half2 a = *(const half2 *) &v.x, b = *(const half2 *) &v.y;
    return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}

// The small steps are latency-bound, but batching their loads (4 tasks per warp) raised register demand and made the
// whole kernel spill: 155 vs 124 us (TRACKER #25). One task per iteration; the start-up and tail bubbles are to be
// filled with independent work (shared expert) in the layer kernel instead.

// A_p[slot][i][:] = H128(x[row_i] * suh_p[e]); warp tasks = (row, 128-column block)
template <class Shape>
__device__ __noinline__ void run_h(const float * __restrict__ x, const Slot & S, const half * __restrict__ suh,
                                   half * __restrict__ A)
{
    constexpr int D = Shape::D_MODEL, HB = D / 128;
    const int lane = threadIdx.x % 32;
    const half * sh = suh + (size_t) S.expert * D;
    for (int t = threadIdx.x / 32; t < S.count * HB; t += WARPS) {
        const int i = t / HB, c0 = (t % HB) * 128 + lane * 4;
        const float4 v = *(const float4 *) (x + (size_t) S.rows[i] * D + c0);
        const uint2 sc = *(const uint2 *) (sh + c0);
        // fp16 pre-scale like the reference (round input to fp16, multiply by fp16 suh), then fp32 Hadamard
        const half2 p01 = __hmul2(__floats2half2_rn(v.x, v.y), *(const half2 *) &sc.x);
        const half2 p23 = __hmul2(__floats2half2_rn(v.z, v.w), *(const half2 *) &sc.y);
        float h0 = __low2float(p01), h1 = __high2float(p01), h2 = __low2float(p23), h3 = __high2float(p23);
        had4x32(h0, h1, h2, h3, lane);
        half2 * o = (half2 *) (A + (size_t) i * D + c0);
        o[0] = __floats2half2_rn(h0, h1);
        o[1] = __floats2half2_rn(h2, h3);
    }
}

// A_d[slot][i] = H128(silu(H128(C_g) * svh_g) * H128(C_u) * svh_u * suh_d); warp tasks = (row, 128-column block)
// Cg, Cu: K part 0 of the slot; part k is k * part_stride floats further (parts summed in order)
template <class Shape>
__device__ __noinline__ void run_mid(const Slot & S, const float * __restrict__ Cg, const float * __restrict__ Cu,
                                     size_t part_stride, const Weights & W, half * __restrict__ A_d)
{
    constexpr int F = Shape::D_FF, HB = F / 128;
    const int lane = threadIdx.x % 32, e = S.expert;
    for (int t = threadIdx.x / 32; t < S.count * HB; t += WARPS) {
        const int i = t / HB, c0 = (t % HB) * 128 + lane * 4;
        const size_t off = (size_t) i * F + c0;
        float4 g = __ldcg((const float4 *) (Cg + off));
        float4 u = __ldcg((const float4 *) (Cu + off));
        for (int k = 1; k < Plan<Shape>::GU_KSPLIT; ++k) {
            const float4 g2 = __ldcg((const float4 *) (Cg + k * part_stride + off));
            const float4 u2 = __ldcg((const float4 *) (Cu + k * part_stride + off));
            g.x += g2.x; g.y += g2.y; g.z += g2.z; g.w += g2.w;
            u.x += u2.x; u.y += u2.y; u.z += u2.z; u.w += u2.w;
        }
        const float4 fg = h4_to_f4(*(const uint2 *) (W.proj[0].svh + (size_t) e * F + c0));
        const float4 fu = h4_to_f4(*(const uint2 *) (W.proj[1].svh + (size_t) e * F + c0));
        const uint2 sd = *(const uint2 *) (W.proj[2].suh + (size_t) e * F + c0);
        had4x32(g.x, g.y, g.z, g.w, lane);
        had4x32(u.x, u.y, u.z, u.w, lane);
        // round to fp16 before the down pre-scale, as the reference feeds fp16 activations into the down GEMV
        const half2 p01 = __hmul2(__floats2half2_rn(silu(g.x * fg.x) * (u.x * fu.x), silu(g.y * fg.y) * (u.y * fu.y)),
                                  *(const half2 *) &sd.x);
        const half2 p23 = __hmul2(__floats2half2_rn(silu(g.z * fg.z) * (u.z * fu.z), silu(g.w * fg.w) * (u.w * fu.w)),
                                  *(const half2 *) &sd.y);
        float h0 = __low2float(p01), h1 = __high2float(p01), h2 = __low2float(p23), h3 = __high2float(p23);
        had4x32(h0, h1, h2, h3, lane);
        half2 * o = (half2 *) (A_d + off);
        o[0] = __floats2half2_rn(h0, h1);
        o[1] = __floats2half2_rn(h2, h3);
    }
}

// out[row][cols of group] = sum over the row's routing choices, in order, of w * H128(C_d) * svh_d
template <class Shape>
__device__ __noinline__ void run_combine(const Route<Shape> & R, int n_rows, int grp, const float * __restrict__ C_d,
                                         const half * __restrict__ svh_d, float * __restrict__ out)
{
    using P = Plan<Shape>;
    constexpr int D = P::D, HB = P::DN::BLOCK_COLS / 128;
    const int lane = threadIdx.x % 32;
    for (int t = threadIdx.x / 32; t < n_rows * HB; t += WARPS) {
        const int row = t / HB, c0 = grp * P::DN::BLOCK_COLS + (t % HB) * 128 + lane * 4;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int s = 0; s < P::TOPK; ++s) {
            const int j = row * P::TOPK + s, sl = R.pair_slot[j], i = R.pair_idx[j];
            const Slot & S = R.slot[sl];
            float4 v = __ldcg((const float4 *) (C_d + ((size_t) sl * MAX_ROWS + i) * D + c0));
            const float4 f = h4_to_f4(*(const uint2 *) (svh_d + (size_t) S.expert * D + c0));
            had4x32(v.x, v.y, v.z, v.w, lane);
            const float w = S.w[i];
            acc.x += w * v.x * f.x;
            acc.y += w * v.y * f.y;
            acc.z += w * v.z * f.z;
            acc.w += w * v.w * f.w;
        }
        *(float4 *) (out + (size_t) row * D + c0) = acc;
    }
}

// ---------------------------------------------------------------------------------------------------------
// work queue: H items of all slots, then GU items slot by slot, then D items slot by slot. (Inserting each slot's
// D items one grid-wave after its GU items measured no gain, TRACKER #23.)

enum ItemKind { ITEM_H = 0, ITEM_GU = 1, ITEM_D = 2, ITEM_NONE = 3 };

struct Item { int kind, slot, proj, group, kpart; };

template <class Shape>
__device__ Item decode_item(int item, int n_slots)
{
    using P = Plan<Shape>;
    const int n_h = 2 * n_slots, n_gu = n_slots * P::GU_ITEMS, n_d = n_slots * P::DN_GROUPS;
    if (item < n_h) return { ITEM_H, item / 2, item % 2, 0, 0 };
    item -= n_h;
    if (item < n_gu) {
        const int r = item % P::GU_ITEMS;   // (proj, group, kpart), kpart fastest
        return { ITEM_GU, item / P::GU_ITEMS, r / (P::GU_GROUPS * P::GU_KSPLIT), (r / P::GU_KSPLIT) % P::GU_GROUPS,
                 r % P::GU_KSPLIT };
    }
    item -= n_gu;
    if (item < n_d) return { ITEM_D, item / P::DN_GROUPS, 2, item % P::DN_GROUPS, 0 };
    return { ITEM_NONE, 0, 0, 0, 0 };
}

template <class Shape>
__global__ __launch_bounds__(THREADS, 2) void window_kernel(Weights W, const float * __restrict__ x,
                                                            const int * __restrict__ ids,
                                                            const float * __restrict__ wts, int n_rows,
                                                            float * __restrict__ out, Workspace<Shape> ws,
                                                            TraceEvent * __restrict__ trace)
{
    using P = Plan<Shape>;
    constexpr int D = P::D, F = P::F, SLOTS = P::SLOTS;
    __shared__ Route<Shape> R;
    __shared__ __align__(16) float red[P::RED_FLOATS];
    __shared__ int s_item, s_last;
    __shared__ Item s_it;
    __shared__ unsigned long long s_t0, s_ready;
    Counters<Shape> & ctr = *ws.ctr;
    const unsigned long long t_enter = trace ? now_ns() : 0;
    const size_t gu_part = (size_t) 2 * SLOTS * MAX_ROWS * F;   // floats between K parts of C_gu

    // the first item's atomic is in flight while the block builds its routing table
    int first_item = 0;
    if (threadIdx.x == 0) first_item = atomicAdd(&ctr.next_item, 1);
    build_route<Shape>(ids, wts, n_rows, W, R);
    const int n = R.n;

    for (bool first = true;; first = false) {
        if (threadIdx.x == 0) {
            s_item = first ? first_item : atomicAdd(&ctr.next_item, 1);
            s_it = decode_item<Shape>(s_item, n);
        }
        __syncthreads();
        const Item it = s_it;
        const int item = s_item;
        __syncthreads();
        if (it.kind == ITEM_NONE) break;
        if (trace && threadIdx.x == 0) s_t0 = s_ready = now_ns();
        auto mark_ready = [&] { if (trace && threadIdx.x == 0) s_ready = now_ns(); };
        const int sl = it.slot;
        const Slot & S = R.slot[sl];

        if (it.kind == ITEM_H) {
            run_h<Shape>(x, S, W.proj[it.proj].suh, ws.A_gu + ((size_t) it.proj * SLOTS + sl) * MAX_ROWS * D);
            block_signal(&ctr.h_done[sl]);
        } else if (it.kind == ITEM_GU) {
            const int p = it.proj;
            float * C = ws.C_gu + it.kpart * gu_part + ((size_t) p * SLOTS + sl) * MAX_ROWS * F;
            gemv_dispatch<typename P::GU>(R.meta[sl][p], W.proj[p], ws.A_gu + ((size_t) p * SLOTS + sl) * MAX_ROWS * D,
                                          D, S.count, C, it.kpart * P::GU_KSLICES, P::GU_KSLICES, F, it.group, red,
                                          [&] { wait_at_least(&ctr.h_done[sl], 2); mark_ready(); });
            if (block_signal(&ctr.gu_done[sl]) == P::GU_ITEMS - 1) {   // last GU item of the slot: mid
                run_mid<Shape>(S, ws.C_gu + (size_t) sl * MAX_ROWS * F, ws.C_gu + ((size_t) SLOTS + sl) * MAX_ROWS * F,
                               gu_part, W, ws.A_d + (size_t) sl * MAX_ROWS * F);
                block_signal(&ctr.mid_done[sl]);
            }
        } else {
            gemv_dispatch<typename P::DN>(R.meta[sl][2], W.proj[2], ws.A_d + (size_t) sl * MAX_ROWS * F, F, S.count,
                                          ws.C_d + (size_t) sl * MAX_ROWS * D, 0, F / 16, D, it.group, red,
                                          [&] { wait_at_least(&ctr.mid_done[sl], 1); mark_ready(); });
            if (block_signal(&ctr.d_done[it.group]) == n - 1)          // last slot of the column group: combine
                run_combine<Shape>(R, n_rows, it.group, ws.C_d, W.proj[2].svh, out);
        }
        if (trace && threadIdx.x == 0 && item < MAX_TRACE_EVENTS)
            trace[item] = { (int) blockIdx.x, it.kind, sl, it.group, t_enter, s_t0, s_ready, now_ns() };
    }

    // the last block to leave resets the counters for the next launch (all items are done by then)
    if (threadIdx.x == 0) s_last = atomicAdd(&ctr.exited, 1) == (int) gridDim.x - 1;
    __syncthreads();
    if (s_last) {
        int * c = (int *) &ctr;
        for (int i = threadIdx.x; i < (int) (sizeof(Counters<Shape>) / sizeof(int)); i += THREADS) c[i] = 0;
    }
}

template <class Shape>
int persistent_grid()
{
    static int g = 0;
    if (!g) {
        int dev, sms, per_sm;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, window_kernel<Shape>, THREADS, 0);
        g = sms * per_sm;
    }
    return g;
}

}  // namespace

template <class Shape>
size_t workspace_bytes()
{
    return Workspace<Shape>(nullptr).bytes;
}

template <class Shape>
void workspace_init(void * ws, cudaStream_t stream)
{
    cudaMemsetAsync(ws, 0, sizeof(Counters<Shape>), stream);
}

template <class Shape>
void window(const Weights & W, const float * x, const int * ids, const float * wts, int n_rows, float * out, void * ws,
            cudaStream_t stream, TraceEvent * trace)
{
    if (n_rows < 1 || n_rows > MAX_ROWS || W.n_expert > MAX_EXPERTS) {
        fprintf(stderr, "truss::moe::window: n_rows %d (max %d), n_expert %d (max %d)\n", n_rows, MAX_ROWS, W.n_expert,
                MAX_EXPERTS);
        abort();
    }
    window_kernel<Shape><<<persistent_grid<Shape>(), THREADS, 0, stream>>>(W, x, ids, wts, n_rows, out,
                                                                          Workspace<Shape>(ws), trace);
}

template size_t workspace_bytes<FlashNext>();
template void workspace_init<FlashNext>(void *, cudaStream_t);
template void window<FlashNext>(const Weights &, const float *, const int *, const float *, int, float *, void *,
                                cudaStream_t, TraceEvent *);

}  // namespace truss::moe
