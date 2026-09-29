// MoE window kernel chain (see moe_window.cuh).
//
// Why not llama-paw's x3m_moe_kernel: it hands one expert at a time to a group of blocks, pads each expert's rows to
// a 16-row GEMM tile and crosses four group barriers per expert, so a 4-row window over ~35 experts reads at 12% of
// the 3090's bandwidth (CP0a). Here every active expert's column groups form one flat work list that persistent
// blocks drain, and the steps between the two GEMVs are small separate kernels (no grid barriers).
//
// Steps (one CUDA graph node each):
//   route    ids -> active expert slots (device side, fixed scan order)
//   had_in   A_g/A_u[slot][i] = H128(x[row] * suh)                         (fp16)
//   gemv_gu  C_gu[slot][g|u][i] = A . W   (codec decodes tiles into mma fragments, rows <= 8)
//   mid      A_d[slot][i] = H128(silu(H128(Cg) * svh_g) * H128(Cu) * svh_u * suh_d)
//   gemv_d   C_d[slot][i] = A_d . W_down
//   combine  out[row] = sum over its routed experts, in routing order, of w * H128(C_d) * svh_d (no atomics)
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

constexpr int THREADS = 256;

template <class Shape> __host__ __device__ constexpr int max_slots() { return MAX_ROWS * Shape::TOPK; }

struct Slot {
    int expert;
    int count;
    int rows[MAX_ROWS];
    float w[MAX_ROWS];
};

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

// One block, one thread per (row, choice) pair j. A pair opens a slot iff no earlier pair routes to its expert; slots
// are numbered in scan order (prefix count of openers), and each opener gathers its expert's rows in scan order.
template <class Shape>
__global__ void route_kernel(const int * __restrict__ ids, const float * __restrict__ wts, int n_rows,
                             Slot * __restrict__ slots, int * __restrict__ n_active, int * __restrict__ e2slot)
{
    constexpr int SLOTS = max_slots<Shape>();
    static_assert(SLOTS <= 128, "one thread per pair, 4 warps");
    __shared__ int s_ids[SLOTS];
    __shared__ int s_open_count[4];
    const int j = threadIdx.x, n_pairs = n_rows * Shape::TOPK;
    const int e = j < n_pairs ? ids[j] : -1;
    if (j < SLOTS) s_ids[j] = e;
    __syncthreads();
    bool opens = j < n_pairs;
    for (int k = 0; k < j && opens; ++k) opens = s_ids[k] != e;
    const unsigned ballot = __ballot_sync(0xffffffffu, opens);
    const int warp = j / 32, lane = j % 32;
    if (lane == 0) s_open_count[warp] = __popc(ballot);
    __syncthreads();
    int slot = __popc(ballot & ((1u << lane) - 1u));
    for (int w = 0; w < warp; ++w) slot += s_open_count[w];
    if (j == 0) *n_active = s_open_count[0] + s_open_count[1] + s_open_count[2] + s_open_count[3];
    if (!opens) return;
    Slot S;
    S.expert = e;
    S.count = 0;
    for (int k = j; k < n_pairs; ++k)
        if (s_ids[k] == e) {
            S.rows[S.count] = k / Shape::TOPK;
            S.w[S.count] = wts[k];
            ++S.count;
        }
    slots[slot] = S;
    e2slot[e] = slot;
}

// grid (D_MODEL/128, max_slots, 2), one warp per row: A[p][slot][i][:] = H(x[row] * suh_p[e])
template <class Shape>
__global__ void had_in_kernel(const float * __restrict__ x, const Slot * __restrict__ slots,
                              const int * __restrict__ n_active, const half * __restrict__ suh_g,
                              const half * __restrict__ suh_u, half * __restrict__ A)
{
    constexpr int D = Shape::D_MODEL;
    const int slot = blockIdx.y;
    if (slot >= *n_active) return;
    const Slot & S = slots[slot];
    const int p = blockIdx.z;
    const half * suh = (p == 0 ? suh_g : suh_u) + (size_t) S.expert * D;
    const int i = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (i >= S.count) return;
    const int c0 = blockIdx.x * 128 + lane * 4;
    const float4 v = *(const float4 *) (x + (size_t) S.rows[i] * D + c0);
    // fp16 pre-scale like the reference (round input to fp16, multiply by fp16 suh), then fp32 Hadamard
    float h0 = __half2float(__hmul(__float2half_rn(v.x), suh[c0 + 0]));
    float h1 = __half2float(__hmul(__float2half_rn(v.y), suh[c0 + 1]));
    float h2 = __half2float(__hmul(__float2half_rn(v.z), suh[c0 + 2]));
    float h3 = __half2float(__hmul(__float2half_rn(v.w), suh[c0 + 3]));
    had4x32(h0, h1, h2, h3, lane);
    half2 * o = (half2 *) (A + (((size_t) p * max_slots<Shape>() + slot) * MAX_ROWS + i) * D + c0);
    o[0] = __floats2half2_rn(h0, h1);
    o[1] = __floats2half2_rn(h2, h3);
}

// GEMV geometry: a block is WK x WG warps; WK warps split the k slices, WG column groups of WNT 16-column tiles;
// PF = k slices in flight per warp (loads ring).
template <int WK_, int WG_, int PF_, int WNT_>
struct Geom {
    static constexpr int WK = WK_, WG = WG_, PF = PF_, WNT = WNT_;
    static constexpr int COLS = WNT * 16;                 // columns per warp
    static constexpr int BLOCK_COLS = WG * COLS;          // columns per work item
    static_assert(WK * WG * 32 == THREADS, "block is 8 warps");
};

// One work item = (slot, projection, BLOCK_COLS columns) over all of K.
template <class Codec, class G>
__device__ __forceinline__ void gemv_item(const half * __restrict__ A, int rows, const uint32_t * __restrict__ B32,
                                          float * __restrict__ C, int size_k, int size_n, int item_group,
                                          float (*sh_red)[MAX_ROWS][G::BLOCK_COLS])
{
    constexpr int WK = G::WK, PF = G::PF, WNT = G::WNT;
    constexpr int FOLD = 2;                                // k slices per fp16 accumulation before the fp32 fold
    constexpr int LOADS = WNT / Codec::TILES_PER_VEC;      // warp-wide word loads per k slice
    static_assert(WNT % Codec::TILES_PER_VEC == 0, "tiles per warp must fill whole loads");
    const int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int warp = wid % WK;                 // k part
    const int wg = wid / WK;                   // column group inside the block
    const int group = item_group * G::WG + wg;
    const int ntiles = size_n / 16, kslices = size_k / 16;
    const int chunk = (kslices + WK - 1) / WK;
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * Codec::TILE_WORDS;
    const half2 * A2 = (const half2 *) A;
    const half2 hzero = __half2half2(__ushort_as_half(0));
    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (size_k / 2);
    const bool r0_ok = r0 < rows;
    const Codec codec(lane);
    const bool lane_loads = codec.loads(lane);

    const uint32_t * bp = B32 + (size_t) ks0 * slice_stride + group * WNT * Codec::TILE_WORDS + lane;
    auto ld_b = [&] (int i, int l) -> uint32_t {
        return lane_loads ? __ldcs(bp + (size_t) i * slice_stride + l * Codec::VEC_WORDS) : 0u;
    };
    // activation fragments ride in the same ring as the trellis words (a load consumed in the same iteration
    // exposes its latency on every k slice)
    auto ld_a = [&] (int i, half2 & lo, half2 & hi) {
        const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
        lo = r0_ok ? A2[a_row0 + a_col] : hzero;
        hi = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
    };
    uint32_t pf[PF][LOADS];
    half2 pa[PF][2];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn) {
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);
            ld_a(d, pa[d][0], pa[d][1]);
        }

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

    // lane holds columns g = lane/4 and g + 8 of each tile, rows 2q and 2q + 1 (q = lane % 4)
    const int g = lane >> 2, q2 = 2 * (lane & 3);
    if constexpr (WK == 1) {
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = group * G::COLS + t * 16 + f * 8 + g;
                if (q2 < rows)     C[(size_t) q2 * size_n + col] = acc[t][f].x;
                if (q2 + 1 < rows) C[(size_t) (q2 + 1) * size_n + col] = acc[t][f].y;
            }
        return;
    }
    #pragma unroll
    for (int t = 0; t < WNT; ++t)
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            const int col = wg * G::COLS + t * 16 + f * 8 + g;
            sh_red[warp][q2][col] = acc[t][f].x;
            sh_red[warp][q2 + 1][col] = acc[t][f].y;
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < G::BLOCK_COLS * rows; idx += THREADS) {
        const int r = idx / G::BLOCK_COLS, c = idx % G::BLOCK_COLS;
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WK; ++j) sum += sh_red[j][r][c];
        C[(size_t) r * size_n + item_group * G::BLOCK_COLS + c] = sum;
    }
    __syncthreads();
}

// Persistent grid: blocks pull (slot, projection, column group) items in slot-major order, so there are no waves
// and neighbouring items read the same expert. The expert's rate picks the codec instantiation.
template <class G>
__global__ __launch_bounds__(THREADS, 2) void gemv_kernel(const Slot * __restrict__ slots,
                                                         const int * __restrict__ n_active, const half * __restrict__ A,
                                                         size_t a_proj_stride, ProjView p0, ProjView p1,
                                                         float * __restrict__ C, size_t c_proj_stride, int size_k,
                                                         int size_n, int n_proj, int max_slot)
{
    // one reduction buffer for all rates (a __shared__ inside each instantiation would triple it)
    __shared__ float sh_red[G::WK == 1 ? 1 : G::WK][MAX_ROWS][G::BLOCK_COLS];
    const int n_groups = size_n / G::BLOCK_COLS;
    const int per_slot = n_proj * n_groups;
    const int n_items = min(*n_active, max_slot) * per_slot;
    for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
        const int slot = item / per_slot, rem = item % per_slot;
        const int proj = rem / n_groups, grp = rem % n_groups;
        const Slot & S = slots[slot];
        const ProjView & P = proj == 0 ? p0 : p1;
        const int e = S.expert;
        const uint32_t * B32 = (const uint32_t *) (P.trellis + P.meta[2 * e + 1]);
        const half * Ab = A + proj * a_proj_stride + (size_t) slot * MAX_ROWS * size_k;
        float * Cb = C + proj * c_proj_stride + (size_t) slot * MAX_ROWS * size_n;
        switch (P.meta[2 * e]) {
            case 2: gemv_item<Mul1<2>, G>(Ab, S.count, B32, Cb, size_k, size_n, grp, sh_red); break;
            case 3: gemv_item<Mul1<3>, G>(Ab, S.count, B32, Cb, size_k, size_n, grp, sh_red); break;
            case 4: gemv_item<Mul1<4>, G>(Ab, S.count, B32, Cb, size_k, size_n, grp, sh_red); break;
            default: break;   // unsupported rate: the pack loader rejects it before a kernel runs
        }
    }
}

template <class G>
int persistent_grid()
{
    static int g = 0;
    if (!g) {
        int dev, sms, per_sm;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemv_kernel<G>, THREADS, 0);
        g = sms * per_sm;
    }
    return g;
}

// grid (D_FF/128, max_slots), one warp per row
template <class Shape>
__global__ void mid_kernel(const Slot * __restrict__ slots, const int * __restrict__ n_active,
                           const float * __restrict__ C_gu, const half * __restrict__ svh_g,
                           const half * __restrict__ svh_u, const half * __restrict__ suh_d, half * __restrict__ A_d)
{
    constexpr int F = Shape::D_FF;
    const int slot = blockIdx.y;
    if (slot >= *n_active) return;
    const Slot & S = slots[slot];
    const int i = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (i >= S.count) return;
    const int e = S.expert;
    const int c0 = blockIdx.x * 128 + lane * 4;
    const size_t off = ((size_t) slot * MAX_ROWS + i) * F + c0;
    float4 g = *(const float4 *) (C_gu + off);
    float4 u = *(const float4 *) (C_gu + (size_t) max_slots<Shape>() * MAX_ROWS * F + off);
    had4x32(g.x, g.y, g.z, g.w, lane);
    had4x32(u.x, u.y, u.z, u.w, lane);
    const half * sg = svh_g + (size_t) e * F + c0;
    const half * su = svh_u + (size_t) e * F + c0;
    const half * sd = suh_d + (size_t) e * F + c0;
    float h[4];
    const float gv[4] = { g.x, g.y, g.z, g.w }, uv[4] = { u.x, u.y, u.z, u.w };
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float gg = gv[j] * __half2float(sg[j]);
        const float uu = uv[j] * __half2float(su[j]);
        // round to fp16 before the down pre-scale, as the reference feeds fp16 activations into the down GEMV
        h[j] = __half2float(__hmul(__float2half_rn(silu(gg) * uu), sd[j]));
    }
    had4x32(h[0], h[1], h[2], h[3], lane);
    half2 * o = (half2 *) (A_d + off);
    o[0] = __floats2half2_rn(h[0], h[1]);
    o[1] = __floats2half2_rn(h[2], h[3]);
}

// grid (D_MODEL/128, n_rows), block TOPK warps: warp s handles routing choice s; partial sums are added in routing
// order through shared memory (deterministic, no atomics)
template <class Shape>
__global__ void combine_kernel(const int * __restrict__ ids, const Slot * __restrict__ slots,
                               const int * __restrict__ e2slot, const float * __restrict__ C_d,
                               const half * __restrict__ svh_d, float * __restrict__ out)
{
    constexpr int D = Shape::D_MODEL, TOPK = Shape::TOPK;
    __shared__ float4 part[TOPK][32];
    const int row = blockIdx.y, s = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int c0 = blockIdx.x * 128 + lane * 4;
    const int e = ids[row * TOPK + s];
    const int sl = e2slot[e];
    const Slot & S = slots[sl];
    int i = 0;
    while (S.rows[i] != row) ++i;
    float4 v = *(const float4 *) (C_d + ((size_t) sl * MAX_ROWS + i) * D + c0);
    had4x32(v.x, v.y, v.z, v.w, lane);
    const half * sv = svh_d + (size_t) e * D + c0;
    const float w = S.w[i];
    part[s][lane] = make_float4(w * v.x * __half2float(sv[0]), w * v.y * __half2float(sv[1]),
                                w * v.z * __half2float(sv[2]), w * v.w * __half2float(sv[3]));
    __syncthreads();
    if (s == 0) {
        float4 acc = part[0][lane];
        #pragma unroll
        for (int k = 1; k < TOPK; ++k) {
            acc.x += part[k][lane].x; acc.y += part[k][lane].y; acc.z += part[k][lane].z; acc.w += part[k][lane].w;
        }
        *(float4 *) (out + (size_t) row * D + c0) = acc;
    }
}

// Per-shape GEMV geometry, best measured on the 3090 (cp3_moe_tune.log v6).
template <class Shape> struct Tune;
template <> struct Tune<FlashNext> {
    using GateUp = Geom<8, 1, 2, 8>;       // 8 k-split warps x 8 tiles
    using Down = Geom<4, 2, 2, 8>;         // 4 k-split warps x 2 groups x 8 tiles
};

// Workspace layout, shared by workspace_bytes() and window()
template <class Shape>
struct Workspace {
    static constexpr size_t SLOTS = max_slots<Shape>(), D = Shape::D_MODEL, F = Shape::D_FF;
    Slot * slots;
    int * n_active;
    int * e2slot;
    half * A_gu;     // [2][SLOTS][MAX_ROWS][D]
    float * C_gu;    // [2][SLOTS][MAX_ROWS][F]
    half * A_d;      // [SLOTS][MAX_ROWS][F]
    float * C_d;     // [SLOTS][MAX_ROWS][D]
    size_t bytes;

    explicit Workspace(void * base, int n_expert)
    {
        char * p = (char *) base;
        auto take = [&] (size_t n) { char * r = p; p += (n + 255) & ~(size_t) 255; return r; };
        slots = (Slot *) take(sizeof(Slot) * SLOTS);
        n_active = (int *) take(sizeof(int));
        e2slot = (int *) take(sizeof(int) * n_expert);
        A_gu = (half *) take(sizeof(half) * 2 * SLOTS * MAX_ROWS * D);
        C_gu = (float *) take(sizeof(float) * 2 * SLOTS * MAX_ROWS * F);
        A_d = (half *) take(sizeof(half) * SLOTS * MAX_ROWS * F);
        C_d = (float *) take(sizeof(float) * SLOTS * MAX_ROWS * D);
        bytes = (size_t) (p - (char *) base);
    }
};

constexpr int MAX_EXPERTS = 1024;          // e2slot capacity in the workspace

}  // namespace

template <class Shape>
size_t workspace_bytes()
{
    static_assert(Shape::D_MODEL % 128 == 0 && Shape::D_FF % 128 == 0, "Hadamard-128 blocks");
    static_assert(Shape::D_FF % Tune<Shape>::GateUp::BLOCK_COLS == 0, "gate/up column groups must tile D_FF");
    static_assert(Shape::D_MODEL % Tune<Shape>::Down::BLOCK_COLS == 0, "down column groups must tile D_MODEL");
    return Workspace<Shape>(nullptr, MAX_EXPERTS).bytes;
}

template <class Shape>
void window(const Weights & W, const float * x, const int * ids, const float * wts, int n_rows, float * out, void * ws,
            cudaStream_t stream)
{
    using GU = typename Tune<Shape>::GateUp;
    using DN = typename Tune<Shape>::Down;
    constexpr int D = Shape::D_MODEL, F = Shape::D_FF, SLOTS = max_slots<Shape>();
    if (n_rows < 1 || n_rows > MAX_ROWS || W.n_expert > MAX_EXPERTS) {
        fprintf(stderr, "truss::moe::window: n_rows %d (max %d), n_expert %d (max %d)\n", n_rows, MAX_ROWS, W.n_expert,
                MAX_EXPERTS);
        abort();
    }
    const Workspace<Shape> w(ws, MAX_EXPERTS);
    const int used_slots = n_rows * Shape::TOPK;

    route_kernel<Shape><<<1, 128, 0, stream>>>(ids, wts, n_rows, w.slots, w.n_active, w.e2slot);
    had_in_kernel<Shape><<<dim3(D / 128, used_slots, 2), 32 * MAX_ROWS, 0, stream>>>(
        x, w.slots, w.n_active, W.proj[0].suh, W.proj[1].suh, w.A_gu);
    gemv_kernel<GU><<<persistent_grid<GU>(), THREADS, 0, stream>>>(
        w.slots, w.n_active, w.A_gu, (size_t) SLOTS * MAX_ROWS * D, W.proj[0], W.proj[1], w.C_gu,
        (size_t) SLOTS * MAX_ROWS * F, D, F, 2, used_slots);
    mid_kernel<Shape><<<dim3(F / 128, used_slots), 32 * MAX_ROWS, 0, stream>>>(
        w.slots, w.n_active, w.C_gu, W.proj[0].svh, W.proj[1].svh, W.proj[2].suh, w.A_d);
    gemv_kernel<DN><<<persistent_grid<DN>(), THREADS, 0, stream>>>(
        w.slots, w.n_active, w.A_d, 0, W.proj[2], W.proj[2], w.C_d, 0, F, D, 1, used_slots);
    combine_kernel<Shape><<<dim3(D / 128, n_rows), 32 * Shape::TOPK, 0, stream>>>(
        ids, w.slots, w.e2slot, w.C_d, W.proj[2].svh, out);
}

template size_t workspace_bytes<FlashNext>();
template void window<FlashNext>(const Weights &, const float *, const int *, const float *, int, float *, void *,
                                cudaStream_t);

}  // namespace truss::moe
