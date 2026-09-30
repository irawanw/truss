// MoE prefill (see moe_prefill.cuh). Kernels, in order, on one stream:
//   route_count / route_scan / route_place   (token, choice) pairs sorted by expert, stable (token order inside an
//                                            expert): pair_tok, pair_w, and pair_pos[t * TOPK + s] = sorted index
//   prep      A_p[pair] = H128(x[token] * suh_p[expert]), p = gate, up; in token order (x read once, coalesced).
//             Building it inside the GEMM instead repeats it in all 5 column blocks: 23 vs 36 TFLOPS (TRACKER #42)
//   gate_up   item (expert, 64-row block, 128 D_FF columns): gate and up GEMMs, then the mid step as the epilogue:
//             A_d = H128(silu(H128(C_g) * svh_g) * H128(C_u) * svh_u * suh_d)   (its 128 columns = one H block)
//   down      item (expert, 64-row block, 256 D_MODEL columns): C_d = w * H128(A_d . W_d) * svh_d, fp16
//   combine   out[t] = sum over s in routing order of C_d[pair_pos[t, s]]
// Deterministic: stable routing, fixed summation orders, no float atomics.
//
// GEMM warp tile: 4 weight tiles (64 columns) x 32 rows. Per 16-deep k slice a warp decodes its 4 tiles once and
// runs them against 4 groups of 8 rows (mma m16n8k16, weights as the A operand, fp32 accumulation). A block is
// 8 warps: 2 row halves x 4 (gate/up: projection x column half; down: column quarter).
#include "moe_prefill.cuh"

#include "core/cuda_check.h"
#include "../trellis/codec_mul1.cuh"
#include "../trellis/hadamard.cuh"
#include "../trellis/mma.cuh"

#include <stdexcept>
#include <string>

namespace truss::moe {
namespace {

constexpr int THREADS = 128, WARPS = THREADS / 32;
constexpr int BM = 64;                  // rows per item
constexpr int KC = 64;                  // k depth of one staged activation chunk (double-buffered, cp.async)
constexpr int AS = KC + 8;              // shared row stride (halfs): 8 rows x 4 k pairs hit 32 distinct banks
constexpr int WNT = 4;                  // weight tiles per warp
constexpr int RG = 8;                   // 8-row groups per warp (all 64 rows of the item)
constexpr int ROUTE_CHUNK = 2048;       // pairs per routing warp
constexpr int MAX_EXPERTS = 1024;

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

__device__ __forceinline__ float4 h4_to_f4(uint2 v)
{
    const half2 a = *(const half2 *) &v.x, b = *(const half2 *) &v.y;
    return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}

// ---------------------------------------------------------------------------------------------------------
// routing: counting sort of pairs by expert, stable. One warp per chunk of ROUTE_CHUNK pairs.

__global__ void route_count(const int * __restrict__ ids, int n_pairs, int n_expert, int * __restrict__ chunk_cnt)
{
    extern __shared__ int cnt[];
    const int lane = threadIdx.x, c = blockIdx.x, p0 = c * ROUTE_CHUNK;
    for (int e = lane; e < n_expert; e += 32) cnt[e] = 0;
    __syncwarp();
    for (int i = 0; i < ROUTE_CHUNK && p0 + i < n_pairs; i += 32) {
        const int p = p0 + i + lane;
        const int e = p < n_pairs ? ids[p] : -1;
        const unsigned peers = __match_any_sync(0xffffffffu, e);
        if (e >= 0 && lane == __ffs(peers) - 1) cnt[e] += __popc(peers);
        __syncwarp();
    }
    for (int e = lane; e < n_expert; e += 32) chunk_cnt[(size_t) c * n_expert + e] = cnt[e];
}

// chunk counts -> each chunk's first index inside its expert; expert and 64-row-block offsets
__global__ void route_scan(int * __restrict__ chunk_cnt, int n_chunks, int n_expert, int * __restrict__ e_off,
                           int * __restrict__ rb_off)
{
    __shared__ int s_n[MAX_EXPERTS], s_rb[MAX_EXPERTS];
    const int e = threadIdx.x;
    int run = 0;
    if (e < n_expert)
        for (int c = 0; c < n_chunks; ++c) {
            const int v = chunk_cnt[(size_t) c * n_expert + e];
            chunk_cnt[(size_t) c * n_expert + e] = run;
            run += v;
        }
    s_n[e] = e < n_expert ? run : 0;
    s_rb[e] = e < n_expert ? (run + BM - 1) / BM : 0;
    __syncthreads();
    for (int d = 1; d < MAX_EXPERTS; d <<= 1) {   // inclusive scan
        const int a = e >= d ? s_n[e - d] : 0, b = e >= d ? s_rb[e - d] : 0;
        __syncthreads();
        s_n[e] += a;
        s_rb[e] += b;
        __syncthreads();
    }
    if (e < n_expert) {
        e_off[e + 1] = s_n[e];
        rb_off[e + 1] = s_rb[e];
    }
    if (e == 0) e_off[0] = rb_off[0] = 0;
}

__global__ void route_place(const int * __restrict__ ids, const float * __restrict__ wts, int n_pairs, int n_expert,
                            int topk, const int * __restrict__ chunk_cnt, const int * __restrict__ e_off,
                            int * __restrict__ pair_tok, float * __restrict__ pair_w, int * __restrict__ pair_pos)
{
    extern __shared__ int run[];
    const int lane = threadIdx.x, c = blockIdx.x, p0 = c * ROUTE_CHUNK;
    for (int e = lane; e < n_expert; e += 32) run[e] = e_off[e] + chunk_cnt[(size_t) c * n_expert + e];
    __syncwarp();
    for (int i = 0; i < ROUTE_CHUNK && p0 + i < n_pairs; i += 32) {
        const int p = p0 + i + lane;
        const int e = p < n_pairs ? ids[p] : -1;
        const unsigned peers = __match_any_sync(0xffffffffu, e);
        if (e >= 0) {
            const int dest = run[e] + __popc(peers & ((1u << lane) - 1));
            pair_tok[dest] = p / topk;
            pair_w[dest] = wts[p];
            pair_pos[p] = dest;
        }
        __syncwarp();
        if (e >= 0 && lane == __ffs(peers) - 1) run[e] += __popc(peers);
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------------------------------------
// GEMM pieces

// async copy of rows [0, BM) x k [0, KC) of a fp16 matrix (row stride ld) into shared [BM][AS]; rows >= n_rows are
// zero-filled (no global read)
__device__ __forceinline__ void stage_async(half * dst, const half * src, int ld, int n_rows)
{
    constexpr int V = KC / 8;   // 16-byte vectors per row
    for (int i = threadIdx.x; i < BM * V; i += THREADS) {
        const int r = i / V, v = i % V;
        const half * g = src + (size_t) (r < n_rows ? r : 0) * ld + v * 8;
        const unsigned d = (unsigned) __cvta_generic_to_shared(dst + r * AS + v * 8);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(g), "r"(r < n_rows ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N> __device__ __forceinline__ void stage_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

struct ItemRows {
    int e, row0, rows, pair0;
};

// item -> (expert, rows); false past the last row block
__device__ __forceinline__ bool item_rows(int rb, int n_expert, const int * __restrict__ e_off,
                                          const int * __restrict__ rb_off, ItemRows & it)
{
    if (rb >= rb_off[n_expert]) return false;
    int lo = 0, hi = n_expert - 1;   // last e with rb_off[e] <= rb
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (rb_off[mid] <= rb) lo = mid; else hi = mid - 1;
    }
    it.e = lo;
    it.row0 = (rb - rb_off[lo]) * BM;
    it.rows = min(BM, e_off[lo + 1] - e_off[lo] - it.row0);
    it.pair0 = e_off[lo] + it.row0;
    return true;
}

// one 128-deep k chunk of the warp tile: acc[t][g] += W(k chunk, tiles nt0..nt0+3)^T . A(rows of the warp)
// As: the warp's 32 rows of the staged chunk; n_rg: valid 8-row groups
template <class Codec>
__device__ __forceinline__ void mma_chunk(const half * __restrict__ As, int n_rg, const uint32_t * __restrict__ B32,
                                          int kc, int nt0, int ntiles, float (&acc)[WNT][RG][4])
{
    constexpr int LOADS = WNT / Codec::TILES_PER_VEC;
    constexpr int KS = KC / 16;
    static_assert(WNT % Codec::TILES_PER_VEC == 0, "tiles per warp must fill whole loads");
    const int lane = threadIdx.x % 32;
    const Codec codec(lane);
    const bool lane_loads = codec.loads(lane);
    // all words of the chunk first: KS slices x LOADS in flight
    uint32_t w[KS][LOADS];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
        const uint32_t * b = B32 + ((size_t) (kc * KS + ks) * ntiles + nt0) * Codec::TILE_WORDS + lane;
#pragma unroll
        for (int l = 0; l < LOADS; ++l) w[ks][l] = lane_loads ? __ldg(b + l * Codec::VEC_WORDS) : 0u;
    }
    const int r = lane >> 2, q2 = 2 * (lane & 3);
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
        FragB f0[WNT], f1[WNT];
#pragma unroll
        for (int t = 0; t < WNT; ++t) codec.tile(w[ks][t / Codec::TILES_PER_VEC], t % Codec::TILES_PER_VEC, f0[t], f1[t]);
#pragma unroll
        for (int g = 0; g < RG; ++g) {
            if (g >= n_rg) break;
            const half * a = As + (g * 8 + r) * AS + ks * 16 + q2;
            const half2 lo = *(const half2 *) a, hi = *(const half2 *) (a + 8);
#pragma unroll
            for (int t = 0; t < WNT; ++t) mma_w_f32(f0[t], f1[t], lo, hi, acc[t][g]);
        }
    }
}

__device__ __forceinline__ void mma_chunk_k(int K, const half * As, int n_rg, const uint32_t * B32, int kc, int nt0,
                                            int ntiles, float (&acc)[WNT][RG][4])
{
    switch (K) {
        case 1: mma_chunk<Mul1<1>>(As, n_rg, B32, kc, nt0, ntiles, acc); break;
        case 2: mma_chunk<Mul1<2>>(As, n_rg, B32, kc, nt0, ntiles, acc); break;
        case 3: mma_chunk<Mul1<3>>(As, n_rg, B32, kc, nt0, ntiles, acc); break;
        case 4: mma_chunk<Mul1<4>>(As, n_rg, B32, kc, nt0, ntiles, acc); break;
        default: break;   // unsupported rate: the pack loader rejects it before a kernel runs
    }
}

// rows [32 h, 32 h + 32) of a warp's accumulators -> shared C[32 rows][cs] at column offset col0
// (row = 8 g + 2 q (+1), col = 16 t + lane/4 (+8))
__device__ __forceinline__ void store_acc(const float (&acc)[WNT][RG][4], int h, float * C, int cs, int col0)
{
    const int lane = threadIdx.x % 32, g8 = lane >> 2, q2 = 2 * (lane & 3);
#pragma unroll
    for (int t = 0; t < WNT; ++t)
#pragma unroll
        for (int g = 0; g < RG; ++g) {
            if (g / 4 != h) continue;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                C[((g % 4) * 8 + q2 + (i & 1)) * cs + col0 + t * 16 + g8 + (i >> 1) * 8] = acc[t][g][i];
        }
}

// ---------------------------------------------------------------------------------------------------------
// prep: warp task = (token, choice, 128-column block), both projections

template <class Shape>
__global__ __launch_bounds__(THREADS) void prep_kernel(Weights W, const float * __restrict__ x,
                                                       const int * __restrict__ ids, const int * __restrict__ pair_pos,
                                                       int n_tokens, half * __restrict__ A_gu, size_t proj_stride)
{
    constexpr int D = Shape::D_MODEL, TOPK = Shape::TOPK, HB = D / 128;
    const int lane = threadIdx.x % 32;
    const int task = blockIdx.x * WARPS + threadIdx.x / 32;
    if (task >= n_tokens * HB * TOPK) return;
    const int t = task / (HB * TOPK), hb = (task / TOPK) % HB, s = task % TOPK;
    const int c0 = hb * 128 + lane * 4, e = ids[t * TOPK + s], dst = pair_pos[t * TOPK + s];
    const float4 v = *(const float4 *) (x + (size_t) t * D + c0);
    const half2 x01 = __floats2half2_rn(v.x, v.y), x23 = __floats2half2_rn(v.z, v.w);
    for (int p = 0; p < 2; ++p) {
        // fp16 pre-scale by suh, fp32 Hadamard, fp16 (as moe_window's H step and the reference)
        const uint2 sc = *(const uint2 *) (W.proj[p].suh + (size_t) e * D + c0);
        const half2 p01 = __hmul2(x01, *(const half2 *) &sc.x), p23 = __hmul2(x23, *(const half2 *) &sc.y);
        float h0 = __low2float(p01), h1 = __high2float(p01), h2 = __low2float(p23), h3 = __high2float(p23);
        had4x32(h0, h1, h2, h3, lane);
        half2 * o = (half2 *) (A_gu + p * proj_stride + (size_t) dst * D + c0);
        o[0] = __floats2half2_rn(h0, h1);
        o[1] = __floats2half2_rn(h2, h3);
    }
}

// ---------------------------------------------------------------------------------------------------------
// gate/up + mid

template <class Shape>
__global__ __launch_bounds__(THREADS) void gate_up_kernel(Weights W, const half * __restrict__ A_gu,
                                                          size_t proj_stride, const int * __restrict__ e_off,
                                                          const int * __restrict__ rb_off, half * __restrict__ A_d)
{
    constexpr int D = Shape::D_MODEL, F = Shape::D_FF, CB = F / 128, CS = 128 + 4;
    static_assert(D % KC == 0 && F % 128 == 0, "k chunks; Hadamard blocks");
    static_assert(WARPS == 4 && 2 * WNT * 16 == 128, "4 warps = 2 proj x 2 column halves (of 128)");
    constexpr int STAGE = 2 * BM * AS;   // halfs: [proj][64 rows][AS]
    constexpr int SMEM_A = 2 * STAGE * 2, SMEM_C = 2 * 32 * CS * 4;
    __shared__ __align__(16) char smem[SMEM_A > SMEM_C ? SMEM_A : SMEM_C];
    half * As = (half *) smem;     // [buffer][proj][64 rows][AS]
    float * Cs = (float *) smem;   // epilogue: [proj][32 rows][CS]

    ItemRows it;
    const int cb = blockIdx.x % CB;
    if (!item_rows(blockIdx.x / CB, W.n_expert, e_off, rb_off, it)) return;
    const int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int proj = wid >> 1, colh = wid & 1;
    const ProjView & P = proj ? W.proj[1] : W.proj[0];   // not W.proj[proj]: a runtime index puts W in local memory
    const int K = P.meta[2 * it.e];
    const uint32_t * B32 = (const uint32_t *) (P.trellis + P.meta[2 * it.e + 1]);
    const int n_rg = (it.rows + 7) / 8;

    const half * src = A_gu + (size_t) it.pair0 * D;
    auto issue = [&](int kc) {
        half * b = As + (kc & 1) * STAGE;
        stage_async(b, src + kc * KC, D, it.rows);
        stage_async(b + BM * AS, src + proj_stride + kc * KC, D, it.rows);
    };
    float acc[WNT][RG][4] = {};
    constexpr int NK = D / KC;
    issue(0);
    for (int kc = 0; kc < NK; ++kc) {
        if (kc + 1 < NK) {
            issue(kc + 1);
            stage_wait<2>();   // this chunk's two groups done, the next chunk's two in flight
        } else {
            stage_wait<0>();
        }
        __syncthreads();
        mma_chunk_k(K, As + (kc & 1) * STAGE + proj * BM * AS, n_rg, B32, kc, cb * 8 + colh * 4, F / 16, acc);
        __syncthreads();
    }

    // mid epilogue, one row half at a time through shared memory
    for (int rh = 0; rh < 2; ++rh) {
        store_acc(acc, rh, Cs + proj * 32 * CS, CS, colh * 64);
        __syncthreads();
        for (int r = wid; r < 32; r += WARPS) {
            const int row = rh * 32 + r;
            if (row >= it.rows) break;
            const int c0 = lane * 4, fc = cb * 128 + c0;
            float4 g = *(const float4 *) (Cs + r * CS + c0);
            float4 u = *(const float4 *) (Cs + (32 + r) * CS + c0);
            const float4 fg = h4_to_f4(*(const uint2 *) (W.proj[0].svh + (size_t) it.e * F + fc));
            const float4 fu = h4_to_f4(*(const uint2 *) (W.proj[1].svh + (size_t) it.e * F + fc));
            const uint2 sd = *(const uint2 *) (W.proj[2].suh + (size_t) it.e * F + fc);
            had4x32(g.x, g.y, g.z, g.w, lane);
            had4x32(u.x, u.y, u.z, u.w, lane);
            const half2 p01 = __hmul2(__floats2half2_rn(silu(g.x * fg.x) * (u.x * fu.x), silu(g.y * fg.y) * (u.y * fu.y)),
                                      *(const half2 *) &sd.x);
            const half2 p23 = __hmul2(__floats2half2_rn(silu(g.z * fg.z) * (u.z * fu.z), silu(g.w * fg.w) * (u.w * fu.w)),
                                      *(const half2 *) &sd.y);
            float h0 = __low2float(p01), h1 = __high2float(p01), h2 = __low2float(p23), h3 = __high2float(p23);
            had4x32(h0, h1, h2, h3, lane);
            half2 * o = (half2 *) (A_d + (size_t) (it.pair0 + row) * F + fc);
            o[0] = __floats2half2_rn(h0, h1);
            o[1] = __floats2half2_rn(h2, h3);
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------------------
// down

template <class Shape>
__global__ __launch_bounds__(THREADS) void down_kernel(Weights W, const half * __restrict__ A_d,
                                                       const int * __restrict__ e_off, const int * __restrict__ rb_off,
                                                       const float * __restrict__ pair_w, half * __restrict__ C_d)
{
    constexpr int D = Shape::D_MODEL, F = Shape::D_FF, BN = 4 * WNT * 16, CB = D / BN, CS = BN + 4;
    static_assert(F % KC == 0 && D % BN == 0 && BN % 128 == 0, "k chunks; Hadamard blocks");
    constexpr int STAGE = BM * AS;
    constexpr int SMEM_A = 2 * STAGE * 2, SMEM_C = 32 * CS * 4;
    __shared__ __align__(16) char smem[SMEM_A > SMEM_C ? SMEM_A : SMEM_C];
    half * As = (half *) smem;     // [buffer][64 rows][AS]
    float * Cs = (float *) smem;   // epilogue: [32 rows][CS]

    ItemRows it;
    const int cb = blockIdx.x % CB;
    if (!item_rows(blockIdx.x / CB, W.n_expert, e_off, rb_off, it)) return;
    const int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int colq = wid;
    const ProjView & P = W.proj[2];
    const int K = P.meta[2 * it.e];
    const uint32_t * B32 = (const uint32_t *) (P.trellis + P.meta[2 * it.e + 1]);
    const int n_rg = (it.rows + 7) / 8;

    const half * src = A_d + (size_t) it.pair0 * F;
    float acc[WNT][RG][4] = {};
    constexpr int NK = F / KC;
    stage_async(As, src, F, it.rows);
    for (int kc = 0; kc < NK; ++kc) {
        if (kc + 1 < NK) {
            stage_async(As + ((kc + 1) & 1) * STAGE, src + (kc + 1) * KC, F, it.rows);
            stage_wait<1>();
        } else {
            stage_wait<0>();
        }
        __syncthreads();
        mma_chunk_k(K, As + (kc & 1) * STAGE, n_rg, B32, kc, cb * (BN / 16) + colq * WNT, D / 16, acc);
        __syncthreads();
    }

    for (int rh = 0; rh < 2; ++rh) {
        store_acc(acc, rh, Cs, CS, colq * WNT * 16);
        __syncthreads();
        for (int task = wid; task < 32 * (BN / 128); task += WARPS) {
            const int r = task / (BN / 128), hb = task % (BN / 128), row = rh * 32 + r;
            if (row >= it.rows) continue;
            const int c = hb * 128 + lane * 4, col = cb * BN + c;
            float4 v = *(const float4 *) (Cs + r * CS + c);
            had4x32(v.x, v.y, v.z, v.w, lane);
            const float4 f = h4_to_f4(*(const uint2 *) (P.svh + (size_t) it.e * D + col));
            const float w = pair_w[it.pair0 + row];
            half2 * o = (half2 *) (C_d + (size_t) (it.pair0 + row) * D + col);
            o[0] = __floats2half2_rn(w * v.x * f.x, w * v.y * f.y);
            o[1] = __floats2half2_rn(w * v.z * f.z, w * v.w * f.w);
        }
        __syncthreads();
    }
}

// out[t][c] = sum over s (routing order) of C_d[pair_pos[t * TOPK + s]][c]; one thread per 4 columns
template <class Shape>
__global__ void combine_kernel(const half * __restrict__ C_d, const int * __restrict__ pair_pos, int n_tokens,
                               float * __restrict__ out)
{
    constexpr int D = Shape::D_MODEL, TOPK = Shape::TOPK;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tokens * (D / 4)) return;
    const int t = i / (D / 4), c = (i % (D / 4)) * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < TOPK; ++s) {
        const float4 v = h4_to_f4(*(const uint2 *) (C_d + (size_t) pair_pos[t * TOPK + s] * D + c));
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    *(float4 *) (out + (size_t) t * D + c) = acc;
}

// ---------------------------------------------------------------------------------------------------------

template <class Shape>
struct PrefillWorkspace {
    int * chunk_cnt;   // [chunks][MAX_EXPERTS]
    int * e_off;       // [MAX_EXPERTS + 1]
    int * rb_off;      // [MAX_EXPERTS + 1]
    int * pair_tok;    // [pairs]
    half * A_gu;       // [2][pairs][D_MODEL]
    float * pair_w;    // [pairs]
    int * pair_pos;    // [pairs]
    half * A_d;        // [pairs][D_FF]
    half * C_d;        // [pairs][D_MODEL]
    size_t bytes;

    PrefillWorkspace(void * base, int max_tokens)
    {
        const size_t pairs = (size_t) max_tokens * Shape::TOPK, chunks = (pairs + ROUTE_CHUNK - 1) / ROUTE_CHUNK;
        char * p = (char *) base;
        auto take = [&](size_t n) { char * r = p; p += (n + 255) & ~(size_t) 255; return r; };
        chunk_cnt = (int *) take(sizeof(int) * chunks * MAX_EXPERTS);
        e_off = (int *) take(sizeof(int) * (MAX_EXPERTS + 1));
        rb_off = (int *) take(sizeof(int) * (MAX_EXPERTS + 1));
        pair_tok = (int *) take(sizeof(int) * pairs);
        pair_w = (float *) take(sizeof(float) * pairs);
        pair_pos = (int *) take(sizeof(int) * pairs);
        A_gu = (half *) take(sizeof(half) * 2 * pairs * Shape::D_MODEL);
        A_d = (half *) take(sizeof(half) * pairs * Shape::D_FF);
        C_d = (half *) take(sizeof(half) * pairs * Shape::D_MODEL);
        bytes = (size_t) (p - (char *) base);
    }
};

}  // namespace

template <class Shape>
size_t prefill_workspace_bytes(int max_tokens)
{
    return PrefillWorkspace<Shape>(nullptr, max_tokens).bytes;
}

template <class Shape>
void prefill(const Weights & W, const float * x, const int * ids, const float * wts, int n_tokens, float * out,
             void * ws, int max_tokens, cudaStream_t stream)
{
    if (n_tokens < 1 || n_tokens > max_tokens)
        throw std::runtime_error("moe::prefill: n_tokens " + std::to_string(n_tokens) + " outside 1.." +
                                 std::to_string(max_tokens));
    if (W.n_expert < 1 || W.n_expert > MAX_EXPERTS) throw std::runtime_error("moe::prefill: n_expert > 1024");
    const PrefillWorkspace<Shape> w(ws, max_tokens);
    const int E = W.n_expert, pairs = n_tokens * Shape::TOPK, chunks = (pairs + ROUTE_CHUNK - 1) / ROUTE_CHUNK;
    route_count<<<chunks, 32, E * sizeof(int), stream>>>(ids, pairs, E, w.chunk_cnt);
    route_scan<<<1, MAX_EXPERTS, 0, stream>>>(w.chunk_cnt, chunks, E, w.e_off, w.rb_off);
    route_place<<<chunks, 32, E * sizeof(int), stream>>>(ids, wts, pairs, E, Shape::TOPK, w.chunk_cnt, w.e_off,
                                                         w.pair_tok, w.pair_w, w.pair_pos);
    const int max_rb = (pairs + BM - 1) / BM + E;   // sum over experts of ceil(rows / BM)
    const size_t proj_stride = (size_t) max_tokens * Shape::TOPK * Shape::D_MODEL;
    const int prep_tasks = n_tokens * (Shape::D_MODEL / 128) * Shape::TOPK;
    prep_kernel<Shape><<<(prep_tasks + WARPS - 1) / WARPS, THREADS, 0, stream>>>(W, x, ids, w.pair_pos, n_tokens,
                                                                                  w.A_gu, proj_stride);
    gate_up_kernel<Shape><<<max_rb * (Shape::D_FF / 128), THREADS, 0, stream>>>(W, w.A_gu, proj_stride, w.e_off,
                                                                                w.rb_off, w.A_d);
    down_kernel<Shape><<<max_rb * (Shape::D_MODEL / (4 * WNT * 16)), THREADS, 0, stream>>>(W, w.A_d, w.e_off, w.rb_off,
                                                                                         w.pair_w, w.C_d);
    const int n4 = n_tokens * (Shape::D_MODEL / 4);
    combine_kernel<Shape><<<(n4 + 255) / 256, 256, 0, stream>>>(w.C_d, w.pair_pos, n_tokens, out);
    TRUSS_CUDA(cudaGetLastError());
}

template size_t prefill_workspace_bytes<FlashNext>(int);
template void prefill<FlashNext>(const Weights &, const float *, const int *, const float *, int, float *, void *,
                                 int, cudaStream_t);

}  // namespace truss::moe
