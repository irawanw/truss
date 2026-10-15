#include "kernels/attn/gqa.cuh"

#include "core/cuda_check.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace truss::attn {

namespace {

constexpr int D = 64;            // head dim (the kernels are written for 64: 2 values per lane, 4 threads x 16 per row)
constexpr int R = 64;            // rows (query, head) per CTA
constexpr int BC = 64;           // cells per K / V tile
constexpr int THREADS = 256;     // 4 threads per row
constexpr int KP = D + 8;        // padded smem row (halves) for K / V
constexpr int MAX_SPLITS = 256;
constexpr int MAX_PART_ROWS = 1024;   // splits x T bound: the partials workspace stays <= 1024 x H x (D + 2) floats
constexpr int SPLIT_ROWS = 32;   // chunks up to this many rows split their cells across CTAs

// ---- prepare: one warp per (row, head slot); slots [0, H) q, [H, H + HKV) k, [H + HKV, H + 2 HKV) v
__global__ void prepare_kernel(GqaShape s, const float * __restrict__ q, const float * __restrict__ k,
                               const float * __restrict__ v, const float * __restrict__ qn, const float * __restrict__ kn,
                               int pos0, int T, half * __restrict__ q16, half * __restrict__ kc, half * __restrict__ vc)
{
    const int slots = s.H + 2 * s.HKV;
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (w >= T * slots) return;
    const int t = w / slots, j = w % slots, pos = pos0 + t;
    const float * src;
    const float * nw = nullptr;
    half * dst;
    bool rope = true;
    if (j < s.H) src = q + ((size_t) t * s.H + j) * D, nw = qn, dst = q16 + ((size_t) t * s.H + j) * D;
    else if (j < s.H + s.HKV) {
        const int hk = j - s.H;
        src = k + ((size_t) t * s.HKV + hk) * D, nw = kn, dst = kc + ((size_t) pos * s.HKV + hk) * D;
    } else {
        const int hk = j - s.H - s.HKV;
        src = v + ((size_t) t * s.HKV + hk) * D, dst = vc + ((size_t) pos * s.HKV + hk) * D, rope = false;
    }
    float a = src[lane], b = src[lane + 32];
    if (nw) {
        float ss = a * a + b * b;
        for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float r = rsqrtf(ss / D + s.eps);
        a *= r * nw[lane], b *= r * nw[lane + 32];
    }
    if (rope) {   // NEOX: pair (i, i + D/2), frequency base^(-2i/D), angle in fp64
        const double ang = (double) pos * pow((double) s.rope_base, -2.0 * lane / D);
        const float c = (float) cos(ang), sn = (float) sin(ang);
        const float a2 = a * c - b * sn, b2 = b * c + a * sn;
        a = a2, b = b2;
    }
    dst[lane] = __float2half(a), dst[lane + 32] = __float2half(b);
}

// ---- attention: CTA = (BQ queries x G heads of one KV head, split); 4 threads per row
struct Smem {
    float q[R][D];
    half k[BC][KP], v[BC][KP];
    float p[R][BC + 1];
};

__global__ void __launch_bounds__(THREADS) attn_kernel(GqaShape s, const half * __restrict__ q16,
                                                       const half * __restrict__ kc, const half * __restrict__ vc,
                                                       int pos0, int T, int splits, half * __restrict__ out16,
                                                       float * __restrict__ part)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem & sm = *reinterpret_cast<Smem *>(smem_raw);
    const int G = s.H / s.HKV, BQ = R / G, hk = blockIdx.y, split = blockIdx.z;
    const int tid = threadIdx.x, r = tid / 4, pt = tid % 4;   // this thread: row r, quarter pt (16 cells / 16 dims)
    const int qi = r / G, g = r % G, t = blockIdx.x * BQ + qi, h = hk * G + g;
    const bool valid = t < T;
    const int qpos = pos0 + t;   // the last cell this row may see
    // q tile, pre-scaled by 1/sqrt(D)
    const float scale = rsqrtf((float) D);
    for (int i = tid; i < R * D; i += THREADS) {
        const int rr = i / D, d = i % D, tt = blockIdx.x * BQ + rr / G, hh = hk * G + rr % G;
        sm.q[rr][d] = tt < T ? __half2float(q16[((size_t) tt * s.H + hh) * D + d]) * scale : 0.f;
    }
    // this split's cells: the chunk's visible range cut into `splits` equal pieces
    const int n_cells = pos0 + T, per = (n_cells + splits - 1) / splits;
    const int c_lo = split * per, cta_hi = min(pos0 + min(T, (int) (blockIdx.x + 1) * BQ), n_cells);
    const int c_hi = min((split + 1) * per, cta_hi);
    float m = -INFINITY, l = 0.f, o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i] = 0.f;
    for (int c0 = c_lo; c0 < c_hi; c0 += BC) {
        __syncthreads();   // the previous tile's p / v are consumed
        for (int i = tid; i < BC * D / 8; i += THREADS) {   // 8 halves per load
            const int cc = i / (D / 8), d8 = (i % (D / 8)) * 8, cell = c0 + cc;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (cell < c_hi) {
                kv = *reinterpret_cast<const uint4 *>(kc + ((size_t) cell * s.HKV + hk) * D + d8);
                vv = *reinterpret_cast<const uint4 *>(vc + ((size_t) cell * s.HKV + hk) * D + d8);
            }
            *reinterpret_cast<uint4 *>(&sm.k[cc][d8]) = kv;
            *reinterpret_cast<uint4 *>(&sm.v[cc][d8]) = vv;
        }
        __syncthreads();
        float sc[16], tmax = -INFINITY;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int cc = pt * 16 + j, cell = c0 + cc;
            float acc = 0.f;
#pragma unroll 16
            for (int d = 0; d < D; ++d) acc += sm.q[r][d] * __half2float(sm.k[cc][d]);
            sc[j] = valid && cell < c_hi && cell <= qpos ? acc : -INFINITY;
            tmax = fmaxf(tmax, sc[j]);
        }
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
        const float m_new = fmaxf(m, tmax);
        const float corr = m_new == -INFINITY ? 1.f : __expf(m - m_new);
        float psum = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float p = m_new == -INFINITY ? 0.f : __expf(sc[j] - m_new);
            sm.p[r][pt * 16 + j] = p;
            psum += p;
        }
        psum += __shfl_xor_sync(0xffffffffu, psum, 1);
        psum += __shfl_xor_sync(0xffffffffu, psum, 2);
        l = l * corr + psum, m = m_new;
        __syncwarp();   // the 4 threads of a row are in one warp: their p row is complete
#pragma unroll
        for (int i = 0; i < 16; ++i) o[i] *= corr;
        for (int cc = 0; cc < BC; ++cc) {
            const float p = sm.p[r][cc];
#pragma unroll
            for (int i = 0; i < 16; ++i) o[i] += p * __half2float(sm.v[cc][pt * 16 + i]);
        }
    }
    if (!valid) return;
    if (splits == 1) {
        const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) out16[((size_t) t * s.H + h) * D + pt * 16 + i] = __float2half(o[i] * inv);
        return;
    }
    float * pr = part + (((size_t) split * T + t) * s.H + h) * (D + 2);
    if (pt == 0) pr[0] = m, pr[1] = l;
#pragma unroll
    for (int i = 0; i < 16; ++i) pr[2 + pt * 16 + i] = o[i];
}

// ---- decode (T x G <= DEC_ROWS rows, e.g. T <= 4 at G = 4): the chunk kernel above keeps 16 of 256 threads busy at
// T = 1 and fits one CTA per SM (51 KB smem). Here a CTA = (KV head, split) with DEC_WARPS warps; a warp takes 32
// cells at a time, lane = cell: the lane holds its K row in registers and scores every row; online softmax per warp
// (shuffles), P V with lane = 2 output dims (V rows read coalesced). Warps merge in smem; splits write the same
// partials as attn_kernel (combine_kernel merges them).
constexpr int DEC_ROWS = 16, DEC_WARPS = 4, DEC_CELLS = DEC_WARPS * 32;   // cells per CTA pass

__global__ void __launch_bounds__(DEC_WARPS * 32) decode_kernel(GqaShape s, const half * __restrict__ q16,
                                                               const half * __restrict__ kc, const half * __restrict__ vc,
                                                               int pos0, int T, int splits, half * __restrict__ out16,
                                                               float * __restrict__ part)
{
    __shared__ float sq[DEC_ROWS][D];
    __shared__ float wm[DEC_WARPS][DEC_ROWS], wl[DEC_WARPS][DEC_ROWS];
    __shared__ float wo[DEC_WARPS][DEC_ROWS][D];
    const int G = s.H / s.HKV, rows = T * G, hk = blockIdx.x, split = blockIdx.y;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const float scale = rsqrtf((float) D);
    for (int i = threadIdx.x; i < rows * D; i += blockDim.x) {
        const int r = i / D, d = i % D;
        sq[r][d] = __half2float(q16[((size_t) (r / G) * s.H + hk * G + r % G) * D + d]) * scale;
    }
    __syncthreads();
    const int n_cells = pos0 + T, per = (n_cells + splits - 1) / splits;
    const int c_lo = split * per, c_hi = min(c_lo + per, n_cells);
    float m[DEC_ROWS], l[DEC_ROWS], a[DEC_ROWS], b[DEC_ROWS];
#pragma unroll
    for (int r = 0; r < DEC_ROWS; ++r) m[r] = -INFINITY, l[r] = 0.f, a[r] = 0.f, b[r] = 0.f;
    for (int c0 = c_lo + warp * 32; c0 < c_hi; c0 += DEC_CELLS) {
        const int cell = c0 + lane;
        const bool in = cell < c_hi;
        uint4 kr[D / 8];
#pragma unroll
        for (int j = 0; j < D / 8; ++j)
            kr[j] = in ? reinterpret_cast<const uint4 *>(kc + ((size_t) cell * s.HKV + hk) * D)[j] : make_uint4(0, 0, 0, 0);
        const half2 * k2 = reinterpret_cast<const half2 *>(kr);
        float p[DEC_ROWS];
#pragma unroll
        for (int r = 0; r < DEC_ROWS; ++r) {
            if (r >= rows) break;
            float acc = 0.f;
#pragma unroll
            for (int j = 0; j < D / 2; ++j) {
                const float2 kk = __half22float2(k2[j]);
                acc += sq[r][2 * j] * kk.x + sq[r][2 * j + 1] * kk.y;
            }
            float sc = in && cell <= pos0 + r / G ? acc : -INFINITY, mx = sc;
#pragma unroll
            for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mn = fmaxf(m[r], mx);
            const float corr = mn == -INFINITY ? 1.f : __expf(m[r] - mn);
            p[r] = mn == -INFINITY ? 0.f : __expf(sc - mn);
            float ps = p[r];
#pragma unroll
            for (int o = 16; o; o >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, o);
            l[r] = l[r] * corr + ps, a[r] *= corr, b[r] *= corr, m[r] = mn;
        }
        const int n = min(32, c_hi - c0);
        for (int j = 0; j < n; ++j) {
            const half * vp = vc + ((size_t) (c0 + j) * s.HKV + hk) * D;
            const float v0 = __half2float(vp[lane]), v1 = __half2float(vp[lane + 32]);
#pragma unroll
            for (int r = 0; r < DEC_ROWS; ++r) {
                if (r >= rows) break;
                const float pj = __shfl_sync(0xffffffffu, p[r], j);
                a[r] += pj * v0, b[r] += pj * v1;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < DEC_ROWS; ++r) {
        if (r >= rows) break;
        if (lane == 0) wm[warp][r] = m[r], wl[warp][r] = l[r];
        wo[warp][r][lane] = a[r], wo[warp][r][lane + 32] = b[r];
    }
    __syncthreads();
    for (int r = warp; r < rows; r += DEC_WARPS) {
        float M = -INFINITY, L = 0.f, A = 0.f, B = 0.f;
        for (int w = 0; w < DEC_WARPS; ++w) M = fmaxf(M, wm[w][r]);
        for (int w = 0; w < DEC_WARPS; ++w) {
            const float f = wm[w][r] == -INFINITY ? 0.f : __expf(wm[w][r] - M);
            L += wl[w][r] * f, A += wo[w][r][lane] * f, B += wo[w][r][lane + 32] * f;
        }
        const int t = r / G, h = hk * G + r % G;
        if (splits == 1) {
            const float inv = L > 0.f ? 1.f / L : 0.f;
            out16[((size_t) t * s.H + h) * D + lane] = __float2half(A * inv);
            out16[((size_t) t * s.H + h) * D + lane + 32] = __float2half(B * inv);
        } else {
            float * pr = part + (((size_t) split * T + t) * s.H + h) * (D + 2);
            if (lane == 0) pr[0] = M, pr[1] = L;
            pr[2 + lane] = A, pr[2 + lane + 32] = B;
        }
    }
}

// merge the splits of each (row, head) in split order
__global__ void combine_kernel(GqaShape s, int T, int splits, const float * __restrict__ part, half * __restrict__ out16)
{
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (w >= T * s.H) return;
    float m = -INFINITY;
    for (int k = lane; k < splits; k += 32) m = fmaxf(m, part[((size_t) k * T * s.H + w) * (D + 2)]);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float l = 0.f, a = 0.f, b = 0.f;
#pragma unroll 4
    for (int k = 0; k < splits; ++k) {
        const float * pr = part + ((size_t) k * T * s.H + w) * (D + 2);
        const float f = pr[0] == -INFINITY ? 0.f : __expf(pr[0] - m);
        l += pr[1] * f, a += pr[2 + lane] * f, b += pr[2 + lane + 32] * f;
    }
    const float inv = l > 0.f ? 1.f / l : 0.f;
    out16[(size_t) w * D + lane] = __float2half(a * inv), out16[(size_t) w * D + lane + 32] = __float2half(b * inv);
}

void check(const GqaShape & s)
{
    if (s.D != D) throw std::invalid_argument("attn::gqa: head dim " + std::to_string(s.D) + " (only 64)");
    if (s.HKV <= 0 || s.H % s.HKV || R % (s.H / s.HKV)) throw std::invalid_argument("attn::gqa: H / HKV must divide 64");
}

bool decode_path(const GqaShape & s, int T) { return T * (s.H / s.HKV) <= DEC_ROWS; }

int split_count(const GqaShape & s, int T, int n_cells)
{
    if (T > SPLIT_ROWS) return 1;
    if (decode_path(s, T))   // a CTA pass (DEC_CELLS) per split
        return std::max(1, std::min({MAX_SPLITS, MAX_PART_ROWS / T, (n_cells + DEC_CELLS - 1) / DEC_CELLS}));
    // a K/V tile per CTA (was cells / 512, at most 32: decode used 8 CTAs on 82 SMs)
    return std::max(1, std::min({MAX_SPLITS, MAX_PART_ROWS / T, (n_cells + BC - 1) / BC}));
}

}  // namespace

void gqa_prepare(const GqaShape & s, const float * q, const float * k, const float * v, const float * q_norm,
                 const float * k_norm, int pos0, int T, half * q16, half * k_cache, half * v_cache, cudaStream_t stream)
{
    check(s);
    if (T <= 0) return;
    const int warps = T * (s.H + 2 * s.HKV);
    prepare_kernel<<<(warps * 32 + 255) / 256, 256, 0, stream>>>(s, q, k, v, q_norm, k_norm, pos0, T, q16, k_cache, v_cache);
    TRUSS_CUDA(cudaGetLastError());
}

size_t gqa_workspace_bytes(const GqaShape & s, int max_T, int n_ctx)
{
    const int T = std::min(max_T, SPLIT_ROWS);
    return (size_t) split_count(s, T, n_ctx) * T * s.H * (s.D + 2) * sizeof(float);
}

void gqa_attention(const GqaShape & s, const half * q16, const half * k_cache, const half * v_cache, int pos0, int T,
                   half * out16, void * ws, size_t ws_bytes, cudaStream_t stream)
{
    check(s);
    if (T <= 0) return;
    const int G = s.H / s.HKV, BQ = R / G;
    int splits = split_count(s, T, pos0 + T);
    while (splits > 1 && (size_t) splits * T * s.H * (D + 2) * sizeof(float) > ws_bytes) --splits;
    static bool attr = false;   // per process: one device type here (sm_86)
    if (!attr) {
        TRUSS_CUDA(cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem)));
        attr = true;
    }
    if (decode_path(s, T)) {
        decode_kernel<<<dim3(s.HKV, splits), DEC_WARPS * 32, 0, stream>>>(s, q16, k_cache, v_cache, pos0, T, splits, out16,
                                                                          static_cast<float *>(ws));
    } else {
        const dim3 grid((T + BQ - 1) / BQ, s.HKV, splits);
        attn_kernel<<<grid, THREADS, sizeof(Smem), stream>>>(s, q16, k_cache, v_cache, pos0, T, splits, out16,
                                                             static_cast<float *>(ws));
    }
    TRUSS_CUDA(cudaGetLastError());
    if (splits > 1) {
        combine_kernel<<<(T * s.H * 32 + 255) / 256, 256, 0, stream>>>(s, T, splits, static_cast<const float *>(ws), out16);
        TRUSS_CUDA(cudaGetLastError());
    }
}

// B1 (TRACKER #167): addresses for graph-node identification (which nodes take pos0 by value). LFM-only file;
// gated by gqa_attn_test. The kernels live in the anonymous namespace above; this accessor is in the same TU.
GqaKernelFns gqa_kernel_fns()
{
    GqaKernelFns f;
    f.prepare = (void *) prepare_kernel;
    f.attn = (void *) attn_kernel;
    f.decode = (void *) decode_kernel;
    f.combine = (void *) combine_kernel;
    return f;
}

}  // namespace truss::attn
