// DeepSeek-V4.1 sparse attention, v5 (sparse_attn.cuh): the v4 shape (32 heads per CTA, KT = 32) with 16 warps
// (512 threads) instead of 8. v4 already loads a row's keys twice instead of four times, but its ~72 KB of dynamic
// shared and 121 registers leave room for one CTA per SM, so the ldmatrix / mma latency is not hidden. Both halves
// of the tile here use all 16 warps: S = Q K^T (32 x 32) is split over the two halves of the 512-dim reduction
// (warp = head half, key octet, k half), each writing one of two partial buffers that the softmax adds, and
// O += P K (32 x 512) gives each warp 16 heads x 64 dims (8 n-tiles). Same tiles, online softmax, mma and
// split/combine path as v2/v4; S rounds as two 256-dim partials instead of one 512-dim sum.
// TRUSS_DS41_ATTN_TC=4 selects it.
#include "kernels/ds41/sparse_attn.cuh"

#include "core/cuda_check.h"

#include <algorithm>
#include <stdexcept>

namespace truss::ds41 {
namespace {

constexpr int HG = 32, KT = 32, D = 512, THREADS = 512, CTHREADS = 256, LDS = D + 8, LDP = KT + 8;
constexpr size_t SMEM = sizeof(half) * (HG * LDS + KT * LDS + HG * LDP)
    + sizeof(float) * (2 * HG * KT + 3 * HG + KT);

__device__ __forceinline__ unsigned smem_u32(const void * p) { return (unsigned) __cvta_generic_to_shared(p); }

__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const half * p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(unsigned (&r)[2], const half * p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2_t(unsigned (&r)[2], const half * p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2])
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ const half * key_ptr(const SparseAttnArgs & a, int t, int j, int nwin, int w0)
{
    if (j < nwin) {
        const int p = w0 + j;
        return p >= a.pos0 ? a.chunk_kv + (size_t) (p - a.pos0) * D : a.ring + (size_t) (p % a.R) * D;
    }
    const int c = a.idx[(size_t) (t + a.row_off) * a.k + (j - nwin)];
    return c >= 0 ? a.comp + (size_t) c * D : nullptr;
}

__global__ void __launch_bounds__(THREADS) sparse_attn_tc_16w_kernel(const SparseAttnArgs a, int tiles_per_split,
                                                                     float * part)
{
    extern __shared__ __align__(16) unsigned char smem[];
    half * Qs = (half *) smem, * Ks = Qs + HG * LDS, * Ps = Ks + KT * LDS;
    float * Ss = (float *) (Ps + HG * LDP), * row_m = Ss + 2 * HG * KT, * row_l = row_m + HG, * row_a = row_l + HG;
    float * valid = row_a + HG;
    const int ngrp = a.H / HG, t = blockIdx.x / ngrp, h0 = (blockIdx.x % ngrp) * HG, split = blockIdx.y;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int ph = warp / 8, pd = warp % 8, pk = pd / 2, kd = warp % 2;   // S split: head half, key octet, k half
    const int g = lane / 4, c2 = (lane % 4) * 2;
    const int p = a.pos0 + t, w0 = max(a.clip, p - a.window + 1), nwin = p - w0 + 1;
    const int nkeys = nwin + (a.idx ? a.k : 0), ntile = (nkeys + KT - 1) / KT;
    const int tb = split * tiles_per_split, te = min(ntile, tb + tiles_per_split);
    // the 32 query rows as fp16
    const float * q = a.q + ((size_t) t * a.H + h0) * D;
    for (int i = threadIdx.x * 4; i < HG * D; i += THREADS * 4) {
        const float4 v = *(const float4 *) (q + i);
        half * d = Qs + (i / D) * LDS + i % D;
        *(half2 *) d = __floats2half2_rn(v.x, v.y);
        *(half2 *) (d + 2) = __floats2half2_rn(v.z, v.w);
    }
    if (threadIdx.x < HG) row_m[threadIdx.x] = -INFINITY, row_l[threadIdx.x] = 0.f;
    float acc[8][4];   // O: this warp's head half x 64 dims (8 n-tiles of 8)
    for (auto & r : acc) r[0] = r[1] = r[2] = r[3] = 0.f;
    for (int tile = tb; tile < te; ++tile) {
        __syncthreads();   // Ks / Ps of the previous tile are consumed (and Qs, row_* written, on the first)
        for (int c = threadIdx.x; c < KT * (D / 8); c += THREADS) {
            const int key = c / (D / 8), ch = c % (D / 8), j = tile * KT + key;
            const half * kp = j < nkeys ? key_ptr(a, t, j, nwin, w0) : nullptr;
            *(uint4 *) (Ks + key * LDS + ch * 8) = kp ? *(const uint4 *) (kp + ch * 8) : make_uint4(0, 0, 0, 0);
            if (ch == 0) valid[key] = kp ? 1.f : 0.f;
        }
        __syncthreads();
        {   // S: warp (ph, pk, kd) takes head half ph, keys pk*8 .. pk*8+7, the kd half of the 512-dim reduction
            float s[4] = { 0.f, 0.f, 0.f, 0.f };
            for (int kk = kd * 256; kk < kd * 256 + 256; kk += 16) {
                unsigned af[4], bf[2];
                ldsm_x4(af, Qs + (ph * 16 + lane % 16) * LDS + kk + (lane / 16) * 8);
                ldsm_x2(bf, Ks + (pk * 8 + lane % 8) * LDS + kk + ((lane / 8) % 2) * 8);
                mma(s, af, bf);
            }
            const int k0 = pk * 8 + c2, r0 = ph * 16 + g, r1 = r0 + 8;
            float * Sb = Ss + (size_t) kd * HG * KT;   // the two k halves meet in the softmax
            Sb[r0 * KT + k0] = s[0], Sb[r0 * KT + k0 + 1] = s[1];
            Sb[r1 * KT + k0] = s[2], Sb[r1 * KT + k0 + 1] = s[3];
        }
        __syncthreads();
        {   // online softmax, 16 threads per head: 2 keys each, then a shuffle reduce over the 16 (keys stay in
            // order per lane); the group of 16 for a head sits inside one warp
            const int r = threadIdx.x / 16, sub = threadIdx.x % 16, j0 = sub * 2, j1 = j0 + 1;
            const float v0 = valid[j0] != 0.f ? (Ss[r * KT + j0] + Ss[HG * KT + r * KT + j0]) * a.scale : -INFINITY;
            const float v1 = valid[j1] != 0.f ? (Ss[r * KT + j1] + Ss[HG * KT + r * KT + j1]) * a.scale : -INFINITY;
            float mt = fmaxf(v0, v1);
            for (int o = 8; o; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o, 16));
            const float mo = row_m[r], mn = fmaxf(mo, mt);
            const float al = mn == -INFINITY ? 1.f : expf(mo - mn);
            const float e0 = mn == -INFINITY ? 0.f : expf(v0 - mn);
            const float e1 = mn == -INFINITY ? 0.f : expf(v1 - mn);
            Ps[r * LDP + j0] = __float2half(e0), Ps[r * LDP + j1] = __float2half(e1);
            float l = e0 + e1;
            for (int o = 8; o; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o, 16);
            if (sub == 0) row_m[r] = mn, row_l[r] = row_l[r] * al + l, row_a[r] = al;
        }
        __syncthreads();
        {   // O = O * alpha + P K over this warp's head half x 64 dims
            const float a0 = row_a[ph * 16 + g], a1 = row_a[ph * 16 + g + 8];
            for (auto & r : acc) r[0] *= a0, r[1] *= a0, r[2] *= a1, r[3] *= a1;
            for (int ks = 0; ks < KT; ks += 16) {
                unsigned af[4];
                ldsm_x4(af, Ps + (ph * 16 + lane % 16) * LDP + ks + (lane / 16) * 8);
                for (int nt = 0; nt < 8; ++nt) {
                    unsigned bf[2];
                    ldsm_x2_t(bf, Ks + (ks + lane % 16) * LDS + pd * 64 + nt * 8);
                    mma(acc[nt], af, bf);
                }
            }
        }
    }
    __syncthreads();
    const int r0 = ph * 16 + g, r1 = r0 + 8;
    const float m0 = row_m[r0], m1 = row_m[r1], l0 = row_l[r0], l1 = row_l[r1];
    if (gridDim.y == 1) {   // final: the sink only enters the denominator (v1 clamps the max at -1e30)
        const float mm0 = fmaxf(m0, -1e30f), mm1 = fmaxf(m1, -1e30f);
        const float i0 = 1.f / (l0 * (m0 == -INFINITY ? 0.f : expf(m0 - mm0)) + expf(a.sinks[h0 + r0] - mm0));
        const float i1 = 1.f / (l1 * (m1 == -INFINITY ? 0.f : expf(m1 - mm1)) + expf(a.sinks[h0 + r1] - mm1));
        float * o0 = a.out + ((size_t) t * a.H + h0 + r0) * D + pd * 64 + c2, * o1 = o0 + (size_t) 8 * D;
        for (int nt = 0; nt < 8; ++nt) {
            *(float2 *) (o0 + nt * 8) = make_float2(acc[nt][0] * i0, acc[nt][1] * i0);
            *(float2 *) (o1 + nt * 8) = make_float2(acc[nt][2] * i1, acc[nt][3] * i1);
        }
        return;
    }
    // partial: [t][h][split] (m, l) then O [t][h][split][D]
    const int S = gridDim.y;
    float * ml = part, * po = part + (size_t) 2 * a.T * a.H * S;
    const size_t p0 = ((size_t) t * a.H + h0 + r0) * S + split, p1 = p0 + (size_t) 8 * S;
    if (lane % 4 == 0 && pd == 0) ml[2 * p0] = m0, ml[2 * p0 + 1] = l0, ml[2 * p1] = m1, ml[2 * p1 + 1] = l1;
    float * o0 = po + p0 * D + pd * 64 + c2, * o1 = po + p1 * D + pd * 64 + c2;
    for (int nt = 0; nt < 8; ++nt) {
        *(float2 *) (o0 + nt * 8) = make_float2(acc[nt][0], acc[nt][1]);
        *(float2 *) (o1 + nt * 8) = make_float2(acc[nt][2], acc[nt][3]);
    }
}

__global__ void __launch_bounds__(CTHREADS) sparse_attn_combine_kernel(const SparseAttnArgs a, int S, const float * part)
{
    const int r = blockIdx.x, h = r % a.H;   // r = t * H + h
    const float * ml = part + (size_t) 2 * r * S, * po = part + (size_t) 2 * a.T * a.H * S + (size_t) r * S * D;
    float M = -1e30f;
    for (int s = 0; s < S; ++s) M = fmaxf(M, ml[2 * s]);
    float den = expf(a.sinks[h] - M);
    for (int s = 0; s < S; ++s) den += ml[2 * s] == -INFINITY ? 0.f : ml[2 * s + 1] * expf(ml[2 * s] - M);
    const float inv = 1.f / den;
    for (int d = threadIdx.x * 4; d < D; d += CTHREADS * 4) {
        float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int s = 0; s < S; ++s) {
            if (ml[2 * s] == -INFINITY) continue;
            const float w = expf(ml[2 * s] - M);
            const float4 v = *(const float4 *) (po + (size_t) s * D + d);
            o.x += v.x * w, o.y += v.y * w, o.z += v.z * w, o.w += v.w * w;
        }
        *(float4 *) (a.out + (size_t) r * D + d) = make_float4(o.x * inv, o.y * inv, o.z * inv, o.w * inv);
    }
}

int splits(const SparseAttnArgs & a)
{
    const int ntile = (a.window + (a.idx ? a.k : 0) + KT - 1) / KT, ctas = a.T * (a.H / HG);
    return std::max(1, std::min(ntile, (160 + ctas - 1) / ctas));   // ~2 CTAs per SM on an 82-SM card
}

}  // namespace

bool sparse_attn_tc_16w_ok(const SparseAttnArgs & a) { return a.D == D && a.H % HG == 0; }

size_t sparse_attn_tc_16w_ws_floats(const SparseAttnArgs & a)
{
    const int S = splits(a);
    return S == 1 ? 0 : (size_t) a.T * a.H * S * (D + 2);
}

void sparse_attn_tc_16w(const SparseAttnArgs & a, float * ws, cudaStream_t stream)
{
    if (!sparse_attn_tc_16w_ok(a)) throw std::invalid_argument("ds41::sparse_attn_tc_16w: needs D = 512 and H % 32 == 0");
    if (a.pos0 > 0 && a.R < a.window) throw std::invalid_argument("ds41::sparse_attn_tc_16w: ring shorter than the window");
    if (a.T <= 0) return;
    static bool attr = false;   // per process: the opt-in is a function attribute (same on every device)
    if (!attr) {
        int dev = 0, n = 0;
        TRUSS_CUDA(cudaGetDevice(&dev));
        TRUSS_CUDA(cudaGetDeviceCount(&n));
        for (int i = 0; i < n; ++i) {
            TRUSS_CUDA(cudaSetDevice(i));
            TRUSS_CUDA(cudaFuncSetAttribute(sparse_attn_tc_16w_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) SMEM));
        }
        TRUSS_CUDA(cudaSetDevice(dev));
        attr = true;
    }
    const int S = splits(a), ntile = (a.window + (a.idx ? a.k : 0) + KT - 1) / KT;
    if (S > 1 && !ws) throw std::invalid_argument("ds41::sparse_attn_tc_16w: split launch without a workspace");
    sparse_attn_tc_16w_kernel<<<dim3((unsigned) (a.T * (a.H / HG)), (unsigned) S), THREADS, SMEM, stream>>>(
        a, (ntile + S - 1) / S, ws);
    TRUSS_CUDA(cudaGetLastError());
    if (S > 1) {
        sparse_attn_combine_kernel<<<(unsigned) (a.T * a.H), CTHREADS, 0, stream>>>(a, S, ws);
        TRUSS_CUDA(cudaGetLastError());
    }
}

}  // namespace truss::ds41
