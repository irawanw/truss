// DeepSeek-V4.1 sparse attention, v3 (sparse_attn.cuh): v2 (sparse_attn_tc.cu) with the key tile double buffered
// through cp.async - the next tile's keys load into the second shared buffer while the current tile computes, so the
// global key reads no longer stall the tensor-core work across the two __syncthreads. Same CTA shape (16 heads per
// CTA, KT = 32), same math, same rounding and the same split/combine path as v2; only the load path differs.
// TRUSS_DS41_ATTN_TC=2 selects it in AttnDs41 (prompt rows; see the dispatcher there).
#include "kernels/ds41/sparse_attn.cuh"

#include "core/cuda_check.h"

#include <algorithm>
#include <stdexcept>

namespace truss::ds41 {
namespace {

constexpr int HG = 16, KT = 32, D = 512, THREADS = 128, LDS = D + 8, LDP = KT + 8;
constexpr size_t SMEM = sizeof(half) * (HG * LDS + 2 * KT * LDS + HG * LDP)
    + sizeof(float) * (HG * KT + 3 * HG + 2 * KT);

__device__ __forceinline__ unsigned smem_u32(const void * p) { return (unsigned) __cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(void * dst, const void * src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;\n"); }

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

__global__ void __launch_bounds__(THREADS) sparse_attn_tc_async_kernel(const SparseAttnArgs a, int tiles_per_split, float * part)
{
    extern __shared__ __align__(16) unsigned char smem[];
    half * Qs = (half *) smem, * Ks = Qs + HG * LDS, * Ps = Ks + 2 * KT * LDS;
    float * Ss = (float *) (Ps + HG * LDP), * row_m = Ss + HG * KT, * row_l = row_m + HG, * row_a = row_l + HG;
    float * valid = row_a + HG;
    const int ngrp = a.H / HG, t = blockIdx.x / ngrp, h0 = (blockIdx.x % ngrp) * HG, split = blockIdx.y;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, c2 = (lane % 4) * 2;
    const int p = a.pos0 + t, w0 = max(a.clip, p - a.window + 1), nwin = p - w0 + 1;
    const int nkeys = nwin + (a.idx ? a.k : 0), ntile = (nkeys + KT - 1) / KT;
    const int tb = split * tiles_per_split, te = min(ntile, tb + tiles_per_split);
    // the 16 query rows as fp16
    const float * q = a.q + ((size_t) t * a.H + h0) * D;
    for (int i = threadIdx.x * 4; i < HG * D; i += THREADS * 4) {
        const float4 v = *(const float4 *) (q + i);
        half * d = Qs + (i / D) * LDS + i % D;
        *(half2 *) d = __floats2half2_rn(v.x, v.y);
        *(half2 *) (d + 2) = __floats2half2_rn(v.z, v.w);
    }
    if (threadIdx.x < HG) row_m[threadIdx.x] = -INFINITY, row_l[threadIdx.x] = 0.f;
    float acc[16][4];   // O: this warp's 128 dims (16 n-tiles of 8) x 16 heads
    for (auto & r : acc) r[0] = r[1] = r[2] = r[3] = 0.f;
    // one key tile into buffer buf: cp.async where the key exists, a zero store where it does not
    auto issue = [&](int tile, int buf) {
        half * Kb = Ks + (size_t) buf * KT * LDS;
        float * vb = valid + buf * KT;
        for (int c = threadIdx.x; c < KT * (D / 8); c += THREADS) {
            const int key = c / (D / 8), ch = c % (D / 8), j = tile * KT + key;
            const half * kp = j < nkeys ? key_ptr(a, t, j, nwin, w0) : nullptr;
            if (kp) cp_async16(Kb + key * LDS + ch * 8, kp + ch * 8);
            else *(uint4 *) (Kb + key * LDS + ch * 8) = make_uint4(0, 0, 0, 0);
            if (ch == 0) vb[key] = kp ? 1.f : 0.f;
        }
    };
    issue(tb, 0);
    cp_async_commit();
    for (int tile = tb; tile < te; ++tile) {
        const int cur = (tile - tb) & 1;
        if (tile + 1 < te) { issue(tile + 1, 1 - cur); cp_async_commit(); cp_async_wait1(); }
        else cp_async_wait0();
        __syncthreads();   // this tile's keys (and Qs, row_*) are visible; the other buffer is free to load
        const half * Kt = Ks + (size_t) cur * KT * LDS;
        const float * vt = valid + cur * KT;
        {   // S: warp w takes keys w*8 .. w*8+7
            float s[4] = { 0.f, 0.f, 0.f, 0.f };
            for (int kk = 0; kk < D; kk += 16) {
                unsigned af[4], bf[2];
                ldsm_x4(af, Qs + (lane % 16) * LDS + kk + (lane / 16) * 8);
                ldsm_x2(bf, Kt + (warp * 8 + lane % 8) * LDS + kk + ((lane / 8) % 2) * 8);
                mma(s, af, bf);
            }
            const int k0 = warp * 8 + c2;
            Ss[g * KT + k0] = vt[k0] != 0.f ? s[0] * a.scale : -INFINITY;
            Ss[g * KT + k0 + 1] = vt[k0 + 1] != 0.f ? s[1] * a.scale : -INFINITY;
            Ss[(g + 8) * KT + k0] = vt[k0] != 0.f ? s[2] * a.scale : -INFINITY;
            Ss[(g + 8) * KT + k0 + 1] = vt[k0 + 1] != 0.f ? s[3] * a.scale : -INFINITY;
        }
        __syncthreads();
        if (threadIdx.x < HG) {   // online softmax, one thread per head (keys in order)
            const int r = threadIdx.x;
            float mt = -INFINITY;
            for (int j = 0; j < KT; ++j) mt = fmaxf(mt, Ss[r * KT + j]);
            const float mo = row_m[r], mn = fmaxf(mo, mt);
            const float al = mn == -INFINITY ? 1.f : expf(mo - mn);
            float l = 0.f;
            for (int j = 0; j < KT; ++j) {
                const float e = mn == -INFINITY ? 0.f : expf(Ss[r * KT + j] - mn);
                l += e;
                Ps[r * LDP + j] = __float2half(e);
            }
            row_m[r] = mn, row_l[r] = row_l[r] * al + l, row_a[r] = al;
        }
        __syncthreads();
        {   // O = O * alpha + P K over this warp's dims
            const float a0 = row_a[g], a1 = row_a[g + 8];
            for (auto & r : acc) r[0] *= a0, r[1] *= a0, r[2] *= a1, r[3] *= a1;
            for (int ks = 0; ks < KT; ks += 16) {
                unsigned af[4];
                ldsm_x4(af, Ps + (lane % 16) * LDP + ks + (lane / 16) * 8);
                for (int nt = 0; nt < 16; ++nt) {
                    unsigned bf[2];
                    ldsm_x2_t(bf, Kt + (ks + lane % 16) * LDS + warp * 128 + nt * 8);
                    mma(acc[nt], af, bf);
                }
            }
        }
        __syncthreads();   // every read of this buffer is done before the next prefetch overwrites it
    }
    __syncthreads();
    const float m0 = row_m[g], m1 = row_m[g + 8], l0 = row_l[g], l1 = row_l[g + 8];
    if (gridDim.y == 1) {   // final: the sink only enters the denominator (v1 clamps the max at -1e30)
        const float mm0 = fmaxf(m0, -1e30f), mm1 = fmaxf(m1, -1e30f);
        const float i0 = 1.f / (l0 * (m0 == -INFINITY ? 0.f : expf(m0 - mm0)) + expf(a.sinks[h0 + g] - mm0));
        const float i1 = 1.f / (l1 * (m1 == -INFINITY ? 0.f : expf(m1 - mm1)) + expf(a.sinks[h0 + g + 8] - mm1));
        float * o0 = a.out + ((size_t) t * a.H + h0 + g) * D + warp * 128 + c2, * o1 = o0 + (size_t) 8 * D;
        for (int nt = 0; nt < 16; ++nt) {
            *(float2 *) (o0 + nt * 8) = make_float2(acc[nt][0] * i0, acc[nt][1] * i0);
            *(float2 *) (o1 + nt * 8) = make_float2(acc[nt][2] * i1, acc[nt][3] * i1);
        }
        return;
    }
    // partial: [t][h][split] (m, l) then O [t][h][split][D]
    const int S = gridDim.y;
    float * ml = part, * po = part + (size_t) 2 * a.T * a.H * S;
    const size_t r0 = ((size_t) t * a.H + h0 + g) * S + split, r1 = r0 + (size_t) 8 * S;
    if (lane % 4 == 0 && warp == 0) ml[2 * r0] = m0, ml[2 * r0 + 1] = l0, ml[2 * r1] = m1, ml[2 * r1 + 1] = l1;
    float * o0 = po + r0 * D + warp * 128 + c2, * o1 = po + r1 * D + warp * 128 + c2;
    for (int nt = 0; nt < 16; ++nt) {
        *(float2 *) (o0 + nt * 8) = make_float2(acc[nt][0], acc[nt][1]);
        *(float2 *) (o1 + nt * 8) = make_float2(acc[nt][2], acc[nt][3]);
    }
}

__global__ void __launch_bounds__(THREADS) sparse_attn_combine_kernel(const SparseAttnArgs a, int S, const float * part)
{
    const int r = blockIdx.x, h = r % a.H;   // r = t * H + h
    const float * ml = part + (size_t) 2 * r * S, * po = part + (size_t) 2 * a.T * a.H * S + (size_t) r * S * D;
    float M = -1e30f;
    for (int s = 0; s < S; ++s) M = fmaxf(M, ml[2 * s]);
    float den = expf(a.sinks[h] - M);
    for (int s = 0; s < S; ++s) den += ml[2 * s] == -INFINITY ? 0.f : ml[2 * s + 1] * expf(ml[2 * s] - M);
    const float inv = 1.f / den;
    for (int d = threadIdx.x * 4; d < D; d += THREADS * 4) {
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

bool sparse_attn_tc_async_ok(const SparseAttnArgs & a) { return a.D == D && a.H % HG == 0; }

size_t sparse_attn_tc_async_ws_floats(const SparseAttnArgs & a)
{
    const int S = splits(a);
    return S == 1 ? 0 : (size_t) a.T * a.H * S * (D + 2);
}

void sparse_attn_tc_async(const SparseAttnArgs & a, float * ws, cudaStream_t stream)
{
    if (!sparse_attn_tc_async_ok(a)) throw std::invalid_argument("ds41::sparse_attn_tc_async: needs D = 512 and H % 16 == 0");
    if (a.pos0 > 0 && a.R < a.window) throw std::invalid_argument("ds41::sparse_attn_tc_async: ring shorter than the window");
    if (a.T <= 0) return;
    static bool attr = false;   // per process: the opt-in is a function attribute (same on every device)
    if (!attr) {
        int dev = 0, n = 0;
        TRUSS_CUDA(cudaGetDevice(&dev));
        TRUSS_CUDA(cudaGetDeviceCount(&n));
        for (int i = 0; i < n; ++i) {
            TRUSS_CUDA(cudaSetDevice(i));
            TRUSS_CUDA(cudaFuncSetAttribute(sparse_attn_tc_async_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) SMEM));
        }
        TRUSS_CUDA(cudaSetDevice(dev));
        attr = true;
    }
    const int S = splits(a), ntile = (a.window + (a.idx ? a.k : 0) + KT - 1) / KT;
    if (S > 1 && !ws) throw std::invalid_argument("ds41::sparse_attn_tc_async: split launch without a workspace");
    sparse_attn_tc_async_kernel<<<dim3((unsigned) (a.T * (a.H / HG)), (unsigned) S), THREADS, SMEM, stream>>>(
        a, (ntile + S - 1) / S, ws);
    TRUSS_CUDA(cudaGetLastError());
    if (S > 1) {
        sparse_attn_combine_kernel<<<(unsigned) (a.T * a.H), THREADS, 0, stream>>>(a, S, ws);
        TRUSS_CUDA(cudaGetLastError());
    }
}

}  // namespace truss::ds41
