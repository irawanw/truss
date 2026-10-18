// DeepSeek-V4.1 indexer kernels (indexer.cuh). v1: CUDA-core scores (one thread per compressed position, the query
// row in shared memory) and a one-CTA-per-row radix select; prefill at long context wants tensor-core scores
// (TRACKER #177 for the plan).
#include "kernels/ds41/indexer.cuh"

#include "core/cuda_check.h"

#include <stdexcept>

namespace truss::ds41 {
namespace {

constexpr int NH = 32, ID = 128;   // DeepSeek-V4.1-Flash indexer heads x dims (host-checked)

__global__ void __launch_bounds__(128) idx_scores_kernel(const float * __restrict__ q, const float * __restrict__ w,
                                                        float w_scale, const float * __restrict__ keys, int pos0, int ratio, int n_max,
                                                        float * __restrict__ s)
{
    __shared__ float sq[NH * ID], sw[NH];
    const int t = blockIdx.y;
    for (int i = threadIdx.x; i < NH * ID; i += blockDim.x) sq[i] = q[(size_t) t * NH * ID + i];
    if (threadIdx.x < NH) sw[threadIdx.x] = w[(size_t) t * NH + threadIdx.x] * w_scale;
    __syncthreads();
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_max) return;
    const int vis = (pos0 + t + 1) / ratio;
    float out = -INFINITY;
    if (n < vis) {
        float acc[NH];
#pragma unroll
        for (int h = 0; h < NH; ++h) acc[h] = 0.f;
        const float4 * kr = (const float4 *) (keys + (size_t) n * ID);
        for (int d4 = 0; d4 < ID / 4; ++d4) {
            const float4 k = __ldg(kr + d4);
#pragma unroll
            for (int h = 0; h < NH; ++h) {
                const float4 qq = *(const float4 *) (sq + h * ID + d4 * 4);
                acc[h] += qq.x * k.x + qq.y * k.y + qq.z * k.z + qq.w * k.w;
            }
        }
        out = 0.f;
#pragma unroll
        for (int h = 0; h < NH; ++h) out += fmaxf(acc[h], 0.f) * sw[h];
    }
    s[(size_t) t * n_max + n] = out;
}

__device__ __forceinline__ unsigned key_of(float f)
{
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ bool live(float f) { return f > -INFINITY; }   // +inf (a pinned block) counts; NaN not

constexpr int SEL_THREADS = 1024;

// block-wide exclusive scan of one int per thread (fixed order); returns the total
__device__ int block_scan(int v, int & excl)
{
    __shared__ int warp_tot[SEL_THREADS / 32];
    const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_tot[wid] = x;
    __syncthreads();
    if (wid == 0) {
        int t = lane < SEL_THREADS / 32 ? warp_tot[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += y;
        }
        if (lane < SEL_THREADS / 32) warp_tot[lane] = t;   // inclusive
    }
    __syncthreads();
    const int before = wid ? warp_tot[wid - 1] : 0;
    excl = before + x - v;
    const int total = warp_tot[SEL_THREADS / 32 - 1];
    __syncthreads();
    return total;
}

__global__ void __launch_bounds__(SEL_THREADS) select_kernel(const float * __restrict__ x, int n, int k,
                                                            int * __restrict__ idx)
{
    const float * row = x + (size_t) blockIdx.x * n;
    int * out = idx + (size_t) blockIdx.x * k;
    __shared__ int hist[256];
    __shared__ unsigned s_prefix;
    __shared__ int s_need;
    // finite count
    int c = 0;
    for (int i = threadIdx.x; i < n; i += SEL_THREADS) c += live(row[i]);
    int dummy;
    const int finite = block_scan(c, dummy);
    const int keff = finite < k ? finite : k;
    if (threadIdx.x == 0) s_prefix = 0, s_need = keff;
    __syncthreads();
    unsigned prefix = 0, mask = 0;
    if (keff > 0) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = threadIdx.x; i < 256; i += SEL_THREADS) hist[i] = 0;
            __syncthreads();
            for (int i = threadIdx.x; i < n; i += SEL_THREADS) {
                const float f = row[i];
                if (!live(f)) continue;
                const unsigned u = key_of(f);
                if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 0xff], 1);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                int need = s_need, b = 255;
                for (; b > 0; --b) {           // from the top: the bin where the remaining count is reached
                    if (hist[b] >= need) break;
                    need -= hist[b];
                }
                s_prefix = prefix | ((unsigned) b << shift);
                s_need = need;                 // how many of the threshold bin's keys are still wanted
            }
            __syncthreads();
            prefix = s_prefix;
            mask |= 0xffu << shift;
        }
    }
    // prefix = the k-th largest key; s_need = how many keys equal to it are taken (lowest indices first)
    const unsigned thr = prefix;
    const int need_eq = s_need;
    const int seg = (n + SEL_THREADS - 1) / SEL_THREADS, lo = threadIdx.x * seg, hi = min(n, lo + seg);
    int gt = 0, eq = 0;
    if (keff > 0)
        for (int i = lo; i < hi; ++i) {
            const float f = row[i];
            if (!live(f)) continue;
            const unsigned u = key_of(f);
            gt += u > thr, eq += u == thr;
        }
    int eq_before;
    block_scan(eq, eq_before);
    int take_eq = need_eq - eq_before;
    take_eq = take_eq < 0 ? 0 : (take_eq > eq ? eq : take_eq);
    int at;
    block_scan(gt + take_eq, at);
    if (keff > 0)
        for (int i = lo; i < hi; ++i) {
            const float f = row[i];
            if (!live(f)) continue;
            const unsigned u = key_of(f);
            if (u > thr || (u == thr && take_eq > 0)) {
                if (u == thr) --take_eq;
                out[at++] = i;
            }
        }
    for (int i = keff + threadIdx.x; i < k; i += SEL_THREADS) out[i] = -1;
}

__global__ void block_max_kernel(const float * __restrict__ s, int n, int block, int nb, int pos0, int ratio,
                                 float * __restrict__ bm)
{
    const int t = blockIdx.y, b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float * r = s + (size_t) t * n;
    float m = -INFINITY;
    for (int i = b * block; i < min(n, (b + 1) * block); ++i) m = fmaxf(m, r[i]);
    const int vis = (pos0 + t + 1) / ratio;
    if (vis > 0 && b == (vis - 1) / block) m = INFINITY;
    bm[(size_t) t * nb + b] = m;
}

__global__ void mask_blocks_kernel(float * __restrict__ s, int n, int block, const int * __restrict__ blocks, int kb)
{
    extern __shared__ unsigned bits[];
    const int t = blockIdx.x, nb = (n + block - 1) / block, words = (nb + 31) / 32;
    for (int i = threadIdx.x; i < words; i += blockDim.x) bits[i] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < kb; i += blockDim.x) {
        const int b = blocks[(size_t) t * kb + i];
        if (b >= 0) atomicOr(&bits[b / 32], 1u << (b % 32));
    }
    __syncthreads();
    float * r = s + (size_t) t * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int b = i / block;
        if (!(bits[b / 32] >> (b % 32) & 1u)) r[i] = -INFINITY;
    }
}

}  // namespace

void idx_scores(const float * q, const float * w, float w_scale, const float * keys, int nh, int id, int pos0, int T,
                int ratio, float * s, cudaStream_t stream)
{
    if (nh != NH || id != ID) throw std::invalid_argument("ds41::idx_scores: needs 32 heads x 128");
    const int n_max = (pos0 + T) / ratio;
    if (T <= 0 || n_max <= 0) return;
    if (T > 65535) throw std::invalid_argument("ds41::idx_scores: T > 65535");
    idx_scores_kernel<<<dim3((n_max + 127) / 128, T), 128, 0, stream>>>(q, w, w_scale, keys, pos0, ratio, n_max, s);
    TRUSS_CUDA(cudaGetLastError());
}

void select_topk(const float * x, int T, int n, int k, int * idx, cudaStream_t stream)
{
    if (T <= 0) return;
    select_kernel<<<T, SEL_THREADS, 0, stream>>>(x, n, k, idx);
    TRUSS_CUDA(cudaGetLastError());
}

void block_max(const float * s, int T, int n, int block, int pos0, int ratio, float * bm, cudaStream_t stream)
{
    const int nb = (n + block - 1) / block;
    if (T <= 0 || nb <= 0) return;
    block_max_kernel<<<dim3((nb + 255) / 256, T), 256, 0, stream>>>(s, n, block, nb, pos0, ratio, bm);
    TRUSS_CUDA(cudaGetLastError());
}

void mask_blocks(float * s, int T, int n, int block, const int * blocks, int kb, cudaStream_t stream)
{
    const int nb = (n + block - 1) / block, words = (nb + 31) / 32;
    if (T <= 0 || n <= 0) return;
    if (words * 4 > 48 * 1024) throw std::invalid_argument("ds41::mask_blocks: too many blocks for shared memory");
    mask_blocks_kernel<<<T, 256, words * 4, stream>>>(s, n, block, blocks, kb);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::ds41
