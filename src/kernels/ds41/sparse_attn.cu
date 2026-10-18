// DeepSeek-V4.1 sparse attention, v1 (sparse_attn.cuh): CTA (row, head), 256 threads; the query row in shared
// memory, one warp per key for the scores (16 dims per lane), softmax in shared memory, then each thread owns two
// output dims and sums p * kv over the keys in key order (deterministic).
#include "kernels/ds41/sparse_attn.cuh"

#include "core/cuda_check.h"

#include <stdexcept>

namespace truss::ds41 {
namespace {

constexpr int THREADS = 256, WARPS = THREADS / 32, D_MAX = 512, KEYS_MAX = 128 + 1024;

__device__ __forceinline__ const half * key_ptr(const SparseAttnArgs & a, int t, int j, int nwin, int w0)
{
    if (j < nwin) {
        const int p = w0 + j;
        return p >= a.pos0 ? a.chunk_kv + (size_t) (p - a.pos0) * a.D : a.ring + (size_t) (p % a.R) * a.D;
    }
    const int c = a.idx[(size_t) t * a.k + (j - nwin)];
    return c >= 0 ? a.comp + (size_t) c * a.D : nullptr;
}

__global__ void __launch_bounds__(THREADS) sparse_attn_kernel(const SparseAttnArgs a)
{
    __shared__ float sq[D_MAX], sp[KEYS_MAX];
    __shared__ float s_red[WARPS];
    const int t = blockIdx.x / a.H, h = blockIdx.x % a.H, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int p = a.pos0 + t, w0 = max(0, p - a.window + 1), nwin = p - w0 + 1;
    const int nkeys = nwin + (a.idx ? a.k : 0);
    const float * q = a.q + ((size_t) t * a.H + h) * a.D;
    for (int i = threadIdx.x; i < a.D; i += THREADS) sq[i] = q[i];
    __syncthreads();
    // scores
    for (int j = warp; j < nkeys; j += WARPS) {
        const half * kv = key_ptr(a, t, j, nwin, w0);
        float s = -INFINITY;
        if (kv) {
            float acc = 0.f;
            for (int d = lane * 2; d < a.D; d += 64) {
                const float2 v = __half22float2(*(const half2 *) (kv + d));
                acc += sq[d] * v.x + sq[d + 1] * v.y;
            }
            for (int o = 16; o; o /= 2) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            s = acc * a.scale;
        }
        if (lane == 0) sp[j] = s;
    }
    __syncthreads();
    // max over the keys (the sink only enters the denominator, as in the reference)
    float m = -INFINITY;
    for (int j = threadIdx.x; j < nkeys; j += THREADS) m = fmaxf(m, sp[j]);
    for (int o = 16; o; o /= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) s_red[warp] = m;
    __syncthreads();
    m = s_red[0];
    for (int w = 1; w < WARPS; ++w) m = fmaxf(m, s_red[w]);
    m = fmaxf(m, -1e30f);
    __syncthreads();
    float den = 0.f;
    for (int j = threadIdx.x; j < nkeys; j += THREADS) {
        const float e = expf(sp[j] - m);
        sp[j] = e;
        den += e;
    }
    for (int o = 16; o; o /= 2) den += __shfl_xor_sync(0xffffffffu, den, o);
    if (lane == 0) s_red[warp] = den;
    __syncthreads();
    den = 0.f;
    for (int w = 0; w < WARPS; ++w) den += s_red[w];
    den += expf(a.sinks[h] - m);
    const float inv = 1.f / den;
    // out: two dims per thread, keys in order
    float * o = a.out + ((size_t) t * a.H + h) * a.D;
    for (int d = threadIdx.x * 2; d < a.D; d += THREADS * 2) {
        float o0 = 0.f, o1 = 0.f;
        for (int j = 0; j < nkeys; ++j) {
            const float pj = sp[j];
            if (pj == 0.f) continue;
            const half * kv = key_ptr(a, t, j, nwin, w0);
            const float2 v = __half22float2(*(const half2 *) (kv + d));
            o0 += pj * v.x, o1 += pj * v.y;
        }
        o[d] = o0 * inv;
        o[d + 1] = o1 * inv;
    }
}

}  // namespace

void sparse_attn(const SparseAttnArgs & a, cudaStream_t stream)
{
    if (a.D > D_MAX || a.D % 64 || a.window + (a.idx ? a.k : 0) > KEYS_MAX)
        throw std::invalid_argument("ds41::sparse_attn: D / window / k beyond the v1 kernel");
    if (a.pos0 > 0 && a.R < a.window) throw std::invalid_argument("ds41::sparse_attn: ring shorter than the window");
    if (a.T <= 0) return;
    sparse_attn_kernel<<<(unsigned) a.T * a.H, THREADS, 0, stream>>>(a);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::ds41
