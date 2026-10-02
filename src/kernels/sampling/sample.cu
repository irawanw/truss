#include "sample.cuh"

#include "core/cuda_check.h"

namespace truss::sampling {
namespace {

constexpr int THREADS = 1024, WARPS = THREADS / 32;
constexpr int BISECT = 40;   // halvings of the logit range: far below fp32 spacing for any real logit spread

// block-wide sum / max / count; every thread gets the result
__device__ float block_sum(float v, float * sh)
{
    for (int o = 16; o; o /= 2) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v;
    __syncthreads();
    float t = 0.f;
    for (int w = 0; w < WARPS; ++w) t += sh[w];   // fixed order: the same result in every thread and run
    return t;
}

__device__ float block_max(float v, float * sh)
{
    for (int o = 16; o; o /= 2) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v;
    __syncthreads();
    float t = -INFINITY;
    for (int w = 0; w < WARPS; ++w) t = fmaxf(t, sh[w]);
    return t;
}

__device__ __forceinline__ uint64_t mix(uint64_t z)   // splitmix64 finalizer
{
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// x / T in logit units; the cut and the draw both work on y = x / T
__global__ void __launch_bounds__(THREADS) sample_kernel(const float * xs, int n, SampleParams p, uint64_t counter, int * out)
{
    __shared__ float sh[WARPS];
    __shared__ float sv[WARPS];
    __shared__ int si[WARPS];
    const float * x = xs + (size_t) blockIdx.x * n;
    const float it = 1.f / p.temperature;
    float m = -INFINITY, lo = INFINITY;
    for (int j = threadIdx.x; j < n; j += THREADS) {
        const float y = x[j] * it;
        m = fmaxf(m, y), lo = fminf(lo, y);
    }
    m = block_max(m, sh);
    lo = -block_max(-lo, sh);
    float cut = lo;   // kept: y >= cut
    if (p.min_p > 0.f) cut = fmaxf(cut, m + logf(p.min_p));
    if (p.top_k > 0 && p.top_k < n) {   // largest c with count(y >= c) >= k
        float a = lo, b = m;
        for (int s = 0; s < BISECT; ++s) {
            const float c = 0.5f * (a + b);
            float k = 0.f;
            for (int j = threadIdx.x; j < n; j += THREADS) k += x[j] * it >= c ? 1.f : 0.f;
            (block_sum(k, sh) >= (float) p.top_k ? a : b) = c;
        }
        cut = fmaxf(cut, a);
    }
    if (p.top_p > 0.f && p.top_p < 1.f) {   // largest c whose kept mass (y >= max(c, cut)) reaches top_p of the total
        float z = 0.f;
        for (int j = threadIdx.x; j < n; j += THREADS) {
            const float y = x[j] * it;
            z += y >= cut ? __expf(y - m) : 0.f;
        }
        z = block_sum(z, sh);
        float a = cut, b = m;
        for (int s = 0; s < BISECT; ++s) {
            const float c = 0.5f * (a + b);
            float q = 0.f;
            for (int j = threadIdx.x; j < n; j += THREADS) {
                const float y = x[j] * it;
                q += y >= c ? __expf(y - m) : 0.f;
            }
            (block_sum(q, sh) >= p.top_p * z ? a : b) = c;
        }
        cut = a;
    }
    // Gumbel-max over the kept tokens: argmax y + g, g = -log(-log u)
    const uint64_t stream = mix(p.seed ^ mix(counter + blockIdx.x + 0x9e3779b97f4a7c15ull));
    float v = -INFINITY;
    int i = n;
    for (int j = threadIdx.x; j < n; j += THREADS) {
        const float y = x[j] * it;
        if (y < cut) continue;
        const uint64_t h = mix(stream + (uint64_t) j * 0x9e3779b97f4a7c15ull);
        const float u = ((float) (h >> 40) + 0.5f) * (1.f / 16777216.f);   // (0, 1)
        const float g = y - logf(-logf(u));
        if (g > v || (g == v && j < i)) v = g, i = j;
    }
    for (int o = 16; o; o /= 2) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, i, o);
        if (ov > v || (ov == v && oi < i)) v = ov, i = oi;
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0) sv[threadIdx.x / 32] = v, si[threadIdx.x / 32] = i;
    __syncthreads();
    if (threadIdx.x) return;
    for (int w = 1; w < WARPS; ++w)
        if (sv[w] > v || (sv[w] == v && si[w] < i)) v = sv[w], i = si[w];
    out[blockIdx.x] = i;
}

}  // namespace

void sample(const float * x, int n, int rows, const SampleParams & p, uint64_t counter, int * out, cudaStream_t stream)
{
    if (rows <= 0) return;
    sample_kernel<<<rows, THREADS, 0, stream>>>(x, n, p, counter, out);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::sampling
