#include "spec.cuh"

#include "core/cuda_check.h"

namespace truss::sampling {
namespace {

constexpr int THREADS = 1024, WARPS = THREADS / 32;
constexpr int BISECT = 40;   // as sample.cu

__device__ float block_sum(float v, float * sh)
{
    for (int o = 16; o; o /= 2) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v;
    __syncthreads();
    float t = 0.f;
    for (int w = 0; w < WARPS; ++w) t += sh[w];
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

__device__ __forceinline__ float unit(uint64_t h) { return ((float) (h >> 40) + 0.5f) * (1.f / 16777216.f); }   // (0, 1)

// salts: the draft, accept and residual draws never share a stream with sample()'s rows or each other
constexpr uint64_t SALT_DRAFT = 0xd1b54a32d192ed03ull, SALT_ACCEPT = 0xa24baed4963ee407ull,
                   SALT_RESID = 0x9fb21c651e98df25ull;

// sample.cu's cut, verbatim: y = x / T kept when y >= cut; m = max y
__device__ float cut_of(const float * x, int n, float it, const SampleParams & p, float * sh, float & m_out)
{
    float m = -INFINITY, lo = INFINITY;
    for (int j = threadIdx.x; j < n; j += THREADS) {
        const float y = x[j] * it;
        m = fmaxf(m, y), lo = fminf(lo, y);
    }
    m = block_max(m, sh);
    lo = -block_max(-lo, sh);
    float cut = lo;
    if (p.min_p > 0.f) cut = fmaxf(cut, m + logf(p.min_p));
    if (p.top_k > 0 && p.top_k < n) {
        float a = lo, b = m;
        for (int s = 0; s < BISECT; ++s) {
            const float c = 0.5f * (a + b);
            float k = 0.f;
            for (int j = threadIdx.x; j < n; j += THREADS) k += x[j] * it >= c ? 1.f : 0.f;
            (block_sum(k, sh) >= (float) p.top_k ? a : b) = c;
        }
        cut = fmaxf(cut, a);
    }
    if (p.top_p > 0.f && p.top_p < 1.f) {
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
    m_out = m;
    return cut;
}

// block-wide argmax of (v, i), ties to the lower index; thread 0 gets the result
__device__ int block_argmax(float v, int i, float * sv, int * si)
{
    for (int o = 16; o; o /= 2) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, i, o);
        if (ov > v || (ov == v && oi < i)) v = ov, i = oi;
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0) sv[threadIdx.x / 32] = v, si[threadIdx.x / 32] = i;
    __syncthreads();
    if (threadIdx.x == 0)
        for (int w = 1; w < WARPS; ++w)
            if (sv[w] > v || (sv[w] == v && si[w] < i)) v = sv[w], i = si[w];
    return i;
}

__global__ void __launch_bounds__(THREADS) probs_kernel(float * xs, int n, SampleParams p)
{
    __shared__ float sh[WARPS];
    float * x = xs + (size_t) blockIdx.x * n;
    const float it = 1.f / p.temperature;
    float m;
    const float cut = cut_of(x, n, it, p, sh, m);
    float z = 0.f;
    for (int j = threadIdx.x; j < n; j += THREADS) {
        const float y = x[j] * it;
        z += y >= cut ? __expf(y - m) : 0.f;
    }
    z = block_sum(z, sh);   // every thread has read its own x[j] above; the writes below touch only those
    const float iz = 1.f / z;
    for (int j = threadIdx.x; j < n; j += THREADS) {
        const float y = x[j] * it;
        x[j] = y >= cut ? __expf(y - m) * iz : 0.f;
    }
}

__global__ void __launch_bounds__(THREADS) draft_kernel(const float * x, int n, const int * map, SampleParams p,
                                                        uint64_t counter, float * q_full, int * out)
{
    __shared__ float sh[WARPS], sv[WARPS];
    __shared__ int si[WARPS];
    const float it = 1.f / p.temperature;
    float m;
    const float cut = cut_of(x, n, it, p, sh, m);
    float z = 0.f;
    for (int j = threadIdx.x; j < n; j += THREADS) {
        const float y = x[j] * it;
        z += y >= cut ? __expf(y - m) : 0.f;
    }
    z = block_sum(z, sh);
    const float iz = 1.f / z;
    const uint64_t stream = mix(p.seed ^ mix(counter + SALT_DRAFT));
    float v = -INFINITY;
    int i = n;
    for (int j = threadIdx.x; j < n; j += THREADS) {
        const float y = x[j] * it;
        if (y < cut) continue;
        const float q = __expf(y - m) * iz;
        q_full[map ? map[j] : j] = q;
        // Gumbel-max on log q (= y - m - log z): the same argmax as on y
        const float g = y - logf(-logf(unit(mix(stream + (uint64_t) j * 0x9e3779b97f4a7c15ull))));
        if (g > v || (g == v && j < i)) v = g, i = j;
    }
    i = block_argmax(v, i, sv, si);
    if (threadIdx.x == 0) out[0] = map ? map[i] : i;
}

// one block: rows in order; thread 0 decides, all threads draw the residual / bonus
__global__ void __launch_bounds__(THREADS) accept_kernel(const float * probs, const float * q_full, const int * drafts,
                                                         int nw, int V, uint64_t seed, uint64_t counter, int * out)
{
    __shared__ float sv[WARPS];
    __shared__ int si[WARPS];
    __shared__ int rej;
    int j = 0;
    for (; j < nw; ++j) {
        if (threadIdx.x == 0) {
            const int x = drafts[j];
            const float px = probs[(size_t) j * V + x], qx = q_full[(size_t) j * V + x];
            const float u = unit(mix(mix(seed ^ mix(counter + j + SALT_ACCEPT))));
            rej = !(u * qx < px);   // accept with probability min(1, px / qx); qx > 0: x was drawn from q
        }
        __syncthreads();
        if (rej) break;
        __syncthreads();
    }
    // j == nw: every draft accepted, the bonus is a sample of p_nw; else the residual max(0, p_j - q_j)
    const float * pr = probs + (size_t) j * V;
    const float * qr = j < nw ? q_full + (size_t) j * V : nullptr;
    const uint64_t stream = mix(seed ^ mix(counter + j + SALT_RESID));
    float v = -INFINITY;
    int i = V;
    for (int t = threadIdx.x; t < V; t += THREADS) {
        const float r = qr ? pr[t] - qr[t] : pr[t];
        if (!(r > 0.f)) continue;
        const float g = logf(r) - logf(-logf(unit(mix(stream + (uint64_t) t * 0x9e3779b97f4a7c15ull))));
        if (g > v || (g == v && t < i)) v = g, i = t;
    }
    i = block_argmax(v, i, sv, si);
    if (threadIdx.x == 0) {
        if (i >= V) {   // empty residual (p_j == q_j up to rounding): fall back to p_j itself
            float bv = -INFINITY;
            for (int t = 0; t < V; ++t)
                if (pr[t] > bv) bv = pr[t], i = t;
        }
        out[0] = j, out[1] = i;
    }
}

}  // namespace

void probs_inplace(float * x, int n, int rows, const SampleParams & p, cudaStream_t stream)
{
    if (rows <= 0) return;
    probs_kernel<<<rows, THREADS, 0, stream>>>(x, n, p);
    TRUSS_CUDA(cudaGetLastError());
}

void draft_sample(const float * x, int n, const int * map, int V, const SampleParams & p, uint64_t counter,
                  float * q_full, int * out, cudaStream_t stream)
{
    TRUSS_CUDA(cudaMemsetAsync(q_full, 0, sizeof(float) * V, stream));
    draft_kernel<<<1, THREADS, 0, stream>>>(x, n, map, p, counter, q_full, out);
    TRUSS_CUDA(cudaGetLastError());
}

void spec_accept(const float * probs, const float * q_full, const int * drafts, int nw, int V, uint64_t seed,
                 uint64_t counter, int * out, cudaStream_t stream)
{
    accept_kernel<<<1, THREADS, 0, stream>>>(probs, q_full, drafts, nw, V, seed, counter, out);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::sampling
