#include "hc_prefill.cuh"

#include "core/cuda_check.h"

namespace truss::hc {
namespace {

constexpr int THREADS = 256;

__device__ __forceinline__ float sigmoid(float v) { return 1.f / (1.f + __expf(-v)); }

// fixed-order block sum
__device__ float block_sum(float v, float * part)
{
    for (int o = 16; o; o /= 2) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x % 32 == 0) part[threadIdx.x / 32] = v;
    __syncthreads();
    float s = 0.f;
    for (int w = 0; w < THREADS / 32; ++w) s += part[w];
    __syncthreads();
    return s;
}

// one block per (token, stream)
__global__ void __launch_bounds__(THREADS) norm_kernel(const float * res, const float * gamma, int hc, int d, float eps,
                                                       half * xn16, float * rstd)
{
    __shared__ float part[THREADS / 32];
    const size_t r = blockIdx.x;
    const float * x = res + r * d;
    float acc = 0.f;
    for (int i = threadIdx.x; i < d; i += THREADS) acc += x[i] * x[i];
    const float inv = rsqrtf(block_sum(acc, part) / d + eps);
    const float * g = gamma + (size_t) (r % hc) * d;
    for (int i = threadIdx.x; i < d; i += THREADS) xn16[r * d + i] = __float2half(x[i] * inv * g[i]);
    if (threadIdx.x == 0) rstd[r] = inv;
}

__global__ void silu_kernel(const float * lo, int n, float inv_hc, half * lo16)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = lo[i] * inv_hc;
    lo16[i] = __float2half(a * sigmoid(a));
}

__global__ void collapse_kernel(const float * res, const float * rstd, const float * gamma, const float * gate, int T,
                                int hc, int d, float * mixed, half * mixed16)
{
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * d) return;
    const int64_t t = i / d, e = i % d;
    float acc = 0.f;
    for (int c = 0; c < hc; ++c) {
        const int64_t j = (t * hc + c) * d + e;
        acc += res[j] * rstd[t * hc + c] * gamma[(int64_t) c * d + e] * sigmoid(gate[j]);
    }
    acc *= 1.f / hc;
    mixed[i] = acc;
    if (mixed16) mixed16[i] = __float2half(acc);
}

__global__ void combine_kernel(float * res, const float * out, const float * inject, int T, int hc, int d)
{
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * hc * d) return;
    const int64_t t = i / ((int64_t) hc * d), c = i / d % hc, e = i % d;
    res[i] += out[t * d + e] * 2.f * sigmoid(inject[t * hc + c] / hc);
}

__global__ void expand_kernel(const float * emb, int T, int hc, int d, float * res)
{
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t) T * hc * d) res[i] = emb[i / ((int64_t) hc * d) * d + i % d];
}

unsigned grid(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

}  // namespace

void expand(const float * emb, int T, int hc, int d, float * res, cudaStream_t stream)
{
    expand_kernel<<<grid((int64_t) T * hc * d), THREADS, 0, stream>>>(emb, T, hc, d, res);
    TRUSS_CUDA(cudaGetLastError());
}

void norm(const float * res, const float * gamma, int T, int hc, int d, float eps, half * xn16, float * rstd,
          cudaStream_t stream)
{
    norm_kernel<<<T * hc, THREADS, 0, stream>>>(res, gamma, hc, d, eps, xn16, rstd);
    TRUSS_CUDA(cudaGetLastError());
}

void silu(const float * lo, int n, float inv_hc, half * lo16, cudaStream_t stream)
{
    silu_kernel<<<grid(n), THREADS, 0, stream>>>(lo, n, inv_hc, lo16);
    TRUSS_CUDA(cudaGetLastError());
}

void collapse(const float * res, const float * rstd, const float * gamma, const float * gate, int T, int hc, int d,
              float * mixed, half * mixed16, cudaStream_t stream)
{
    collapse_kernel<<<grid((int64_t) T * d), THREADS, 0, stream>>>(res, rstd, gamma, gate, T, hc, d, mixed, mixed16);
    TRUSS_CUDA(cudaGetLastError());
}

void combine(float * res, const float * out, const float * inject, int T, int hc, int d, cudaStream_t stream)
{
    combine_kernel<<<grid((int64_t) T * hc * d), THREADS, 0, stream>>>(res, out, inject, T, hc, d);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::hc
