#include "argmax.cuh"

#include "core/cuda_check.h"

namespace truss::sampling {
namespace {

constexpr int THREADS = 1024;

__device__ __forceinline__ void better(float & v, int & i, float ov, int oi)
{
    if (ov > v || (ov == v && oi < i)) v = ov, i = oi;
}

__global__ void __launch_bounds__(THREADS) argmax_kernel(const float * x, int n, int * out)
{
    __shared__ float sv[THREADS / 32];
    __shared__ int si[THREADS / 32];
    float v = -INFINITY;
    int i = n;
    for (int j = threadIdx.x; j < n; j += THREADS) better(v, i, x[j], j);
    for (int o = 16; o; o /= 2) better(v, i, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, i, o));
    if (threadIdx.x % 32 == 0) sv[threadIdx.x / 32] = v, si[threadIdx.x / 32] = i;
    __syncthreads();
    if (threadIdx.x) return;
    for (int w = 1; w < THREADS / 32; ++w) better(v, i, sv[w], si[w]);
    out[0] = i;
}

// out = map ? map[argmax] : argmax; prob = softmax probability of the maximum (two passes over x in one block)
__global__ void __launch_bounds__(THREADS) argmax_prob_kernel(const float * x, int n, const int * map, int * out,
                                                              float * prob)
{
    __shared__ float sv[THREADS / 32];
    __shared__ int si[THREADS / 32];
    __shared__ float s_max, s_sum[THREADS / 32];
    float v = -INFINITY;
    int i = n;
    for (int j = threadIdx.x; j < n; j += THREADS) better(v, i, x[j], j);
    for (int o = 16; o; o /= 2) better(v, i, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, i, o));
    if (threadIdx.x % 32 == 0) sv[threadIdx.x / 32] = v, si[threadIdx.x / 32] = i;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < THREADS / 32; ++w) better(v, i, sv[w], si[w]);
        s_max = v;
        out[0] = map ? map[i] : i;
    }
    __syncthreads();
    float z = 0.f;
    for (int j = threadIdx.x; j < n; j += THREADS) z += __expf(x[j] - s_max);
    for (int o = 16; o; o /= 2) z += __shfl_xor_sync(0xffffffffu, z, o);
    if (threadIdx.x % 32 == 0) s_sum[threadIdx.x / 32] = z;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int w = 0; w < THREADS / 32; ++w) t += s_sum[w];
        prob[0] = 1.f / t;
    }
}

}  // namespace

void argmax_prob(const float * x, int n, const int * map, int * out, float * prob, cudaStream_t stream)
{
    argmax_prob_kernel<<<1, THREADS, 0, stream>>>(x, n, map, out, prob);
    TRUSS_CUDA(cudaGetLastError());
}

void argmax(const float * x, int n, int * out, cudaStream_t stream)
{
    argmax_kernel<<<1, THREADS, 0, stream>>>(x, n, out);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::sampling
