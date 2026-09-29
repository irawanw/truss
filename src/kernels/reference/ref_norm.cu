#include "core/cuda_check.h"
#include "kernels/reference/ref.cuh"

namespace truss::ref {

namespace {

// one block per row; per-thread partial sums reduced in shared memory in a fixed tree
__global__ void rms_norm_kernel(const float * x, const float * gamma, int gamma_rows, float * y, int n, float eps)
{
    __shared__ float part[256];
    const int r = blockIdx.x;
    const float * xr = x + (size_t) r * n;
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += xr[i] * xr[i];
    part[threadIdx.x] = acc;
    __syncthreads();
    for (int m = blockDim.x / 2; m; m >>= 1) {
        if (threadIdx.x < m) part[threadIdx.x] += part[threadIdx.x + m];
        __syncthreads();
    }
    const float inv = rsqrtf(part[0] / n + eps);
    const float * g = gamma ? gamma + (size_t) (r % gamma_rows) * n : nullptr;
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[(size_t) r * n + i] = xr[i] * inv * (g ? g[i] : 1.f);
}

}  // namespace

void rms_norm(const float * x, const float * gamma, int gamma_rows, float * y, int rows, int n, float eps,
              cudaStream_t s)
{
    rms_norm_kernel<<<rows, 256, 0, s>>>(x, gamma, gamma_rows, y, n, eps);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::ref
