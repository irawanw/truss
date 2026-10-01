#include "runtime/doorbell.cuh"

#include "core/cuda_check.h"

namespace truss::runtime {
namespace {

__global__ void publish_kernel(const int * ids, const int * pred, const float * wts, int n_ids, const float * x, int n_x,
                               int * dst_ids, int * dst_pred, float * dst_wts, float * dst_x, volatile int * doorbell,
                               int seq)
{
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        dst_ids[i] = ids[i];
        if (pred) dst_pred[i] = pred[i];
        dst_wts[i] = wts[i];
    }
    for (int i = threadIdx.x; i < n_x / 4; i += blockDim.x)
        reinterpret_cast<float4 *>(dst_x)[i] = reinterpret_cast<const float4 *>(x)[i];
    __threadfence_system();   // the data reaches host memory before the doorbell does
    __syncthreads();
    if (threadIdx.x == 0) *doorbell = seq;
}

__global__ void spin_kernel(const volatile int * flag, int seq)
{
    while (*flag < seq) __nanosleep(256);
    __threadfence();
}

__global__ void wait_plan_kernel(const volatile int * plan, const volatile int * need_copy, const volatile int * go,
                                 int seq, const volatile int * mapped_ids, int * dst, int n)
{
    __shared__ int need;
    if (threadIdx.x == 0) {
        while (*plan < seq) __nanosleep(128);
        __threadfence_system();
        need = *need_copy == seq;
        if (need)
            while (*go < seq) __nanosleep(128);
        __threadfence();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = mapped_ids[i];
}

__global__ void add_mapped_kernel(float * y, const float * x, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

}  // namespace

void publish(const int * ids, const int * pred, const float * wts, int n_ids, const float * x, int n_x, int * dst_ids,
             int * dst_pred, float * dst_wts, float * dst_x, volatile int * doorbell, int seq, cudaStream_t stream)
{
    publish_kernel<<<1, 1024, 0, stream>>>(ids, pred, wts, n_ids, x, n_x, dst_ids, dst_pred, dst_wts, dst_x, doorbell,
                                           seq);
    TRUSS_CUDA(cudaGetLastError());
}

void spin_until(const volatile int * flag, int seq, cudaStream_t stream)
{
    spin_kernel<<<1, 1, 0, stream>>>(flag, seq);
    TRUSS_CUDA(cudaGetLastError());
}

void wait_plan(const volatile int * plan, const volatile int * need_copy, const volatile int * go, int seq,
               const int * mapped_ids, int * dst, int n, cudaStream_t stream)
{
    wait_plan_kernel<<<1, 128, 0, stream>>>(plan, need_copy, go, seq, mapped_ids, dst, n);
    TRUSS_CUDA(cudaGetLastError());
}

void add_mapped(float * y, const float * x, int n, cudaStream_t stream)
{
    add_mapped_kernel<<<(n + 255) / 256, 256, 0, stream>>>(y, x, n);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::runtime
