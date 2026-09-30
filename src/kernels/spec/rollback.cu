#include "kernels/spec/rollback.cuh"

#include "core/cuda_check.h"

#include <cstdint>

namespace truss::spec {
namespace {

__global__ void tail_rows_kernel(const float * old, int H, const float * rows, int n, int width, float * out)
{
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) H * width) return;
    const int r = (int) (i / width), c = (int) (i % width);
    const int src = n + r;   // row r of the result is row n + r of concat(old, rows)
    out[i] = src < H ? old[(int64_t) src * width + c] : rows[(int64_t) (src - H) * width + c];
}

}  // namespace

void tail_rows(const float * old, int H, const float * rows, int n, int width, float * out, cudaStream_t stream)
{
    const int64_t total = (int64_t) H * width;
    tail_rows_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, stream>>>(old, H, rows, n, width, out);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::spec
