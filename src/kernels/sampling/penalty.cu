#include "penalty.cuh"

#include "core/cuda_check.h"

namespace truss::sampling {
namespace {

constexpr int THREADS = 256;

__global__ void count_kernel(int n, const int * hist, int h, int * cnt)
{
    const int * row = hist + (size_t) blockIdx.x * h;
    for (int i = threadIdx.x; i < h; i += THREADS) {
        const int t = row[i];
        if (t >= 0 && t < n) atomicAdd(cnt + (size_t) blockIdx.x * n + t, 1);
    }
}

// the one thread that takes a token's count back to zero applies that token's penalty, so each token once
__global__ void apply_kernel(float * x, int n, const int * hist, int h, PenaltyParams p, int * cnt)
{
    const int * row = hist + (size_t) blockIdx.x * h;
    for (int i = threadIdx.x; i < h; i += THREADS) {
        const int t = row[i];
        if (t < 0 || t >= n) continue;
        const size_t k = (size_t) blockIdx.x * n + t;
        const int c = atomicExch(cnt + k, 0);
        if (c <= 0) continue;
        float v = x[k];
        if (p.repeat != 1.f) v = v > 0.f ? v / p.repeat : v * p.repeat;
        x[k] = v - (float) c * p.frequency - p.presence;
    }
}

}  // namespace

void penalize(float * x, int n, int rows, const int * hist, int h, const PenaltyParams & p, int * cnt,
              cudaStream_t stream)
{
    if (rows <= 0 || h <= 0 || !penalties_on(p)) return;
    count_kernel<<<rows, THREADS, 0, stream>>>(n, hist, h, cnt);
    TRUSS_CUDA(cudaGetLastError());
    apply_kernel<<<rows, THREADS, 0, stream>>>(x, n, hist, h, p, cnt);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::sampling
