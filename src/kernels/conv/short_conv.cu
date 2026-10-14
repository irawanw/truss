#include "kernels/conv/short_conv.cuh"

#include "core/cuda_check.h"

#include <stdexcept>

namespace truss::conv {

namespace {

constexpr int MAX_L = 8;

__device__ __forceinline__ float bx_at(const float * bcx, const float * state, int t, int c, int d, int L)
{
    if (t < 0) return state[(size_t) (L - 1 + t) * d + c];   // t = -(L-1) .. -1: the carried rows
    return bcx[(size_t) t * 3 * d + c] * bcx[(size_t) t * 3 * d + 2 * d + c];
}

// one thread per (row, channel): every output reads its L inputs (rows before the chunk from state)
__global__ void conv_kernel(const float * __restrict__ bcx, const float * __restrict__ w, const float * __restrict__ state,
                            int T, int d, int L, half * __restrict__ y16, float * __restrict__ bx_out)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * d) return;
    const int t = (int) (i / d), c = (int) (i % d);
    float acc = 0.f;
    for (int k = 0; k < L; ++k) acc += w[(size_t) c * L + k] * bx_at(bcx, state, t - (L - 1) + k, c, d, L);
    const float C = bcx[(size_t) t * 3 * d + d + c];
    y16[i] = __float2half(C * acc);
    if (bx_out) bx_out[i] = bx_at(bcx, state, t, c, d, L);
}

// state <- the last L - 1 bx rows of [state | chunk]; separate launch: conv_kernel reads the old state
__global__ void advance_kernel(const float * __restrict__ bcx, float * __restrict__ state, int T, int d, int L,
                               float * __restrict__ tmp)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d) return;
    float v[MAX_L];
    for (int j = 0; j < L - 1; ++j) v[j] = bx_at(bcx, state, T - (L - 1) + j, c, d, L);
    for (int j = 0; j < L - 1; ++j) state[(size_t) j * d + c] = v[j];
    (void) tmp;
}

}  // namespace

void short_conv(const float * bcx, const float * w, float * state, int T, int d, int L, half * y16, float * bx_out,
                cudaStream_t stream)
{
    if (L < 2 || L > MAX_L) throw std::invalid_argument("conv::short_conv: L must be 2..8");
    if (T <= 0) return;
    const size_t n = (size_t) T * d;
    conv_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(bcx, w, state, T, d, L, y16, bx_out);
    TRUSS_CUDA(cudaGetLastError());
    advance_kernel<<<(d + 255) / 256, 256, 0, stream>>>(bcx, state, T, d, L, nullptr);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::conv
