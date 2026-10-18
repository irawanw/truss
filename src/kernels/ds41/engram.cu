// DeepSeek-V4.1 Engram device ops (engram.cuh).
#include "kernels/ds41/engram.cuh"

#include "core/cuda_check.h"

#include <stdexcept>

namespace truss::ds41 {
namespace {

// one warp per row: lane l owns values 8l .. 8l + 7 (one group's quarter), 4 bytes of nibbles
__global__ void dequant_q4_kernel(const uint8_t * __restrict__ rows, int n, half * __restrict__ out)
{
    const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= n) return;
    const uint8_t * row = rows + (size_t) r * ENGRAM_Q4_ROW;
    const float s = __half2float(((const half *) row)[lane / 4]);
    const uint32_t q = *(const uint32_t *) (row + 16 + lane * 4);
    half2 * o = (half2 *) (out + (size_t) r * ENGRAM_DIM + lane * 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int b = (q >> (8 * i)) & 0xff;
        o[i] = __floats2half2_rn(((b & 15) - 8) * s, ((b >> 4) - 8) * s);
    }
}

constexpr int GT = 256;

__device__ float block_sum(float v, float * red)
{
    for (int o = 16; o; o /= 2) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = v;
    __syncthreads();
    float s = 0.f;
    for (int w = 0; w < GT / 32; ++w) s += red[w];
    return s;
}

__global__ void __launch_bounds__(GT) gate_kernel(float * res, int res_ld, const float * __restrict__ kv,
                                                  const float * __restrict__ qk, int hc, int d, float eps)
{
    __shared__ float red[GT / 32];
    const int t = blockIdx.x / hc, c = blockIdx.x % hc;
    float * h = res + (size_t) t * res_ld + (size_t) c * d;
    const float * key = kv + (size_t) t * (hc + 1) * d + (size_t) c * d;
    const float * val = kv + (size_t) t * (hc + 1) * d + (size_t) hc * d;
    const float * w = qk + (size_t) c * d;
    float hh = 0.f, kk = 0.f, hk = 0.f;
    for (int i = threadIdx.x; i < d; i += GT) {
        const float a = h[i], k = key[i];
        hh += a * a, kk += k * k, hk += a * w[i] * k;
    }
    hh = block_sum(hh, red);
    kk = block_sum(kk, red);
    hk = block_sum(hk, red);
    const float rstd = rsqrtf(hh / d + eps) * rsqrtf(kk / d + eps);
    const float dot = hk * rstd * rsqrtf((float) d);
    const float g = 1.f / (1.f + expf(-copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot)));
    for (int i = threadIdx.x; i < d; i += GT) h[i] += g * val[i];
}

__global__ void mul_kernel(const float * a, const float * b, int n, float * y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}

}  // namespace

void engram_dequant_q4(const uint8_t * rows, int n, half * out, cudaStream_t stream)
{
    if (n <= 0) return;
    dequant_q4_kernel<<<(n + 7) / 8, 256, 0, stream>>>(rows, n, out);
    TRUSS_CUDA(cudaGetLastError());
}

void engram_gate(float * res, int res_ld, const float * kv, const float * qk, int T, int hc, int d, float eps,
                 cudaStream_t stream)
{
    if (T <= 0) return;
    gate_kernel<<<(unsigned) T * hc, GT, 0, stream>>>(res, res_ld, kv, qk, hc, d, eps);
    TRUSS_CUDA(cudaGetLastError());
}

void mul(const float * a, const float * b, int n, float * y, cudaStream_t stream)
{
    if (n <= 0) return;
    mul_kernel<<<(n + 255) / 256, 256, 0, stream>>>(a, b, n, y);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::ds41
