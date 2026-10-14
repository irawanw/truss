#include "kernels/norm/rms_norm.cuh"

#include "core/cuda_check.h"

#include <stdexcept>

namespace truss::norm {

namespace {

constexpr int THREADS = 256;

// a != nullptr: x += a first (the residual add, written back; each thread re-reads only what it wrote)
__global__ void __launch_bounds__(THREADS) rms_kernel(float * __restrict__ x, const float * __restrict__ a,
                                                      const float * __restrict__ w, int d, float eps,
                                                      float * __restrict__ y32, half * __restrict__ y16)
{
    const size_t row = blockIdx.x;
    float4 * xr = reinterpret_cast<float4 *>(x + row * d);
    float s = 0.f;
    for (int i = threadIdx.x; i < d / 4; i += THREADS) {
        float4 v = xr[i];
        if (a) {
            const float4 b = reinterpret_cast<const float4 *>(a + row * d)[i];
            v.x += b.x, v.y += b.y, v.z += b.z, v.w += b.w;
            xr[i] = v;
        }
        s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    __shared__ float part[THREADS / 32];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x / 32] = s;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < THREADS / 32; ++i) tot += part[i];   // fixed order: every thread gets the same sum
    const float r = rsqrtf(tot / d + eps);
    for (int i = threadIdx.x; i < d / 4; i += THREADS) {
        const float4 v = xr[i], g = reinterpret_cast<const float4 *>(w)[i];
        const float4 o = { v.x * r * g.x, v.y * r * g.y, v.z * r * g.z, v.w * r * g.w };
        if (y32) reinterpret_cast<float4 *>(y32 + row * d)[i] = o;
        if (y16) {
            half2 * h = reinterpret_cast<half2 *>(y16 + row * d) + 2 * i;
            h[0] = __floats2half2_rn(o.x, o.y), h[1] = __floats2half2_rn(o.z, o.w);
        }
    }
}

// one warp per (row, head); D / 32 values per lane
template <int PER>
__global__ void rms_heads_kernel(float * __restrict__ x, const float * __restrict__ w, int n, int D, float eps)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (warp >= n) return;
    float * v = x + (size_t) warp * D;
    float a[PER], s = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) a[i] = v[lane + 32 * i], s += a[i] * a[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float r = rsqrtf(s / D + eps);
#pragma unroll
    for (int i = 0; i < PER; ++i) v[lane + 32 * i] = a[i] * r * w[lane + 32 * i];
}

}  // namespace

void rms(const float * x, const float * w, int T, int d, float eps, float * y32, half * y16, cudaStream_t stream)
{
    if (d % 4) throw std::invalid_argument("norm::rms: d must be a multiple of 4");
    if (T <= 0) return;
    rms_kernel<<<T, THREADS, 0, stream>>>(const_cast<float *>(x), nullptr, w, d, eps, y32, y16);
    TRUSS_CUDA(cudaGetLastError());
}

void add_rms(float * x, const float * a, const float * w, int T, int d, float eps, float * y32, half * y16,
             cudaStream_t stream)
{
    if (d % 4) throw std::invalid_argument("norm::add_rms: d must be a multiple of 4");
    if (T <= 0) return;
    rms_kernel<<<T, THREADS, 0, stream>>>(x, a, w, d, eps, y32, y16);
    TRUSS_CUDA(cudaGetLastError());
}

void rms_heads(float * x, const float * w, int T, int H, int D, float eps, cudaStream_t stream)
{
    if (D % 32 || D > 256) throw std::invalid_argument("norm::rms_heads: D must be a multiple of 32, <= 256");
    const int n = T * H, blocks = (n * 32 + 255) / 256;
    if (n <= 0) return;
    switch (D / 32) {
    case 1: rms_heads_kernel<1><<<blocks, 256, 0, stream>>>(x, w, n, D, eps); break;
    case 2: rms_heads_kernel<2><<<blocks, 256, 0, stream>>>(x, w, n, D, eps); break;
    case 4: rms_heads_kernel<4><<<blocks, 256, 0, stream>>>(x, w, n, D, eps); break;
    case 8: rms_heads_kernel<8><<<blocks, 256, 0, stream>>>(x, w, n, D, eps); break;
    default: throw std::invalid_argument("norm::rms_heads: D / 32 must be 1, 2, 4 or 8");
    }
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::norm
