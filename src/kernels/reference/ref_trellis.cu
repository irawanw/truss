#include "core/cuda_check.h"
#include "kernels/reference/ref.cuh"
#include "formats/trellis_k.h"

#include <cuda_fp16.h>

namespace truss::ref {

namespace {

// y[i] = 1/sqrt(128) * sum_j (-1)^popcount(i & j) x[j], one block of 128 threads per 128 values
__global__ void hadamard128_kernel(float * x)
{
    __shared__ float v[128];
    float * b = x + (size_t) blockIdx.x * 128;
    v[threadIdx.x] = b[threadIdx.x];
    __syncthreads();
    float acc = 0.f;
    for (int j = 0; j < 128; ++j) acc += (__popc(threadIdx.x & j) & 1) ? -v[j] : v[j];
    b[threadIdx.x] = acc * 0.088388347648f;
}

__device__ float mul1(uint32_t state)
{
    const uint32_t x = state * 0x83DCD12Du;
    const uint32_t sum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24) + 0x6400u;
    const half h = __ushort_as_half((unsigned short) sum);
    return __half2float(__hfma(h, __ushort_as_half(0x1eee), __ushort_as_half(0xc931)));
}

__global__ void trellis_dequant_kernel(const uint32_t * w32, int K, int in, int out, float * W)
{
    const int64_t idx = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (idx >= (int64_t) in * out) return;
    const int o = (int) (idx / in), i = (int) (idx % in);
    const uint32_t * tile = w32 + ((size_t) (i / 16) * (out / 16) + o / 16) * (formats::k_tile_u16(K) / 2);
    // (n, k) inside the tile -> lane and value index of the fragment decode
    const int n = o % 16, k = i % 16;
    const int lane = (n % 8) * 4 + (k % 8) / 2;
    const int j = lane * 8 + (k & 1) + (k >= 8 ? 2 : 0) + (n >= 8 ? 4 : 0);
    const int bits = formats::k_tile_bits(K), end = formats::k_window_end(K, j);
    uint32_t state = 0;
    for (int b = 0; b < 16; ++b) {
        const int p = ((end - 16 + b) % bits + bits) % bits;
        state = (state << 1) | ((tile[p / 32] >> (31 - p % 32)) & 1u);
    }
    W[idx] = mul1(state);
}

}  // namespace

void hadamard128(float * x, int64_t n, cudaStream_t s)
{
    hadamard128_kernel<<<(unsigned) (n / 128), 128, 0, s>>>(x);
    TRUSS_CUDA(cudaGetLastError());
}

void trellis_dequant(const uint16_t * words, int K, int in, int out, float * W, cudaStream_t s)
{
    const int64_t n = (int64_t) in * out;
    trellis_dequant_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>((const uint32_t *) words, K, in, out, W);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::ref
