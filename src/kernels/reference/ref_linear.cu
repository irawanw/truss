#include "core/cuda_check.h"
#include "kernels/reference/ref.cuh"

#include <cuda_fp16.h>

#include <stdexcept>

namespace truss::ref {

namespace {

struct BlockQ8_0 {                         // ggml block_q8_0: 32 weights
    half d;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34);

template <gguf::Type T> __device__ float weight(const void * row, int i);
template <> __device__ float weight<gguf::Type::F32>(const void * row, int i) { return ((const float *) row)[i]; }
template <> __device__ float weight<gguf::Type::F16>(const void * row, int i)
{
    return __half2float(((const half *) row)[i]);
}
template <> __device__ float weight<gguf::Type::Q8_0>(const void * row, int i)
{
    const BlockQ8_0 & b = ((const BlockQ8_0 *) row)[i / 32];
    return __half2float(b.d) * (float) b.qs[i % 32];
}

template <gguf::Type T> __host__ __device__ constexpr size_t row_bytes(int64_t n)
{
    if constexpr (T == gguf::Type::F32) return n * 4;
    if constexpr (T == gguf::Type::F16) return n * 2;
    return n / 32 * sizeof(BlockQ8_0);
}

// one warp per (row, output); lanes stride the input, then a fixed xor-tree reduction
template <gguf::Type T>
__global__ void linear_kernel(const void * W, const float * x, float * y, int rows, int in, int out)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (warp >= rows * out) return;
    const int r = warp / out, o = warp % out;
    const void * wrow = (const char *) W + (size_t) o * row_bytes<T>(in);
    const float * xr = x + (size_t) r * in;
    float acc = 0.f;
    for (int i = lane; i < in; i += 32) acc += weight<T>(wrow, i) * xr[i];
    for (int m = 16; m; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) r * out + o] = acc;
}

// Q8_0 weights x Q8_1 activations, block by block as ggml's quantize_q8_1 + vec_dot_q8_0_q8_1: q uses the fp32
// scale, the dot uses the fp16-rounded one
__global__ void linear_q8_0_q8_1_kernel(const void * W, const float * x, float * y, int rows, int in, int out)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (warp >= rows * out) return;
    const int r = warp / out, o = warp % out;
    const BlockQ8_0 * wrow = (const BlockQ8_0 *) ((const char *) W + (size_t) o * row_bytes<gguf::Type::Q8_0>(in));
    const float * xr = x + (size_t) r * in;
    float acc = 0.f;
    for (int b = lane; b < in / 32; b += 32) {
        const float * xb = xr + b * 32;
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(xb[i]));
        const float d = amax / 127.f;
        int sumi = 0;
        for (int i = 0; i < 32; ++i) sumi += (int) wrow[b].qs[i] * (amax == 0.f ? 0 : (int) roundf(xb[i] / d));
        acc += __half2float(wrow[b].d) * __half2float(__float2half(d)) * (float) sumi;
    }
    for (int m = 16; m; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) r * out + o] = acc;
}

}  // namespace

void linear(const DTensor & W, const float * x, float * y, int rows, cudaStream_t s, ActQuant aq)
{
    const int in = (int) W.ne[0], out = (int) W.ne[1];
    if (W.ne[2] != 1 || W.ne[3] != 1) throw std::runtime_error("ref::linear: weight is not 2-D");
    const int threads = 256, blocks = (int) (((int64_t) rows * out * 32 + threads - 1) / threads);
    switch (W.type) {
        case gguf::Type::F32: linear_kernel<gguf::Type::F32><<<blocks, threads, 0, s>>>(W.data, x, y, rows, in, out); break;
        case gguf::Type::F16: linear_kernel<gguf::Type::F16><<<blocks, threads, 0, s>>>(W.data, x, y, rows, in, out); break;
        case gguf::Type::Q8_0:
            if (in % 32) throw std::runtime_error("ref::linear: Q8_0 row not a multiple of 32");
            if (aq == ActQuant::Q8_1)
                linear_q8_0_q8_1_kernel<<<blocks, threads, 0, s>>>(W.data, x, y, rows, in, out);
            else
                linear_kernel<gguf::Type::Q8_0><<<blocks, threads, 0, s>>>(W.data, x, y, rows, in, out);
            break;
        default: throw std::runtime_error(std::string("ref::linear: unsupported weight type ") +
                                          gguf::type_info(W.type)->name);
    }
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::ref
