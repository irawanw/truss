#include "kernels/mtp/mtp_ops.cuh"

#include "core/cuda_check.h"

namespace truss::mtp {
namespace {

// one block per token: the embedding row's rms, then every stream's [e_norm | hn] row
__global__ void join_kernel(const float * emb, const float * enorm, const half * hn, int hc, int d, float eps, half * out)
{
    __shared__ float red[32];
    const int t = blockIdx.x;
    const float * e = emb + (size_t) t * d;
    float ss = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) ss += e[i] * e[i];
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float r = rsqrtf(red[0] / d + eps);
    for (int i = threadIdx.x; i < hc * d; i += blockDim.x) {
        const int h = i / d, j = i % d;
        half * o = out + ((size_t) t * hc + h) * 2 * d;
        o[j] = __float2half(e[j] * r * enorm[j]);
        o[d + j] = hn[((size_t) t * hc + h) * d + j];
    }
}

}  // namespace

void join(const float * emb, const float * enorm, const half * hn, int T, int hc, int d, float eps, half * out,
          cudaStream_t stream)
{
    join_kernel<<<T, 256, 0, stream>>>(emb, enorm, hn, hc, d, eps, out);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::mtp
