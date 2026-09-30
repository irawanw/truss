#include "core/cuda_check.h"
#include "kernels/reference/ref.cuh"

#include <stdexcept>

namespace truss::ref {

namespace {

__global__ void conv_silu_kernel(const float * x, const float * w, const float * state, float * y, int T, int C, int kc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    const int t = i / C, c = i % C;
    float acc = 0.f;
    for (int j = 0; j < kc; ++j) {
        const int src = t - (kc - 1) + j;   // < 0: from the state rows
        const float v = src >= 0 ? x[(size_t) src * C + c] : (state ? state[(size_t) (kc - 1 + src) * C + c] : 0.f);
        acc += w[(size_t) c * kc + j] * v;
    }
    y[i] = acc / (1.f + expf(-acc));
}

// one block per row; fixed-tree reduction
__global__ void l2_norm_kernel(const float * x, float * y, int n, float eps)
{
    __shared__ float part[128];
    const float * xr = x + (size_t) blockIdx.x * n;
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += xr[i] * xr[i];
    part[threadIdx.x] = acc;
    __syncthreads();
    for (int m = blockDim.x / 2; m; m >>= 1) {
        if (threadIdx.x < m) part[threadIdx.x] += part[threadIdx.x + m];
        __syncthreads();
    }
    // as llama: rms_norm with eps / n, then * 1 / sqrt(n)
    const float inv = rsqrtf(part[0] / n + eps / n) * rsqrtf((float) n);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[(size_t) blockIdx.x * n + i] = xr[i] * inv;
}

// one block per value head, one thread per state column (value index); the column lives in local memory
template <int S>
__global__ void delta_rule_kernel(const float * q, const float * k, const float * v, const float * g,
                                  const float * beta, float * state, float * out, int T, int Hk, int Hv)
{
    __shared__ float sk[S], sq[S];
    const int h = blockIdx.x, hk = h % Hk, col = threadIdx.x;
    float col_s[S];
    for (int i = 0; i < S; ++i) col_s[i] = state ? state[((size_t) h * S + i) * S + col] : 0.f;
    const float scale = rsqrtf((float) S);
    for (int t = 0; t < T; ++t) {
        __syncthreads();
        sk[col] = k[((size_t) t * Hk + hk) * S + col];
        sq[col] = q[((size_t) t * Hk + hk) * S + col];
        __syncthreads();
        const float decay = expf(g[(size_t) t * Hv + h]), b = beta[(size_t) t * Hv + h];
        float kv = 0.f;
        for (int i = 0; i < S; ++i) {
            col_s[i] *= decay;
            kv += col_s[i] * sk[i];
        }
        const float delta = b * (v[((size_t) t * Hv + h) * S + col] - kv);
        float o = 0.f;
        for (int i = 0; i < S; ++i) {
            col_s[i] += sk[i] * delta;
            o += col_s[i] * sq[i];
        }
        out[((size_t) t * Hv + h) * S + col] = o * scale;
    }
    if (state)
        for (int i = 0; i < S; ++i) state[((size_t) h * S + i) * S + col] = col_s[i];
}

}  // namespace

void causal_conv_silu(const float * x, const float * w, const float * state, float * y, int T, int C, int kc,
                      cudaStream_t s)
{
    conv_silu_kernel<<<(T * C + 255) / 256, 256, 0, s>>>(x, w, state, y, T, C, kc);
    TRUSS_CUDA(cudaGetLastError());
}

void l2_norm(const float * x, float * y, int rows, int n, float eps, cudaStream_t s)
{
    l2_norm_kernel<<<rows, 128, 0, s>>>(x, y, n, eps);
    TRUSS_CUDA(cudaGetLastError());
}

void gated_delta_rule(const float * q, const float * k, const float * v, const float * g, const float * beta,
                      float * state, float * out, int T, int Hk, int Hv, int S, cudaStream_t s)
{
    if (S != 128) throw std::runtime_error("ref::gated_delta_rule: head size 128 only");
    delta_rule_kernel<128><<<Hv, 128, 0, s>>>(q, k, v, g, beta, state, out, T, Hk, Hv);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::ref
