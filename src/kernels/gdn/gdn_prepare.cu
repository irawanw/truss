#include "gdn_prepare.cuh"

#include "core/cuda_check.h"

#include <stdexcept>

namespace truss::gdn {
namespace {

constexpr int THREADS = 256;

__device__ __forceinline__ float sigmoid(float v) { return 1.f / (1.f + __expf(-v)); }

unsigned grid(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

// conv + silu per (token, channel), scattered to q, k, v
__global__ void conv_kernel(Dims d, const float * qkv, const float * w, const float * state, int T, float * q,
                            float * k, float * v)
{
    const int C = d.channels(), kd = d.Hk * d.S;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * C) return;
    const int t = (int) (i / C), c = (int) (i % C);
    float acc = 0.f;
    for (int j = 0; j < d.K; ++j) {
        const int src = t - (d.K - 1) + j;
        acc += w[(size_t) c * d.K + j] * (src >= 0 ? qkv[(size_t) src * C + c] : state[(size_t) (d.K - 1 + src) * C + c]);
    }
    const float y = acc * sigmoid(acc);
    if (c < kd) q[(size_t) t * kd + c] = y;
    else if (c < 2 * kd) k[(size_t) t * kd + c - kd] = y;
    else v[(size_t) t * (C - 2 * kd) + c - 2 * kd] = y;
}

// state <- the last K - 1 rows of [state; qkv]
__global__ void state_kernel(Dims d, const float * qkv, float * state, int T)
{
    const int C = d.channels(), R = d.K - 1;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C) return;
    float row[8];
    for (int r = 0; r < R; ++r) {
        const int src = T - R + r;   // index into [state; qkv] minus R
        row[r] = src >= 0 ? qkv[(size_t) src * C + i] : state[(size_t) (R + src) * C + i];
    }
    for (int r = 0; r < R; ++r) state[(size_t) r * C + i] = row[r];
}

// l2 norm of q and k rows (one warp per (token, key head, q|k)), as llama: rms_norm(eps / n) / sqrt(n)
__global__ void l2_kernel(float * q, float * k, int rows, int S, float eps)
{
    const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= 2 * rows) return;
    float * x = (r < rows ? q : k) + (size_t) (r % rows) * S;
    float acc = 0.f;
    for (int i = lane; i < S; i += 32) acc += x[i] * x[i];
    for (int o = 16; o; o /= 2) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    const float inv = rsqrtf(acc / S + eps / S) * rsqrtf((float) S);
    for (int i = lane; i < S; i += 32) x[i] *= inv;
}

__global__ void gates_kernel(const float * alpha, const float * beta_raw, const float * dt_bias, const float * a,
                             int n, int H, float * g, float * beta)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = i % H;
    const float x = alpha[i] + dt_bias[h];
    g[i] = (x > 20.f ? x : log1pf(expf(x))) * a[h];
    beta[i] = sigmoid(beta_raw[i]);
}

// one warp per (token, value head)
__global__ void output_norm_kernel(const float * core, const float * z, const float * gamma, int rows, int S,
                                   float eps, half * out16)
{
    const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= rows) return;
    const float * x = core + (size_t) r * S;
    float acc = 0.f;
    for (int i = lane; i < S; i += 32) acc += x[i] * x[i];
    for (int o = 16; o; o /= 2) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    const float inv = rsqrtf(acc / S + eps);
    for (int i = lane; i < S; i += 32)
        out16[(size_t) r * S + i] = __float2half(x[i] * inv * gamma[i] * sigmoid(z[(size_t) r * S + i]));
}

}  // namespace

void prepare(const Dims & d, const float * qkv, const float * conv_w, float * conv_state, const float * alpha,
             const float * beta_raw, const float * dt_bias, const float * a, int T, float eps, float * q, float * k,
             float * v, float * g, float * beta, cudaStream_t stream)
{
    if (d.K < 2 || d.K > 9) throw std::invalid_argument("gdn::prepare: conv kernel size out of range");
    conv_kernel<<<grid((int64_t) T * d.channels()), THREADS, 0, stream>>>(d, qkv, conv_w, conv_state, T, q, k, v);
    state_kernel<<<grid(d.channels()), THREADS, 0, stream>>>(d, qkv, conv_state, T);
    l2_kernel<<<grid((int64_t) 2 * T * d.Hk * 32), THREADS, 0, stream>>>(q, k, T * d.Hk, d.S, eps);
    gates_kernel<<<grid((int64_t) T * d.Hv), THREADS, 0, stream>>>(alpha, beta_raw, dt_bias, a, T * d.Hv, d.Hv, g, beta);
    TRUSS_CUDA(cudaGetLastError());
}

void output_norm(const Dims & d, const float * core, const float * z, const float * gamma, int T, float eps,
                 half * out16, cudaStream_t stream)
{
    output_norm_kernel<<<grid((int64_t) T * d.Hv * 32), THREADS, 0, stream>>>(core, z, gamma, T * d.Hv, d.S, eps, out16);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::gdn
