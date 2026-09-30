#include "ple_prefill.cuh"

#include "core/cuda_check.h"

#include <stdexcept>

namespace truss::ple {
namespace {

constexpr int THREADS = 256, MAX_HIST = 16;

__device__ __forceinline__ float sigmoid(float v) { return 1.f / (1.f + __expf(-v)); }

unsigned grid(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

// fixed-order block sums of three values
__device__ float3 block_sum3(float3 v, float3 * part)
{
    for (int o = 16; o; o /= 2) {
        v.x += __shfl_xor_sync(0xffffffffu, v.x, o);
        v.y += __shfl_xor_sync(0xffffffffu, v.y, o);
        v.z += __shfl_xor_sync(0xffffffffu, v.z, o);
    }
    if (threadIdx.x % 32 == 0) part[threadIdx.x / 32] = v;
    __syncthreads();
    float3 s = make_float3(0.f, 0.f, 0.f);
    for (int w = 0; w < THREADS / 32; ++w) s.x += part[w].x, s.y += part[w].y, s.z += part[w].z;
    __syncthreads();
    return s;
}

// one block per (token, stream)
__global__ void __launch_bounds__(THREADS) gate_kernel(const float * key, const float * res, const float * gk,
                                                       const float * gq, int hc, int d, float eps, float * gate)
{
    __shared__ float3 part[THREADS / 32];
    const size_t r = blockIdx.x;
    const float * k = key + r * d, * x = res + r * d, * a = gk + (r % hc) * d, * b = gq + (r % hc) * d;
    float3 acc = make_float3(0.f, 0.f, 0.f);
    for (int i = threadIdx.x; i < d; i += THREADS) acc.x += k[i] * k[i], acc.y += x[i] * x[i], acc.z += k[i] * a[i] * x[i] * b[i];
    const float3 s3 = block_sum3(acc, part);
    if (threadIdx.x) return;
    const float s = s3.z * rsqrtf(s3.x / d + eps) * rsqrtf(s3.y / d + eps) / sqrtf((float) d);
    const float sg = s > 0.f ? 1.f : s < 0.f ? -1.f : 0.f;
    gate[r] = sigmoid(sg * sqrtf(fmaxf(fabsf(s), 1e-6f)));
}

// normed [T][hc][d] = RMSNorm(value * gate) * gamma, one block per (token, stream)
__global__ void __launch_bounds__(THREADS) norm_kernel(const float * value, const float * gate, const float * gamma,
                                                       int hc, int d, float eps, float * normed)
{
    __shared__ float3 part[THREADS / 32];
    const size_t r = blockIdx.x;
    const float g = gate[r], * v = value + r / hc * d, * gm = gamma + (r % hc) * d;
    float3 acc = make_float3(0.f, 0.f, 0.f);
    for (int i = threadIdx.x; i < d; i += THREADS) acc.x += (v[i] * g) * (v[i] * g);
    const float inv = rsqrtf(block_sum3(acc, part).x / d + eps);
    for (int i = threadIdx.x; i < d; i += THREADS) normed[r * d + i] = v[i] * g * inv * gm[i];
}

// res[t][c] += gated + silu(sum_k w[c][k] normed[t - (K - 1 - k) dil][c])
__global__ void conv_kernel(const float * value, const float * gate, const float * normed, const half * w,
                            const float * hist, int T, int hc, int d, int K, int dil, float * res)
{
    const int C = hc * d, HL = (K - 1) * dil;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * C) return;
    const int t = (int) (i / C), c = (int) (i % C);
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int src = t - (K - 1 - k) * dil;
        acc += __half2float(w[(size_t) c * K + k]) * (src >= 0 ? normed[(size_t) src * C + c] : hist[(size_t) (HL + src) * C + c]);
    }
    res[i] += value[(size_t) t * d + c % d] * gate[(size_t) t * hc + c / d] + acc * sigmoid(acc);
}

// hist <- the last HL rows of [hist; normed]
__global__ void hist_kernel(const float * normed, float * hist, int T, int C, int HL)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float row[MAX_HIST];
    for (int r = 0; r < HL; ++r) {
        const int src = T - HL + r;
        row[r] = src >= 0 ? normed[(size_t) src * C + c] : hist[(size_t) (HL + src) * C + c];
    }
    for (int r = 0; r < HL; ++r) hist[(size_t) r * C + c] = row[r];
}

}  // namespace

void gate(const float * key, const float * res, const float * norm_key, const float * norm_query, int T, int hc,
          int d, float eps, float * gate, cudaStream_t stream)
{
    gate_kernel<<<T * hc, THREADS, 0, stream>>>(key, res, norm_key, norm_query, hc, d, eps, gate);
    TRUSS_CUDA(cudaGetLastError());
}

void apply(const float * value, const float * gate, const float * norm_conv, const half * conv_w, float * hist, int T,
           int hc, int d, int K, int dil, float eps, float * normed, float * res, cudaStream_t stream)
{
    const int HL = (K - 1) * dil;
    if (HL > MAX_HIST) throw std::invalid_argument("ple::apply: conv history longer than supported");
    norm_kernel<<<T * hc, THREADS, 0, stream>>>(value, gate, norm_conv, hc, d, eps, normed);
    conv_kernel<<<grid((int64_t) T * hc * d), THREADS, 0, stream>>>(value, gate, normed, conv_w, hist, T, hc, d, K, dil,
                                                                    res);
    hist_kernel<<<grid(hc * d), THREADS, 0, stream>>>(normed, hist, T, hc * d, HL);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::ple
