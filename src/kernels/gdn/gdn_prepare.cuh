// GDN mixer ops around the delta rule, for prefill (the math of qwen4exp::reference::gdn; the projections are
// dense GEMMs run by the caller):
//   prepare      qkv (raw projection) -> causal depthwise conv + silu, carried across chunks by conv_state ->
//                q, k l2-normed per key head, v; alpha, beta (raw) -> g = softplus(alpha + dt_bias) a, beta = sigmoid
//   output_norm  core -> RMSNorm per value head * gamma * sigmoid(z) -> fp16 (the out projection's input)
// Channel order of qkv: q [Hk][S], k [Hk][S], v [Hv][S].
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace truss::gdn {

struct Dims {
    int Hk, Hv, S, K;   // key heads, value heads, head size, conv kernel
    __host__ __device__ int channels() const { return (2 * Hk + Hv) * S; }
};

// conv_w [channels][K] fp32; conv_state [K - 1][channels]: the previous K - 1 raw qkv rows, oldest first (zeros at
// the sequence start), replaced by this chunk's last K - 1 rows. dt_bias, a [Hv].
// -> q, k [T][Hk][S], v [T][Hv][S], g, beta [T][Hv]
void prepare(const Dims & d, const float * qkv, const float * conv_w, float * conv_state, const float * alpha,
             const float * beta_raw, const float * dt_bias, const float * a, int T, float eps, float * q, float * k,
             float * v, float * g, float * beta, cudaStream_t stream);

// core [T][Hv][S], z [T][Hv][S], gamma [S] -> out16 [T][Hv][S]
void output_norm(const Dims & d, const float * core, const float * z, const float * gamma, int T, float eps,
                 half * out16, cudaStream_t stream);

}  // namespace truss::gdn
