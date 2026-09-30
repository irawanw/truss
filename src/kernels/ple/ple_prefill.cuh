// PLE block ops for prefill (the math of qwen4exp::reference::ple; the key and value projections of the gathered
// n-gram embedding are dense GEMMs run by the caller):
//   gate   per stream: s = RMSNorm(key) . RMSNorm(res) / sqrt(d), gate = sigmoid(sign(s) sqrt(max(|s|, 1e-6)))
//   apply  gated = value * gate; normed = RMSNorm(gated) per stream; res += gated + silu(dilated causal depthwise
//          conv of normed), the conv history carried across chunks in hist
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace truss::ple {

// key, res [T][hc][d]; norm_key, norm_query [hc][d] -> gate [T][hc]
void gate(const float * key, const float * res, const float * norm_key, const float * norm_query, int T, int hc,
          int d, float eps, float * gate, cudaStream_t stream);

// value [T][d]; norm_conv [hc][d]; conv_w [hc d][K] fp16; hist [(K - 1) dil][hc d]: the previous normed rows, oldest
// first (zeros at the sequence start), replaced by this chunk's; normed [T][hc][d] scratch; res [T][hc][d] in/out.
void apply(const float * value, const float * gate, const float * norm_conv, const half * conv_w, float * hist, int T,
           int hc, int d, int K, int dil, float eps, float * normed, float * res, cudaStream_t stream);

}  // namespace truss::ple
