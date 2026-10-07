// Hyper-connection elementwise ops for prefill (the math of qwen4exp::reference::hc_mix / hc_combine; the low-rank
// and inject projections are dense GEMMs run by the caller between them):
//   norm      xn = RMSNorm per stream (gamma [hc][d]) -> fp16 (the GEMM input) + the per-stream rstd
//   silu      lo16 = silu(lo / hc) -> fp16 (the up GEMM input)
//   collapse  mixed = (1 / hc) sum_c xn[c] * sigmoid(gate[c]), xn recomputed from res and rstd in fp32
//   combine   res[c] += out * 2 sigmoid(inject[c] / hc)
// All memory-bound; rows are processed by one block each so the norm reduction order is fixed.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace truss::hc {

// res [T][hc][d] = emb [T][d] in every stream (the residual's start)
void expand(const float * emb, int T, int hc, int d, float * res, cudaStream_t stream);

// res [T][hc][d] -> xn16 [T][hc][d], rstd [T][hc]
void norm(const float * res, const float * gamma, int T, int hc, int d, float eps, half * xn16, float * rstd,
          cudaStream_t stream);

// lo [n] -> lo16 [n] = silu(lo * inv_hc)
void silu(const float * lo, int n, float inv_hc, half * lo16, cudaStream_t stream);

// mixed [T][d] (and mixed16, the next GEMM's input, when non-null)
void collapse(const float * res, const float * rstd, const float * gamma, const float * gate, int T, int hc, int d,
              float * mixed, half * mixed16, cudaStream_t stream);

// res [T][hc][d] += out [T][d] * 2 sigmoid(inject [T][hc] / hc)
void combine(float * res, const float * out, const float * inject, int T, int hc, int d, cudaStream_t stream);

// Q8_1 fused variants (L3, Order 11): norm (+ optional combine) with the activation quantization the caller
// would run on xn16 next. Bit-identical to the separate kernels (q8_gemm quantize_kernel<__half> tree/scale/
// rounding reproduced per warp); xn16 is never materialized.
void norm_q8(const float * res, const float * gamma, int T, int hc, int d, float eps, float * rstd,
             int8_t * q, half * dsc, cudaStream_t stream);
void combine_norm_q8(float * res, const float * out, const float * inject, const float * gamma, int T, int hc, int d,
                     float eps, float * rstd, int8_t * q, half * dsc, cudaStream_t stream);

}  // namespace truss::hc
