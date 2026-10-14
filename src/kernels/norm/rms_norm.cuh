// Row RMSNorm: y[t] = w * x[t] / sqrt(mean(x[t]^2) + eps), fp32 math (HF Lfm2MoeRMSNorm / LlamaRMSNorm; the weight
// multiplies directly, no 1 + w). Outputs fp32 and / or fp16 (either may be nullptr). One CTA per row; the sum of
// squares is reduced in a fixed order, so a row's result does not depend on T.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace truss::norm {

// x [T][d], w [d]; y32 [T][d], y16 [T][d]. d % 4 == 0.
void rms(const float * x, const float * w, int T, int d, float eps, float * y32, half * y16, cudaStream_t stream);
// x += a (residual add, in place, fp32 as ffn::add), then rms of the sum: one launch for add + norm, bit-equal to the two
void add_rms(float * x, const float * a, const float * w, int T, int d, float eps, float * y32, half * y16,
             cudaStream_t stream);

// per head: x [T][H][D] in place (fp32), w [D] shared by the heads (q / k layer norms). D <= 256, D % 32 == 0.
void rms_heads(float * x, const float * w, int T, int H, int D, float eps, cudaStream_t stream);

}  // namespace truss::norm
