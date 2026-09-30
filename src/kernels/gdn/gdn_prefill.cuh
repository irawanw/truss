// Gated delta rule over a prompt chunk (the GDN mixer's recurrent core), one sequence.
//
// Same contract as ref::gated_delta_rule (kernels/reference/ref.cuh): q, k [T][Hk][128] (l2-normed), v [T][Hv][128],
// g, beta [T][Hv]; value head h reads key head h % Hk; state [Hv][128 (key)][128 (value)] in/out; out [T][Hv][128].
// Token-sequential (exact recurrence, fp32), parallel over heads x state columns: FLOPs are negligible (~3 MFLOP per
// token per layer), so the cost is per-step latency.
#pragma once
#include <cuda_runtime.h>

namespace truss::gdn {

// state may be nullptr (zero start, final state discarded)
void delta_rule(const float * q, const float * k, const float * v, const float * g, const float * beta, float * state,
                float * out, int T, int Hk, int Hv, cudaStream_t stream);

}  // namespace truss::gdn
