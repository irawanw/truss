// Speculative sampling on the device (Leviathan et al. 2023, Chen et al. 2023): the draft head SAMPLES its token
// from q (its own distribution under the request's sampler settings) instead of taking the argmax, and the verify
// window accepts draft i with probability min(1, p_i(x_i) / q_i(x_i)); the first rejected row draws from the residual
// max(0, p_i - q_i) (normalized), and a window with every draft accepted draws its bonus token from p_nw. Each
// emitted token is then an exact sample of p, the model's own distribution after the same temperature / top-k / top-p
// / min-p cut as sample() - lossless, like the sampled verify it replaces - while the acceptance rate rises from
// p_i(argmax q_i) to sum_v min(p_i(v), q_i(v)).
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

#include "sample.cuh"

namespace truss::sampling {

// x[r][n] (logits) -> x[r][n] = the kept probabilities of softmax(x / T) after sample()'s cut (0 outside the cut),
// in place, for r in [0, rows)
void probs_inplace(float * x, int n, int rows, const SampleParams & p, cudaStream_t stream);

// the draft head's row: x[n] logits over the draft vocabulary (map[j] = token id, nullptr: identity), q_full[V] is
// zeroed and then gets q at the kept tokens' ids; out[0] = a sample of q (token id). counter selects the stream.
void draft_sample(const float * x, int n, const int * map, int V, const SampleParams & p, uint64_t counter,
                  float * q_full, int * out, cudaStream_t stream);

// probs[nw + 1][V] (probs_inplace of the verify rows), q_full[nw][V], drafts[nw] (token ids): out[0] = accepted
// drafts j, out[1] = the token after them (residual sample at row j, or p_nw's sample when j == nw)
void spec_accept(const float * probs, const float * q_full, const int * drafts, int nw, int V, uint64_t seed,
                 uint64_t counter, int * out, cudaStream_t stream);

}  // namespace truss::sampling
