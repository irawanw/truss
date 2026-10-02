// Sampling on the device: one launch samples `rows` rows of logits with temperature, top-k, top-p and min-p, so a
// speculative verify window can be sampled without copying its logits to the host (Strata's sampled verify:
// accept drafts while the row's sampled token equals the draft; each emitted token is then an exact sample of the
// model's own distribution, whatever the drafts were).
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

namespace truss::sampling {

struct SampleParams {
    float temperature = 1.f;   // > 0 (greedy callers use argmax)
    float top_p = 1.f;         // keep the most probable tokens whose mass reaches top_p (>= 1: off)
    int top_k = 0;             // keep the k most probable (<= 0: off)
    float min_p = 0.f;         // drop tokens below min_p x the top probability (<= 0: off)
    uint64_t seed = 0;
};

// out[r] = a sample of softmax(x[r] / temperature) restricted to the kept tokens, for r in [0, rows); x is
// [rows][n]. counter + r selects the random stream of row r (the caller advances counter by rows per call). The cut
// is a logit threshold (kept: x >= cut) found by bisection, so tokens tied at the cut are all kept; the draw is
// Gumbel-max over the kept tokens with a counter-based hash, deterministic for (seed, counter, row, token).
void sample(const float * x, int n, int rows, const SampleParams & p, uint64_t counter, int * out,
            cudaStream_t stream);

}  // namespace truss::sampling
