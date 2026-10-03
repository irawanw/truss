// Repetition / frequency / presence penalties on the device, applied in place to logits before sampling (llama.cpp's
// order and formulas, which Strata's sampler follows): for a token seen c > 0 times in the row's history,
//   x = x > 0 ? x / repeat : x * repeat;   x -= c * frequency + presence.
// Each row has its own history (a speculative verify window's row t also counts the window's tokens 0..t).
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

namespace truss::sampling {

struct PenaltyParams {
    float repeat = 1.f;     // repetition_penalty (1: off)
    float frequency = 0.f;  // frequency_penalty, per occurrence (0: off)
    float presence = 0.f;   // presence_penalty, once per token present (0: off)
};

inline bool penalties_on(const PenaltyParams & p) { return p.repeat != 1.f || p.frequency != 0.f || p.presence != 0.f; }

// The verify rows' histories on the host (Strata's penalty_rows): row t of `out` ([rows][w]) is the last w tokens of
// (history[0 .. nh), feed[0 .. min(nfeed, t + 1))), -1 in front where the sequence is shorter than w.
inline void penalty_rows(const int32_t * history, int nh, const int32_t * feed, int nfeed, int rows, int w, int * out)
{
    for (int t = 0; t < rows; ++t) {
        const int nf = nfeed < t + 1 ? nfeed : t + 1, total = nh + nf;
        for (int i = 0; i < w; ++i) {
            const int k = total - w + i;
            out[(size_t) t * w + i] = k < 0 ? -1 : k < nh ? history[k] : feed[k - nh];
        }
    }
}

// x is [rows][n]; hist is [rows][h] token ids on the device (entries < 0 or >= n are ignored, so rows may be padded
// with -1). cnt is device scratch of rows * n ints that must be zero on entry; it is zero again on return.
void penalize(float * x, int n, int rows, const int * hist, int h, const PenaltyParams & p, int * cnt,
              cudaStream_t stream);

}  // namespace truss::sampling
