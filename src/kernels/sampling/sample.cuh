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

// Tokens a sampled row may not produce (the end-of-turn tokens inside an open reasoning block): x[r][id[i]] = BANNED
// for r in [0, rows). A large finite logit, not -inf: sample() bisects between the row's min and max logit, and
// exp(BANNED / T - max) is exactly 0 in fp32 for any sane temperature, so a banned token is never drawn and gets
// probability 0 in probs_inplace (a draft of it is then always rejected by spec_accept).
constexpr int BAN_MAX = 8;
constexpr float BANNED = -1e4f;
struct BanList {
    int n = 0;
    int id[BAN_MAX] = {};
};
void ban(float * x, int n, int rows, const BanList & b, cudaStream_t stream);

// Word-repeat penalty (the reasoning's "maybe maybe maybe" loop): cls[v] is token v's word class (the lowercased
// letters of the token without its leading space, so "Maybe" and " maybe" share one; < 0: not penalized, see
// truss_set_word_classes), prev.c[r] the class of the token row r follows (-1: none). x[r][v] -= pen for every v
// with cls[v] == prev.c[r] >= 0, r in [0, rows): a word is not repeated back to back. Rows of a verify window follow different tokens, so each row has its own class.
// Filler recency (10-22, "Hmm ... Hmm ... Hmm" loops): row r also takes fp[r][k] logits off token ft[r][k], k < nf[r]
// (the host's per-row list: the next token of a filler word's token sequence whose prefix ends the row's context,
// penalty x the filler's uses in the row's recent window). Both rules in one launch.
constexpr int WORD_ROWS = 16, FILL_MAX = 12;
struct WordPrev {
    int c[WORD_ROWS] = {};
    int nf[WORD_ROWS] = {};
    int ft[WORD_ROWS][FILL_MAX] = {};
    float fp[WORD_ROWS][FILL_MAX] = {};
};
void word_repeat(float * x, int n, int rows, const int * cls, const WordPrev & prev, float pen, cudaStream_t stream);

}  // namespace truss::sampling
