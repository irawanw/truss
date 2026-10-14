// PLE n-gram hash embedding, host side (llama-paw llm_graph_input_ple::set_input, build_inp_ple).
//
// Each token looks up heads() rows of one shared int8 table (320M rows of head_dim, 48 GiB for Flash-Next: it stays
// in host memory and only the gathered rows go to the device). For n = 2 .. ngram the window (token, its n - 1
// predecessors) is hashed, mixed = xor_j window[j] * multiplier[j] (uint64, wrapping), and head h of that n-gram
// reads row mixed % vocab[h] + offset[h]. A predecessor before the sequence start, or at or before an eos in the
// window, reads as eos; the token's own eos does not cut its window.
#pragma once
#include "formats/gguf.h"

#include <cstdint>
#include <vector>

namespace truss::blocks {

struct PleDims {
    int d_model, hc, head_dim, conv, ngram, heads_per_ngram;
    std::vector<int64_t> head_offsets, head_vocab, multipliers;
    int64_t eos;
    float eps;
    int heads() const { return (ngram - 1) * heads_per_ngram; }
    int hc_dim() const { return hc * d_model; }
};

// rows [T][heads()] for tokens[0 .. T), one sequence whose first token is tokens[0]
void ple_rows(const PleDims & c, const int32_t * tokens, int T, int32_t * rows);

// emb [T][heads() * head_dim] fp32 = table row * its fp16 scale (heads outermost within a token)
void ple_gather(const PleDims & c, const gguf::Tensor * table, const gguf::Tensor * scale, const int32_t * rows, int T,
                float * emb);

}  // namespace truss::blocks
