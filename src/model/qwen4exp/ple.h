// PLE n-gram hash embedding, host side (llama-paw llm_graph_input_ple::set_input, build_inp_ple).
//
// Each token looks up ple_heads() rows of one shared int8 table (320M rows of ple_head_dim, 48 GiB: it stays in host
// memory and only the gathered rows go to the device). For n = 2 .. ple_ngram the window (token, its n - 1
// predecessors) is hashed, mixed = xor_j window[j] * multiplier[j] (uint64, wrapping), and head h of that n-gram
// reads row mixed % vocab[h] + offset[h]. A predecessor before the sequence start, or at or before a ple_eos in the
// window, reads as ple_eos; the token's own ple_eos does not cut its window.
#pragma once
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/weights.h"

#include <cstdint>

namespace truss::qwen4exp {

// rows [T][ple_heads()] for tokens[0 .. T), one sequence whose first token is tokens[0]
void ple_rows(const Config & c, const int32_t * tokens, int T, int32_t * rows);

// emb [T][ple_heads() * ple_head_dim] fp32 = table row * its fp16 scale (heads outermost within a token)
void ple_gather(const Config & c, const Weights & w, const int32_t * rows, int T, float * emb);

}  // namespace truss::qwen4exp
