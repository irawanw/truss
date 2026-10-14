// Output head: logits [n][n_vocab] = output x16 for already-normed rows. Shared by every model (the norm before it is
// the model's). A drafter may score only a vocabulary subset (draft_vocab: the subset's output rows gathered once at
// set_draft_vocab; 16% of the matrix for data/draft_vocab_en.bin), verify keeps the full output exact.
// Q8_1 activations: one q8_gemv / gemm over the whole vocab (decode: 0.9 ms vs 4 ms, TRACKER #56). FP16: the fused
// gemv at decode rows, else the output matrix in vocab tiles that fit Linear's dequantized-weight scratch.
#pragma once
#include "blocks/linear.h"

#include <vector>

namespace truss::blocks {

class Head {
public:
    Head(engine::Gpu & g, Linear & lin, T output) : g_(g), lin_(lin), out_(output) {}
    void set_draft_vocab(const std::vector<int32_t> & ids);   // after upload
    const int * draft_map() const { return draft_map_; }        // device [draft rows]: row -> token id (nullptr: all)
    int draft_rows() const { return draft_.q ? draft_.out : lin_.q8(out_).out; }
    void logits(const half * x16, int n, float * logits, bool draft);

private:
    engine::Gpu & g_;
    Linear & lin_;
    const T out_;
    dense::Q8Matrix draft_{};
    int * draft_map_ = nullptr;
};

}  // namespace truss::blocks
