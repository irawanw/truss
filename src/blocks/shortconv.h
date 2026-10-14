// Short-convolution mixer (LFM2's conv layers): in_proj -> [B | C | x] -> causal depthwise conv of B*x over the last
// L rows -> C * conv -> out_proj (kernels/conv/short_conv.cuh).
//
// Per slot: the last L - 1 B*x rows (conv state). A verify window keeps the state from before it and its own B*x
// rows, so accept(n < T) rebuilds the state as the last L - 1 rows of [old state | accepted rows] (spec::tail_rows,
// the same rollback as PLE's history).
#pragma once
#include "blocks/linear.h"
#include "engine/block.h"

#include <vector>

namespace truss::blocks {

struct ShortConvWeights {
    T in_proj = nullptr, conv = nullptr, out_proj = nullptr;   // [d -> 3d], [d][L] fp32, [d -> d]
};

class ShortConv : public engine::Stateful {
public:
    ShortConv(Linear & lin, const ShortConvWeights & w, int d, int L) : lin_(lin), w_(w), d_(d), L_(L) {}
    static void add_tensors(Linear & lin, const ShortConvWeights & w)
    {
        lin.add(w.in_proj), lin.add(w.conv), lin.add(w.out_proj);
    }
    static size_t scratch_per_row(int d) { return (size_t) d * (3 * 4 + 2 + 2) + (size_t) d / 16 * 2; }
    // in16 [T][d] (normed) -> out [T][d]
    void forward(engine::Gpu & g, int slot, const half * in16, int T, float * out, bool tentative);

    void alloc_slots(engine::Stage & st, int n_slots, int seq_ctx, int spec_rows) override;
    void reset(engine::Stage & st, int slot) override;
    void accept(engine::Stage & st, int slot, int n, const engine::Window & w) override;
    void ck_buffers(int slot, std::vector<std::pair<float *, size_t>> & out) override;

private:
    struct Slot {
        float * state = nullptr, * snap = nullptr, * rows = nullptr;   // [L-1][d], [L-1][d], [window][d]
    };
    size_t state_floats() const { return (size_t) (L_ - 1) * d_; }
    Linear & lin_;
    const ShortConvWeights w_;
    const int d_, L_;
    std::vector<Slot> s_;
};

}  // namespace truss::blocks
