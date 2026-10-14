// Gated delta net mixer (Flash-Next's 36 linear-attention layers): qkv / gate / alpha / beta projections, causal conv
// over the last `conv` rows, the delta-rule recurrence, gated RMSNorm, out projection.
//
// Per slot: the recurrent state (two buffers when verify windows run: a window writes the other one, so accepting the
// whole window is a flip and a rollback costs nothing) and the conv rows. A window also keeps the conv rows from
// before it and its raw projection rows, so accept(n < T) restores the conv rows and replays the n accepted rows.
#pragma once
#include "blocks/weights.h"
#include "blocks/linear.h"
#include "engine/block.h"

#include <vector>

namespace truss::blocks {


struct GdnDims {
    int d_model, groups, v_heads, state, conv;
    float eps;
    int key_dim() const { return state * groups; }
    int value_dim() const { return state * v_heads; }
    int conv_dim() const { return 2 * key_dim() + value_dim(); }
    size_t state_floats() const { return (size_t) v_heads * state * state; }
    size_t conv_floats() const { return (size_t) (conv - 1) * conv_dim(); }
    size_t scratch_per_row() const   // bytes of temporaries per row in forward()
    {
        return (size_t) conv_dim() * 4 + (size_t) value_dim() * (4 + 4 + 4 + 2) + (size_t) key_dim() * 8 + (size_t) v_heads * 16;
    }
};

class Gdn : public engine::Stateful {
public:
    Gdn(Linear & lin, const GdnWeights & w, const GdnDims & d) : lin_(lin), w_(w), d_(d) {}
    static void add_tensors(Linear & lin, const GdnWeights & w);
    void forward(engine::Gpu & g, int slot, const half * in16, int T, float * out, bool tentative);

    void alloc_slots(engine::Stage & st, int n_slots, int seq_ctx, int spec_rows) override;
    void reset(engine::Stage & st, int slot) override;
    void accept(engine::Stage & st, int slot, int n, const engine::Window & w) override;
    void ck_buffers(int slot, std::vector<std::pair<float *, size_t>> & out) override;

private:
    struct Slot {
        float * state_buf[2] = {};
        int cur = 0;
        float * conv = nullptr;
        float * conv_snap = nullptr, * raw_qkv = nullptr, * raw_alpha = nullptr, * raw_beta = nullptr;
        float * state() const { return state_buf[cur]; }
    };
    Linear & lin_;
    const GdnWeights w_;
    const GdnDims d_;
    std::vector<Slot> s_;
};

}  // namespace truss::blocks
