// Dense causal grouped-query attention mixer (LFM2's attention layers; any GQA model with per-head q / k norms and
// NEOX rope): q / k / v projections (one launch at decode rows), kernels/attn/gqa (norm + rope + cache write, then
// attention), out projection.
//
// Per slot: the K / V cache is a sub-range of one n_ctx array (positions past the committed one are rewritten before
// any query reads them, so a verify window needs no undo and a checkpoint copies nothing).
#pragma once
#include "blocks/linear.h"
#include "engine/block.h"
#include "kernels/attn/gqa.cuh"

#include <vector>

namespace truss::blocks {

struct GqaWeights {
    T q = nullptr, k = nullptr, v = nullptr, out = nullptr, q_norm = nullptr, k_norm = nullptr;   // norms optional
};

class AttnGqa : public engine::Stateful {
public:
    AttnGqa(Linear & lin, const GqaWeights & w, const attn::GqaShape & s, int d_model)
        : lin_(lin), w_(w), s_(s), d_(d_model)
    {
    }
    static void add_tensors(Linear & lin, const GqaWeights & w)
    {
        for (T t : { w.q, w.k, w.v, w.out, w.q_norm, w.k_norm }) lin.add(t);
    }
    static size_t scratch_per_row(const attn::GqaShape & s, int d)
    {
        const size_t qd = (size_t) s.H * s.D, kd = (size_t) s.HKV * s.D;
        return qd * (4 + 2 + 2) + kd * 8 + (size_t) (d + d / 16) * 2;
    }
    static size_t scratch_fixed(const attn::GqaShape & s, int T, int n_ctx) { return attn::gqa_workspace_bytes(s, T, n_ctx); }
    // in16 [T][d] (normed) at positions pos0 .. -> out [T][d]; tab [T][32]: the shared rope table (B2),
    // or nullptr for the inline trig
    void forward(engine::Gpu & g, int slot, const half * in16, int pos0, int T, float * out,
                 const float2 * tab = nullptr);

    void alloc_slots(engine::Stage & st, int n_slots, int seq_ctx, int spec_rows) override;
    void positional(std::vector<Positional> & out) const override;

private:
    struct Slot {
        half * k = nullptr, * v = nullptr;   // [seq_ctx][HKV][D] views of the shared arrays
    };
    Linear & lin_;
    const GqaWeights w_;
    const attn::GqaShape s_;
    const int d_;
    int n_ctx_ = 0;
    std::vector<Slot> sl_;
};

}  // namespace truss::blocks
