// DSA mixer (Flash-Next's 12 sparse-attention layers and its MTP block): q / k / v / indexer projections, per-head
// RMSNorm + NEOX rope, the indexer's block selection (top 512 blocks of 4 cells), gated attention over the chosen
// blocks, out projection (math: docs/reference/kernels-mixers.md). Shape: a dsa:: shape struct (head counts, dims).
//
// Per slot: the K/V cache and the indexer's block cache are sub-ranges of one n_ctx array (positions past the
// committed one are simply rewritten, so a verify window needs no undo there); the indexer's open block (raw keys of
// the last < 4 cells) is per slot, with its snapshot and the window's raw keys for accept(n < T).
#pragma once
#include "blocks/weights.h"
#include "blocks/linear.h"
#include "engine/block.h"
#include "kernels/dsa/dsa_prepare.cuh"
#include "kernels/dsa/dsa_prefill.cuh"

#include <vector>

namespace truss::blocks {


struct DsaDims {
    int d_model;
    float rope_base, eps;
    const int * rope_sec = nullptr;   // device [ROPE_DIMS / 2]: mrope pair -> axis (null: text-only)
};

template <class Shape> class Dsa : public engine::Stateful {
public:
    Dsa(Linear & lin, const DsaWeights & w, const DsaDims & d, bool kv_int8) : lin_(lin), w_(w), d_(d), int8_(kv_int8) {}
    static void add_tensors(Linear & lin, const DsaWeights & w);
    static size_t scratch_per_row();
    static size_t scratch_fixed(int T, int n_ctx);
    // rp: 3-D positions of a multimodal prompt (rp.pos3 / stride / row0) and the decode rope offset (rp.delta);
    // default = text (1-D, pos0 + t). rp.sec is filled from d_.rope_sec.
    void forward(engine::Gpu & g, int slot, const half * in16, int pos0, int T, float * out, bool tentative,
                 const dsa::RopePos & rp = {});

    // the indexer's open block of a slot: a drafter's chain snapshots and restores it around its tentative rows
    float * partial(int slot) const { return s_[slot].idx_partial; }
    static constexpr size_t partial_floats() { return (size_t) (Shape::RATIO - 1) * Shape::ID; }

    void alloc_slots(engine::Stage & st, int n_slots, int seq_ctx, int spec_rows) override;
    void accept(engine::Stage & st, int slot, int n, const engine::Window & w) override;
    void ck_buffers(int slot, std::vector<std::pair<float *, size_t>> & out) override;
    void positional(std::vector<Positional> & out) const override;

private:
    struct Slot {
        dsa::KvCache kv;
        half * idx_k = nullptr;
        float * idx_partial = nullptr, * partial_snap = nullptr, * raw_ik = nullptr;
    };
    Linear & lin_;
    const DsaWeights w_;
    const DsaDims d_;
    const bool int8_;
    int n_ctx_ = 0;
    std::vector<Slot> s_;
};

}  // namespace truss::blocks
