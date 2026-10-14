// MoE FFN block: router (fp32 projection, so routing keeps full precision), routed experts through the stage's
// runtime::MoeExec, an optional gated shared expert, and the sum. The routing rule is a parameter (softmax top-k for
// Flash-Next; LFM's sigmoid + expert-bias top-k is another RouteFn), so is the presence of the shared expert.
#pragma once
#include "blocks/weights.h"
#include "blocks/linear.h"
#include "formats/trellis_table.h"
#include "runtime/moe_exec.h"

namespace truss::blocks {


struct MoeDims {
    int d_model, n_expert, topk, d_ff_shexp;   // d_ff_shexp 0: no shared expert
    size_t scratch_per_row() const
    {
        return (size_t) n_expert * 4 + topk * 12 + (size_t) d_model * 12 + (size_t) d_ff_shexp * 10 + 4;
    }
};

// routing rules (moe.cu): softmax top-k renormalized (Flash-Next); sigmoid + selection bias top-k renormalized (LFM2)
void route_softmax(const float * logits, const float * bias, int T, int E, int K, int * ids, float * wts, cudaStream_t s);
void route_sigmoid_bias(const float * logits, const float * bias, int T, int E, int K, int * ids, float * wts,
                        cudaStream_t s);

class Moe {
public:
    // sl: its store layer in the stage's MoeExec; next: the next layer's MoE weights on this stage (its router is the
    // pre-gating hint, TRACKER #58), or nullptr; softmax_route: route is the softmax top-k (the fused route+publish
    // may replace it)
    Moe(Linear & lin, runtime::MoeExec & exec, int sl, int model_layer, const MoeWeights & w, const MoeDims & d,
        const MoeWeights * next, runtime::MoeExec::RouteFn route, bool softmax_route, engine::Profile * prof)
        : lin_(lin), exec_(exec), sl_(sl), ml_(model_layer), w_(w), d_(d), next_(next), route_(route),
          softmax_(softmax_route), prof_(prof)
    {
    }
    static void add_tensors(Linear & lin, const MoeWeights & w);
    // out [T][d] = routed experts (+ shared expert) of in [T][d] (fp32 and fp16 copies of the same rows)
    void forward(engine::Gpu & g, const float * in, const half * in16, int T, float * out, bool stream, void * moe_ws,
                 int moe_rows);

private:
    Linear & lin_;
    runtime::MoeExec & exec_;
    const int sl_, ml_;
    const MoeWeights & w_;
    const MoeDims d_;
    const MoeWeights * next_;
    const runtime::MoeExec::RouteFn route_;
    const bool softmax_;
    engine::Profile * prof_;
};

}  // namespace truss::blocks
