// DeepSeek-V4 hyper-connections around one sub-block (kernels/hc/hc_sinkhorn): mix (the next sub-block's collapse
// weights, this one's expand weights), collapse (copies -> one sub-block input), post (expand + mix the copies).
// Stateless; one per (layer, sub-block) weight set. Row strides come with the dims (deepseek41 keeps each residual
// row's next-collapse coef in the row's tail).
#pragma once
#include "blocks/linear.h"
#include "blocks/weights.h"
#include "kernels/hc/hc_sinkhorn.cuh"

namespace truss::blocks {

class HcSinkhorn {
public:
    HcSinkhorn(Linear & lin, const SkHcWeights & w, const hc::SkDims & k) : lin_(lin), w_(w), k_(k) {}
    static void add_tensors(Linear & lin, const SkHcWeights & w) { lin.add(w.fn), lin.add(w.base), lin.add(w.scale); }
    static size_t scratch_per_row(const hc::SkDims & k) { return hc::sk_mix_ws_bytes(1, k) + 64; }
    // coef [T][dims().coef_ld] of res
    void mix(engine::Gpu & g, const float * res, int T, float * coef) const
    {
        Scratch & sc = *g.sc;
        const size_t m = sc.mark();
        void * ws = sc.alloc<unsigned char>(hc::sk_mix_ws_bytes(T, k_));
        hc::sk_mix(res, lin_.f32(w_.fn), lin_.f32(w_.scale), lin_.f32(w_.base), T, k_, ws, coef, g.s);
        sc.release(m);
    }
    // y [T][d] (and y16) = sum_c pre[c] res[c], pre = the first hc floats of each coef row (coef_ld apart)
    void collapse(engine::Gpu & g, const float * res, const float * coef, int coef_ld, int T, float * y, half * y16) const
    {
        hc::sk_collapse(res, coef, T, with(coef_ld), y, y16, g.s);
    }
    // res = post[j] x + sum_i comb[i][j] res[i], in place
    void post(engine::Gpu & g, float * res, const float * x, const float * coef, int coef_ld, int T) const
    {
        hc::sk_post(res, x, coef, T, with(coef_ld), res, g.s);
    }
    const hc::SkDims & dims() const { return k_; }

private:
    hc::SkDims with(int coef_ld) const
    {
        hc::SkDims k = k_;
        k.coef_ld = coef_ld;
        return k;
    }
    Linear & lin_;
    const SkHcWeights w_;
    const hc::SkDims k_;
};

}  // namespace truss::blocks
