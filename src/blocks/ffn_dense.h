// Dense SwiGLU FFN: out = down(silu(gate x) * up x). Stateless. (LFM2's first layers; any dense model's FFN.)
#pragma once
#include "blocks/linear.h"
#include "kernels/ffn/ffn_ops.cuh"

namespace truss::blocks {

struct DenseFfnWeights {
    T gate = nullptr, up = nullptr, down = nullptr;
};

class DenseFfn {
public:
    DenseFfn(Linear & lin, const DenseFfnWeights & w, int d_model, int d_ff) : lin_(lin), w_(w), d_(d_model), f_(d_ff) {}
    static void add_tensors(Linear & lin, const DenseFfnWeights & w) { lin.add(w.gate), lin.add(w.up), lin.add(w.down); }
    static size_t scratch_per_row(int d, int f) { return (size_t) f * 10 + (size_t) (d + d / 16) * 2; }
    // in16 [T][d] (normed) -> out [T][d]
    void forward(engine::Gpu & g, const half * in16, int T, float * out)
    {
        Scratch & sc = *g.sc;
        const size_t m = sc.mark();
        float * gt = sc.alloc((size_t) T * f_), * up = sc.alloc((size_t) T * f_);
        half * mid = sc.alloc<half>((size_t) T * f_);
        const Linear::Act a = lin_.quant(in16, T, d_);
        lin_.lin_multi({ { w_.gate, gt }, { w_.up, up } }, a, T);
        ffn::swiglu(gt, up, T * f_, mid, g.s);
        lin_.lin(w_.down, mid, T, out);
        sc.release(m);
    }

private:
    Linear & lin_;
    const DenseFfnWeights w_;
    const int d_, f_;
};

}  // namespace truss::blocks
