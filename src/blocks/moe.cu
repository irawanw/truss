#include "blocks/moe.h"

#include "kernels/ffn/ffn_ops.cuh"

#include <vector>

namespace truss::blocks {

void route_softmax(const float * logits, const float *, int T, int E, int K, int * ids, float * wts, cudaStream_t s)
{
    ffn::route(logits, T, E, K, ids, wts, s);
}

void route_sigmoid_bias(const float * logits, const float * bias, int T, int E, int K, int * ids, float * wts,
                        cudaStream_t s)
{
    ffn::route_sigmoid(logits, bias, T, E, K, true, 1.f, ids, wts, s);
}

void Moe::add_tensors(Linear & lin, const MoeWeights & w)
{
    lin.add(w.router), lin.add(w.shexp_gate_inp), lin.add(w.shexp_gate), lin.add(w.shexp_up), lin.add(w.shexp_down);
    lin.add(w.exp_bias);
}

void Moe::forward(engine::Gpu & g, const float * in, const half * in16, int T, float * out, bool stream, void * moe_ws,
                  int moe_rows)
{
    Scratch & sc = *g.sc;
    const size_t m = sc.mark();
    if (prof_) prof_->record(ml_, 3, g.s);   // router + routed MoE (its copies and the CPU tier's start are inside)
    const int E = d_.n_expert, K = d_.topk, F = d_.d_ff_shexp, dm = d_.d_model;
    const bool has_shared = F > 0;
    float * logits = sc.alloc((size_t) T * E), * wts = sc.alloc((size_t) T * K);
    int * ids = sc.alloc<int>((size_t) T * K);
    float * routed = sc.alloc((size_t) T * dm), * y = sc.alloc((size_t) T * dm);
    float * gu = has_shared ? sc.alloc((size_t) T * F) : nullptr, * uu = has_shared ? sc.alloc((size_t) T * F) : nullptr;
    float * sg = has_shared ? sc.alloc(T) : nullptr;
    half * mid16 = has_shared ? sc.alloc<half>((size_t) T * F) : nullptr;
    auto shared = [&] {   // y = shared expert (sg, its gate, comes with the router)
        const size_t m2 = sc.mark();
        const Linear::Act ia = lin_.quant(in16, T, dm);
        lin_.lin_multi({ { w_.shexp_gate, gu }, { w_.shexp_up, uu } }, ia, T);
        ffn::swiglu(gu, uu, T * F, mid16, g.s);
        lin_.lin(w_.shexp_down, mid16, T, y);
        sc.release(m2);
    };
    // the router, the next layer's router on this input (pre-gating hint) and the shared-expert gate read the same
    // fp32 rows: one launch at decode sizes
    const bool hint = next_ && exec_.wants_hint(sl_, T, stream);
    float * pl = hint ? sc.alloc((size_t) T * E) : nullptr;
    {
        std::vector<Linear::Proj> ps{ { w_.router, logits } };
        if (has_shared) ps.push_back({ w_.shexp_gate_inp, sg });
        if (hint) ps.push_back({ next_->router, pl });
        lin_.lin32_multi(ps, in, T);
    }
    const float * bias = w_.exp_bias ? lin_.f32(w_.exp_bias) : nullptr;
    const float * pbias = hint && next_->exp_bias ? lin_.f32(next_->exp_bias) : nullptr;
    runtime::MoeExec::Call c{ sl_, in, T, logits, pl, bias, pbias, ids, wts, has_shared ? routed : out, stream, moe_ws, moe_rows,
                              route_, softmax_, has_shared ? std::function<void()>(shared) : std::function<void()>() };
    exec_.run(c);
    if (has_shared) ffn::shared_add(routed, y, sg, T, dm, out, g.s);
    sc.release(m);
}

}  // namespace truss::blocks
