#include "blocks/moe.h"

#include "kernels/ffn/ffn_ops.cuh"

#include <stdexcept>
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

void route_sqrtsoftplus_bias(const float * logits, const float * bias, int T, int E, int K, int * ids, float * wts,
                             cudaStream_t s)
{
    ffn::route_biased(logits, bias, T, E, K, ffn::Score::SqrtSoftplus, true, 1e-20f, 1.f, ids, wts, s);
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
    const bool gated = has_shared && w_.shexp_gate_inp;
    float * sg = gated ? sc.alloc(T) : nullptr;
    half * mid16 = has_shared ? sc.alloc<half>((size_t) T * F) : nullptr;
    auto shared = [&] {   // y = shared expert (sg, its gate, comes with the router)
        const size_t m2 = sc.mark();
        const Linear::Act ia = lin_.quant(in16, T, dm);
        lin_.lin_multi({ { w_.shexp_gate, gu }, { w_.shexp_up, uu } }, ia, T);
        ffn::swiglu(gu, uu, T * F, mid16, g.s, d_.swiglu_limit);
        lin_.lin(w_.shexp_down, mid16, T, y);
        sc.release(m2);
    };
    // the router, the next layer's router on this input (pre-gating hint) and the shared-expert gate read the same
    // fp32 rows: one launch at decode sizes
    const bool hint = next_ && exec_.wants_hint(sl_, T, stream);
    float * pl = hint ? sc.alloc((size_t) T * E) : nullptr;
    // B2: the lone-router fast path (one T == 1 row, nothing batched with it): projection + route in one launch,
    // bit-identical logits/ids/wts. Anything else (shared gate, hint, prefill/verify rows) keeps the old path.
    const bool fuse_route = T == 1 && !has_shared && !hint && dm >= 2048 && dm % 128 == 0 && E % 32 == 0 && E <= 1024
                            && K <= 32 && K <= E && route_ == route_sigmoid_bias;
    if (fuse_route) {
        ffn::route_sigmoid_gemv(lin_.f32(w_.router), dm, w_.exp_bias ? lin_.f32(w_.exp_bias) : nullptr, in, E, K,
                                logits, ids, wts, exec_.route_arrive(), g.s);
    } else {
        std::vector<Linear::Proj> ps{ { w_.router, logits } };
        if (gated) ps.push_back({ w_.shexp_gate_inp, sg });
        if (hint) ps.push_back({ next_->router, pl });
        lin_.lin32_multi(ps, in, T);
    }
    const float * bias = w_.exp_bias ? lin_.f32(w_.exp_bias) : nullptr;
    const float * pbias = hint && next_->exp_bias ? lin_.f32(next_->exp_bias) : nullptr;
    const bool scaled = d_.route_scale != 1.f;
    if (gated && scaled) throw std::runtime_error("Moe: a gated shared expert with a routed scale is not wired");
    runtime::MoeExec::Call c{ sl_, in, T, logits, pl, bias, pbias, ids, wts, has_shared || scaled ? routed : out, stream,
                              moe_ws, moe_rows, route_, softmax_, fuse_route,
                              has_shared ? std::function<void()>(shared) : std::function<void()>() };
    exec_.run(c);
    if (gated) ffn::shared_add(routed, y, sg, T, dm, out, g.s);
    else if (has_shared || scaled) ffn::scaled_add(routed, d_.route_scale, has_shared ? y : nullptr, T, dm, out, g.s);
    sc.release(m);
}

}  // namespace truss::blocks
