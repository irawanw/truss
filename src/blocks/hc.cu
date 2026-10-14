#include "blocks/hc.h"

#include "kernels/hc/hc_prefill.cuh"

#include <cstdlib>
#include <vector>

namespace truss::blocks {

namespace {
// Phase 1's two hyper-connection fusions, each off until its gate: TRUSS_HC_SILU folds the down projection's silu
// into the gemv that produces its input, TRUSS_HC_COLLAPSE folds the stream collapse into the up projection's launch.
bool hc_silu()
{
    static const bool v = [] { const char * e = std::getenv("TRUSS_HC_SILU"); return e && std::atoi(e); }();
    return v;
}
bool hc_collapse()
{
    static const bool v = [] { const char * e = std::getenv("TRUSS_HC_COLLAPSE"); return e && std::atoi(e); }();
    return v;
}
}  // namespace

void Hc::mix(const HcWeights & h, const float * res, int T, float * mixed, half * mixed16, float * inject,
             const float * cout, const float * cinj)
{
    Scratch & sc = *g_.sc;
    const size_t m = sc.mark();
    // both fusions live in the fp16-activation gemv at decode rows
    const bool gemv_ok = lin_.act() == Activations::FP16 && T <= dense::GEMV_ROWS && dense::mma_on();
    const bool silu_epi = gemv_ok && hc_silu();
    // the head's connection has its up matrix in the q8 table but not as an uploaded fp16 tensor: shapes from there
    const dense::Q8Matrix & wup = lin_.q8(h.up);
    const bool collapse_epi = gemv_ok && hc_collapse() && dense::up_collapse_ok(wup.in, wup.out, d_.hc, d_.d_model);
    float * rstd = sc.alloc((size_t) T * d_.hc);
    float * lo = silu_epi ? nullptr : sc.alloc((size_t) T * d_.hc_rank);   // the epilogue makes it dead
    half * lo16 = sc.alloc<half>((size_t) T * d_.hc_rank);
    float * gate = collapse_epi ? nullptr : sc.alloc((size_t) T * d_.hc_dim());
    Linear::Act xa;
    if (lin_.act() == Activations::Q8_1) {
        xa.q = sc.alloc<int8_t>((size_t) T * d_.hc_dim());
        xa.d = sc.alloc<half>((size_t) T * d_.hc_dim() / 32);
        if (cout) hc::combine_norm_q8(const_cast<float *>(res), cout, cinj, lin_.f32(h.norm), T, d_.hc, d_.d_model, d_.eps,
                                      rstd, xa.q, xa.d, g_.s);
        else hc::norm_q8(res, lin_.f32(h.norm), T, d_.hc, d_.d_model, d_.eps, rstd, xa.q, xa.d, g_.s);
    } else {
        half * xn16 = sc.alloc<half>((size_t) T * d_.hc_dim());
        if (cout) hc::combine_norm(const_cast<float *>(res), cout, cinj, lin_.f32(h.norm), T, d_.hc, d_.d_model, d_.eps,
                                   xn16, rstd, g_.s);
        else hc::norm(res, lin_.f32(h.norm), T, d_.hc, d_.d_model, d_.eps, xn16, rstd, g_.s);
        xa = lin_.quant(xn16, T, d_.hc_dim());
    }
    const float inv_hc = 1.f / d_.hc;
    if (silu_epi) {   // the down projection writes silu(lo) as fp16 itself; inject shares the launch
        std::vector<Linear::Proj> ps = { { h.down, nullptr } };
        if (h.inject) ps.push_back({ h.inject, inject });
        lin_.lin_multi_silu(ps, lo16, inv_hc, xa, T);
    } else {
        if (h.inject) lin_.lin_multi({ { h.down, lo }, { h.inject, inject } }, xa, T);   // inject reads the same xa
        else lin_.lin(h.down, xa, T, lo);
        hc::silu(lo, T * d_.hc_rank, inv_hc, lo16, g_.s);
    }
    if (collapse_epi)   // gate stays in the CTA that computed it; mixed and mixed16 come out of the gemv
        dense::q8_gemv_a16_up_collapse(wup, lo16, T, res, rstd, lin_.f32(h.norm), mixed, mixed16, d_.hc, d_.d_model, g_.s);
    else {
        lin_.lin(h.up, lo16, T, gate);
        hc::collapse(res, rstd, lin_.f32(h.norm), gate, T, d_.hc, d_.d_model, mixed, mixed16, g_.s);
    }
    sc.release(m);
}

void Hc::combine(float * res, const float * out, const float * inject, int T)
{
    hc::combine(res, out, inject, T, d_.hc, d_.d_model, g_.s);
}

void Hc::expand(const float * emb, int T, float * res) { hc::expand(emb, T, d_.hc, d_.d_model, res, g_.s); }

}  // namespace truss::blocks
