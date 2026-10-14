// Hyper-connections (Flash-Next, likely Qwen4): hc parallel residual streams of d_model, collapsed into one mixed row
// for a sub-block and written back with a combine. Stateless: a helper the model calls around each mixer and FFN.
//
//   mix      rstd-normalized streams -> rank-hc_rank projection (silu) -> gate -> collapse into mixed [T][d] (fp32 +
//            fp16), and the inject weights [T][hc] when the connection has them. With Q8_1 activations the norm fuses
//            the activation quantize (and the previous combine when cout / cinj are given): one launch, same math.
//   combine  res[s] += inject[s] * out for each stream s
//   expand   embedding rows -> hc copies
// TRUSS_HC_SILU / TRUSS_HC_COLLAPSE fold the silu / the collapse into the decode gemvs (fp16 activations only).
#pragma once
#include "blocks/weights.h"
#include "blocks/linear.h"

namespace truss::blocks {


struct HcDims {
    int hc, hc_rank, d_model;
    float eps;
    int hc_dim() const { return hc * d_model; }
};

class Hc {
public:
    Hc(engine::Gpu & g, Linear & lin, const HcDims & d) : g_(g), lin_(lin), d_(d) {}
    void add_tensors(const HcWeights & h) { lin_.add(h.norm), lin_.add(h.down), lin_.add(h.up), lin_.add(h.inject); }
    void mix(const HcWeights & h, const float * res, int T, float * mixed, half * mixed16, float * inject,
             const float * cout = nullptr, const float * cinj = nullptr);
    void combine(float * res, const float * out, const float * inject, int T);
    void expand(const float * emb, int T, float * res);
    const HcDims & dims() const { return d_; }

private:
    engine::Gpu & g_;
    Linear & lin_;
    const HcDims d_;
};

}  // namespace truss::blocks
