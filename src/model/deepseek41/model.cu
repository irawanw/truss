#include "model/deepseek41/model.h"

#include "engine/stage.h"
#include "kernels/dense/q8_gemm.cuh"
#include "kernels/ds41/attn_ops.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace truss::deepseek41 {

Family::Family(const gguf::File & f, const engine::DrafterFiles & d) : c_(Config::from_gguf(f)), w_(bind(f, c_))
{
    engine::require(d.path.empty(), "deepseek41: the DSpark drafter is not wired yet");
    engine::require(c_.d_model == MoeShape::D_MODEL && c_.d_ff_exp == MoeShape::D_FF && c_.n_expert_used == MoeShape::TOPK
                        && c_.swiglu_limit == MoeShape::SWIGLU_LIMIT,
                    "deepseek41: routed-expert shape differs from moe::DsV41");
    for (int l = 0; l < c_.n_layer; ++l)
        if (c_.engram.index(l) >= 0)
            engine::require(w_.layers[l].engram.table != nullptr,
                            "deepseek41: no Engram table for layer " + std::to_string(l)
                                + " (add ds41-engram-L" + std::to_string(l) + ".gguf to TRUSS_EXPERT_OVERLAY)");
}

std::unique_ptr<engine::StageModel> Family::stage(int lb, int le, int n_ctx) const
{
    const Config & c = c_;
    for (int l = lb; l < le; ++l) {   // sources and consumers on one stage (blocks/attn_ds41.h)
        const int ks = c.kv_source_of(l), is = c.index_source_of(l);
        engine::require((ks < 0 || ks >= lb) && (is < 0 || is >= lb),
                        "deepseek41: stage [" + std::to_string(lb) + ", " + std::to_string(le)
                            + ") starts after an attention source its layers read; split at 2, 8, 14 or 20");
    }
    return std::make_unique<StageModel>(*this, lb, le, n_ctx);
}

StageModel::StageModel(const Family & fam, int lb, int le, int n_ctx)
    : c_(fam.config()), w_(fam.weights()), lb_(lb), le_(le), n_ctx_(n_ctx)
{
}

StageModel::~StageModel() = default;

hc::SkDims StageModel::sk_dims() const
{
    hc::SkDims k { c_.hc, c_.d_model, c_.rms_eps, c_.hc_eps, c_.sinkhorn_iters };
    k.res_ld = c_.hc * c_.d_model + k.n_mix();
    k.coef_ld = k.n_mix();   // mix() writes a separate [T][24]; collapse / post take their coef_ld per call
    return k;
}

blocks::Ds41AttnDims StageModel::attn_dims(int l) const
{
    const int ratio = (int) c_.compress_ratio[l];
    const bool uses_cand = ratio > 0 && c_.candidate_source >= 0 && l > c_.candidate_source && c_.is_index_source(l);
    return { c_.d_model, c_.n_head, c_.head_dim, c_.rope_dim, c_.q_rank, c_.o_rank, c_.o_groups, c_.window,
             c_.index_heads, c_.index_dim, c_.index_top_k, c_.candidate_top_k_blocks, c_.candidate_block, ratio,
             ratio > 0 && c_.is_kv_source(l), ratio > 0 && c_.is_index_source(l), l == c_.candidate_source, uses_cand,
             c_.rms_eps, ratio ? inv_freq_comp_ : inv_freq_plain_ };
}

void StageModel::add_tensors(engine::Stage & st)
{
    blocks::Linear & lin = st.lin();
    if (lb_ == 0) lin.add(w_.token_embd);
    if (le_ == c_.n_layer) lin.add(w_.final_norm), lin.add(w_.output);
    for (int l = lb_; l < le_; ++l) {
        const Layer & L = w_.layers[l];
        lin.add(L.attn_norm), lin.add(L.ffn_norm);
        blocks::HcSinkhorn::add_tensors(lin, L.hc_attn);
        blocks::HcSinkhorn::add_tensors(lin, L.hc_ffn);
        blocks::AttnDs41::add_tensors(lin, L.attn);
        blocks::Moe::add_tensors(lin, L.moe);
        if (L.engram.wkv) blocks::Engram::add_tensors(lin, L.engram);
    }
}

std::vector<runtime::MoeLayer> StageModel::moe_layers() const
{
    std::vector<runtime::MoeLayer> out;
    for (int l = lb_; l < le_; ++l) {
        const blocks::MoeWeights & m = w_.layers[l].moe;
        out.push_back({ { &m.gate, &m.up, &m.down }, l, true, l + 1 < le_, true });   // CPU tier allowed (#191)
    }
    return out;
}

size_t StageModel::scratch_bytes(int T) const
{
    const size_t d = c_.d_model, rw = (size_t) c_.hc * d + 24;
    size_t attn = 0, fixed = 0;
    for (int l = lb_; l < le_; ++l) {
        const auto ad = attn_dims(l);
        attn = std::max(attn, blocks::AttnDs41::scratch_per_row(ad));
        fixed = std::max(fixed, blocks::AttnDs41::scratch_fixed(ad, n_ctx_));
    }
    blocks::MoeDims md { c_.d_model, c_.n_expert, c_.n_expert_used, c_.d_ff_exp * c_.n_shexp };
    const blocks::EngramDims ed { c_.hc, c_.d_model, c_.engram.heads, c_.engram.head_dim, c_.engram.max_ngram, 0, 0,
                                  nullptr, nullptr, nullptr, nullptr, 0 };
    const size_t base = d * (4 + 4 + 2) + 2 * 24 * 4 + 64;   // collapse out fp32, sub-block out, normed fp16, coefs
    const size_t per_row = base + std::max({ attn, md.scratch_per_row(), blocks::Engram::scratch_per_row(ed),
                                             blocks::HcSinkhorn::scratch_per_row(sk_dims()) });
    (void) rw;
    return (size_t) T * per_row + fixed + (64ull << 20);
}

void StageModel::build(engine::Stage & st)
{
    blocks::Linear & lin = st.lin();
    engine::Gpu & g = st.gpu();
    auto table = [&] (bool comp) {
        const auto v = ds41::rope_inv_freq(c_.rope_dim, comp ? c_.compress_rope_base : c_.rope_base,
                                           comp ? c_.rope_ctx_orig : 0, c_.rope_factor, c_.beta_fast, c_.beta_slow);
        float * p = g.alloc<float>(v.size());
        TRUSS_CUDA(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        return p;
    };
    inv_freq_plain_ = table(false), inv_freq_comp_ = table(true);
    const hc::SkDims k = sk_dims();
    for (int l = lb_; l < le_; ++l) {
        const Layer & L = w_.layers[l];
        attn_norm_.push_back(std::make_unique<blocks::RmsNorm>(lin, L.attn_norm, c_.d_model, c_.rms_eps));
        ffn_norm_.push_back(std::make_unique<blocks::RmsNorm>(lin, L.ffn_norm, c_.d_model, c_.rms_eps));
        hc_attn_.push_back(std::make_unique<blocks::HcSinkhorn>(lin, L.hc_attn, k));
        hc_ffn_.push_back(std::make_unique<blocks::HcSinkhorn>(lin, L.hc_ffn, k));
        attn_.push_back(std::make_unique<blocks::AttnDs41>(lin, L.attn, attn_dims(l)));
        const int ei = c_.engram.index(l);
        if (ei >= 0) {
            const auto & e = c_.engram;
            const int B = e.buckets();
            const blocks::EngramDims ed { c_.hc, c_.d_model, e.heads, e.head_dim, e.max_ngram, e.pad_id, c_.rms_eps,
                                          e.multipliers.data() + (size_t) ei * e.max_ngram,
                                          e.primes.data() + (size_t) ei * B, e.offsets.data() + (size_t) ei * B,
                                          e.token_map.data(), (int64_t) e.token_map.size() };
            engram_.push_back(std::make_unique<blocks::Engram>(lin, L.engram, ed, *L.engram.table));
            engram_.back()->build(g);
        } else {
            engram_.push_back(nullptr);
        }
    }
    for (int l = lb_; l < le_; ++l) {   // attention sources -> consumers (same stage, Family::stage checked)
        auto at = [&] (int s) -> const blocks::AttnDs41 * { return s < 0 || s == l ? nullptr : attn_[s - lb_].get(); };
        const auto ad = attn_dims(l);
        attn_[l - lb_]->set_sources(ad.ratio ? at(c_.kv_source_of(l)) : nullptr, ad.ratio ? at(c_.index_source_of(l)) : nullptr,
                                    ad.uses_candidates ? at(c_.candidate_source) : nullptr);
    }
    if (le_ == c_.n_layer) {
        final_norm_ = std::make_unique<blocks::RmsNorm>(lin, w_.final_norm, c_.d_model, c_.rms_eps);
        head_hc_ = std::make_unique<blocks::HcSinkhorn>(lin, blocks::SkHcWeights {}, k);
        head_ = std::make_unique<blocks::Head>(g, lin, w_.output);
    }
}

void StageModel::bind_experts(engine::Stage & st)
{
    blocks::MoeDims md { c_.d_model, c_.n_expert, c_.n_expert_used, c_.d_ff_exp * c_.n_shexp };
    md.swiglu_limit = c_.swiglu_limit;
    md.route_scale = c_.expert_scale;
    for (int l = lb_; l < le_; ++l) {
        const blocks::MoeWeights * next = l + 1 < le_ ? &w_.layers[l + 1].moe : nullptr;
        moe_.push_back(std::make_unique<blocks::Moe>(st.lin(), *st.moe(), l - lb_, l, w_.layers[l].moe, md, next,
                                                     &blocks::route_sqrtsoftplus_bias, false, &st.prof()));
    }
}

void StageModel::embed(engine::Stage & st, const engine::Chunk & c, float * res)
{
    engine::Gpu & g = st.gpu();
    Scratch & sc = *g.sc;
    int * ids = sc.alloc<int>(c.T);
    float * emb = sc.alloc((size_t) c.T * c_.d_model);
    TRUSS_CUDA(cudaMemcpyAsync(ids, c.tokens, (size_t) c.T * 4, cudaMemcpyHostToDevice, g.s));
    dense::q8_rows(st.lin().q8(w_.token_embd), ids, c.T, emb, g.s);
    hc::SkDims k = sk_dims();
    k.coef_ld = k.res_ld;
    hc::sk_expand(emb, c.T, k, res, res + (size_t) c_.hc * c_.d_model, g.s);
}

// TRUSS_DS41_DUMP=<dir> (parity debugging, tests/layer/ds41_stage_dump.cu): append T x n fp32 rows of a layer's
// tensor to <dir>/<name>_L<nn>.f32 (synchronous; off unless the variable is set)
static void dump(engine::Gpu & g, const char * name, int l, const float * p, int T, int n, int ld)
{
    static const char * dir = std::getenv("TRUSS_DS41_DUMP");
    if (!dir) return;
    std::vector<float> h((size_t) T * n);
    TRUSS_CUDA(cudaMemcpy2DAsync(h.data(), (size_t) n * 4, p, (size_t) ld * 4, (size_t) n * 4, T, cudaMemcpyDeviceToHost, g.s));
    g.sync();
    char path[512];
    std::snprintf(path, sizeof path, "%s/%s_L%02d.f32", dir, name, l);
    if (FILE * f = std::fopen(path, "ab")) std::fwrite(h.data(), 4, h.size(), f), std::fclose(f);
}

void StageModel::layers(engine::Stage & st, const engine::Chunk & c, float * res, const engine::LayerHook & hook)
{
    engine::Gpu & g = st.gpu();
    Scratch & sc = *g.sc;
    const int T = c.T, d = c_.d_model, W = c_.hc * d + 24, NM = 24;
    float * row_coef = res + (size_t) c_.hc * d;   // each row's next-collapse coef (W floats apart)
    float * A = sc.alloc((size_t) T * NM), * F = sc.alloc((size_t) T * NM);
    const size_t base = sc.mark();
    for (int l = lb_; l < le_; ++l) {
        sc.release(base);
        const int i = l - lb_;
        float * y = sc.alloc((size_t) T * d), * x = sc.alloc((size_t) T * d), * out = sc.alloc((size_t) T * d);
        half * x16 = sc.alloc<half>((size_t) T * d);
        st.prof().record(l, 0, g.s);
        dump(g, "res_in", l, res, T, c_.hc * d + NM, W);
        if (engram_[i]) engram_[i]->forward(g, c.slot, c.tokens, c.pos0, T, res, W), dump(g, "res_eng", l, res, T, c_.hc * d + NM, W);
        // attention
        hc_attn_[i]->mix(g, res, T, A);
        hc_attn_[i]->collapse(g, res, row_coef, W, T, y, nullptr);
        attn_norm_[i]->forward(g, y, T, x, x16);
        st.prof().record(l, 1, g.s);
        dump(g, "attn_in", l, x, T, d, d);
        attn_[i]->forward(g, c.slot, x, x16, c.pos0, T, out);
        dump(g, "attn_out", l, out, T, d, d);
        st.prof().record(l, 2, g.s);
        hc_attn_[i]->post(g, res, out, A, NM, T);
        // FFN
        hc_ffn_[i]->mix(g, res, T, F);
        hc_ffn_[i]->collapse(g, res, A, NM, T, y, nullptr);
        ffn_norm_[i]->forward(g, y, T, x, x16);   // fp32 for the router, fp16 for the projections
        dump(g, "moe_in", l, x, T, d, d);
        moe_[i]->forward(g, x, x16, T, out, c.stream, st.moe_ws(), st.moe_rows());
        dump(g, "moe_out", l, out, T, d, d);
        hc_ffn_[i]->post(g, res, out, F, NM, T);
        TRUSS_CUDA(cudaMemcpy2DAsync(row_coef, (size_t) W * 4, F, (size_t) NM * 4, (size_t) NM * 4, T,
                                     cudaMemcpyDeviceToDevice, g.s));
        st.prof().record(l, 6, g.s);
        if (hook) {
            g.sync();
            hook(l, res, T);
        }
    }
}

void StageModel::head(engine::Stage & st, const float * rows, int n, float * logits, bool draft)
{
    engine::Gpu & g = st.gpu();
    Scratch & sc = *g.sc;
    const size_t m = sc.mark();
    const int d = c_.d_model, W = c_.hc * d + 24;
    float * y = sc.alloc((size_t) n * d);
    half * x16 = sc.alloc<half>((size_t) n * d);
    head_hc_->collapse(g, rows, rows + (size_t) c_.hc * d, W, n, y, nullptr);
    final_norm_->forward(g, y, n, nullptr, x16);
    head_->logits(x16, n, logits, draft);
    sc.release(m);
}

std::vector<engine::Stateful *> StageModel::states()
{
    std::vector<engine::Stateful *> out;
    for (size_t i = 0; i < attn_.size(); ++i) {
        if (engram_[i]) out.push_back(engram_[i].get());
        out.push_back(attn_[i].get());
    }
    return out;
}

}  // namespace truss::deepseek41
