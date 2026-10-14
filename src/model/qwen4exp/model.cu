#include "model/qwen4exp/model.h"

#include "engine/stage.h"
#include "kernels/ffn/ffn_ops.cuh"
#include "model/qwen4exp/mtp.h"

#include <algorithm>

namespace truss::qwen4exp {

using engine::require;

Family::Family(const gguf::File & f, const engine::DrafterFiles & d)
    : f_(f), c_(Config::from_gguf(f)), w_(bind(f, c_)), d_(d)
{
    require(c_.d_model == MoeShape::D_MODEL && c_.d_ff_exp == MoeShape::D_FF && c_.n_expert_used == MoeShape::TOPK,
            "routed-expert shape differs from moe::FlashNext");
    require(c_.n_head == DsaShape::H && c_.n_head_kv == DsaShape::HKV && c_.head_dim == DsaShape::D &&
                c_.idx_heads == DsaShape::IH && c_.idx_head_dim == DsaShape::ID && c_.rope_dims == DsaShape::ROPE_DIMS &&
                c_.idx_top_k == DsaShape::TOP_BLOCKS * DsaShape::RATIO,
            "DSA shape differs from dsa::FlashNext");
    for (int l = 0; l < c_.n_layer; ++l)
        require(c_.mixer[l] != Mixer::DSA || c_.compress_ratio[l] == DsaShape::RATIO, "DSA ratio differs");
    if (!d_.path.empty()) {
        mtp_file_ = gguf::File::open(d_.path);
        mtp_ = std::make_unique<Mtp>(bind_mtp(*mtp_file_, c_));
    }
}

Family::~Family() = default;

std::unique_ptr<engine::StageModel> Family::stage(int lb, int le, int n_ctx) const
{
    return std::make_unique<StageModel>(*this, lb, le, n_ctx);
}

StageModel::StageModel(const Family & fam, int lb, int le, int n_ctx)
    : fam_(fam), c_(fam.config()), w_(fam.weights()), mtp_(le == fam.config().n_layer ? fam.mtp() : nullptr), lb_(lb),
      le_(le), n_ctx_(n_ctx)
{
}

StageModel::~StageModel() = default;

void StageModel::add_tensors(engine::Stage & st)
{
    blocks::Linear & lin = st.lin();
    auto add_hc = [&](const HyperConnection & h) { lin.add(h.norm), lin.add(h.down), lin.add(h.up), lin.add(h.inject); };
    // a split stage: its own layers; the token embedding where it is read (the first stage, the MTP block), the head
    // on the last stage
    if (lb_ == 0 || mtp_) lin.add(w_.token_embd);
    if (le_ == c_.n_layer) lin.add(w_.output), add_hc(w_.hc_head);
    std::vector<const Layer *> layers;
    for (int l = lb_; l < le_; ++l) layers.push_back(&w_.layers[l]);
    if (mtp_) {
        layers.push_back(&mtp_->layer);
        lin.add(mtp_->eh_proj), lin.add(mtp_->enorm), lin.add(mtp_->hnorm);
    }
    for (const Layer * L : layers) {
        add_hc(L->hc_attn), add_hc(L->hc_ffn);
        blocks::Gdn::add_tensors(lin, L->gdn);
        blocks::Dsa<DsaShape>::add_tensors(lin, L->dsa);
        blocks::Ple::add_tensors(lin, L->ple);
        blocks::Moe::add_tensors(lin, L->moe);
    }
}

// The token embedding is read one row per token: in pinned host memory, gathered over PCIe (2.5 KB per token),
// leaving its 0.68 GB of VRAM to the expert cache (TRACKER #60). Tied to the output: stays on the device.
const gguf::Tensor * StageModel::host_rows() const
{
    return w_.token_embd != w_.output && (lb_ == 0 || mtp_) ? w_.token_embd : nullptr;
}

std::vector<runtime::MoeLayer> StageModel::moe_layers() const
{
    std::vector<runtime::MoeLayer> out;
    for (int l = lb_; l < le_; ++l) {
        const Moe & m = w_.layers[l].moe;
        out.push_back({ { &m.gate, &m.up, &m.down }, l, true, l + 1 < le_, true });
    }
    if (mtp_) {
        const Moe & m = mtp_->layer.moe;
        out.push_back({ { &m.gate, &m.up, &m.down }, c_.n_layer, false, false, false });
    }
    return out;
}

// peak per-layer temporaries: the layer's own buffers plus the largest of hc mix / PLE / GDN / DSA / FFN (with the
// MTP block, its join and the layer's own peak are live together: added)
size_t StageModel::scratch_bytes(int T) const
{
    const size_t hcd = c_.hc_dim(), dm = c_.d_model;
    const size_t base = dm * 4 + 4 + dm * (4 + 4 + 2) + c_.hc * 4;   // embedding + ids, then the layer's own
    const size_t mix = hcd * (2 + 4) + c_.hc * 4 + c_.hc_rank * (4 + 2);
    const size_t ple = blocks::Ple::scratch_per_row(c_.ple_dims());
    const blocks::GdnDims gd{ c_.d_model, c_.ssm_groups, c_.ssm_v_heads, c_.ssm_state, c_.ssm_conv, c_.rms_eps };
    const size_t gdn = gd.scratch_per_row();
    const size_t dsa = blocks::Dsa<DsaShape>::scratch_per_row();
    const size_t ffn = blocks::MoeDims{ c_.d_model, c_.n_expert, c_.n_expert_used, c_.d_ff_shexp }.scratch_per_row();
    const size_t mtp = hcd * 2 + c_.hc * 4 + (size_t) c_.hc * 2 * dm * (2 + 1) + (size_t) c_.hc * 2 * dm / 16;
    const size_t q8_act = 2 * (hcd + hcd / 16);   // Q8_1: a shared quantized input + one per-call input
    const size_t per_token = base + q8_act + std::max({ mix, ple, gdn, dsa, ffn }) + (fam_.has_drafter() ? mtp : 0);
    const size_t fixed = blocks::Dsa<DsaShape>::scratch_fixed(T, n_ctx_) + (64ull << 20);   // + alignment slack
    return (size_t) T * per_token + fixed;
}

void StageModel::build(engine::Stage & st)
{
    blocks::Linear & lin = st.lin();
    engine::Gpu & g = st.gpu();
    hc_ = std::make_unique<blocks::Hc>(g, lin, blocks::HcDims{ c_.hc, c_.hc_rank, c_.d_model, c_.rms_eps });
    bool any_ple = false;
    for (int l = lb_; l < le_; ++l) any_ple |= c_.is_ple(l);
    if (any_ple) {
        ple_table_ = std::make_unique<blocks::PleTable>(c_.ple_dims(), w_.ple_table, w_.ple_scale,
                                                        st.opts().direct_tables ? &fam_.file() : nullptr, st.max_rows());
    }
    const blocks::GdnDims gd{ c_.d_model, c_.ssm_groups, c_.ssm_v_heads, c_.ssm_state, c_.ssm_conv, c_.rms_eps };
    const blocks::DsaDims dd{ c_.d_model, c_.rope_base, c_.rms_eps };
    for (int l = lb_; l < le_; ++l) {
        const Layer & L = w_.layers[l];
        gdn_.push_back(L.mixer == Mixer::GDN ? std::make_unique<blocks::Gdn>(lin, L.gdn, gd) : nullptr);
        dsa_.push_back(L.mixer == Mixer::DSA ? std::make_unique<blocks::Dsa<DsaShape>>(lin, L.dsa, dd, st.opts().kv_int8)
                                             : nullptr);
        ple_.push_back(c_.is_ple(l) ? std::make_unique<blocks::Ple>(lin, L.ple, *ple_table_) : nullptr);
    }
    if (le_ == c_.n_layer) head_ = std::make_unique<blocks::Head>(g, lin, w_.output);
    if (mtp_) drafter_ = std::make_unique<MtpDrafter>(st, *this, fam_);
}

void StageModel::bind_experts(engine::Stage & st)
{
    const blocks::MoeDims md{ c_.d_model, c_.n_expert, c_.n_expert_used, c_.d_ff_shexp };
    for (int l = lb_; l < le_; ++l)
        moe_.push_back(std::make_unique<blocks::Moe>(st.lin(), *st.moe(), l - lb_, l, w_.layers[l].moe, md,
                                                     l + 1 < le_ ? &w_.layers[l + 1].moe : nullptr,
                                                     &blocks::route_softmax, true, &st.prof()));
    if (drafter_) drafter_->bind_experts(st);
}

void StageModel::embed(engine::Stage & st, const engine::Chunk & c, float * res)
{
    engine::Gpu & g = st.gpu();
    int * ids = g.sc->alloc<int>(c.T);
    float * emb = g.sc->alloc((size_t) c.T * c_.d_model);
    TRUSS_CUDA(cudaMemcpyAsync(ids, c.tokens, (size_t) c.T * 4, cudaMemcpyHostToDevice, g.s));
    dense::q8_rows(st.lin().q8(w_.token_embd), ids, c.T, emb, g.s);
    hc_->expand(emb, c.T, res);
}

void StageModel::layers(engine::Stage & st, const engine::Chunk & c, float * res, const engine::LayerHook & hook)
{
    engine::Gpu & g = st.gpu();
    Scratch & sc = *g.sc;
    const int T = c.T;
    if (ple_table_) ple_table_->set_next(c.next, c.next_T);
    const size_t base = sc.mark();
    // a layer's closing combine is folded into the next layer's first norm: out and inject sit at the same scratch
    // offsets every layer and are read by that first kernel before anything rewrites them
    bool pend_combine = false;
    for (int l = lb_; l < le_; ++l) {
        sc.release(base);
        const Layer & L = w_.layers[l];
        const int i = l - lb_;
        float * mixed = sc.alloc((size_t) T * c_.d_model), * out = sc.alloc((size_t) T * c_.d_model);
        float * inject = sc.alloc((size_t) T * c_.hc);
        half * mixed16 = sc.alloc<half>((size_t) T * c_.d_model);
        st.prof().record(l, 0, g.s);
        if (ple_[i]) ple_[i]->forward(g, c.slot, c.tokens, T, c.pos0, res, c.tentative);
        hc_->mix(L.hc_attn, res, T, mixed, mixed16, inject, pend_combine ? out : nullptr, pend_combine ? inject : nullptr);
        pend_combine = false;
        st.prof().record(l, 1, g.s);
        if (gdn_[i]) gdn_[i]->forward(g, c.slot, mixed16, T, out, c.tentative);
        else dsa_[i]->forward(g, c.slot, mixed16, c.pos0, T, out, c.tentative);
        st.prof().record(l, 2, g.s);
        hc_->mix(L.hc_ffn, res, T, mixed, mixed16, inject, out, inject);   // combine #1 fused into the norm (L3)
        moe_[i]->forward(g, mixed, mixed16, T, out, c.stream, st.moe_ws(), st.moe_rows());
        st.prof().record(l, 6, g.s);
        if (!hook && l + 1 < le_ && !c_.is_ple(l + 1)) pend_combine = true;
        else hc_->combine(res, out, inject, T);
        if (hook) {
            g.sync();
            hook(l, res, T);
        }
    }
}

// logits [n][vocab] of residual rows [n][hc][d]: the head hc mix, then the output projection
void StageModel::head(engine::Stage & st, const float * rows, int n, float * logits, bool draft)
{
    Scratch & sc = *st.gpu().sc;
    const size_t m = sc.mark();
    float * mixed = sc.alloc((size_t) n * c_.d_model);
    half * mixed16 = sc.alloc<half>((size_t) n * c_.d_model);
    hc_->mix(w_.hc_head, rows, n, mixed, mixed16, nullptr);
    head_->logits(mixed16, n, logits, draft);
    sc.release(m);
}

engine::Drafter * StageModel::drafter() { return drafter_.get(); }

std::vector<engine::Stateful *> StageModel::states()
{
    std::vector<engine::Stateful *> out;
    for (size_t i = 0; i < gdn_.size(); ++i) {
        if (gdn_[i]) out.push_back(gdn_[i].get());
        if (dsa_[i]) out.push_back(dsa_[i].get());
        if (ple_[i]) out.push_back(ple_[i].get());
    }
    if (ple_table_) out.push_back(ple_table_.get());
    return out;
}

}  // namespace truss::qwen4exp
