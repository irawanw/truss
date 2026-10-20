#include "model/lfm2moe/model.h"

#include "engine/stage.h"
#include "kernels/attn/gqa.cuh"
#include "kernels/dense/q8_gemm.cuh"
#include "kernels/ffn/ffn_ops.cuh"

#include <algorithm>

namespace truss::lfm2moe {

Family::Family(const gguf::File & f, const engine::DrafterFiles & d) : c_(Config::from_gguf(f)), w_(bind(f, c_))
{
    engine::require(d.path.empty(), "lfm2moe: no speculative drafter for this family yet");
    engine::require(c_.d_model == MoeShape::D_MODEL && c_.d_ff_exp == MoeShape::D_FF && c_.n_expert_used == MoeShape::TOPK,
                    "lfm2moe: routed-expert shape differs from moe::Lfm2");
}

std::unique_ptr<engine::StageModel> Family::stage(int lb, int le, int n_ctx) const
{
    return std::make_unique<StageModel>(*this, lb, le, n_ctx);
}

StageModel::StageModel(const Family & fam, int lb, int le, int n_ctx)
    : c_(fam.config()), w_(fam.weights()), lb_(lb), le_(le), n_ctx_(n_ctx)
{
    int s = 0;
    for (int l = lb; l < le; ++l) sl_.push_back(c_.is_moe(l) ? s++ : -1);
}

StageModel::~StageModel() = default;

attn::GqaShape StageModel::gqa_shape() const
{
    return { c_.n_head, c_.kv_heads(), c_.head_dim, c_.rope_base, c_.rms_eps };
}

void StageModel::add_tensors(engine::Stage & st)
{
    blocks::Linear & lin = st.lin();
    lin.add(w_.token_embd);   // the embedding rows and the tied head
    if (le_ == c_.n_layer) lin.add(w_.final_norm);
    for (int l = lb_; l < le_; ++l) {
        const Layer & L = w_.layers[l];
        lin.add(L.op_norm), lin.add(L.ffn_norm);
        if (c_.is_attn(l)) blocks::AttnGqa::add_tensors(lin, L.attn);
        else blocks::ShortConv::add_tensors(lin, L.conv);
        if (c_.is_moe(l)) blocks::Moe::add_tensors(lin, L.moe);
        else blocks::DenseFfn::add_tensors(lin, L.dense);
    }
}

std::vector<runtime::MoeLayer> StageModel::moe_layers() const
{
    std::vector<runtime::MoeLayer> out;
    for (int l = lb_; l < le_; ++l)
        if (c_.is_moe(l)) {
            const blocks::MoeWeights & m = w_.layers[l].moe;
            out.push_back({ { &m.gate, &m.up, &m.down }, l, true, l + 1 < le_ && c_.is_moe(l + 1), true });
        }
    return out;
}

size_t StageModel::scratch_bytes(int T, bool) const
{
    const size_t d = c_.d_model;
    const size_t base = d * (4 + 4 + 4 + 2) + 4;   // two op / ffn outputs, normed rows fp32 + fp16, ids
    const size_t moe = blocks::MoeDims{ c_.d_model, c_.n_expert, c_.n_expert_used, 0 }.scratch_per_row();
    const size_t per_row = base + std::max({ blocks::ShortConv::scratch_per_row(c_.d_model),
                                             blocks::AttnGqa::scratch_per_row(gqa_shape(), c_.d_model),
                                             blocks::DenseFfn::scratch_per_row(c_.d_model, c_.d_ff), moe });
    return (size_t) T * per_row + blocks::AttnGqa::scratch_fixed(gqa_shape(), T, n_ctx_) + (64ull << 20);   // + slack
}

void StageModel::build(engine::Stage & st)
{
    blocks::Linear & lin = st.lin();
    for (int l = lb_; l < le_; ++l) {
        const Layer & L = w_.layers[l];
        op_norm_.push_back(std::make_unique<blocks::RmsNorm>(lin, L.op_norm, c_.d_model, c_.rms_eps));
        ffn_norm_.push_back(std::make_unique<blocks::RmsNorm>(lin, L.ffn_norm, c_.d_model, c_.rms_eps));
        attn_.push_back(c_.is_attn(l) ? std::make_unique<blocks::AttnGqa>(lin, L.attn, gqa_shape(), c_.d_model) : nullptr);
        conv_.push_back(c_.is_attn(l) ? nullptr : std::make_unique<blocks::ShortConv>(lin, L.conv, c_.d_model, c_.conv_L));
        dense_.push_back(c_.is_moe(l) ? nullptr : std::make_unique<blocks::DenseFfn>(lin, L.dense, c_.d_model, c_.d_ff));
    }
    if (le_ == c_.n_layer) {
        final_norm_ = std::make_unique<blocks::RmsNorm>(lin, w_.final_norm, c_.d_model, c_.rms_eps);
        head_ = std::make_unique<blocks::Head>(st.gpu(), lin, w_.token_embd);
    }
}

void StageModel::bind_experts(engine::Stage & st)
{
    const blocks::MoeDims md{ c_.d_model, c_.n_expert, c_.n_expert_used, 0 };
    for (int l = lb_; l < le_; ++l) {
        const int i = l - lb_;
        if (sl_[i] < 0) {
            moe_.push_back(nullptr);
            continue;
        }
        const blocks::MoeWeights * next = l + 1 < le_ && c_.is_moe(l + 1) ? &w_.layers[l + 1].moe : nullptr;
        moe_.push_back(std::make_unique<blocks::Moe>(st.lin(), *st.moe(), sl_[i], l, w_.layers[l].moe, md, next,
                                                     &blocks::route_sigmoid_bias, false, &st.prof()));
    }
}

void StageModel::embed(engine::Stage & st, const engine::Chunk & c, float * res)
{
    engine::Gpu & g = st.gpu();
    int * ids = g.sc->alloc<int>(c.T);
    TRUSS_CUDA(cudaMemcpyAsync(ids, c.tokens, (size_t) c.T * 4, cudaMemcpyHostToDevice, g.s));
    dense::q8_rows(st.lin().q8(w_.token_embd), ids, c.T, res, g.s);
}

void StageModel::layers(engine::Stage & st, const engine::Chunk & c, float * res, const engine::LayerHook & hook)
{
    engine::Gpu & g = st.gpu();
    Scratch & sc = *g.sc;
    const int T = c.T, d = c_.d_model;
    // residual adds fused into the next RMSNorm (norm::add_rms): the op output into ffn_norm, the FFN output into the
    // next layer's op_norm (plain add at the stage's last layer and when a hook reads res). Outputs alternate between
    // two buffers so a pending one survives the next layer's scratch.
    float * outs[2] = { sc.alloc((size_t) T * d), sc.alloc((size_t) T * d) };
    // B2: one rope table per chunk for all attention layers; the per-lane fp64 trig is identical to prepare's
    // inline computation, so the layers read bit-identical angles. Allocated before the per-layer mark: the loop's
    // release(base) must not free it. Even the largest prefill chunk needs only T * 256 B (1 MB at T = 4096).
    float2 * tab = sc.alloc<float2>((size_t) T * 32);
    attn::gqa_rope_table(c_.rope_base, c.pos0, T, tab, g.s);
    const size_t base = sc.mark();
    const float * pend = nullptr;
    for (int l = lb_; l < le_; ++l) {
        sc.release(base);
        const int i = l - lb_;
        float * x = sc.alloc((size_t) T * d), * out = outs[i & 1];
        half * x16 = sc.alloc<half>((size_t) T * d);
        st.prof().record(l, 0, g.s);
        op_norm_[i]->forward_add(g, res, pend, T, nullptr, x16);
        st.prof().record(l, 1, g.s);
        if (attn_[i]) attn_[i]->forward(g, c.slot, x16, c.pos0, T, out, tab);
        else conv_[i]->forward(g, c.slot, x16, T, out, c.tentative);
        st.prof().record(l, 2, g.s);
        ffn_norm_[i]->forward_add(g, res, out, T, x, x16);   // fp32 for the router, fp16 for the projections
        if (moe_[i]) moe_[i]->forward(g, x, x16, T, out, c.stream, st.moe_ws(), st.moe_rows());
        else dense_[i]->forward(g, x16, T, out);
        pend = out;
        if (hook || l + 1 == le_) ffn::add(res, out, T * d, g.s), pend = nullptr;
        st.prof().record(l, 6, g.s);
        if (hook) {
            g.sync();
            hook(l, res, T);
        }
    }
}

void StageModel::head(engine::Stage & st, const float * rows, int n, float * logits, bool draft)
{
    Scratch & sc = *st.gpu().sc;
    const size_t m = sc.mark();
    half * x16 = sc.alloc<half>((size_t) n * c_.d_model);
    final_norm_->forward(st.gpu(), rows, n, nullptr, x16);
    head_->logits(x16, n, logits, draft);
    sc.release(m);
}

std::vector<engine::Stateful *> StageModel::states()
{
    std::vector<engine::Stateful *> out;
    for (size_t i = 0; i < conv_.size(); ++i) {
        if (conv_[i]) out.push_back(conv_[i].get());
        if (attn_[i]) out.push_back(attn_[i].get());
    }
    return out;
}

// B1: GQA prepare (arg 6 of 11) and attention (arg 4 of 9) take pos0 by value; decode splits change every 128
// cells, so a capture is optimal for its own 128-block (stale splits stay correct, TRACKER #168)
engine::StageModel::GraphInfo StageModel::graph_info() const
{
    const attn::GqaKernelFns kf = attn::gqa_kernel_fns();
    engine::StageModel::GraphInfo gi;
    gi.pos_funcs = { { kf.prepare, 6, 12 }, { kf.rope_table, 1, 4 }, { kf.decode, 4, 9 }, { kf.attn, 4, 9 } };
    gi.epoch = [](int pos) { return pos / 128; };
    return gi;
}

}  // namespace truss::lfm2moe
