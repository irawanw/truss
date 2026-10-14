// lfm2moe as an engine family: the composition of its blocks (docs/ENGINE.md).
//
// Per layer: res += op(op_norm(res)) with op = ShortConv or AttnGqa; res += ffn(ffn_norm(res)) with ffn = DenseFfn
// (leading layers) or Moe (sigmoid + expert-bias top-4 of 64, no shared expert). Embedding rows in, final norm +
// tied output head out. Dense projections Q8_0 (blocks::Linear), router fp32, routed experts trellis.
#pragma once
#include "blocks/attn_gqa.h"
#include "blocks/ffn_dense.h"
#include "blocks/head.h"
#include "blocks/moe.h"
#include "blocks/norm.h"
#include "blocks/shortconv.h"
#include "engine/model.h"
#include "kernels/moe/moe_weights.cuh"
#include "model/lfm2moe/config.h"
#include "model/lfm2moe/weights.h"

#include <memory>
#include <vector>

namespace truss::lfm2moe {

using MoeShape = moe::Lfm2;

class Family : public engine::Family {
public:
    Family(const gguf::File & f, const engine::DrafterFiles & d);
    int n_layer() const override { return c_.n_layer; }
    int n_vocab() const override { return c_.n_vocab; }
    int n_expert() const override { return c_.n_expert; }
    int res_width() const override { return c_.d_model; }
    std::unique_ptr<engine::StageModel> stage(int lb, int le, int n_ctx) const override;
    const Config & config() const { return c_; }
    const Weights & weights() const { return w_; }

private:
    Config c_;
    Weights w_;
};

class StageModel : public engine::StageModel {
public:
    StageModel(const Family & fam, int lb, int le, int n_ctx);
    ~StageModel() override;
    void add_tensors(engine::Stage & st) override;
    std::vector<const gguf::Tensor *> no_w16() const override { return { w_.token_embd }; }
    std::vector<runtime::MoeLayer> moe_layers() const override;
    runtime::MoeKernels moe_kernels() const override { return runtime::MoeKernels::of<MoeShape>(); }
    size_t scratch_bytes(int T) const override;
    void build(engine::Stage & st) override;
    void bind_experts(engine::Stage & st) override;
    void embed(engine::Stage & st, const engine::Chunk & c, float * res) override;
    void layers(engine::Stage & st, const engine::Chunk & c, float * res, const engine::LayerHook & hook) override;
    void head(engine::Stage & st, const float * rows, int n, float * logits, bool draft) override;
    std::vector<engine::Stateful *> states() override;

private:
    attn::GqaShape gqa_shape() const;
    const Config & c_;
    const Weights & w_;
    const int lb_, le_, n_ctx_;
    std::vector<int> sl_;   // [l - lb] the layer's store layer in the expert runtime (-1: dense FFN)
    std::vector<std::unique_ptr<blocks::RmsNorm>> op_norm_, ffn_norm_;   // [l - lb]
    std::vector<std::unique_ptr<blocks::ShortConv>> conv_;               // [l - lb] (null on attention layers)
    std::vector<std::unique_ptr<blocks::AttnGqa>> attn_;                 // [l - lb] (null on conv layers)
    std::vector<std::unique_ptr<blocks::DenseFfn>> dense_;               // [l - lb] (null on MoE layers)
    std::vector<std::unique_ptr<blocks::Moe>> moe_;                      // [l - lb] (null on dense layers)
    std::unique_ptr<blocks::RmsNorm> final_norm_;
    std::unique_ptr<blocks::Head> head_;
};

}  // namespace truss::lfm2moe
