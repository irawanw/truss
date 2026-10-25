// deepseek41 as an engine family: the composition of its blocks (docs/ENGINE.md; math dsref/model.py Transformer).
//
// Residual row = hc copies of d_model, then the (2 + hc) hc coef of the next collapse (res_width = hc d + 24): the
// mix a sub-block computes feeds the NEXT sub-block's collapse, and keeping it in the row carries it across a layer
// split like the residual itself. Per layer:
//   [Engram]  res += gated n-gram lookup (layers 1, 14)
//   attn      A = mix_attn(res); y = collapse(res, row coef); res = post_A(res, attn(attn_norm(y)))
//   ffn       F = mix_ffn(res);  y = collapse(res, A);        res = post_F(res, moe(ffn_norm(y)));  row coef = F
// Embedding rows into every copy with a one-hot coef; head = collapse(res, row coef) -> norm -> output (Q8_0).
// MoE: sqrt(softplus) + bias top-6 of the pack's 256 experts, renormalized, x1.5; ungated shared expert; SwiGLU
// clamp 10 (moe::DsV41). Cold experts run on the CPU trellis tier (shape-generic since #181, on since #191)
// or stream by PCIe.
// A stage boundary must fall on an attention source layer (blocks/attn_ds41.h).
// Image input (TRUSS_MMPROJ = ds41-mmproj-f16.gguf, docs/reference/model-deepseek41.md "Vision"): the prompt's image
// span rows (all the placeholder id) take the vision tower's rows as embeddings (Chunk::x.img), route with the
// image router bias (blk.L.bias_vl) and get no Engram lookup; positions stay 1-D.
#pragma once
#include "blocks/attn_ds41.h"
#include "blocks/engram.h"
#include "blocks/hc_sinkhorn.h"
#include "blocks/head.h"
#include "blocks/moe.h"
#include "blocks/norm.h"
#include "engine/model.h"
#include "kernels/moe/moe_weights.cuh"
#include "model/deepseek41/config.h"
#include "model/deepseek41/dspark.h"
#include "model/deepseek41/weights.h"

#include <memory>
#include <vector>

namespace truss::deepseek41 {

using MoeShape = moe::DsV41;

class Family : public engine::Family {
public:
    Family(const gguf::File & f, const engine::DrafterFiles & d);
    int n_layer() const override { return c_.n_layer; }
    int n_vocab() const override { return c_.n_vocab; }
    int n_expert() const override { return c_.n_expert; }
    int res_width() const override { return c_.hc * c_.d_model + (2 + c_.hc) * c_.hc; }
    std::unique_ptr<engine::StageModel> stage(int lb, int le, int n_ctx) const override;
    const Config & config() const { return c_; }
    const Weights & weights() const { return w_; }
    bool has_drafter() const override { return dspark_c_ != nullptr; }
    const DSparkConfig * dspark_config() const { return dspark_c_.get(); }
    const DSparkWeights * dspark_weights() const { return dspark_w_.get(); }
    float drafter_min_p() const { return drafter_min_p_; }
    void release_drafter_pages() const { if (dspark_file_) dspark_file_->release_pages(); }
    // image input (TRUSS_MMPROJ set): the placeholder id of image-span rows (-1: text only) and the image router bias
    // of layer l over the pack's experts (blk.L.bias_vl renumbered by expert_keep)
    int32_t image_token() const { return image_token_; }
    const float * bias_vl(int l) const { return bias_vl_.empty() ? nullptr : bias_vl_.data() + (size_t) l * c_.n_expert; }

private:
    Config c_;
    Weights w_;
    std::unique_ptr<gguf::File> dspark_file_;
    std::unique_ptr<DSparkConfig> dspark_c_;
    std::unique_ptr<DSparkWeights> dspark_w_;
    float drafter_min_p_ = 0.f;
    int32_t image_token_ = -1;
    std::vector<float> bias_vl_;   // [n_layer][n_expert]
};

class StageModel : public engine::StageModel {
public:
    StageModel(const Family & fam, int lb, int le, int n_ctx);
    ~StageModel() override;
    void add_tensors(engine::Stage & st) override;
    std::vector<const gguf::Tensor *> no_w16() const override { return { w_.token_embd, w_.output }; }
    const gguf::Tensor * host_rows() const override
    {
        return fam_.has_drafter() && le_ == c_.n_layer && lb_ != 0 ? w_.token_embd : nullptr;
    }
    engine::Drafter * drafter() override { return dspark_.get(); }
    std::vector<runtime::MoeLayer> moe_layers() const override;
    runtime::MoeKernels moe_kernels() const override { return runtime::MoeKernels::of<MoeShape>(); }
    size_t scratch_bytes(int T, bool decode) const override;
    void build(engine::Stage & st) override;
    void bind_experts(engine::Stage & st) override;
    void embed(engine::Stage & st, const engine::Chunk & c, float * res) override;
    void layers(engine::Stage & st, const engine::Chunk & c, float * res, const engine::LayerHook & hook) override;
    void head(engine::Stage & st, const float * rows, int n, float * logits, bool draft) override;
    std::vector<engine::Stateful *> states() override;

private:
    blocks::Ds41AttnDims attn_dims(int l) const;
    hc::SkDims sk_dims() const;
    const Family & fam_;
    const Config & c_;
    const Weights & w_;
    const int lb_, le_, n_ctx_;
    float * inv_freq_plain_ = nullptr, * inv_freq_comp_ = nullptr;   // device rope tables (window-only / compressing)
    std::vector<std::unique_ptr<blocks::RmsNorm>> attn_norm_, ffn_norm_;    // [l - lb]
    std::vector<std::unique_ptr<blocks::HcSinkhorn>> hc_attn_, hc_ffn_;
    std::vector<std::unique_ptr<blocks::AttnDs41>> attn_;
    std::vector<std::unique_ptr<blocks::Engram>> engram_;                  // null off the Engram layers
    std::vector<std::unique_ptr<blocks::Moe>> moe_;
    std::unique_ptr<blocks::RmsNorm> final_norm_;
    std::unique_ptr<blocks::HcSinkhorn> head_hc_;                           // collapse only (no weights read)
    std::unique_ptr<blocks::Head> head_;
    std::unique_ptr<DSpark> dspark_;
};

// [begin, end) runs of image-span rows (token == image) in tokens[0 .. n), flattened; empty when image < 0
std::vector<int> image_runs(const int32_t * tokens, int n, int32_t image);

}  // namespace truss::deepseek41
