// qwen4exp (Qwen3.8 Flash-Next) as an engine family: the composition of its blocks (docs/ENGINE.md).
//
// Per layer: [PLE] -> hc mix -> GDN or DSA mixer -> hc combine -> hc mix -> MoE (routed trellis experts + gated shared
// expert, softmax top-10 of 512) -> hc combine; the token embedding expanded into hc streams before, the head's hc mix
// + output projection after; the MTP block (mtp.h) drafts on the last stage. A layer's closing combine is folded into
// the next layer's first norm (one launch fewer per layer) except before a PLE layer, with a hook, or at the stage's
// end. Dense projections run Q8_0 (blocks::Linear), the router and the shared-expert gate fp32.
#pragma once
#include "blocks/attn_dsa.h"
#include "blocks/gdn.h"
#include "blocks/hc.h"
#include "blocks/head.h"
#include "blocks/moe.h"
#include "blocks/ple.h"
#include "engine/model.h"
#include "kernels/moe/moe_weights.cuh"
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/weights.h"

#include <memory>
#include <vector>

namespace truss::qwen4exp {

using DsaShape = dsa::FlashNext;
using MoeShape = moe::FlashNext;

class MtpDrafter;

class Family : public engine::Family {
public:
    Family(const gguf::File & f, const engine::DrafterFiles & d);
    ~Family() override;
    int n_layer() const override { return c_.n_layer; }
    int n_vocab() const override { return c_.n_vocab; }
    int n_expert() const override { return c_.n_expert; }
    int res_width() const override { return c_.hc_dim(); }
    int chunk_align() const override { return DsaShape::RATIO; }
    bool has_drafter() const override { return mtp_ != nullptr; }
    std::unique_ptr<engine::StageModel> stage(int lb, int le, int n_ctx) const override;

    const Config & config() const { return c_; }
    const Weights & weights() const { return w_; }
    const Mtp * mtp() const { return mtp_.get(); }
    const gguf::File & file() const { return f_; }
    const engine::DrafterFiles & drafter_files() const { return d_; }

private:
    const gguf::File & f_;
    Config c_;
    Weights w_;
    engine::DrafterFiles d_;
    std::unique_ptr<gguf::File> mtp_file_;
    std::unique_ptr<Mtp> mtp_;
};

class StageModel : public engine::StageModel {
public:
    StageModel(const Family & fam, int lb, int le, int n_ctx);
    ~StageModel() override;
    void add_tensors(engine::Stage & st) override;
    const gguf::Tensor * host_rows() const override;
    std::vector<const gguf::Tensor *> no_w16() const override { return { w_.token_embd, w_.output }; }
    std::vector<runtime::MoeLayer> moe_layers() const override;
    runtime::MoeKernels moe_kernels() const override { return runtime::MoeKernels::of<MoeShape>(); }
    size_t scratch_bytes(int T) const override;
    int extra_buffers() const override { return mtp_ ? 2 : 0; }   // the MTP block's input rows and its residual
    void build(engine::Stage & st) override;
    void bind_experts(engine::Stage & st) override;
    void embed(engine::Stage & st, const engine::Chunk & c, float * res) override;
    void layers(engine::Stage & st, const engine::Chunk & c, float * res, const engine::LayerHook & hook) override;
    void head(engine::Stage & st, const float * rows, int n, float * logits, bool draft) override;
    engine::Drafter * drafter() override;
    std::vector<engine::Stateful *> states() override;
    void table_stats(double & ms, long & calls, long & rows, bool reset, double * wait_ms, double * phase_ms) override
    {
        if (ple_table_) ple_table_->stats(ms, calls, rows, reset, wait_ms, phase_ms);
    }

    // for the drafter (same stage, same blocks)
    blocks::Hc & hc() { return *hc_; }
    blocks::Head & out_head() { return *head_; }

private:
    const Family & fam_;
    const Config & c_;
    const Weights & w_;
    const Mtp * mtp_;
    const int lb_, le_, n_ctx_;
    std::unique_ptr<blocks::Hc> hc_;
    std::unique_ptr<blocks::PleTable> ple_table_;
    std::vector<std::unique_ptr<blocks::Gdn>> gdn_;               // [l - lb] (null on DSA layers)
    std::vector<std::unique_ptr<blocks::Dsa<DsaShape>>> dsa_;     // [l - lb] (null on GDN layers)
    std::vector<std::unique_ptr<blocks::Ple>> ple_;               // [l - lb] (null off PLE layers)
    std::vector<std::unique_ptr<blocks::Moe>> moe_;               // [l - lb]
    std::unique_ptr<blocks::Head> head_;
    std::unique_ptr<MtpDrafter> drafter_;
};

}  // namespace truss::qwen4exp
