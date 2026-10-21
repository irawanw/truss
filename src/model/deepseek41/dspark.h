// DeepSeek-V4.1-Flash DSpark drafter (mtp.0-2) as an engine::Drafter on the last stage.
//
// Math: dsref/model.py DSparkBlock / DSparkAttention / forward_spec. Three 5120-wide layers, each a window-only
// attention (128-slot ring, noncausal over the window + the 5 draft rows) and a 128-expert top-3 MoE + shared
// expert, then the main model's output head with the markov bias. The target's hidden at the input of layers
// 37/38/39 (mean over the hc copies) is projected by main_proj/layer 0's main_norm into the drafter's main_x.
//
// The per-slot ring stores the target KV latents (one per DSpark layer) of the committed positions; on_commit and
// on_accept seed it from the captured target hidden. draft() is one 5-row block forward from the anchor token and
// four noise tokens, positions pos .. pos+4.
#pragma once
#include "blocks/hc_sinkhorn.h"
#include "blocks/moe.h"
#include "blocks/norm.h"
#include "blocks/weights.h"
#include "engine/model.h"
#include "model/deepseek41/config.h"
#include "model/deepseek41/weights.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace truss::deepseek41 {

struct DSparkConfig {
    int n_layer = 0, d_model = 0, n_head = 0, head_dim = 0, rope_dim = 0, q_rank = 0, o_rank = 0, o_groups = 0;
    int window = 0, n_expert = 0, n_expert_used = 0, d_ff = 0, markov_rank = 0, noise_token = 0, block_size = 0;
    int hc = 0, n_shexp = 1, n_vocab = 0;
    float rms_eps = 0, hc_eps = 0, rope_base = 0, route_scale = 1.f, swiglu_limit = 0.f;
    std::vector<int> target_layers;
    static DSparkConfig from_gguf(const gguf::File & f, const Config & c);
};

struct DSparkWeights {
    struct Layer {
        T attn_norm = nullptr, ffn_norm = nullptr;
        blocks::SkHcWeights hc_attn, hc_ffn;
        blocks::Ds41AttnWeights attn;
        blocks::MoeWeights moe;
        T bias_vl = nullptr;
        T main_proj = nullptr, main_norm = nullptr;   // layer 0
        T norm = nullptr, markov_embed = nullptr, markov_head = nullptr, confidence_proj = nullptr;   // last layer
    };
    std::vector<Layer> layers;
};

DSparkWeights bind_dspark(const gguf::File & f, const Config & c, const DSparkConfig & d);

struct DSparkAttnDims {
    int d_model, n_head, head_dim, rope_dim, q_rank, o_rank, o_groups, window;
    float eps;
    const float * inv_freq;
};

// Window-only DSpark attention: q/kv projections, rope, noncausal attention over the ring + this chunk's kv,
// inverse rope, grouped output LoRA. The ring holds one DSpark layer's target KV latents.
class DSparkAttn {
public:
    DSparkAttn(blocks::Linear & lin, const blocks::Ds41AttnWeights & w, const DSparkAttnDims & d)
        : lin_(lin), w_(w), d_(d)
    {
    }
    static void add_tensors(blocks::Linear & lin, const blocks::Ds41AttnWeights & w);
    void alloc(engine::Gpu & g, int n_slots);
    half * ring(int slot) const { return sl_[slot]; }
    // main_x16 [n][d] at positions pos0 .. -> this layer's ring
    void seed(engine::Gpu & g, int slot, const half * main_x16, int n, int pos0);
    // x16 [T][d] at positions pos0 .. (draft rows) -> out [T][d]; reads the ring
    void forward(engine::Gpu & g, int slot, const half * x16, int pos0, int T, float * out);

private:
    blocks::Linear & lin_;
    const blocks::Ds41AttnWeights w_;
    const DSparkAttnDims d_;
    std::vector<half *> sl_;
    half * comb_ = nullptr;   // [window + MAX_DRAFT][D]
    int * idx_ = nullptr;     // [MAX_DRAFT][window + MAX_DRAFT]
};

class DSpark : public engine::Drafter, public engine::Stateful {
public:
    static constexpr int RING = 128;
    static constexpr int MAX_DRAFT = 8;
    static constexpr int CAP_ROWS = 128;   // target hiddens kept per slot (the window size)

    DSpark(engine::Stage & st, const Config & c, const Weights & w, const DSparkConfig & d, const DSparkWeights & dw,
           float min_p);
    ~DSpark() override;

    static void add_tensors(blocks::Linear & lin, const DSparkWeights & w);

    // captured target hidden of layers 37/38/39 (device kernel; called from StageModel::layers before the layer)
    void capture(engine::Gpu & g, int slot, int layer, const float * res, int W, int T, int row_off = 0);

    // Drafter
    void on_commit(engine::Stage & st, const engine::Chunk & c) override;
    void on_accept(engine::Stage & st, int slot, int n, const engine::Window & w) override;
    void flush(engine::Stage &, int) override {}
    void park(engine::Stage &, int) override {}
    void unpark(engine::Stage &, int) override {}
    int draft(engine::Stage & st, int slot, int32_t next, int pos, int n, int32_t * out, const Sampling & s) override;
    // the per-draft confidence p_i of the last draft call (the acceptance probability of draft i)
    const float * draft_confidence(int * n) const override { *n = (int) conf_.size(); return conf_.data(); }
    engine::Stateful * state() override { return this; }

    // Stateful
    void alloc_slots(engine::Stage & st, int n_slots, int seq_ctx, int spec_rows) override;
    void reset(engine::Stage & st, int slot) override;
    void ck_buffers(int slot, std::vector<std::pair<float *, size_t>> & out) override;

private:
    struct Slot {
        float * cap = nullptr;  // [CAP_ROWS][3][d_model], fp32: some rows exceed the fp16 range
        int cap_n = 0;          // rows captured
        int cap_start = 0;      // the chunk row the capture starts at (T > CAP_ROWS: the tail)
        bool seeded = false;
    };
    void seed(engine::Stage & st, int slot, const float * hid, int n, int pos0);
    hc::SkDims sk_dims() const;

    const Config & c_;
    const Weights & w_;
    const DSparkConfig d_;
    const DSparkWeights dw_;
    blocks::Linear & lin_;
    engine::Gpu & g_;
    float min_p_ = 0.f;
    float * inv_freq_ = nullptr;
    std::unique_ptr<runtime::MoeExec> exec_;
    std::vector<std::unique_ptr<blocks::RmsNorm>> attn_norm_, ffn_norm_;
    std::vector<std::unique_ptr<blocks::HcSinkhorn>> hc_attn_, hc_ffn_;
    std::vector<std::unique_ptr<DSparkAttn>> attn_;
    std::vector<std::unique_ptr<blocks::Moe>> moe_;
    std::unique_ptr<blocks::RmsNorm> last_norm_;
    std::vector<Slot> sl_;
    float * mproj_ = nullptr, * mx_ = nullptr, * kv_ = nullptr, * logits_ = nullptr, * me_ = nullptr, * mbias_ = nullptr;
    half * mx16_ = nullptr, * kv16_ = nullptr, * m16_ = nullptr;
    half * hid16_ = nullptr;   // [CAP_ROWS][3][d_model]: seed()'s fp16 rows for main_proj (own memory, not scratch)
    // the confidence head: the [hidden; markov_emb] input (fp32), the logits, the host p_i
    float * xconf_ = nullptr, * conf_logit_ = nullptr;
    std::vector<float> conf_;
    int * ids_ = nullptr, * draft_ids_ = nullptr;
    int32_t * ids_host_ = nullptr, * draft_host_ = nullptr;
    float * draft_prob_ = nullptr, * qfull_ = nullptr;
    float * emb_ = nullptr, * res_ = nullptr, * y_ = nullptr, * x_ = nullptr, * out_ = nullptr;
    float * coef_pre_ = nullptr, * coef_a_ = nullptr, * coef_f_ = nullptr;
    half * x16_ = nullptr, * y16_ = nullptr;
};

}  // namespace truss::deepseek41
