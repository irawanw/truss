// lfm2moe (LiquidAI LFM2 / LFM2.5 MoE, e.g. LFM2.5-24B-A2B): hyperparameters from GGUF metadata (llama.cpp's
// convert_hf_to_gguf LFM2MoeModel keys). Per layer the operator is a short convolution or GQA attention (attention
// where attention.head_count_kv[l] > 0); the first leading_dense_block_count layers have a dense SwiGLU FFN, the rest
// a routed MoE with a sigmoid + expert-bias router (expert_gating_func 2).
#pragma once
#include "formats/gguf.h"

#include <cstdint>
#include <vector>

namespace truss::lfm2moe {

struct Config {
    int n_layer, d_model, n_vocab, n_ctx_train;
    float rms_eps;
    int n_head, head_dim;
    std::vector<int64_t> n_head_kv;   // [n_layer]: 0 on short-conv layers
    float rope_base;
    int conv_L;                       // short conv kernel length (l_cache)
    int d_ff;                         // dense FFN (leading layers)
    int n_dense_lead;
    int n_expert, n_expert_used, d_ff_exp;

    bool is_attn(int l) const { return n_head_kv[l] > 0; }
    bool is_moe(int l) const { return l >= n_dense_lead; }
    int kv_heads() const;             // the attention layers' KV head count (all equal)

    static Config from_gguf(const gguf::File & f);   // throws on a missing or inconsistent key
};

}  // namespace truss::lfm2moe
