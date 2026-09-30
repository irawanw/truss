// qwen4exp (Qwen3.8 Flash-Next): hyperparameters from GGUF metadata. Per layer the token mixer is GDN (linear
// attention) or DSA (full attention with a sparse indexer); every layer has hyper-connections around the mixer and
// the MoE; a few layers add the n-gram PLE block.
#pragma once
#include "formats/gguf.h"

#include <cstdint>
#include <vector>

namespace truss::qwen4exp {

enum class Mixer { GDN, DSA };

struct Config {
    int n_layer, d_model, n_vocab, n_ctx_train;
    float rms_eps;
    std::vector<Mixer> mixer;              // [n_layer]

    // DSA layers
    int n_head, n_head_kv, head_dim;       // q projection holds [q | gate] per head: 2 * n_head * head_dim rows
    int rope_dims;
    float rope_base;
    std::vector<int64_t> rope_sections;    // imrope
    std::vector<int64_t> compress_ratio;   // [n_layer], 0 on GDN layers
    int idx_heads, idx_head_dim, idx_top_k;

    // GDN layers
    int ssm_conv, ssm_state, ssm_groups, ssm_v_heads;
    int key_dim() const { return ssm_state * ssm_groups; }
    int value_dim() const { return ssm_state * ssm_v_heads; }
    int conv_dim() const { return 2 * key_dim() + value_dim(); }

    // MoE
    int n_expert, n_expert_used, d_ff_exp, d_ff_shexp;

    // hyper-connections: hc parallel residual streams of d_model, mixed through a rank-hc_rank projection
    int hc, hc_rank;
    int hc_dim() const { return hc * d_model; }

    // PLE: per-layer n-gram embedding table
    std::vector<int64_t> ple_layers;
    int ple_head_dim, ple_conv, ple_ngram, ple_heads_per_ngram;
    std::vector<int64_t> ple_head_offsets, ple_head_vocab;
    std::vector<int64_t> ple_multipliers;  // n-gram hash: one per window position (ple_ngram)
    int64_t ple_eos;                       // resets the n-gram window (not the tokenizer's EOS)
    int ple_heads() const { return (ple_ngram - 1) * ple_heads_per_ngram; }
    bool is_ple(int layer) const;

    // Throws if the architecture is not qwen4exp or a key is missing or inconsistent.
    static Config from_gguf(const gguf::File & f);
};

}  // namespace truss::qwen4exp
