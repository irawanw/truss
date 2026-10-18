// deepseek41 (DeepSeek-V4.1-Flash-1017 as packed by tools/tk-pack/ds41_dense.py + ds41_experts.py): hyperparameters
// from GGUF metadata, keys as mmnga/Strata `deepseek` src/deepseek41/model.cpp reads them (plus the source-layer
// lists and candidate settings that file infers from tensor presence). Math reference: dsref/model.py.
//
// Per layer: mHC (hc copies) around an attention and a MoE FFN. Attention = sliding window of `window` raw KV, plus,
// where compress_ratio[l] > 0, compressed KV shared by layer groups (the source is the last kv_source layer at or
// before l) and indexer top-k (the last index_source layer at or before l; candidate blocks from candidate_source).
// Engram at engram_layers.
#pragma once
#include "formats/gguf.h"

#include <cstdint>
#include <vector>

namespace truss::deepseek41 {

struct Engram {
    int heads, head_dim, max_ngram, pad_id;
    std::vector<int64_t> layers, multipliers, primes, offsets, token_map, table_rows;
    int buckets() const { return (max_ngram - 1) * heads; }   // rows per token per engram layer (24)
    int index(int layer) const;                               // position in `layers`, -1: no engram there
};

struct Config {
    int n_layer, d_model, n_vocab, n_ctx_train;
    int n_head, head_dim, rope_dim, q_rank, o_rank, o_groups, window;
    int n_expert, n_expert_used, d_ff_exp, n_shexp;
    float expert_scale, swiglu_limit;
    int hc, sinkhorn_iters;
    float hc_eps, rms_eps;
    int index_heads, index_dim, index_top_k;
    int candidate_source, candidate_top_k_blocks, candidate_block;
    float rope_base, compress_rope_base, rope_factor, beta_fast, beta_slow;
    int rope_ctx_orig;
    std::vector<int64_t> compress_ratio, kv_sources, index_sources, expert_keep;   // keep: [n_layer][n_expert]
    Engram engram;

    bool is_kv_source(int l) const;
    bool is_index_source(int l) const;
    int kv_source_of(int l) const;      // the layer whose compressed KV l reads, -1 if compress_ratio[l] == 0
    int index_source_of(int l) const;   // the layer whose top-k l uses, -1 if compress_ratio[l] == 0

    static Config from_gguf(const gguf::File & f);   // throws on a missing or inconsistent key
};

}  // namespace truss::deepseek41
