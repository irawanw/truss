#include "model/deepseek41/config.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace truss::deepseek41 {

namespace {
bool contains(const std::vector<int64_t> & v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }
int last_at_or_before(const std::vector<int64_t> & v, int l)
{
    int best = -1;
    for (int64_t s : v)
        if (s <= l) best = std::max(best, (int) s);
    return best;
}
}  // namespace

int Engram::index(int layer) const
{
    const auto it = std::find(layers.begin(), layers.end(), layer);
    return it == layers.end() ? -1 : (int) (it - layers.begin());
}

bool Config::is_kv_source(int l) const { return contains(kv_sources, l); }
bool Config::is_index_source(int l) const { return contains(index_sources, l); }
int Config::kv_source_of(int l) const { return compress_ratio[l] ? last_at_or_before(kv_sources, l) : -1; }
int Config::index_source_of(int l) const { return compress_ratio[l] ? last_at_or_before(index_sources, l) : -1; }

Config Config::from_gguf(const gguf::File & f)
{
    const std::string arch = f.get_string("general.architecture");
    if (arch != "deepseek41") throw std::runtime_error("deepseek41: file architecture is " + arch);
    auto K = [](const char * k) { return "deepseek41." + std::string(k); };
    auto I = [&](const char * k) { return (int) f.get_int(K(k)); };
    auto R = [&](const char * k) { return (float) f.get_float(K(k)); };
    auto fail = [](const std::string & why) { throw std::runtime_error("deepseek41 config: " + why); };
    Config c;
    c.n_layer = I("block_count");
    c.d_model = I("embedding_length");
    c.n_ctx_train = I("context_length");
    c.n_vocab = (int) f.at("token_embd.weight").shape.at(1);
    c.n_head = I("attention.head_count");
    c.head_dim = I("attention.key_length");
    c.rope_dim = I("rope.dimension_count");
    c.q_rank = I("attention.q_lora_rank");
    c.o_rank = I("attention.output_lora_rank");
    c.o_groups = I("attention.output_group_count");
    c.window = I("attention.sliding_window");
    c.n_expert = I("expert_count");
    c.n_expert_used = I("expert_used_count");
    c.d_ff_exp = I("expert_feed_forward_length");
    c.n_shexp = I("expert_shared_count");
    c.expert_scale = R("expert_weights_scale");
    c.hc = I("hyper_connection.count");
    c.sinkhorn_iters = I("hyper_connection.sinkhorn_iterations");
    c.hc_eps = R("hyper_connection.epsilon");
    c.rms_eps = R("attention.layer_norm_rms_epsilon");
    c.index_heads = I("attention.indexer.head_count");
    c.index_dim = I("attention.indexer.key_length");
    c.index_top_k = I("attention.indexer.top_k");
    c.candidate_source = I("attention.candidate_source_layer");
    c.candidate_top_k_blocks = I("attention.candidate_top_k_blocks");
    c.candidate_block = I("attention.candidate_block_size");
    c.rope_base = R("rope.freq_base");
    c.compress_rope_base = R("attention.compress_rope_freq_base");
    c.rope_factor = R("rope.scaling.factor");
    c.beta_fast = R("rope.scaling.yarn_beta_fast");
    c.beta_slow = R("rope.scaling.yarn_beta_slow");
    c.rope_ctx_orig = I("rope.scaling.original_context_length");
    c.compress_ratio = f.get_ints(K("attention.compress_ratios"));
    c.kv_sources = f.get_ints(K("attention.kv_source_layers"));
    c.index_sources = f.get_ints(K("attention.index_source_layers"));
    c.expert_keep = f.get_ints(K("expert_keep"));
    const auto & clamp = f.get_floats(K("swiglu_clamp_exp"));
    auto & e = c.engram;
    e.heads = I("engram.head_count");
    e.head_dim = I("engram.key_length");
    e.max_ngram = I("engram.max_ngram_size");
    e.pad_id = I("engram.pad_id");
    e.layers = f.get_ints(K("engram.layer_ids"));
    e.multipliers = f.get_ints(K("engram.multipliers"));
    e.primes = f.get_ints(K("engram.primes"));
    e.offsets = f.get_ints(K("engram.offsets"));
    e.token_map = f.get_ints(K("engram.token_map"));
    e.table_rows = f.get_ints(K("engram.table_rows"));

    if (f.get_string(K("rope.scaling.type")) != "yarn") fail("rope.scaling.type is not yarn");
    if (f.get_int(K("expert_gating_func")) != 4) fail("expert_gating_func is not sqrtsoftplus (4)");
    if (f.get_int(K("attention.head_count_kv")) != 1) fail("attention.head_count_kv != 1");
    if (f.get_int(K("attention.value_length")) != c.head_dim) fail("key/value lengths differ");
    if ((int) c.compress_ratio.size() != c.n_layer) fail("compress_ratios length");
    for (int64_t r : c.compress_ratio)
        if (r < 0 || r > 2) fail("compress ratio not 0, 1, 2");
    if ((int) clamp.size() != c.n_layer) fail("swiglu_clamp_exp length");
    for (double v : clamp)
        if (v != clamp[0]) fail("per-layer SwiGLU clamps differ (the kernel shape has one)");
    c.swiglu_limit = (float) clamp[0];
    if ((int64_t) c.expert_keep.size() != (int64_t) c.n_layer * c.n_expert) fail("expert_keep length");
    if (c.n_head % c.o_groups || c.rope_dim % 2 || c.rope_dim > c.head_dim) fail("attention dimensions");
    for (int l = 0; l < c.n_layer; ++l) {
        if (!c.compress_ratio[l]) continue;
        const int kv = c.kv_source_of(l), ix = c.index_source_of(l);
        if (kv < 0 || ix < 0) fail("layer " + std::to_string(l) + " reads compressed state before a source");
        if (c.compress_ratio[kv] != c.compress_ratio[l]) fail("layer " + std::to_string(l) + ": ratio != its source's");
    }
    const size_t nb = e.layers.size() * (size_t) e.buckets();
    if (e.multipliers.size() != e.layers.size() * (size_t) e.max_ngram || e.primes.size() != nb ||
        e.offsets.size() != nb || (int) e.token_map.size() != c.n_vocab || e.table_rows.size() != e.layers.size())
        fail("engram hash layout sizes");
    return c;
}

}  // namespace truss::deepseek41
