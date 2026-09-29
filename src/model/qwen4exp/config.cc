#include "model/qwen4exp/config.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace truss::qwen4exp {

namespace {

// a key written as a scalar by one converter and as a one-element array by another
std::vector<int64_t> int_list(const gguf::File & f, const std::string & key)
{
    const gguf::Value * v = f.get(key);
    if (v && std::holds_alternative<int64_t>(*v)) return { std::get<int64_t>(*v) };
    return f.get_ints(key);
}

}  // namespace

bool Config::is_ple(int layer) const
{
    return std::find(ple_layers.begin(), ple_layers.end(), layer) != ple_layers.end();
}

Config Config::from_gguf(const gguf::File & f)
{
    const std::string arch = f.get_string("general.architecture");
    if (arch != "qwen4exp") throw std::runtime_error("qwen4exp: file architecture is " + arch);
    auto I = [&] (const char * k) { return (int) f.get_int("qwen4exp." + std::string(k)); };
    auto fail = [] (const std::string & why) { throw std::runtime_error("qwen4exp config: " + why); };

    Config c;
    c.n_layer = I("block_count");
    c.d_model = I("embedding_length");
    c.n_ctx_train = I("context_length");
    c.rms_eps = (float) f.get_float("qwen4exp.attention.layer_norm_rms_epsilon");
    c.n_vocab = (int) f.at("token_embd.weight").shape.at(1);

    const auto & recurrent = f.get_bools("qwen4exp.attention.recurrent_layers");
    if ((int) recurrent.size() != c.n_layer) fail("recurrent_layers has " + std::to_string(recurrent.size()));
    for (bool r : recurrent) c.mixer.push_back(r ? Mixer::GDN : Mixer::DSA);

    c.n_head = I("attention.head_count");
    c.n_head_kv = I("attention.head_count_kv");
    c.head_dim = I("attention.key_length");
    if (I("attention.value_length") != c.head_dim) fail("key_length != value_length");
    c.rope_dims = I("rope.dimension_count");
    c.rope_base = (float) f.get_float("qwen4exp.rope.freq_base");
    c.rope_sections = f.get_ints("qwen4exp.rope.dimension_sections");
    c.compress_ratio = f.get_ints("qwen4exp.attention.compress_ratios");
    if ((int) c.compress_ratio.size() != c.n_layer) fail("compress_ratios length");
    c.idx_heads = I("attention.indexer.head_count");
    c.idx_head_dim = I("attention.indexer.key_length");
    c.idx_top_k = I("attention.indexer.top_k");

    c.ssm_conv = I("ssm.conv_kernel");
    c.ssm_state = I("ssm.state_size");
    c.ssm_groups = I("ssm.group_count");
    c.ssm_v_heads = I("ssm.time_step_rank");
    if (I("ssm.inner_size") != c.value_dim()) fail("ssm.inner_size != state_size * time_step_rank");

    c.n_expert = I("expert_count");
    c.n_expert_used = I("expert_used_count");
    c.d_ff_exp = I("expert_feed_forward_length");
    c.d_ff_shexp = I("expert_shared_feed_forward_length");

    c.hc = I("hyper_connection.count");
    c.hc_rank = I("hyper_connection.low_rank");

    c.ple_layers = int_list(f, "qwen4exp.ple.layers");
    c.ple_head_dim = I("embedding_length_per_layer_input");
    c.ple_conv = I("ple.conv_kernel");
    c.ple_ngram = I("ple.ngram_size");
    c.ple_heads_per_ngram = I("ple.heads_per_ngram");
    c.ple_head_offsets = f.get_ints("qwen4exp.ple.head_offsets");
    c.ple_head_vocab = f.get_ints("qwen4exp.ple.head_vocab_sizes");
    if (c.ple_head_offsets.size() != c.ple_head_vocab.size()) fail("PLE head offsets / sizes differ in length");
    return c;
}

}  // namespace truss::qwen4exp
