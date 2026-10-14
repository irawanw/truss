#include "model/lfm2moe/config.h"

#include <stdexcept>
#include <string>
#include <variant>

namespace truss::lfm2moe {

namespace {
std::vector<int64_t> int_list(const gguf::File & f, const std::string & key, int n)
{
    const gguf::Value * v = f.get(key);
    if (v && std::holds_alternative<int64_t>(*v)) return std::vector<int64_t>(n, std::get<int64_t>(*v));
    return f.get_ints(key);
}
}  // namespace

int Config::kv_heads() const
{
    for (int64_t h : n_head_kv)
        if (h > 0) return (int) h;
    return 0;
}

Config Config::from_gguf(const gguf::File & f)
{
    const std::string arch = f.get_string("general.architecture");
    if (arch != "lfm2moe") throw std::runtime_error("lfm2moe: file architecture is " + arch);
    auto I = [&](const char * k) { return (int) f.get_int("lfm2moe." + std::string(k)); };
    auto fail = [](const std::string & why) { throw std::runtime_error("lfm2moe config: " + why); };
    Config c;
    c.n_layer = I("block_count");
    c.d_model = I("embedding_length");
    c.n_ctx_train = I("context_length");
    c.rms_eps = (float) f.get_float("lfm2moe.attention.layer_norm_rms_epsilon");
    c.n_vocab = (int) f.at("token_embd.weight").shape.at(1);
    c.n_head = I("attention.head_count");
    c.head_dim = c.d_model / c.n_head;
    c.n_head_kv = int_list(f, "lfm2moe.attention.head_count_kv", c.n_layer);
    c.rope_base = (float) f.get_float("lfm2moe.rope.freq_base");
    c.conv_L = I("shortconv.l_cache");
    c.d_ff = I("feed_forward_length");
    c.n_dense_lead = I("leading_dense_block_count");
    c.n_expert = I("expert_count");
    c.n_expert_used = I("expert_used_count");
    c.d_ff_exp = I("expert_feed_forward_length");
    if ((int) c.n_head_kv.size() != c.n_layer) fail("head_count_kv has " + std::to_string(c.n_head_kv.size()));
    for (int64_t h : c.n_head_kv)
        if (h != 0 && h != c.kv_heads()) fail("attention layers differ in KV heads");
    if (c.head_dim * c.n_head != c.d_model) fail("head_count does not divide embedding_length");
    if (f.get_int("lfm2moe.expert_gating_func") != 2) fail("expert_gating_func is not sigmoid (2)");
    return c;
}

}  // namespace truss::lfm2moe
