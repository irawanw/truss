// DSpark drafter config and tensor binding (dspark.<i>.*, tools/tk-pack/ds41_dspark.py + ds41_experts.py overlay).
#include "model/deepseek41/dspark.h"

#include "formats/binder.h"

#include <string>

namespace truss::deepseek41 {

DSparkConfig DSparkConfig::from_gguf(const gguf::File & f, const Config & c)
{
    auto K = [](const char * k) { return "deepseek41." + std::string(k); };
    auto I = [&](const char * k) { return (int) f.get_int(K(k)); };
    auto R = [&](const char * k) { return (float) f.get_float(K(k)); };
    DSparkConfig d;
    d.n_layer = I("dspark.n_layer");
    d.block_size = I("dspark.block_size");
    d.window = I("dspark.window");
    d.markov_rank = I("dspark.markov_rank");
    d.noise_token = I("dspark.noise_token");
    d.n_expert = I("dspark.n_expert");
    d.n_expert_used = I("dspark.n_expert_used");
    d.d_model = I("embedding_length");
    d.n_head = I("attention.head_count");
    d.head_dim = I("attention.key_length");
    d.rope_dim = I("rope.dimension_count");
    d.q_rank = I("attention.q_lora_rank");
    d.o_rank = I("attention.output_lora_rank");
    d.o_groups = I("attention.output_group_count");
    d.d_ff = I("expert_feed_forward_length");
    d.hc = I("hyper_connection.count");
    d.rms_eps = R("attention.layer_norm_rms_epsilon");
    d.hc_eps = R("hyper_connection.epsilon");
    d.rope_base = R("rope.freq_base");
    const auto & tl = f.get_ints(K("dspark.target_layers"));
    d.target_layers.assign(tl.begin(), tl.end());
    d.n_vocab = c.n_vocab;
    d.route_scale = c.expert_scale;
    d.swiglu_limit = c.swiglu_limit;
    if (d.n_layer < 1 || d.block_size < 1 || d.block_size > 8) throw std::runtime_error("dspark: bad block size");
    if (d.window < 1 || d.window > 128) throw std::runtime_error("dspark: window must be 1..128");
    if (d.n_expert != 128 || d.n_expert_used != 3)
        throw std::runtime_error("dspark: this drafter expects 128 experts top-3");
    if ((int) d.target_layers.size() != 3) throw std::runtime_error("dspark: expected 3 target layers");
    return d;
}

DSparkWeights bind_dspark(const gguf::File & f, const Config & c, const DSparkConfig & d)
{
    formats::Binder b(f, "deepseek41.dspark");
    DSparkWeights w;
    const int64_t H = d.d_model, A = d.n_head, D = d.head_dim, Q = d.q_rank, G = d.o_groups, R = d.o_rank,
                  HC = d.hc, E = d.n_expert, F = d.d_ff, C = (2 + HC) * HC;
    for (int i = 0; i < d.n_layer; ++i) {
        const std::string p = "dspark." + std::to_string(i) + ".";
        DSparkWeights::Layer L;
        L.attn_norm = b.need(p + "attn_norm.weight", { H });
        L.ffn_norm = b.need(p + "ffn_norm.weight", { H });
        for (auto [hw, s] : { std::pair { &L.hc_attn, "attn" }, std::pair { &L.hc_ffn, "ffn" } }) {
            const std::string h = p + "hc_" + s;
            hw->fn = b.need(h + "_fn.weight", { H * HC, C });
            hw->base = b.need(h + "_base.weight", { C });
            hw->scale = b.need(h + "_scale.weight", { 3 });
        }
        auto & a = L.attn;
        a.q_a = b.need(p + "attn_q_a.weight", { H, Q });
        a.q_a_norm = b.need(p + "attn_q_a_norm.weight", { Q });
        a.q_b = b.need(p + "attn_q_b.weight", { Q, A * D });
        a.kv = b.need(p + "attn_kv.weight", { H, D });
        a.kv_norm = b.need(p + "attn_kv_a_norm.weight", { D });
        a.out_a = b.need(p + "attn_output_a.weight", { A * D / G, R * G });
        a.out_b = b.need(p + "attn_output_b.weight", { R * G, H });
        a.sinks = b.need(p + "attn_sinks.weight", { A });
        auto & m = L.moe;
        m.router = b.need(p + "ffn_gate_inp.weight", { H, E });
        m.exp_bias = b.need(p + "ffn_gate_inp.bias", { E });
        L.bias_vl = b.maybe(p + "ffn_gate_inp.bias_vl", { E });
        m.gate = b.experts(p + "ffn_gate_exps", H, F, (int) E);
        m.up = b.experts(p + "ffn_up_exps", H, F, (int) E);
        m.down = b.experts(p + "ffn_down_exps", F, H, (int) E);
        m.shexp_gate = b.need(p + "ffn_gate_shexp.weight", { H, F });
        m.shexp_up = b.need(p + "ffn_up_shexp.weight", { H, F });
        m.shexp_down = b.need(p + "ffn_down_shexp.weight", { F, H });
        if (i == 0) {
            L.main_proj = b.need(p + "main_proj.weight", { (int64_t) d.target_layers.size() * H, H });
            L.main_norm = b.need(p + "main_norm.weight", { H });
        }
        if (i + 1 == d.n_layer) {
            L.norm = b.need(p + "norm.weight", { H });
            L.markov_embed = b.need(p + "markov_embed.weight", { d.markov_rank, c.n_vocab });
            L.markov_head = b.need(p + "markov_head.weight", { d.markov_rank, c.n_vocab });
            L.confidence_proj = b.need(p + "confidence_proj.weight", { H + d.markov_rank, 1 });
        }
        w.layers.push_back(L);
    }
    b.check_all_used();
    return w;
}

}  // namespace truss::deepseek41
