// Tensor names and shapes follow llama-paw src/models/qwen4exp.cpp (load_arch_tensors); shapes are ggml order
// (contiguous dimension first) as stored in the file.
#include "model/qwen4exp/weights.h"

#include "formats/binder.h"

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace truss::qwen4exp {

namespace {

using formats::Binder;

HyperConnection bind_hc(Binder & b, const std::string & p, const Config & c, bool inject)
{
    HyperConnection h;
    h.norm = b.need(p + "_norm.weight", { c.hc_dim() });
    h.down = b.need(p + "_down.weight", { c.hc_dim(), c.hc_rank });
    h.up = b.need(p + "_up.weight", { c.hc_rank, c.hc_dim() });
    if (inject) h.inject = b.need(p + "_inject.weight", { c.hc_dim(), c.hc });
    return h;
}

Layer bind_layer(Binder & b, const Config & c, int l, Mixer mixer, bool ple)
{
    const std::string p = "blk." + std::to_string(l) + ".";
    const int64_t d = c.d_model;
    Layer L;
    L.mixer = mixer;
    L.hc_attn = bind_hc(b, p + "hc_attn", c, true);
    L.hc_ffn = bind_hc(b, p + "hc_ffn", c, true);

    if (L.mixer == Mixer::GDN) {
        Gdn & g = L.gdn;
        g.qkv = b.need(p + "attn_qkv.weight", { d, c.conv_dim() });
        g.gate = b.need(p + "attn_gate.weight", { d, c.value_dim() });
        g.conv1d = b.need(p + "ssm_conv1d.weight", { c.ssm_conv, c.conv_dim() });
        g.dt_bias = b.need(p + "ssm_dt.bias", { c.ssm_v_heads });
        g.a = b.need(p + "ssm_a", { c.ssm_v_heads });
        g.beta = b.need(p + "ssm_beta.weight", { d, c.ssm_v_heads });
        g.alpha = b.need(p + "ssm_alpha.weight", { d, c.ssm_v_heads });
        g.norm = b.need(p + "ssm_norm.weight", { c.ssm_state });
        g.out = b.need(p + "ssm_out.weight", { c.value_dim(), d });
    } else {
        Dsa & a = L.dsa;
        const int64_t q_all = (int64_t) c.n_head * c.head_dim, kv_all = (int64_t) c.n_head_kv * c.head_dim;
        a.q = b.need(p + "attn_q.weight", { d, 2 * q_all });
        a.k = b.need(p + "attn_k.weight", { d, kv_all });
        a.v = b.need(p + "attn_v.weight", { d, kv_all });
        a.out = b.need(p + "attn_output.weight", { q_all, d });
        a.q_norm = b.need(p + "attn_q_norm.weight", { c.head_dim });
        a.k_norm = b.need(p + "attn_k_norm.weight", { c.head_dim });
        a.idx_q = b.need(p + "indexer.q_proj.weight", { d, (int64_t) c.idx_heads * c.idx_head_dim });
        a.idx_k = b.need(p + "indexer.k_proj.weight", { d, c.idx_head_dim });
        a.idx_q_norm = b.need(p + "indexer.q_norm.weight", { c.idx_head_dim });
        a.idx_k_norm = b.need(p + "indexer.k_norm.weight", { c.idx_head_dim });
    }

    if (ple) {
        Ple & e = L.ple;
        e.key = b.need(p + "ple_key.weight", { d, c.hc_dim() });
        e.value = b.need(p + "ple_value.weight", { d, d });
        e.norm_key = b.need(p + "ple_norm_key.weight", { c.hc_dim() });
        e.norm_query = b.need(p + "ple_norm_query.weight", { c.hc_dim() });
        e.norm_conv = b.need(p + "ple_norm_conv.weight", { c.hc_dim() });
        e.conv1d = b.need(p + "ple_conv1d.weight", { c.ple_conv, c.hc_dim() });
    }

    Moe & m = L.moe;
    m.router = b.need(p + "ffn_gate_inp.weight", { d, c.n_expert });
    m.gate = b.experts(p + "ffn_gate_exps", d, c.d_ff_exp, c.n_expert);
    m.up = b.experts(p + "ffn_up_exps", d, c.d_ff_exp, c.n_expert);
    m.down = b.experts(p + "ffn_down_exps", c.d_ff_exp, d, c.n_expert);
    m.shexp_gate_inp = b.need(p + "ffn_gate_inp_shexp.weight", { d });
    m.shexp_gate = b.need(p + "ffn_gate_shexp.weight", { d, c.d_ff_shexp });
    m.shexp_up = b.need(p + "ffn_up_shexp.weight", { d, c.d_ff_shexp });
    m.shexp_down = b.need(p + "ffn_down_shexp.weight", { c.d_ff_shexp, d });
    return L;
}

}  // namespace

Weights bind(const gguf::File & f, const Config & c)
{
    Binder b(f, "qwen4exp");
    Weights w;
    const int64_t d = c.d_model;
    w.token_embd = b.need("token_embd.weight", { d, c.n_vocab });
    w.output = f.find("output.weight") ? b.need("output.weight", { d, c.n_vocab }) : w.token_embd;   // tied
    w.hc_head = bind_hc(b, "output_hc", c, false);

    int64_t ple_rows = 0;
    for (size_t h = 0; h < c.ple_head_offsets.size(); ++h)
        ple_rows = std::max(ple_rows, c.ple_head_offsets[h] + c.ple_head_vocab[h]);
    if (!c.ple_layers.empty()) {
        const gguf::Tensor * q8 = f.find("per_layer_token_embd.q8");
        if (!q8) throw std::runtime_error("qwen4exp: only the PAW Q8 PLE table (per_layer_token_embd.q8) is supported");
        if (q8->shape.size() != 2 || q8->shape[1] < ple_rows)
            throw std::runtime_error("qwen4exp: PLE table has too few rows for its head ranges");
        w.ple_table = b.need("per_layer_token_embd.q8", { c.ple_head_dim, q8->shape[1] });
        w.ple_scale = b.need("per_layer_token_embd.scale", { 1, q8->shape[1] });
    }

    for (int l = 0; l < c.n_layer; ++l) w.layers.push_back(bind_layer(b, c, l, c.mixer[l], c.is_ple(l)));
    b.check_all_used();
    return w;
}

Mtp bind_mtp(const gguf::File & f, const Config & c)
{
    Binder b(f, "qwen4exp");
    Mtp m;
    const int l = c.n_layer;
    const std::string p = "blk." + std::to_string(l) + ".";
    m.layer = bind_layer(b, c, l, Mixer::DSA, false);
    m.eh_proj = b.need(p + "nextn.eh_proj.weight", { 2 * c.d_model, c.d_model });
    m.enorm = b.need(p + "nextn.enorm.weight", { c.d_model });
    m.hnorm = b.need(p + "nextn.hnorm.weight", { c.hc_dim() });
    b.check_all_used();
    return m;
}

}  // namespace truss::qwen4exp
