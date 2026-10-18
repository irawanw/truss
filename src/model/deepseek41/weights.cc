// Tensor names and shapes (ggml order: ne0 = input) follow mmnga/Strata deepseek41 Model::validate().
#include "model/deepseek41/weights.h"

#include "formats/binder.h"

#include <string>

namespace truss::deepseek41 {

Weights bind(const gguf::File & f, const Config & c)
{
    formats::Binder b(f, "deepseek41");
    Weights w;
    const int64_t H = c.d_model, A = c.n_head, D = c.head_dim, Q = c.q_rank, G = c.o_groups, R = c.o_rank,
                  HC = c.hc, E = c.n_expert, F = c.d_ff_exp, NI = c.index_heads, I = c.index_dim,
                  C = (2 + HC) * HC;
    w.token_embd = b.need("token_embd.weight", { H, c.n_vocab });
    w.final_norm = b.need("output_norm.weight", { H });
    w.output = b.need("output.weight", { H, c.n_vocab });
    for (int l = 0; l < c.n_layer; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        Layer L;
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
        if (c.is_kv_source(l)) {
            a.cmp_kv = b.need(p + "attn_compressor_kv.weight", { H, D });
            a.cmp_norm = b.need(p + "attn_compressor_norm.weight", { D });
            if (c.compress_ratio[l] > 1) a.cmp_gate = b.need(p + "attn_compressor_gate.weight", { H, D });
            a.idx_k = b.need(p + "indexer.attn_k.weight", { D, I });
            a.idx_k_norm = b.need(p + "indexer.k_norm.weight", { I });
        }
        if (c.is_index_source(l)) {
            a.idx_q_b = b.need(p + "indexer.attn_q_b.weight", { Q, NI * I });
            a.idx_proj = b.need(p + "indexer.proj.weight", { H, NI });
        }
        auto & m = L.moe;
        m.router = b.need(p + "ffn_gate_inp.weight", { H, E });
        m.exp_bias = b.need(p + "exp_probs_b.bias", { E });
        m.gate = b.experts(p + "ffn_gate_exps", H, F, (int) E);
        m.up = b.experts(p + "ffn_up_exps", H, F, (int) E);
        m.down = b.experts(p + "ffn_down_exps", F, H, (int) E);
        m.shexp_gate = b.need(p + "ffn_gate_shexp.weight", { H, F * c.n_shexp });
        m.shexp_up = b.need(p + "ffn_up_shexp.weight", { H, F * c.n_shexp });
        m.shexp_down = b.need(p + "ffn_down_shexp.weight", { F * c.n_shexp, H });
        if (c.engram.index(l) >= 0) {
            const int64_t nb = c.engram.buckets() * (int64_t) c.engram.head_dim;
            L.engram.wkv = b.need(p + "engram_wkv.weight", { nb, H * (HC + 1) });
            L.engram.q = b.need(p + "engram_q.weight", { H, HC });
            L.engram.k = b.need(p + "engram_k.weight", { H, HC });
        }
        w.layers.push_back(L);
    }
    b.check_all_used();
    return w;
}

}  // namespace truss::deepseek41
