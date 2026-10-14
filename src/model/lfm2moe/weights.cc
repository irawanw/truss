// Tensor names and shapes follow llama-paw src/models/lfm2moe.cpp (load_arch_tensors), ggml order.
#include "model/lfm2moe/weights.h"

#include "formats/binder.h"

#include <string>

namespace truss::lfm2moe {

Weights bind(const gguf::File & f, const Config & c)
{
    formats::Binder b(f, "lfm2moe");
    Weights w;
    const int64_t d = c.d_model, hd = c.head_dim;
    w.token_embd = b.need("token_embd.weight", { d, c.n_vocab });
    w.final_norm = b.need("token_embd_norm.weight", { d });
    for (int l = 0; l < c.n_layer; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        Layer L;
        L.op_norm = b.need(p + "attn_norm.weight", { d });
        L.ffn_norm = b.need(p + "ffn_norm.weight", { d });
        if (c.is_attn(l)) {
            const int64_t kv = c.n_head_kv[l] * hd;
            L.attn.q = b.need(p + "attn_q.weight", { d, c.n_head * hd });
            L.attn.k = b.need(p + "attn_k.weight", { d, kv });
            L.attn.v = b.need(p + "attn_v.weight", { d, kv });
            L.attn.out = b.need(p + "attn_output.weight", { c.n_head * hd, d });
            L.attn.q_norm = b.need(p + "attn_q_norm.weight", { hd });
            L.attn.k_norm = b.need(p + "attn_k_norm.weight", { hd });
        } else {
            L.conv.in_proj = b.need(p + "shortconv.in_proj.weight", { d, 3 * d });
            L.conv.conv = b.need(p + "shortconv.conv.weight", { c.conv_L, d });
            L.conv.out_proj = b.need(p + "shortconv.out_proj.weight", { d, d });
        }
        if (!c.is_moe(l)) {
            L.dense.gate = b.need(p + "ffn_gate.weight", { d, c.d_ff });
            L.dense.up = b.need(p + "ffn_up.weight", { d, c.d_ff });
            L.dense.down = b.need(p + "ffn_down.weight", { c.d_ff, d });
        } else {
            L.moe.router = b.need(p + "ffn_gate_inp.weight", { d, c.n_expert });
            L.moe.exp_bias = b.need(p + "exp_probs_b.bias", { c.n_expert });
            L.moe.gate = b.experts(p + "ffn_gate_exps", d, c.d_ff_exp, c.n_expert);
            L.moe.up = b.experts(p + "ffn_up_exps", d, c.d_ff_exp, c.n_expert);
            L.moe.down = b.experts(p + "ffn_down_exps", c.d_ff_exp, d, c.n_expert);
        }
        w.layers.push_back(L);
    }
    b.check_all_used();
    return w;
}

}  // namespace truss::lfm2moe
