#include "blocks/attn_gqa.h"

#include "engine/stage.h"

namespace truss::blocks {

void AttnGqa::alloc_slots(engine::Stage & st, int n_slots, int seq_ctx, int)
{
    n_ctx_ = st.n_ctx();
    const size_t row = (size_t) s_.HKV * s_.D;
    half * k = st.gpu().alloc<half>((size_t) n_ctx_ * row), * v = st.gpu().alloc<half>((size_t) n_ctx_ * row);
    sl_.resize(n_slots);
    for (int i = 0; i < n_slots; ++i) sl_[i] = { k + (size_t) i * seq_ctx * row, v + (size_t) i * seq_ctx * row };
}

void AttnGqa::forward(engine::Gpu & g, int slot, const half * in16, int pos0, int T, float * out,
                       const float2 * tab)
{
    const Slot & L = sl_[slot];
    Scratch & sc = *g.sc;
    const size_t m = sc.mark();
    const size_t qd = (size_t) s_.H * s_.D, kd = (size_t) s_.HKV * s_.D;
    float * q = sc.alloc((size_t) T * qd), * k = sc.alloc((size_t) T * kd), * v = sc.alloc((size_t) T * kd);
    half * q16 = sc.alloc<half>((size_t) T * qd), * att16 = sc.alloc<half>((size_t) T * qd);
    const Linear::Act a = lin_.quant(in16, T, d_);
    lin_.lin_multi({ { w_.q, q }, { w_.k, k }, { w_.v, v } }, a, T);
    attn::gqa_prepare(s_, q, k, v, w_.q_norm ? lin_.f32(w_.q_norm) : nullptr, w_.k_norm ? lin_.f32(w_.k_norm) : nullptr,
                      pos0, T, q16, L.k, L.v, g.s, tab);
    const size_t ws_bytes = attn::gqa_workspace_bytes(s_, T, pos0 + T);
    void * ws = ws_bytes ? sc.alloc<unsigned char>(ws_bytes) : nullptr;
    attn::gqa_attention(s_, q16, L.k, L.v, pos0, T, att16, ws, ws_bytes, g.s);
    lin_.lin(w_.out, att16, T, out);
    sc.release(m);
}

void AttnGqa::positional(std::vector<Positional> & out) const
{
    const size_t row = (size_t) s_.HKV * s_.D * sizeof(half);
    out.push_back({ sl_[0].k, (size_t) n_ctx_ * row, row });
    out.push_back({ sl_[0].v, (size_t) n_ctx_ * row, row });
}

}  // namespace truss::blocks
