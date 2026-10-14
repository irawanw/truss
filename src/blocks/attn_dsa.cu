#include "blocks/attn_dsa.h"

#include "engine/stage.h"
#include "kernels/dsa/dsa_prepare.cuh"

namespace truss::blocks {

template <class S> void Dsa<S>::add_tensors(Linear & lin, const DsaWeights & w)
{
    for (T t : { w.q, w.k, w.v, w.out, w.q_norm, w.k_norm, w.idx_q, w.idx_k, w.idx_q_norm, w.idx_k_norm }) lin.add(t);
}

template <class S> size_t Dsa<S>::scratch_per_row()
{
    const size_t qd = (size_t) S::H * S::D;
    return qd * (8 + 2 + 4 + 2) + (size_t) S::HKV * S::D * 8 + (size_t) S::IH * S::ID * 6 + S::ID * 4 +
           S::ROPE_DIMS * 4 + S::TOP_BLOCKS * 4 + 4;
}

template <class S> size_t Dsa<S>::scratch_fixed(int T, int n_ctx)
{
    return dsa::select_workspace_bytes<S>(T, n_ctx) + dsa::attention_workspace_bytes<S>(T);
}

template <class S> void Dsa<S>::alloc_slots(engine::Stage & st, int n_slots, int seq_ctx, int spec_rows)
{
    engine::Gpu & g = st.gpu();
    n_ctx_ = st.n_ctx();
    const size_t kvn = (size_t) n_ctx_ * S::HKV * S::D, rowkv = (size_t) S::HKV * S::D;
    Slot s0;
    if (int8_) {   // Strata's int8 KV: 1,056 B per cell per layer instead of 2,048 (TRACKER #87)
        s0.kv.kq = g.alloc<int8_t>(kvn), s0.kv.vq = g.alloc<int8_t>(kvn);
        s0.kv.ks = g.alloc<half>(kvn / dsa::KV_GROUP), s0.kv.vs = g.alloc<half>(kvn / dsa::KV_GROUP);
    } else {
        s0.kv.k16 = g.alloc<half>(kvn), s0.kv.v16 = g.alloc<half>(kvn);
    }
    s0.idx_k = g.alloc<half>((size_t) (n_ctx_ / S::RATIO) * S::ID);
    s_.assign(n_slots, s0);
    for (int k = 0; k < n_slots; ++k) {   // slot k: positions k * seq_ctx .. of the shared arrays
        Slot & s = s_[k];
        const size_t off = (size_t) k * seq_ctx;
        if (s.kv.kq) s.kv.kq += off * rowkv, s.kv.vq += off * rowkv, s.kv.ks += off * rowkv / dsa::KV_GROUP,
                     s.kv.vs += off * rowkv / dsa::KV_GROUP;
        if (s.kv.k16) s.kv.k16 += off * rowkv, s.kv.v16 += off * rowkv;
        s.idx_k += off / S::RATIO * S::ID;
        s.idx_partial = g.alloc<float>(partial_floats());
        if (spec_rows) s.partial_snap = g.alloc<float>(partial_floats()), s.raw_ik = g.alloc<float>((size_t) spec_rows * S::ID);
    }
}

template <class S>
void Dsa<S>::forward(engine::Gpu & g, int slot, const half * in16, int pos0, int T, float * out, bool tentative)
{
    Slot & L = s_[slot];
    Scratch & sc = *g.sc;
    const size_t m = sc.mark();
    constexpr int H = S::H, D = S::D, Hkv = S::HKV, IH = S::IH, ID = S::ID;
    float * qfull = sc.alloc((size_t) T * H * 2 * D);
    float * k = sc.alloc((size_t) T * Hkv * D), * v = sc.alloc((size_t) T * Hkv * D);
    float * iq = sc.alloc((size_t) T * IH * ID), * ik = sc.alloc((size_t) T * ID);
    float2 * cs = sc.alloc<float2>((size_t) T * S::ROPE_DIMS / 2);
    half * q16 = sc.alloc<half>((size_t) T * H * D), * iq16 = sc.alloc<half>((size_t) T * IH * ID);
    float * gate = sc.alloc((size_t) T * H * D);
    int * blocks = sc.alloc<int>((size_t) T * S::TOP_BLOCKS), * n_blocks = sc.alloc<int>(T);
    half * att16 = sc.alloc<half>((size_t) T * H * D);
    const Linear::Act ia = lin_.quant(in16, T, d_.d_model);
    lin_.lin_multi({ { w_.q, qfull }, { w_.k, k }, { w_.v, v }, { w_.idx_q, iq }, { w_.idx_k, ik } }, ia, T);
    if (tentative) {
        g.copy(L.partial_snap, L.idx_partial, partial_floats());
        g.copy(L.raw_ik, ik, (size_t) T * ID);
    }
    dsa::rope_table<S>(pos0, T, d_.rope_base, cs, g.s);
    dsa::prepare_qkv<S>(qfull, k, v, lin_.f32(w_.q_norm), lin_.f32(w_.k_norm), cs, pos0, T, d_.eps, q16, gate, L.kv, g.s);
    dsa::prepare_index<S>(iq, ik, lin_.f32(w_.idx_q_norm), lin_.f32(w_.idx_k_norm), cs, d_.rope_base, pos0, T, d_.eps,
                          L.idx_partial, iq16, L.idx_k, g.s);
    const size_t ws_bytes = dsa::select_workspace_bytes<S>(T, pos0 + T);
    void * ws = sc.alloc<unsigned char>(ws_bytes);
    dsa::select<S>(iq16, L.idx_k, pos0, T, blocks, n_blocks, ws, ws_bytes, g.s);
    const size_t aws_bytes = dsa::attention_workspace_bytes<S>(T);
    void * aws = aws_bytes ? sc.alloc<unsigned char>(aws_bytes) : nullptr;
    dsa::attention<S>(q16, gate, L.kv, blocks, n_blocks, pos0, T, att16, aws, aws_bytes, g.s);
    lin_.lin(w_.out, att16, T, out);
    sc.release(m);
}

// the indexer's open block: the snapshot plus the accepted rows' raw keys (K/V rows need nothing: rewritten later)
template <class S> void Dsa<S>::accept(engine::Stage & st, int slot, int n, const engine::Window & w)
{
    if (n == w.T) return;
    Slot & L = s_[slot];
    st.gpu().copy(L.idx_partial, L.partial_snap, partial_floats());
    if (n > 0) dsa::carry_partial<S>(L.raw_ik, w.pos0, n, L.idx_partial, st.gpu().s);
}

template <class S> void Dsa<S>::ck_buffers(int slot, std::vector<std::pair<float *, size_t>> & out)
{
    out.push_back({ s_[slot].idx_partial, partial_floats() });
}

template <class S> void Dsa<S>::positional(std::vector<Positional> & out) const
{
    const Slot & s = s_[0];
    const size_t row = (size_t) S::HKV * S::D * (s.kv.int8() ? 1 : 2);
    for (void * p : { s.kv.int8() ? (void *) s.kv.kq : (void *) s.kv.k16, s.kv.int8() ? (void *) s.kv.vq : (void *) s.kv.v16 })
        out.push_back({ p, (size_t) n_ctx_ * row, row });
}

template class Dsa<dsa::FlashNext>;

}  // namespace truss::blocks
