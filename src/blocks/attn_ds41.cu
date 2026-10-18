#include "blocks/attn_ds41.h"

#include "engine/stage.h"
#include "kernels/ds41/attn_ops.cuh"
#include "kernels/ds41/indexer.cuh"
#include "kernels/ds41/sparse_attn.cuh"
#include "kernels/norm/rms_norm.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace truss::blocks {

namespace {
constexpr int IDX_ROWS = 64;   // index rows scored at once: the score rows are [IDX_ROWS][n_ctx / ratio] fp32
}

void AttnDs41::add_tensors(Linear & lin, const Ds41AttnWeights & w)
{
    for (T t : { w.q_a, w.q_a_norm, w.q_b, w.kv, w.kv_norm, w.out_a, w.out_b, w.sinks, w.cmp_kv, w.cmp_gate,
                 w.cmp_norm, w.idx_k, w.idx_k_norm, w.idx_q_b, w.idx_proj })
        lin.add(t);
}

size_t AttnDs41::scratch_per_row(const Ds41AttnDims & d)
{
    const size_t Q = d.q_rank, HD = (size_t) d.n_head * d.head_dim, D = d.head_dim, G = d.o_groups,
                 GR = (size_t) d.o_groups * d.o_rank, ID = (size_t) d.index_heads * d.index_dim;
    const size_t f = Q * 2 + HD * 2 + D * 5 + G * GR + GR * 2 + ID + d.index_heads + D * 8 + d.index_dim * 2;
    const size_t h = Q + D + HD + GR + D;
    return f * 4 + h * 2 + 64 * 256;
}

size_t AttnDs41::scratch_fixed(const Ds41AttnDims & d, int n_ctx)
{
    if (!d.ratio || !d.index_source) return 0;
    const size_t n = (size_t) n_ctx / d.ratio;
    return (size_t) IDX_ROWS * (n + n / d.candidate_block + 2) * 4 + 4096;
}

void AttnDs41::alloc_slots(engine::Stage & st, int n_slots, int seq_ctx, int spec_rows)
{
    alloc(st.gpu(), st.n_ctx(), st.max_rows(), n_slots, seq_ctx, spec_rows);
}

void AttnDs41::alloc(engine::Gpu & gpu, int n_ctx, int max_rows, int n_slots, int seq_ctx, int spec_rows)
{
    if (spec_rows + d_.window > RING) throw std::runtime_error("AttnDs41: verify window + window > ring");
    n_ctx_ = n_ctx, seq_ctx_ = seq_ctx;
    const size_t D = d_.head_dim;
    sl_.resize(n_slots);
    half * ring = gpu.alloc<half>((size_t) n_slots * RING * D);
    for (int i = 0; i < n_slots; ++i) sl_[i].ring = ring + (size_t) i * RING * D;
    if (d_.kv_source) {
        const size_t rows = (size_t) n_ctx_ / d_.ratio, per = (size_t) seq_ctx / d_.ratio;
        half * comp = gpu.alloc<half>(rows * D);
        float * ik = gpu.alloc<float>(rows * d_.index_dim);
        float * rk = nullptr, * rs = nullptr;
        if (d_.ratio > 1) {
            rk = gpu.alloc<float>((size_t) n_slots * RING * D);
            rs = gpu.alloc<float>((size_t) n_slots * RING * D);
        }
        for (int i = 0; i < n_slots; ++i) {
            sl_[i].comp = comp + i * per * D;
            sl_[i].idx_k = ik + i * per * d_.index_dim;
            if (rk) sl_[i].raw_kv = rk + (size_t) i * RING * D, sl_[i].raw_sc = rs + (size_t) i * RING * D;
        }
    }
    if (d_.index_source) topk_ = gpu.alloc<int>((size_t) max_rows * d_.index_top_k);
    if (d_.candidate_source) cand_ = gpu.alloc<int>((size_t) max_rows * d_.candidate_top_blocks);
}

void AttnDs41::positional(std::vector<Positional> & out) const
{
    if (!d_.kv_source) return;
    const size_t rows = (size_t) n_ctx_ / d_.ratio;
    out.push_back({ sl_[0].comp, rows * d_.head_dim * sizeof(half), d_.head_dim * sizeof(half) });
    out.push_back({ sl_[0].idx_k, rows * d_.index_dim * sizeof(float), d_.index_dim * sizeof(float) });
}

// this chunk's compressed latents -> index keys and compressed cache of the slot (a KV source)
void AttnDs41::compress(engine::Gpu & g, const Slot & S, const float * in, const half * in16, int pos0, int T)
{
    Scratch & sc = *g.sc;
    const int D = d_.head_dim, ID = d_.index_dim, r = d_.ratio;
    int j0, n;
    ds41::groups_done(pos0, T, r, j0, n);
    float * lat = sc.alloc((size_t) std::max(n, 1) * D), * latn = sc.alloc((size_t) std::max(n, 1) * D);
    if (r == 1) {
        lin_.lin(w_.cmp_kv, in16, T, lat);
    } else {
        float * kv = sc.alloc((size_t) T * D), * scr = sc.alloc((size_t) T * D);
        lin_.lin32(w_.cmp_kv, in, T, kv);
        lin_.lin32(w_.cmp_gate, in, T, scr);
        ds41::pool(kv, scr, S.raw_kv, S.raw_sc, RING, D, r, pos0, j0, n, lat, g.s);
        ds41::ring_write_f32(S.raw_kv, RING, kv, D, pos0, T, g.s);
        ds41::ring_write_f32(S.raw_sc, RING, scr, D, pos0, T, g.s);
    }
    if (n <= 0) return;
    norm::rms(lat, lin_.f32(w_.cmp_norm), n, D, d_.eps, latn, nullptr, g.s);
    // index keys from the latent before its rope
    half * lat16 = sc.alloc<half>((size_t) n * D);
    ds41::to_half(latn, (size_t) n * D, lat16, g.s);
    float * kk = sc.alloc((size_t) n * ID);
    lin_.lin(w_.idx_k, lat16, n, kk);
    float * ik = S.idx_k + (size_t) j0 * ID;
    norm::rms(kk, lin_.f32(w_.idx_k_norm), n, ID, d_.eps, ik, nullptr, g.s);
    ds41::rope_tail(ik, n, 1, ID, d_.rope_dim, j0 * r, r, d_.inv_freq, false, g.s);
    ds41::rope_tail(latn, n, 1, D, d_.rope_dim, j0 * r, r, d_.inv_freq, false, g.s);
    ds41::to_half(latn, (size_t) n * D, S.comp + (size_t) j0 * D, g.s);
}

// this layer's top-k (and, on the candidate source, its candidate blocks) for the chunk's rows
void AttnDs41::index(engine::Gpu & g, int slot, const half * qr16, const half * in16, int pos0, int T)
{
    Scratch & sc = *g.sc;
    const int NH = d_.index_heads, ID = d_.index_dim, r = d_.ratio, B = d_.candidate_block;
    float * q = sc.alloc((size_t) T * NH * ID), * w = sc.alloc((size_t) T * NH);
    lin_.lin(w_.idx_q_b, qr16, T, q);
    ds41::rope_tail(q, T, NH, ID, d_.rope_dim, pos0, 1, d_.inv_freq, false, g.s);
    lin_.lin(w_.idx_proj, in16, T, w);
    const float wscale = 1.0f / std::sqrt((float) ID) / std::sqrt((float) NH);
    const size_t nmax_all = (size_t) (pos0 + T) / r;
    float * s = sc.alloc((size_t) IDX_ROWS * std::max<size_t>(nmax_all, 1));
    float * bm = d_.candidate_source ? sc.alloc((size_t) IDX_ROWS * (nmax_all / B + 1)) : nullptr;
    const AttnDs41 & ks = kv_src_ ? *kv_src_ : *this;
    const float * k_base = ks.sl_[slot].idx_k;
    for (int r0 = 0; r0 < T; r0 += IDX_ROWS) {
        const int rb = std::min(IDX_ROWS, T - r0), n = (pos0 + r0 + rb) / r;
        int * top = topk_ + (size_t) r0 * d_.index_top_k;
        if (n <= 0) {
            TRUSS_CUDA(cudaMemsetAsync(top, 0xff, sizeof(int) * rb * d_.index_top_k, g.s));
            continue;
        }
        ds41::idx_scores(q + (size_t) r0 * NH * ID, w + (size_t) r0 * NH, wscale, k_base, NH, ID, pos0 + r0, rb, r, s,
                         g.s);
        if (d_.candidate_source) {
            const int nb = (n + B - 1) / B;
            ds41::block_max(s, rb, n, B, pos0 + r0, r, bm, g.s);
            ds41::select_topk(bm, rb, nb, d_.candidate_top_blocks, cand_ + (size_t) r0 * d_.candidate_top_blocks,
                              g.s);
        }
        if (d_.uses_candidates)
            ds41::mask_blocks(s, rb, n, B, cand_src_->cand_ + (size_t) r0 * d_.candidate_top_blocks,
                              d_.candidate_top_blocks, g.s);
        ds41::select_topk(s, rb, n, d_.index_top_k, top, g.s);
    }
}

void AttnDs41::forward(engine::Gpu & g, int slot, const float * in, const half * in16, int pos0, int T, float * out)
{
    Scratch & sc = *g.sc;
    const size_t m = sc.mark();
    const Slot & S = sl_[slot];
    const int D = d_.head_dim, H = d_.n_head, Q = d_.q_rank, G = d_.o_groups, R = d_.o_rank, rd = d_.rope_dim;
    // q
    float * qa = sc.alloc((size_t) T * Q), * q = sc.alloc((size_t) T * H * D);
    half * qr16 = sc.alloc<half>((size_t) T * Q);
    lin_.lin(w_.q_a, in16, T, qa);
    norm::rms(qa, lin_.f32(w_.q_a_norm), T, Q, d_.eps, nullptr, qr16, g.s);
    lin_.lin(w_.q_b, qr16, T, q);
    ds41::rope_tail(q, T, H, D, rd, pos0, 1, d_.inv_freq, false, g.s);
    // window kv
    float * kv = sc.alloc((size_t) T * D), * kvn = sc.alloc((size_t) T * D);
    half * kv16 = sc.alloc<half>((size_t) T * D);
    lin_.lin(w_.kv, in16, T, kv);
    norm::rms(kv, lin_.f32(w_.kv_norm), T, D, d_.eps, kvn, nullptr, g.s);
    ds41::rope_tail(kvn, T, 1, D, rd, pos0, 1, d_.inv_freq, false, g.s);
    ds41::to_half(kvn, (size_t) T * D, kv16, g.s);
    // compressed kv and top-k
    if (d_.ratio) {
        if (d_.kv_source) compress(g, S, in, in16, pos0, T);
        if (d_.index_source) index(g, slot, qr16, in16, pos0, T);
    }
    // attention
    float * o = sc.alloc((size_t) T * H * D);
    ds41::SparseAttnArgs a;
    a.q = q, a.chunk_kv = kv16, a.ring = S.ring, a.R = RING;
    if (d_.ratio) {
        const AttnDs41 & ks = kv_src_ ? *kv_src_ : *this, & is = idx_src_ ? *idx_src_ : *this;
        a.comp = ks.sl_[slot].comp, a.idx = is.topk_, a.k = d_.index_top_k;
    } else {
        a.comp = nullptr, a.idx = nullptr, a.k = 0;
    }
    a.sinks = lin_.f32(w_.sinks), a.H = H, a.D = D, a.window = d_.window, a.pos0 = pos0, a.T = T;
    a.scale = 1.0f / std::sqrt((float) D), a.out = o;
    ds41::sparse_attn(a, g.s);
    ds41::ring_write(S.ring, RING, kvn, D, pos0, T, g.s);
    ds41::rope_tail(o, T, H, D, rd, pos0, 1, d_.inv_freq, true, g.s);
    // grouped output LoRA: o as [T * G][H D / G] rows through the whole wo_a, then each row's own group block
    half * o16 = sc.alloc<half>((size_t) T * H * D);
    ds41::to_half(o, (size_t) T * H * D, o16, g.s);
    float * ya = sc.alloc((size_t) T * G * G * R), * yo = sc.alloc((size_t) T * G * R);
    half * yo16 = sc.alloc<half>((size_t) T * G * R);
    lin_.lin(w_.out_a, o16, T * G, ya);
    ds41::pick_groups(ya, T, G, R, yo, g.s);
    ds41::to_half(yo, (size_t) T * G * R, yo16, g.s);
    lin_.lin(w_.out_b, yo16, T, out);
    sc.release(m);
}

}  // namespace truss::blocks
