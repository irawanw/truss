#include "blocks/attn_ds41.h"

#include "engine/limits.h"
#include "engine/stage.h"
#include "kernels/ds41/attn_ops.cuh"
#include "kernels/ds41/indexer.cuh"
#include "kernels/ds41/sparse_attn.cuh"
#include "kernels/norm/rms_norm.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

// the prompt path's peak temporaries per row (bytes): the q projection's fp32 rows and its fp16 copy, q, the kv
// rows, o16 (the grouped LoRA's input), the LoRA's yo and yo16, the Q8_1 stacked rows' quantization, and (on a KV
// source / an index source) the compressor's and the indexer's rows. The attention's output o is a piece
// (engine::PREFILL_PIECE rows), not a row: scratch_piece() adds it.
size_t AttnDs41::scratch_per_row(const Ds41AttnDims & d)
{
    const size_t Q = d.q_rank, HD = (size_t) d.n_head * d.head_dim, D = d.head_dim, G = d.o_groups,
                 GR = (size_t) d.o_groups * d.o_rank, ID = (size_t) d.index_heads * d.index_dim;
    size_t b = Q * 6              // qa (fp32) + qr16 (fp16)
             + HD * 4             // q (fp32)
             + D * 10             // kv, kvn (fp32) + kv16 (fp16)
             + HD * 2             // o16 (fp16)
             + GR * 6             // yo (fp32) + yo16 (fp16)
             + HD * 17 / 16;      // the Q8_1 stacked rows (int8 + the fp16 scales)
    if (d.kv_source) b += D * 10 + ID * 4;                                  // the compressor's rows + the index keys
    if (d.index_source) b += (size_t) d.index_heads * d.index_dim * 4 + d.index_heads * 4;   // the indexer's rows
    return b;
}

// the piece's output buffer (the attention's o): engine::PREFILL_PIECE rows
size_t AttnDs41::scratch_piece(const Ds41AttnDims & d)
{
    const size_t HD = (size_t) d.n_head * d.head_dim;
    return (size_t) engine::prefill_piece() * HD * 4;
}

// the decode path: the same rows plus the grouped LoRA's stacked rows (the gemv path's ya)
size_t AttnDs41::scratch_decode_per_row(const Ds41AttnDims & d)
{
    const size_t G = d.o_groups, GR = (size_t) d.o_groups * d.o_rank;
    return scratch_per_row(d) + G * GR * 4;
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
    // The attention and the grouped output LoRA in row pieces: the chunk's per-row scratch (the attention's output o
    // and the LoRA's yo) must not scale with the chunk - the prompt path's buffers are carved from the expert ring's
    // spare region, and a chunk of 8K rows would take 6 GB of it (the ring's size decides how many experts stay
    // resident, TRACKER #201). The dense projections stay whole-chunk: cuBLAS picks its kernel by the row count, so
    // splitting them would change the last bits (tests/unit/gemm_piece_test.cu). The pieces themselves are exact: the
    // attention kernel's rows are independent, and the grouped GEMM's values do not depend on the row count.
    const int P = engine::prefill_piece();
    half * o16 = sc.alloc<half>((size_t) T * H * D);
    float * yo = sc.alloc((size_t) T * G * R);
    half * yo16 = sc.alloc<half>((size_t) T * G * R);
    const int W_in = lin_.q8(w_.out_a).in;   // the grouped LoRA's K (the group's slice of the row)
    const bool grouped = T * G > dense::GEMV_ROWS;   // more rows than a gemv: the per-group GEMMs (#194)
    Linear::Act ia;   // the Q8_1 path's quantized stacked rows (the groups' slices)
    if (grouped && lin_.act() != Activations::FP16) ia = lin_.quant(o16, T * G, W_in);
    for (int r0 = 0; r0 < T; r0 += P) {
        const int rb = std::min(P, T - r0);
        const size_t m2 = sc.mark();
        float * o = sc.alloc((size_t) rb * H * D);
        ds41::SparseAttnArgs a;
        // the piece's keys: the rows at or after the piece's start are in the piece's kv, the earlier ones in the
        // window ring (the chunk's rows are written to the ring before the piece's attention reads them)
        a.q = q + (size_t) r0 * H * D, a.chunk_kv = kv16 + (size_t) r0 * D, a.ring = S.ring, a.R = RING;
        if (d_.ratio) {
            const AttnDs41 & ks = kv_src_ ? *kv_src_ : *this, & is = idx_src_ ? *idx_src_ : *this;
            a.comp = ks.sl_[slot].comp, a.idx = is.topk_ + (size_t) r0 * d_.index_top_k, a.k = d_.index_top_k;
        } else {
            a.comp = nullptr, a.idx = nullptr, a.k = 0;
        }
        ds41::ring_write(S.ring, RING, kvn + (size_t) r0 * D, D, pos0 + r0, rb, g.s);
        a.sinks = lin_.f32(w_.sinks), a.H = H, a.D = D, a.window = d_.window, a.pos0 = pos0 + r0, a.T = rb;
        a.scale = 1.0f / std::sqrt((float) D), a.out = o;
        static const bool v1 = std::getenv("TRUSS_DS41_ATTN_V1") && std::atoi(std::getenv("TRUSS_DS41_ATTN_V1"));
        // v5 (32 heads per CTA, 16 warps) is the default when the shape allows; 1 = v2, 2 = v3 (cp.async), 3 = v4, 4 = v5
        static const int tc = std::getenv("TRUSS_DS41_ATTN_TC") ? std::atoi(std::getenv("TRUSS_DS41_ATTN_TC")) : 4;
        if (!v1 && tc == 2 && ds41::sparse_attn_tc_async_ok(a)) {   // v3: v2 with a cp.async double-buffered key tile
            const size_t n = ds41::sparse_attn_tc_async_ws_floats(a);
            ds41::sparse_attn_tc_async(a, n ? sc.alloc(n) : nullptr, g.s);
        } else if (!v1 && tc == 3 && ds41::sparse_attn_tc_h32_ok(a)) {   // v4: 32 heads per CTA, 8 warps
            const size_t n = ds41::sparse_attn_tc_h32_ws_floats(a);
            ds41::sparse_attn_tc_h32(a, n ? sc.alloc(n) : nullptr, g.s);
        } else if (!v1 && tc == 4 && ds41::sparse_attn_tc_16w_ok(a)) {   // v5: 32 heads per CTA, 16 warps, S split over k
            const size_t n = ds41::sparse_attn_tc_16w_ws_floats(a);
            ds41::sparse_attn_tc_16w(a, n ? sc.alloc(n) : nullptr, g.s);
        } else if (!v1 && ds41::sparse_attn_tc_ok(a)) {   // v2, tensor cores (#189); TRUSS_DS41_ATTN_V1=1: the v1 kernel
            const size_t n = ds41::sparse_attn_tc_ws_floats(a);
            ds41::sparse_attn_tc(a, n ? sc.alloc(n) : nullptr, g.s);
        } else {
            ds41::sparse_attn(a, g.s);
        }
        ds41::rope_tail(o, rb, H, D, rd, pos0 + r0, 1, d_.inv_freq, true, g.s);
        ds41::to_half(o, (size_t) rb * H * D, o16 + (size_t) r0 * H * D, g.s);
        sc.release(m2);
    }
    // grouped output LoRA: each row's group g of o (H D / G dims) through wo_a's block g (R outputs)
    if (grouped && lin_.act() == Activations::FP16) {
        for (int r0 = 0; r0 < T; r0 += P) {
            const int rb = std::min(P, T - r0);
            dense::q8_gemm_a16_grouped(lin_.q8(w_.out_a), G, o16 + (size_t) r0 * H * D, rb, yo + (size_t) r0 * G * R,
                                       lin_.w16(), g.blas, g.s);
        }
    } else if (grouped) {   // Q8_1: the quantized stacked rows, then the grouped int8 GEMM: the same values as the
        for (int r0 = 0; r0 < T; r0 += P) {   // dense product of the stacked rows with all of W, G times fewer FLOPs
            const int rb = std::min(P, T - r0);
            dense::q8_gemm_grouped(lin_.q8(w_.out_a), G, ia.q + (size_t) r0 * G * W_in,
                                   ia.d + (size_t) r0 * G * (W_in / 32), rb, yo + (size_t) r0 * G * R, g.s);
        }
    } else {   // decode rows: one gemv of the T x G stacked rows reads wo_a once, then each row's own block
        float * ya = sc.alloc((size_t) T * G * G * R);
        lin_.lin(w_.out_a, o16, T * G, ya);
        ds41::pick_groups(ya, T, G, R, yo, g.s);
    }
    ds41::to_half(yo, (size_t) T * G * R, yo16, g.s);
    lin_.lin(w_.out_b, yo16, T, out);
    sc.release(m);
}

}  // namespace truss::blocks
