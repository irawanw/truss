#include "model/qwen4exp/mtp.h"

#include "engine/stage.h"
#include "kernels/ffn/ffn_ops.cuh"
#include "kernels/hc/hc_prefill.cuh"
#include "kernels/mtp/mtp_ops.cuh"
#include "kernels/sampling/argmax.cuh"
#include "kernels/sampling/spec.cuh"

#include <algorithm>

namespace truss::qwen4exp {

using engine::require;
constexpr int W = moe::MAX_ROWS + 1;   // the largest draft chain / commit staged

MtpDrafter::MtpDrafter(engine::Stage & st, StageModel & sm, const Family & fam)
    : sm_(sm), c_(fam.config()), w_(fam.weights()), m_(*fam.mtp())
{
    engine::Gpu & g = st.gpu();
    dsa_ = std::make_unique<blocks::Dsa<DsaShape>>(st.lin(), m_.layer.dsa, blocks::DsaDims{ c_.d_model, c_.rope_base, c_.rms_eps },
                                                   st.opts().kv_int8);
    partial_snap_ = g.alloc<float>(blocks::Dsa<DsaShape>::partial_floats());
    logits_ = g.alloc<float>(c_.n_vocab);
    draft_ids_ = g.alloc<int>(W);
    TRUSS_CUDA(cudaMallocHost(&draft_host_, sizeof(int32_t) * W));
    TRUSS_CUDA(cudaMallocHost(&commit_host_, sizeof(int32_t) * 2 * W));
    draft_prob_ = g.alloc<float>(W);
    TRUSS_CUDA(cudaMallocHost(&draft_prob_host_, sizeof(float) * W));
    sm_.out_head().set_draft_vocab(fam.drafter_files().vocab);   // gather the subset's output rows once
    min_p_ = fam.drafter_files().min_p;
    if (const char * p = std::getenv("TRUSS_MTP_CALIB")) {
        calib_f_ = std::fopen(p, "ab");
        require(calib_f_ != nullptr, "cannot open TRUSS_MTP_CALIB file");
        TRUSS_CUDA(cudaMallocHost(&calib_host_, sizeof(float) * (size_t) st.max_rows() * c_.d_model));
    }
}

MtpDrafter::~MtpDrafter()
{
    for (void * p : { (void *) draft_host_, (void *) commit_host_, (void *) draft_prob_host_, (void *) calib_host_ })
        if (p) cudaFreeHost(p);
    if (calib_f_) std::fclose(calib_f_);
}

void MtpDrafter::bind_experts(engine::Stage & st)
{
    const blocks::MoeDims md{ c_.d_model, c_.n_expert, c_.n_expert_used, c_.d_ff_shexp };
    moe_ = std::make_unique<blocks::Moe>(st.lin(), *st.moe(), st.moe()->n_store() - 1, c_.n_layer, m_.layer.moe, md,
                                         nullptr, &blocks::route_softmax, true, nullptr);
}

void MtpDrafter::alloc_slots(engine::Stage & st, int n_slots, int seq_ctx, int spec_rows)
{
    dsa_->alloc_slots(st, n_slots, seq_ctx, spec_rows);
    s_.resize(n_slots);
    for (Slot & s : s_) {
        s.pending_h = st.gpu().alloc<float>(c_.hc_dim());
        if (n_slots > 1) s.res_keep = st.gpu().alloc<float>((size_t) W * c_.hc_dim());
    }
}

// The MTP block on T rows: hidden states h [T][hc][d], next tokens d_ids [T] (device) at MTP positions pos0 ..; its
// residual in the stage's extra(1); logits of the last row into logits_ when asked. calib: TRUSS_MTP_CALIB capture.
void MtpDrafter::rows(engine::Stage & st, int slot, const float * h, const int * d_ids, int pos0, int T, bool stream,
                      bool logits, bool calib)
{
    engine::Gpu & g = st.gpu();
    Scratch & sc = *g.sc;
    blocks::Linear & lin = st.lin();
    blocks::Hc & hc = sm_.hc();
    const Layer & L = m_.layer;
    const int nhc = c_.hc, dm = c_.d_model;
    float * mres = st.extra(1);
    const size_t m = sc.mark();
    float * emb = sc.alloc((size_t) T * dm), * rstd = sc.alloc((size_t) T * nhc);
    half * hn16 = sc.alloc<half>((size_t) T * c_.hc_dim()), * cat16 = sc.alloc<half>((size_t) T * nhc * 2 * dm);
    dense::q8_rows(lin.q8(w_.token_embd), d_ids, T, emb, g.s);
    hc::norm(h, lin.f32(m_.hnorm), T, nhc, dm, c_.rms_eps, hn16, rstd, g.s);
    mtp::join(emb, lin.f32(m_.enorm), hn16, T, nhc, dm, c_.rms_eps, cat16, g.s);
    lin.lin(m_.eh_proj, cat16, T * nhc, mres);   // per stream: [2d] -> [d]
    float * mixed = sc.alloc((size_t) T * dm), * out = sc.alloc((size_t) T * dm);
    float * inject = sc.alloc((size_t) T * nhc);
    half * mixed16 = sc.alloc<half>((size_t) T * dm);
    hc.mix(L.hc_attn, mres, T, mixed, mixed16, inject);
    dsa_->forward(g, slot, mixed16, pos0, T, out, false);
    hc.combine(mres, out, inject, T);
    hc.mix(L.hc_ffn, mres, T, mixed, mixed16, inject);
    if (calib && calib_f_) {   // the routed experts' input rows, true path only
        TRUSS_CUDA(cudaMemcpyAsync(calib_host_, mixed, sizeof(float) * (size_t) T * dm, cudaMemcpyDeviceToHost, g.s));
        g.sync();
        require(std::fwrite(calib_host_, sizeof(float) * dm, T, calib_f_) == (size_t) T, "TRUSS_MTP_CALIB write");
    }
    moe_->forward(g, mixed, mixed16, T, out, stream, st.moe_ws(), st.moe_rows());
    hc.combine(mres, out, inject, T);
    if (logits) sm_.head(st, mres + (size_t) (T - 1) * c_.hc_dim(), 1, logits_, true);
    sc.release(m);
}

// After committing rows at positions pos0 .. pos0 + T - 1: the block at those positions, each from the previous
// position's hidden state (pending for the first, the target's residual rows after)
void MtpDrafter::commit(engine::Stage & st, int slot, const int32_t * tokens, int pos0, int T, bool stream)
{
    Slot & s = s_[slot];
    engine::Gpu & g = st.gpu();
    const size_t row = c_.hc_dim();
    const bool ahead_hit = s.ahead && T > 0 && tokens[0] == s.ahead_tok && pos0 == s.ahead_pos;
    s.ahead = false;
    const int skip = s.has_pending && !ahead_hit ? 0 : 1;
    const int n = T - skip;
    if (n > 0) {
        float * mh = st.extra(0);
        if (!skip) g.copy(mh, s.pending_h, row);
        if (T > 1) g.copy(mh + (size_t) (1 - skip) * row, st.res(), (size_t) (T - 1) * row);
        const size_t m = g.sc->mark();
        int * d_ids = g.sc->alloc<int>(n);
        // a decode commit's tokens go through pinned staging (alternating halves), so accept() returns without
        // waiting for the GPU; a prompt chunk's tokens may be a host temporary: that copy is waited for
        const bool staged = n <= W;
        const int32_t * src = tokens + skip;
        if (staged) {
            int32_t * h = commit_host_ + (size_t) commit_flip_ * W;
            commit_flip_ ^= 1;
            std::copy(src, src + n, h);
            src = h;
        }
        TRUSS_CUDA(cudaMemcpyAsync(d_ids, src, sizeof(int32_t) * n, cudaMemcpyHostToDevice, g.s));
        rows(st, slot, mh, d_ids, pos0 + skip, n, stream, false, true);
        if (!staged) g.sync();
        g.sc->release(m);
    }
    g.copy(s.pending_h, st.res() + (size_t) (T - 1) * row, row);
    s.has_pending = true;
}

void MtpDrafter::on_commit(engine::Stage & st, const engine::Chunk & c)
{
    commit(st, c.slot, c.tokens, c.pos0, c.T, c.stream);
}

void MtpDrafter::on_accept(engine::Stage &, int slot, int n, const engine::Window & w)
{
    Slot & s = s_[slot];
    s.dc_on = n > 0, s.dc_pos0 = w.pos0, s.dc_T = n;
    s.dc_tokens.assign(w.tokens, w.tokens + n);
}

void MtpDrafter::flush(engine::Stage & st, int slot)
{
    Slot & s = s_[slot];
    if (!s.dc_on) return;
    s.dc_on = false;
    // a decode pass's commit fetches in ring mode; with two slots the store may be in stream mode now
    st.moe()->begin_ring();
    commit(st, slot, s.dc_tokens.data(), s.dc_pos0, s.dc_T, false);
}

// a deferred commit reads its rows from the stage's small buffer, which the other slot's calls overwrite: parked
void MtpDrafter::park(engine::Stage & st, int slot)
{
    const Slot & s = s_[slot];
    if (s.dc_on) st.gpu().copy(s.res_keep, st.small_res(), (size_t) s.dc_T * c_.hc_dim());
}

void MtpDrafter::unpark(engine::Stage & st, int slot)
{
    const Slot & s = s_[slot];
    if (!s.dc_on) return;
    st.gpu().copy(st.small_res(), s.res_keep, (size_t) s.dc_T * c_.hc_dim());
    st.use_small();
}

void MtpDrafter::reset(engine::Stage &, int slot)
{
    Slot & s = s_[slot];
    s.has_pending = false, s.dc_on = false, s.ahead = false;
}

void MtpDrafter::accept(engine::Stage &, int, int, const engine::Window &) {}   // the block never runs a window

void MtpDrafter::ck_buffers(int slot, std::vector<std::pair<float *, size_t>> & out)
{
    dsa_->ck_buffers(slot, out);
    out.push_back({ s_[slot].pending_h, (size_t) c_.hc_dim() });
}

void MtpDrafter::ck_save(int slot, std::vector<int32_t> & host) { host.push_back(s_[slot].has_pending); }

void MtpDrafter::ck_load(int slot, const std::vector<int32_t> & host, size_t & at)
{
    Slot & s = s_[slot];
    s.has_pending = host[at++] != 0;
    s.dc_on = false, s.ahead = false;   // the next commit recomputes its first row
}

}  // namespace truss::qwen4exp
