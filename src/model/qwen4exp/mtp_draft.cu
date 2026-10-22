// MtpDrafter::draft: n drafts after `next` (the token at position pos), a chain of single-row MTP steps, each from
// the previous step's residual and argmax (or a sample of q with draft sampling). The chain's cache rows and indexer
// block are tentative: the snapshot is restored at the end, and the next commit rewrites those positions from the
// target's states. A chain stops early when the block's top probability drops below min_p.
#include "model/qwen4exp/mtp.h"

#include "engine/round_trace.h"
#include "engine/stage.h"
#include "kernels/sampling/argmax.cuh"
#include "kernels/sampling/spec.cuh"

#include <algorithm>
#include <cstdlib>

namespace truss::qwen4exp {

using engine::require;

int MtpDrafter::draft(engine::Stage & st, int slot, int32_t next, int pos, int n, int32_t * out, const Sampling & smp,
                      int rope_delta)
{
    constexpr int W = moe::MAX_ROWS + 1;
    Slot & s = s_[slot];
    require(s.has_pending, "draft() needs a committed position");
    require(n >= 1 && n <= moe::MAX_ROWS, "draft count");
    engine::Gpu & g = st.gpu();
    st.kv_reserve(pos + n + 1, false);
    st.use_small();
    st.moe()->begin_ring();
    g.sc->reset();
    const size_t row = c_.hc_dim();
    const auto * ds_p = static_cast<const sampling::SampleParams *>(smp.params);
    // the deferred commit and the first draft row (pending state = the last accepted row, token = next) as one MTP
    // pass when they fit a window; that row is true, not tentative, so it is kept (ahead)
    int merged = 0;   // rows of the merged pass (0: none)
    if (s.dc_on) {
        s.dc_on = false;
        const int T = s.dc_T;
        const bool ahead_hit = s.ahead && T > 0 && s.dc_tokens[0] == s.ahead_tok && s.dc_pos0 == s.ahead_pos;
        const int skip = s.has_pending && !ahead_hit ? 0 : 1, nc = T - skip;
        static const bool merge_on = [] { const char * e = std::getenv("TRUSS_MTP_MERGE"); return !e || std::atoi(e); }();
        if (merge_on && nc + 1 <= moe::MAX_ROWS && s.dc_pos0 + T == pos && T >= 1) {
            s.ahead = false;
            float * mh = st.extra(0);
            if (!skip) g.copy(mh, s.pending_h, row);
            g.copy(mh + (size_t) (1 - skip) * row, st.res(), (size_t) T * row);
            int32_t * h = commit_host_ + (size_t) commit_flip_ * W;
            commit_flip_ ^= 1;
            std::copy(s.dc_tokens.begin() + skip, s.dc_tokens.begin() + T, h);
            h[nc] = next;
            int * d_ids = g.sc->alloc<int>(nc + 1);
            TRUSS_CUDA(cudaMemcpyAsync(d_ids, h, sizeof(int32_t) * (nc + 1), cudaMemcpyHostToDevice, g.s));
            rows(st, slot, mh, d_ids, s.dc_pos0 + skip, nc + 1, false, true, true, s.dc_delta);
            g.copy(s.pending_h, st.res() + (size_t) (T - 1) * row, row);
            s.has_pending = true;
            s.ahead = true, s.ahead_tok = next, s.ahead_pos = pos;
            merged = nc + 1;
        } else {
            commit(st, slot, s.dc_tokens.data(), s.dc_pos0, T, false, s.dc_delta);
        }
    }
    g.copy(partial_snap_, dsa_->partial(slot), blocks::Dsa<DsaShape>::partial_floats());
    draft_host_[0] = next;
    TRUSS_CUDA(cudaMemcpyAsync(draft_ids_, draft_host_, sizeof(int32_t), cudaMemcpyHostToDevice, g.s));
    if (!merged) g.copy(st.extra(0), s.pending_h, row);
    const blocks::Head & hd = sm_.out_head();
    const int nv = hd.draft_rows();
    int made = 0;
    for (int i = 0; i < n; ++i) {
        if (i > 0 || !merged) rows(st, slot, st.extra(0), draft_ids_ + i, pos + i, 1, false, true, false, rope_delta);
        sampling::argmax_prob(logits_, nv, hd.draft_map(), draft_ids_ + i + 1, draft_prob_ + i, g.s);
        if (ds_p)   // the guess becomes a sample of q (the argmax probability above still decides the stop)
            sampling::draft_sample(logits_, nv, hd.draft_map(), c_.n_vocab, *ds_p, smp.counter + i,
                                   smp.q + (size_t) i * c_.n_vocab, draft_ids_ + i + 1, g.s);
        if (min_p_ > 0.f) {   // an unsure guess does not enter the window, and ends the chain
            TRUSS_RT(engine::rt::kDraftMinP);   // one host round trip per draft step, with the stage held through it
            engine::rt::Wait w(st.rt_st);
            TRUSS_CUDA(cudaMemcpyAsync(draft_prob_host_ + i, draft_prob_ + i, sizeof(float), cudaMemcpyDeviceToHost, g.s));
            g.sync();
            if (draft_prob_host_[i] < min_p_) break;
        }
        ++made;
        if (i + 1 < n) g.copy(st.extra(0), st.extra(1) + (size_t) (i == 0 && merged ? merged - 1 : 0) * row, row);
    }
    g.copy(dsa_->partial(slot), partial_snap_, blocks::Dsa<DsaShape>::partial_floats());
    if (made) TRUSS_CUDA(cudaMemcpyAsync(draft_host_, draft_ids_ + 1, sizeof(int32_t) * made, cudaMemcpyDeviceToHost, g.s));
    TRUSS_RT(engine::rt::kDraftEnd);
    engine::rt::Wait we(st.rt_st);
    g.sync();
    std::copy(draft_host_, draft_host_ + made, out);
    return made;
}

}  // namespace truss::qwen4exp
