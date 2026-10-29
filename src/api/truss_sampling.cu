// The C API's device sampling and speculative steps (include/truss/truss.h): truss_eval_sample, truss_spec_step
// (greedy) and truss_spec_step_sampled (Strata's sampled verify), with penalties, the token ban and the word-repeat penalty.
#include "api/api_internal.h"

namespace truss::api {
sampling::SampleParams params_of(const truss_sampling * sp)
{
    if (!sp || !(sp->temperature > 0.f)) throw std::invalid_argument("truss sampling: temperature must be > 0");
    // Reject a penalty set that penalize_rows() would throw on, here, where no engine state is open yet. Its check
    // stays (defense in depth), but from a spec step it would fire inside Forward::round's pick() - after the verify
    // window is open - which is the state review C2 found the engine could not recover from (review R3: a request
    // with repetition_penalty: 0 must fail, not poison the slot or the one next to it).
    sampling::PenaltyParams pp;
    pp.repeat = sp->repetition_penalty, pp.frequency = sp->frequency_penalty, pp.presence = sp->presence_penalty;
    if (sampling::penalties_on(pp) && !(pp.repeat > 0.f))
        throw std::invalid_argument("truss sampling: repetition_penalty must be > 0");
    sampling::SampleParams p;
    p.temperature = sp->temperature, p.top_p = sp->top_p, p.top_k = sp->top_k, p.min_p = sp->min_p, p.seed = sp->seed;
    return p;
}

// Penalties on rows [0, rows) of m->logits: row t counts the last w of (history, feed[0 .. t]) (feed: the tokens
// the window feeds; nfeed 0 for truss_eval_sample, whose caller puts what counts in history).
void penalize_rows(truss_model & m, const truss_sampling & sp, const int32_t * feed, int nfeed, int rows)
{
    sampling::PenaltyParams pp;
    pp.repeat = sp.repetition_penalty, pp.frequency = sp.frequency_penalty, pp.presence = sp.presence_penalty;
    if (!sampling::penalties_on(pp)) return;
    if (!(pp.repeat > 0.f)) throw std::invalid_argument("truss sampling: repetition_penalty must be > 0");
    const int w = sp.penalty_last_n > 0 && sp.penalty_last_n < TRUSS_PENALTY_CAP ? sp.penalty_last_n : TRUSS_PENALTY_CAP;
    const int nh = sp.history ? std::max(0, sp.n_history) : 0;
    sampling::penalty_rows(sp.history, nh, feed, nfeed, rows, w, m.pen_host);
    TRUSS_CUDA(cudaMemcpyAsync(m.pen_hist, m.pen_host, sizeof(int) * rows * w, cudaMemcpyHostToDevice, m.fwd->stream()));
    sampling::penalize(m.logits, m.n_vocab, rows, m.pen_hist, w, pp, m.pen_cnt, m.fwd->stream());
}
// The ban (truss_sampling::ban) on rows [0, rows) of m->logits. Row r has been fed feed[0 .. r + nfeed - rows]
// (a verify window feeds one token per row; truss_eval_sample feeds n tokens for its one row); the ban holds for the
// rows before the first one whose fed tokens include ban_lift.
void ban_rows(truss_model & m, const truss_sampling & sp, const int32_t * feed, int nfeed, int rows)
{
    if (!sp.ban || sp.n_ban <= 0) return;
    if (sp.n_ban > sampling::BAN_MAX) throw std::invalid_argument("truss sampling: n_ban must be <= 8");
    sampling::BanList b;
    b.n = sp.n_ban;
    for (int i = 0; i < b.n; ++i) b.id[i] = sp.ban[i];
    int k = rows;   // rows [0, k) are banned
    if (sp.ban_lift >= 0)
        for (int t = 0; t < nfeed; ++t)
            if (feed[t] == sp.ban_lift) { k = std::max(0, std::min(rows, t - (nfeed - rows))); break; }
    sampling::ban(m.logits, m.n_vocab, k, b, m.fwd->stream());
}
// The word-repeat penalty (truss_sampling::word_repeat) on rows [0, rows): row r follows feed[r + nfeed - rows]
// (nfeed 0: history's last token), and stops at the first row whose fed tokens include word_lift, as the ban does.
void word_rows(truss_model & m, const truss_sampling & sp, const int32_t * feed, int nfeed, int rows)
{
    const bool fill = sp.filler > 0.f && !m.filler_seqs.empty();
    if ((!(sp.word_repeat > 0.f) && !fill) || !m.word_cls) return;
    if (rows > sampling::WORD_ROWS) throw std::invalid_argument("truss sampling: too many rows for word_repeat");
    int k = rows;
    if (sp.word_lift >= 0)
        for (int t = 0; t < nfeed; ++t)
            if (feed[t] == sp.word_lift) { k = std::max(0, std::min(rows, t - (nfeed - rows))); break; }
    sampling::WordPrev prev;
    const int V = (int) m.word_cls_host.size();
    for (int r = 0; r < k; ++r) {
        const int i = r + nfeed - rows;
        const int32_t t = i >= 0 && i < nfeed ? feed[i] : (sp.history && sp.n_history > 0 ? sp.history[sp.n_history - 1] : -1);
        const int c = t >= 0 && t < V ? m.word_cls_host[t] : -1;
        prev.c[r] = !(sp.word_repeat > 0.f) ? -1 : c <= -2 ? -c - 2 : c;   // a glued piece (<= -2) names its word
    }
    if (fill) {   // row r's context: (history, feed[0 .. r + nfeed - rows]); its window: the last filler_window of it
        const int W = sp.filler_window > 0 ? sp.filler_window : 64, nh = sp.history ? std::max(0, sp.n_history) : 0;
        std::vector<int32_t> ctx;
        for (int r = 0; r < k; ++r) {
            const int nfr = std::max(0, std::min(nfeed, r + nfeed - rows + 1)), total = nh + nfr;
            ctx.clear();
            for (int i = std::max(0, total - W); i < total; ++i) ctx.push_back(i < nh ? sp.history[i] : feed[i - nh]);
            const int L = (int) ctx.size();
            for (const auto & q : m.filler_seqs) {
                const int ql = (int) q.size();
                int uses = 0;   // the sequence's uses in the window
                for (int i = 0; i + ql <= L; ++i) {
                    int j = 0;
                    while (j < ql && ctx[i + j] == q[j]) ++j;
                    uses += j == ql;
                }
                if (!uses) continue;
                // the word is chosen at its first token (" Hmm" = " H" + "mm": once " H" is out, "mm" is near certain),
                // so the first token carries the penalty; the last one too when the context ends with the prefix
                bool pre = ql - 1 <= L;   // the context ends with the sequence's prefix
                for (int j = 0; j < ql - 1 && pre; ++j) pre = ctx[L - (ql - 1) + j] == q[j];
                if (ql > 1 && prev.nf[r] < sampling::FILL_MAX) {
                    prev.ft[r][prev.nf[r]] = q[0];
                    prev.fp[r][prev.nf[r]++] = sp.filler * uses;
                }
                if (pre && prev.nf[r] < sampling::FILL_MAX) {
                    prev.ft[r][prev.nf[r]] = q[ql - 1];
                    prev.fp[r][prev.nf[r]++] = sp.filler * uses;
                }
            }
        }
    }
    sampling::word_repeat(m.logits, m.n_vocab, k, m.word_cls, prev, sp.word_repeat, m.fwd->stream());
}
}  // namespace truss::api

using namespace truss::api;

extern "C" {

int truss_eval_sample_s(truss_model * m, int slot, const int32_t * tokens, int n, const truss_sampling * sp,
                        int32_t * next)
{
    return guarded(
        [&] {
            const sampling::SampleParams p = params_of(sp);
            eval(*m, slot, tokens, n, [&] {
                penalize_rows(*m, *sp, nullptr, 0, 1);
                ban_rows(*m, *sp, tokens, n, 1);
                word_rows(*m, *sp, tokens, n, 1);
                sampling::sample(m->logits, m->n_vocab, 1, p, m->counter[slot]++, m->next, m->fwd->stream());
                TRUSS_CUDA(cudaMemcpyAsync(next, m->next, sizeof(int), cudaMemcpyDeviceToHost, m->fwd->stream()));
                sync(*m);
            });
            return 0;
        },
        -1);
}

int truss_eval_img_sample_s(truss_model * m, int slot, const int32_t * tokens, int n, const float * embeds,
                            int n_img, int cols, int img_row0, int nh, int nw, const truss_sampling * sp,
                            int32_t * next)
{
    return guarded(
        [&] {
            if (n_img <= 0 || nh <= 0 || nw <= 0 || nh * nw != n_img || cols <= 0)
                throw std::invalid_argument("truss_eval_img_sample_s: bad image grid/width");
            const sampling::SampleParams p = params_of(sp);
            const engine::ImageIn img{ embeds, img_row0, n_img, cols, nh, nw };
            eval(*m, slot, tokens, n, [&] {
                penalize_rows(*m, *sp, nullptr, 0, 1);
                ban_rows(*m, *sp, tokens, n, 1);
                word_rows(*m, *sp, tokens, n, 1);
                sampling::sample(m->logits, m->n_vocab, 1, p, m->counter[slot]++, m->next, m->fwd->stream());
                TRUSS_CUDA(cudaMemcpyAsync(next, m->next, sizeof(int), cudaMemcpyDeviceToHost, m->fwd->stream()));
                sync(*m);
            }, 0, &img);
            return 0;
        },
        -1);
}

int truss_eval_imgs_sample_s(truss_model * m, int slot, const int32_t * tokens, int n, const float * embeds, int cols,
                             int n_img, const int * rows0, const int * lens, const truss_sampling * sp, int32_t * next)
{
    return guarded(
        [&] {
            const auto imgs = images_in(embeds, cols, n_img, rows0, lens);
            const sampling::SampleParams p = params_of(sp);
            eval(*m, slot, tokens, n, [&] {
                penalize_rows(*m, *sp, nullptr, 0, 1);
                ban_rows(*m, *sp, tokens, n, 1);
                word_rows(*m, *sp, tokens, n, 1);
                sampling::sample(m->logits, m->n_vocab, 1, p, m->counter[slot]++, m->next, m->fwd->stream());
                TRUSS_CUDA(cudaMemcpyAsync(next, m->next, sizeof(int), cudaMemcpyDeviceToHost, m->fwd->stream()));
                sync(*m);
            }, 0, imgs.data(), n_img);
            return 0;
        },
        -1);
}

int truss_eval_sample(truss_model * m, const int32_t * tokens, int n, const truss_sampling * sp, int32_t * next)
{
    return truss_eval_sample_s(m, m->cur, tokens, n, sp, next);
}

int truss_spec_step_sampled_s(truss_model * m, int slot, int32_t next, const truss_sampling * sp, int32_t * emitted,
                              int * n_emitted, int32_t * new_next)
{
    return guarded(
        [&] {
            if (!m->drafts) throw std::logic_error("truss_spec_step_sampled: opened without an MTP block");
            check_slot(*m, slot);
            const sampling::SampleParams p = params_of(sp);
            const int V = m->n_vocab;
            const int nd = std::min(m->drafts, m->n_ctx - m->fwd->position(slot) - 1);
            uint64_t & ctr = m->counter[slot];
            if (nd < 1) {   // no room for a window: one plain step
                eval(*m, slot, &next, 1, [&] {
                    penalize_rows(*m, *sp, &next, 1, 1);
                    ban_rows(*m, *sp, &next, 1, 1);
                    word_rows(*m, *sp, &next, 1, 1);
                    sampling::sample(m->logits, V, 1, p, ctr++, m->next, m->fwd->stream());
                    TRUSS_CUDA(cudaMemcpyAsync(new_next, m->next, sizeof(int), cudaMemcpyDeviceToHost, m->fwd->stream()));
                    { TRUSS_RT(rt::kPlainSync); rt::Wait w(*m->fwd->rt_last_stage()); sync(*m); }
                });
                emitted[0] = next, *n_emitted = 1;
                return 0;
            }
            int32_t win[WINDOW_ROWS];
            if (m->spec_sample) {   // speculative sampling: lossless, accepts sum min(p, q) instead of p(argmax q)
                float * qf = m->q_full[slot];
                const uint64_t c0 = ctr;
                const int keep = m->fwd->round(slot, next, nd, win, m->logits, engine::Drafter::Sampling{ &p, c0, qf }, [&](int nw) {
                    ctr = c0 + nd;
                    penalize_rows(*m, *sp, win, nw + 1, nw + 1);
                    ban_rows(*m, *sp, win, nw + 1, nw + 1);
                    word_rows(*m, *sp, win, nw + 1, nw + 1);
                    sampling::probs_inplace(m->logits, V, nw + 1, p, m->fwd->stream());
                    if (nw) TRUSS_CUDA(cudaMemcpyAsync(m->spec_drafts, win + 1, sizeof(int) * nw, cudaMemcpyHostToDevice,
                                                       m->fwd->stream()));
                    sampling::spec_accept(m->logits, qf, m->spec_drafts, nw, V, p.seed, ctr, m->spec_next,
                                          m->fwd->stream());
                    ctr += nw + 1;
                    TRUSS_CUDA(cudaMemcpyAsync(m->spec_host, m->spec_next, sizeof(int) * 2, cudaMemcpyDeviceToHost,
                                               m->fwd->stream()));
                    { TRUSS_RT(rt::kPickSync); rt::Wait w(*m->fwd->rt_last_stage());   // the sampled token: the
                      sync(*m); }                                                             // last stage held through it
                    *new_next = m->spec_host[1];
                    return m->spec_host[0] + 1;
                });
                for (int i = 0; i < keep; ++i) emitted[i] = win[i];
                *n_emitted = keep;
                return 0;
            }
            const int keep = m->fwd->round(slot, next, nd, win, m->logits, engine::Drafter::Sampling{}, [&](int nw) {
                penalize_rows(*m, *sp, win, nw + 1, nw + 1);
                ban_rows(*m, *sp, win, nw + 1, nw + 1);
                word_rows(*m, *sp, win, nw + 1, nw + 1);
                sampling::sample(m->logits, V, nw + 1, p, ctr, m->spec_next, m->fwd->stream());
                ctr += nw + 1;
                TRUSS_CUDA(cudaMemcpyAsync(m->spec_host, m->spec_next, sizeof(int) * (nw + 1), cudaMemcpyDeviceToHost,
                                           m->fwd->stream()));
                { TRUSS_RT(rt::kPickSyncQ); rt::Wait w(*m->fwd->rt_last_stage()); sync(*m); }
                int j = 0;
                while (j < nw && m->spec_host[j] == win[j + 1]) ++j;
                *new_next = m->spec_host[j];
                return j + 1;
            });
            for (int i = 0; i < keep; ++i) emitted[i] = win[i];
            *n_emitted = keep;
            return 0;
        },
        -1);
}

int truss_spec_step_sampled(truss_model * m, int32_t next, const truss_sampling * sp, int32_t * emitted,
                            int * n_emitted, int32_t * new_next)
{
    return truss_spec_step_sampled_s(m, m->cur, next, sp, emitted, n_emitted, new_next);
}

int truss_spec_step_s(truss_model * m, int slot, int32_t next, int32_t * emitted, int * n_emitted, int32_t * new_next)
{
    return guarded(
        [&] {
            if (!m->drafts) throw std::logic_error("truss_spec_step: opened without an MTP block");
            check_slot(*m, slot);
            const int nd = std::min(m->drafts, m->n_ctx - m->fwd->position(slot) - 1);
            if (nd < 1) {   // no room for a window: one plain step
                eval(*m, slot, &next, 1, [&] {
                    sampling::argmax(m->logits, m->n_vocab, m->next, m->fwd->stream());
                    TRUSS_CUDA(cudaMemcpyAsync(new_next, m->next, sizeof(int), cudaMemcpyDeviceToHost, m->fwd->stream()));
                    sync(*m);
                });
                emitted[0] = next, *n_emitted = 1;
                return 0;
            }
            int32_t win[WINDOW_ROWS];
            const int V = m->n_vocab;
            const int keep = m->fwd->round(slot, next, nd, win, m->logits, engine::Drafter::Sampling{}, [&](int nw) {
                for (int r = 0; r <= nw; ++r)
                    sampling::argmax(m->logits + (size_t) r * V, V, m->spec_next + r, m->fwd->stream());
                TRUSS_CUDA(cudaMemcpyAsync(m->spec_host, m->spec_next, sizeof(int) * (nw + 1), cudaMemcpyDeviceToHost,
                                           m->fwd->stream()));
                sync(*m);
                int j = 0;
                while (j < nw && m->spec_host[j] == win[j + 1]) ++j;
                *new_next = m->spec_host[j];
                return j + 1;
            });
            for (int i = 0; i < keep; ++i) emitted[i] = win[i];
            *n_emitted = keep;
            return 0;
        },
        -1);
}

int truss_spec_step(truss_model * m, int32_t next, int32_t * emitted, int * n_emitted, int32_t * new_next)
{
    return truss_spec_step_s(m, m->cur, next, emitted, n_emitted, new_next);
}

}  // extern "C"
