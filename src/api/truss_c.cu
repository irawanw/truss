// The C API (include/truss/truss.h) over qwen4exp::Forward. The only architecture today is qwen4exp; a second one
// becomes a dispatch on general.architecture here, behind the same functions.
#include "truss/truss.h"

#include "core/cuda_check.h"
#include "formats/gguf.h"
#include "kernels/sampling/argmax.cuh"
#include "kernels/sampling/penalty.cuh"
#include "kernels/sampling/sample.cuh"
#include "kernels/sampling/spec.cuh"
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/forward.h"
#include "model/qwen4exp/weights.h"
#include "runtime/expert_store.h"

#include <atomic>
#include <functional>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace truss;
namespace q = truss::qwen4exp;

struct truss_model {
    std::unique_ptr<gguf::File> file, mtp_file;
    std::unique_ptr<q::Mtp> mtp;
    int drafts = 0;
    int * spec_next = nullptr, * spec_host = nullptr;   // device / pinned argmax of each verify row
    bool spec_sample = false;                // TRUSS_SPEC_SAMPLE=1: sampled drafts + min(1, p/q) acceptance
    std::vector<float *> q_full;             // per slot, device [drafts][n_vocab]: the sampled drafts' q
    int * spec_drafts = nullptr;             // device [drafts]: the window's guesses (used under the last stage's hold)
    q::Config config;
    q::Weights weights;
    float * logits = nullptr;                // device [n_vocab]
    int * next = nullptr;                    // device, argmax
    int n_ctx = 0, max_chunk = 0;
    // Per slot (TRUSS_SLOTS): the random stream of the next sampled row, and the running eval's progress (tokens run /
    // tokens asked, truss_progress). The device buffers above and below are shared: they are only used while the
    // engine holds its last stage for one slot (Forward::prompt's after(), Forward::round's pick()).
    std::vector<uint64_t> counter;
    std::unique_ptr<std::atomic<int>[]> prog_done, prog_total;
    int cur = 0;                             // truss_select(): the slot of the calls without one
    int * pen_cnt = nullptr;                 // device [drafts + 1][n_vocab] penalty counts, zero between calls
    int * pen_hist = nullptr, * pen_host = nullptr;   // device / pinned [drafts + 1][TRUSS_PENALTY_CAP] row histories
    std::unique_ptr<q::Forward> fwd;         // last: its expert budget takes the memory left

    ~truss_model()
    {
        fwd.reset();
        cudaFree(logits);
        cudaFree(next);
        cudaFree(spec_next);
        for (float * q : q_full) cudaFree(q);
        cudaFree(spec_drafts);
        if (spec_host) cudaFreeHost(spec_host);
        cudaFree(pen_cnt);
        cudaFree(pen_hist);
        if (pen_host) cudaFreeHost(pen_host);
    }
};

namespace {

thread_local std::string last_error;

template <class F> auto guarded(F && f, decltype(f()) fail)
{
    try {
        return f();
    } catch (const std::exception & e) {
        last_error = e.what();
        return fail;
    }
}

void check_slot(const truss_model & m, int slot)
{
    if (slot < 0 || slot >= (int) m.counter.size())
        throw std::invalid_argument("truss: slot " + std::to_string(slot) + " of " + std::to_string(m.counter.size()));
}

// append tokens in chunks; the last chunk's last row feeds the head (m.logits), then after() runs while the engine
// still holds the last stage for this slot (it reads m.logits)
void eval(truss_model & m, int slot, const int32_t * tokens, int n, const std::function<void()> & after, int step_max = 0)
{
    check_slot(m, slot);
    if (n <= 0) throw std::invalid_argument("truss_eval: no tokens");
    if (m.fwd->position(slot) + n > m.n_ctx)
        throw std::length_error("truss_eval: sequence of " + std::to_string(m.fwd->position(slot) + n) +
                                " tokens exceeds n_ctx " + std::to_string(m.n_ctx));
    // A short prompt runs as decode-sized chunks (fetch path: only the experts its rows route to move), not as a
    // prompt chunk that streams every cold expert layer (~2.3 s at 256K whatever the length; TRACKER #89).
    const int fetch_max = std::getenv("TRUSS_FETCH_PROMPT") ? std::atoi(std::getenv("TRUSS_FETCH_PROMPT")) : 0;
    int step = n <= fetch_max ? std::min(m.max_chunk, q::Forward::fetch_rows()) : m.max_chunk;
    if (step_max > 0) step = std::min(step, std::max(4, step_max / 4 * 4));   // a smaller chunk (truss_eval_step_s)
    m.prog_done[slot] = 0, m.prog_total[slot] = n;
    // in chunks with the PLE lookahead (the next chunk's trigram rows are read while this chunk computes); a layer
    // split runs its stages a chunk apart (Forward::prompt)
    m.fwd->prompt(slot, tokens, n, step, m.logits, after, [&](int done) { m.prog_done[slot] = done; });
}

void sync(truss_model & m) { TRUSS_CUDA(cudaStreamSynchronize(m.fwd->stream())); }

}  // namespace

extern "C" {

truss_params truss_default_params(void)
{
    truss_params p{};
    p.n_ctx = 65536;
    p.max_chunk = 8192;
    p.drafts = 3;
    p.cpu_share = 0.5f;
    p.cpu_threads = 12;
    p.draft_min_p = 0.5f;
    return p;
}

truss_model * truss_open_params(const char * gguf_path, const truss_params * p)
{
    return guarded(
        [&]() -> truss_model * {
            auto m = std::make_unique<truss_model>();
            m->file = gguf::File::open(gguf_path);
            if (const char * ov = std::getenv("TRUSS_EXPERT_OVERLAY")) {   // X3.1 experts over the served GGUF (#118)
                m->file->overlay(ov);
                std::fprintf(stderr, "expert overlay %s: %zu tensors replaced\n", ov, m->file->overlaid());
            }
            if (m->file->get_string("general.architecture") != "qwen4exp")
                throw std::runtime_error("unsupported architecture " + m->file->get_string("general.architecture"));
            m->config = q::Config::from_gguf(*m->file);
            m->weights = q::bind(*m->file, m->config);
            m->n_ctx = p->n_ctx;
            m->max_chunk = p->max_chunk / 4 * 4;
            if (m->max_chunk <= 0 || m->n_ctx <= 0) throw std::invalid_argument("n_ctx and max_chunk must be positive");
            q::Forward::Options o;
            if (p->expert_usage && *p->expert_usage)
                o.expert_usage = runtime::ExpertStore::load_usage(p->expert_usage, m->config.n_layer, m->config.n_expert);
            if (p->mtp && *p->mtp) {
                if (p->drafts < 1 || p->drafts > 7) throw std::invalid_argument("drafts must be 1..7");
                m->mtp_file = gguf::File::open(p->mtp);
                m->mtp = std::make_unique<q::Mtp>(q::bind_mtp(*m->mtp_file, m->config));
                m->drafts = p->drafts;
                o.mtp = m->mtp.get();
                o.spec_rows = p->drafts + 1;
                o.draft_min_p = p->draft_min_p;
                if (p->draft_vocab && *p->draft_vocab) {
                    FILE * f = std::fopen(p->draft_vocab, "rb");
                    if (!f) throw std::runtime_error(std::string("cannot open ") + p->draft_vocab);
                    int32_t t;
                    while (std::fread(&t, 4, 1, f) == 1) o.draft_vocab.push_back(t);
                    std::fclose(f);
                }
            }
            if (p->cpu_dir && *p->cpu_dir) {
                o.cpu_dir = p->cpu_dir;
                o.cpu_share = p->cpu_share;
                o.cpu_threads = p->cpu_threads;
            }
            o.ple_file = m->file.get();   // PLE rows by O_DIRECT reads (TRUSS_PLE_DIRECT=0: the mapping)
            o.act = q::Activations::FP16;   // served default: 16-bit dense activations (KL 0.040 vs 0.065); TRUSS_ACT=q8 for the old 8-bit path
            q::apply_env(o);   // the bench's decode-tier knobs (TRUSS_CPU_TRELLIS, TRUSS_PCIE_FRAC, TRUSS_HINT_K, ...)
            // drop the shards' file pages once the weights are on the device (a ~47 GB resident peak otherwise, TRACKER #72)
            o.after_upload = [&] {
                m->file->release_pages();
                if (m->mtp_file) m->mtp_file->release_pages();
            };
            // device buffers before the engine: its expert budget takes the memory left
            TRUSS_CUDA(cudaMalloc(&m->logits, sizeof(float) * m->config.n_vocab * (m->drafts + 1)));
            TRUSS_CUDA(cudaMalloc(&m->next, sizeof(int)));
            TRUSS_CUDA(cudaMalloc(&m->spec_next, sizeof(int) * (m->drafts + 1)));
            TRUSS_CUDA(cudaMallocHost(&m->spec_host, sizeof(int) * (m->drafts + 1)));
            if (const char * e = std::getenv("TRUSS_SPEC_SAMPLE")) m->spec_sample = std::atoi(e) != 0 && m->drafts > 0;
            const int n_slots = std::max(1, std::getenv("TRUSS_SLOTS") ? std::atoi(std::getenv("TRUSS_SLOTS")) : 1);
            if (m->spec_sample) {   // 3 MB a slot at 3 drafts
                m->q_full.assign(n_slots, nullptr);
                for (float *& qf : m->q_full) TRUSS_CUDA(cudaMalloc(&qf, sizeof(float) * m->config.n_vocab * m->drafts));
                TRUSS_CUDA(cudaMalloc(&m->spec_drafts, sizeof(int) * m->drafts));
            }
            const size_t rows = m->drafts + 1;   // penalties: 4 MB of counts at 3 drafts
            TRUSS_CUDA(cudaMalloc(&m->pen_cnt, sizeof(int) * rows * m->config.n_vocab));
            TRUSS_CUDA(cudaMemset(m->pen_cnt, 0, sizeof(int) * rows * m->config.n_vocab));
            TRUSS_CUDA(cudaMalloc(&m->pen_hist, sizeof(int) * rows * TRUSS_PENALTY_CAP));
            TRUSS_CUDA(cudaMallocHost(&m->pen_host, sizeof(int) * rows * TRUSS_PENALTY_CAP));
            m->fwd = std::make_unique<q::Forward>(m->config, m->weights, m->n_ctx, m->max_chunk, o);
            m->n_ctx = m->fwd->seq_ctx();   // TRUSS_SLOTS: every limit below is per sequence
            const int ns = m->fwd->slots();
            if (m->spec_sample && ns != n_slots) throw std::logic_error("TRUSS_SLOTS: slot count mismatch");
            m->counter.assign(ns, 0);
            m->prog_done.reset(new std::atomic<int>[ns]);
            m->prog_total.reset(new std::atomic<int>[ns]);
            for (int i = 0; i < ns; ++i) m->prog_done[i] = 0, m->prog_total[i] = 0;
            m->file->release_pages();
            return m.release();
        },
        nullptr);
}

truss_model * truss_open(const char * gguf_path, int n_ctx, int max_chunk, const char * expert_usage)
{
    truss_params p = truss_default_params();
    p.n_ctx = n_ctx, p.max_chunk = max_chunk, p.expert_usage = expert_usage;
    return truss_open_params(gguf_path, &p);
}

void truss_close(truss_model * m) { delete m; }

const char * truss_last_error(void) { return last_error.c_str(); }

int truss_n_vocab(const truss_model * m) { return m->config.n_vocab; }

int truss_n_ctx(const truss_model * m) { return m->n_ctx; }

int truss_position(const truss_model * m) { return m->fwd->position(m->cur); }

int truss_position_s(const truss_model * m, int slot) { return m->fwd->position(slot); }

const char * truss_meta_string(const truss_model * m, const char * key)
{
    const gguf::Value * v = m->file->get(key);
    const auto * s = v ? std::get_if<std::string>(v) : nullptr;
    return s ? s->c_str() : nullptr;
}

int64_t truss_meta_int(const truss_model * m, const char * key, int64_t def)
{
    const gguf::Value * v = m->file->get(key);
    const auto * i = v ? std::get_if<int64_t>(v) : nullptr;
    return i ? *i : def;
}

int truss_reset_s(truss_model * m, int slot)
{
    return guarded(
        [&] {
            check_slot(*m, slot);
            m->fwd->reset(slot);
            // A fresh sequence starts a fresh random stream. Every sampled row draws from (seed, counter), so without
            // this a run's samples depend on how many sampled rows earlier runs consumed.
            m->counter[slot] = 0;
            return 0;
        },
        -1);
}

int truss_reset(truss_model * m) { return truss_reset_s(m, m->cur); }

void truss_progress_s(const truss_model * m, int slot, int * done, int * total)
{
    if (slot < 0 || slot >= (int) m->counter.size()) {
        *done = *total = 0;
        return;
    }
    *done = m->prog_done[slot].load(std::memory_order_relaxed), *total = m->prog_total[slot].load(std::memory_order_relaxed);
}

void truss_progress(const truss_model * m, int * done, int * total) { truss_progress_s(m, m->cur, done, total); }

int truss_eval_s(truss_model * m, int slot, const int32_t * tokens, int n, float * logits)
{
    return guarded(
        [&] {
            eval(*m, slot, tokens, n, [&] {
                if (logits) TRUSS_CUDA(cudaMemcpyAsync(logits, m->logits, sizeof(float) * m->config.n_vocab,
                                                       cudaMemcpyDeviceToHost, m->fwd->stream()));
                sync(*m);
            });
            return 0;
        },
        -1);
}

int truss_eval(truss_model * m, const int32_t * tokens, int n, float * logits) { return truss_eval_s(m, m->cur, tokens, n, logits); }

int truss_eval_step_s(truss_model * m, int slot, const int32_t * tokens, int n, int step, float * logits)
{
    return guarded(
        [&] {
            eval(*m, slot, tokens, n, [&] {
                if (logits) TRUSS_CUDA(cudaMemcpyAsync(logits, m->logits, sizeof(float) * m->config.n_vocab,
                                                       cudaMemcpyDeviceToHost, m->fwd->stream()));
                sync(*m);
            }, step);
            return 0;
        },
        -1);
}

int truss_eval_argmax_s(truss_model * m, int slot, const int32_t * tokens, int n, int32_t * next)
{
    return guarded(
        [&] {
            eval(*m, slot, tokens, n, [&] {
                sampling::argmax(m->logits, m->config.n_vocab, m->next, m->fwd->stream());
                TRUSS_CUDA(cudaMemcpyAsync(next, m->next, sizeof(int), cudaMemcpyDeviceToHost, m->fwd->stream()));
                sync(*m);
            });
            return 0;
        },
        -1);
}

int truss_eval_argmax(truss_model * m, const int32_t * tokens, int n, int32_t * next)
{
    return truss_eval_argmax_s(m, m->cur, tokens, n, next);
}

int truss_drafts(const truss_model * m) { return m->drafts; }

int truss_slots(const truss_model * m) { return m->fwd->slots(); }

int truss_select(truss_model * m, int slot)
{
    return guarded(
        [&] {
            check_slot(*m, slot);
            m->fwd->select(slot);
            m->cur = slot;
            return 0;
        },
        -1);
}

int truss_checkpoint_s(truss_model * m, int slot)
{
    return guarded([&] { check_slot(*m, slot); m->fwd->checkpoint(slot); return 0; }, -1);
}

int truss_checkpoint(truss_model * m) { return truss_checkpoint_s(m, m->cur); }

int truss_restore_s(truss_model * m, int slot)
{
    return guarded([&] { check_slot(*m, slot); return m->fwd->restore(slot); }, -2);
}

int truss_restore(truss_model * m) { return truss_restore_s(m, m->cur); }

namespace {
sampling::SampleParams params_of(const truss_sampling * sp)
{
    if (!sp || !(sp->temperature > 0.f)) throw std::invalid_argument("truss sampling: temperature must be > 0");
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
    sampling::penalize(m.logits, m.config.n_vocab, rows, m.pen_hist, w, pp, m.pen_cnt, m.fwd->stream());
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
    sampling::ban(m.logits, m.config.n_vocab, k, b, m.fwd->stream());
}
}  // namespace

int truss_eval_sample_s(truss_model * m, int slot, const int32_t * tokens, int n, const truss_sampling * sp,
                        int32_t * next)
{
    return guarded(
        [&] {
            const sampling::SampleParams p = params_of(sp);
            eval(*m, slot, tokens, n, [&] {
                penalize_rows(*m, *sp, nullptr, 0, 1);
                ban_rows(*m, *sp, tokens, n, 1);
                sampling::sample(m->logits, m->config.n_vocab, 1, p, m->counter[slot]++, m->next, m->fwd->stream());
                TRUSS_CUDA(cudaMemcpyAsync(next, m->next, sizeof(int), cudaMemcpyDeviceToHost, m->fwd->stream()));
                sync(*m);
            });
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
            const int V = m->config.n_vocab;
            const int nd = std::min(m->drafts, m->n_ctx - m->fwd->position(slot) - 1);
            uint64_t & ctr = m->counter[slot];
            if (nd < 1) {   // no room for a window: one plain step
                eval(*m, slot, &next, 1, [&] {
                    penalize_rows(*m, *sp, &next, 1, 1);
                    ban_rows(*m, *sp, &next, 1, 1);
                    sampling::sample(m->logits, V, 1, p, ctr++, m->next, m->fwd->stream());
                    TRUSS_CUDA(cudaMemcpyAsync(new_next, m->next, sizeof(int), cudaMemcpyDeviceToHost, m->fwd->stream()));
                    sync(*m);
                });
                emitted[0] = next, *n_emitted = 1;
                return 0;
            }
            int32_t win[8];
            if (m->spec_sample) {   // speculative sampling: lossless, accepts sum min(p, q) instead of p(argmax q)
                float * qf = m->q_full[slot];
                const uint64_t c0 = ctr;
                const int keep = m->fwd->round(slot, next, nd, win, m->logits, &p, c0, qf, [&](int nw) {
                    ctr = c0 + nd;
                    penalize_rows(*m, *sp, win, nw + 1, nw + 1);
                    ban_rows(*m, *sp, win, nw + 1, nw + 1);
                    sampling::probs_inplace(m->logits, V, nw + 1, p, m->fwd->stream());
                    if (nw) TRUSS_CUDA(cudaMemcpyAsync(m->spec_drafts, win + 1, sizeof(int) * nw, cudaMemcpyHostToDevice,
                                                       m->fwd->stream()));
                    sampling::spec_accept(m->logits, qf, m->spec_drafts, nw, V, p.seed, ctr, m->spec_next,
                                          m->fwd->stream());
                    ctr += nw + 1;
                    TRUSS_CUDA(cudaMemcpyAsync(m->spec_host, m->spec_next, sizeof(int) * 2, cudaMemcpyDeviceToHost,
                                               m->fwd->stream()));
                    sync(*m);
                    *new_next = m->spec_host[1];
                    return m->spec_host[0] + 1;
                });
                for (int i = 0; i < keep; ++i) emitted[i] = win[i];
                *n_emitted = keep;
                return 0;
            }
            const int keep = m->fwd->round(slot, next, nd, win, m->logits, nullptr, 0, nullptr, [&](int nw) {
                penalize_rows(*m, *sp, win, nw + 1, nw + 1);
                ban_rows(*m, *sp, win, nw + 1, nw + 1);
                sampling::sample(m->logits, V, nw + 1, p, ctr, m->spec_next, m->fwd->stream());
                ctr += nw + 1;
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
                    sampling::argmax(m->logits, m->config.n_vocab, m->next, m->fwd->stream());
                    TRUSS_CUDA(cudaMemcpyAsync(new_next, m->next, sizeof(int), cudaMemcpyDeviceToHost, m->fwd->stream()));
                    sync(*m);
                });
                emitted[0] = next, *n_emitted = 1;
                return 0;
            }
            int32_t win[8];
            const int V = m->config.n_vocab;
            const int keep = m->fwd->round(slot, next, nd, win, m->logits, nullptr, 0, nullptr, [&](int nw) {
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
