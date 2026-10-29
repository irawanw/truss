// engine::Pipeline::prompt and ::round: the concurrent slot calls. With a layer split the first stage runs a chunk
// ahead of the last one on its own thread; a round holds each stage only while its rows are enqueued there, and any
// throw part way through is unwound (accept(0) of each open window, drain of this round's hand-off) so no slot is
// wedged behind it (review C1/C2).
#include "engine/pipeline.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <thread>

namespace truss::engine {

namespace {

// the confidence-scheduled verify (the DeepInfer recipe, item 8b): off by default.
bool varlen_on()
{
    static const bool on = [] {
        const char * e = std::getenv("TRUSS_DSPARK_VARLEN");
        return e && std::atoi(e) != 0;
    }();
    return on;
}

// the cost of a verify pass with n+1 verified rows, in ms: cost(n) = c0 + c1 * n (measured in item 8a)
void varlen_cost(double & c0, double & c1)
{
    static const double k0 = [] {
        const char * e = std::getenv("TRUSS_VARLEN_C0");
        return e ? std::atof(e) : 0.0;
    }();
    static const double k1 = [] {
        const char * e = std::getenv("TRUSS_VARLEN_C1");
        return e ? std::atof(e) : 0.0;
    }();
    c0 = k0;
    c1 = k1;
}

// the prefix n that maximizes (1 + sum cumprod(p)_1..n) / cost(n); returns nw if the model has no confidence
int varlen_pick(const float * p, int cn, int nw, int & nw_drafted)
{
    double c0 = 0, c1 = 0;
    varlen_cost(c0, c1);
    if (!p || cn <= 0 || c1 <= 0) return nw;
    double cum = 1.0, toks = 1.0, best = -1.0;
    int best_n = nw;
    for (int n = 0; n <= nw; ++n) {
        if (n > 0) {
            cum *= p[std::min(n - 1, cn - 1)];
            toks += cum;
        }
        const double val = toks / (c0 + c1 * n);
        if (val > best) {
            best = val;
            best_n = n;
        }
    }
    nw_drafted = nw;
    return std::min(nw, best_n);
}

}  // namespace

void Pipeline::prompt(int slot, const int32_t * tokens, int n, int step, float * logits,
                      const std::function<void()> & after, const std::function<void(int)> & done, const ImageIn * img,
                      int n_img)
{
    require(n > 0 && step > 0, "run_prompt: no tokens");
    require(slot >= 0 && slot < (int) seq_pos_.size(), "prompt(): no such slot");
    rt::ctx().slot = slot, ++rt::ctx().round;   // TRUSS_ROUND_TRACE: the records of this call belong to this slot
    auto chunk = [&](int s, int & T, const int32_t *& nxt, int & nxt_T) {   // the chunks and their lookahead
        T = std::min(step, n - s);
        nxt_T = std::max(0, std::min(step, n - s - T));
        nxt = nxt_T > 0 ? tokens + s + T : nullptr;
    };
    const int pos0 = seq_pos_[slot];
    // 3-D rope positions of an image prompt (HF get_rope_index, one image): text before the image at t, the image rows
    // t = s, h = s + k / nw, w = s + k % nw (merge-block order), text after at s + max(nh, nw); rope_delta for decode
    std::vector<int> p3;
    int delta = rope_delta_[slot];
    if (!img) n_img = 0;
    for (int i = 0; i < n_img; ++i) {
        const ImageIn & g = img[i];
        require(g.n > 0 && g.nh > 0 && g.nw > 0 && g.nh * g.nw == g.n && g.cols > 0 && g.row0 >= 0 && g.row0 + g.n <= n,
                "prompt(): bad image rows");
        require(g.nh == 1 || n_img == 1, "prompt(): several images are 1-D only (nh 1)");
        require(i == 0 || g.row0 >= img[i - 1].row0 + img[i - 1].n, "prompt(): images out of order or overlapping");
    }
    const bool flat = n_img > 0 && img[0].nh == 1;
    // nh == 1: rows of a model with 1-D positions (DS41's image spans, any number per call): no 3-D positions, any
    // start position
    if (n_img > 0 && !flat) {
        require(pos0 == 0, "prompt(): an image prompt must start at position 0 (no prefix restore)");
        const int is = img->row0, iw = img->nw, span = std::max(img->nh, img->nw);
        p3.resize((size_t) 3 * n);
        int mx = 0;
        for (int t = 0; t < n; ++t) {
            int vt, vh, vw;
            if (t < is) vt = vh = vw = t;
            else if (t < is + img->n) {
                const int k = t - is;
                vt = is, vh = is + k / iw, vw = is + k % iw;
            } else vt = vh = vw = is + span + (t - is - img->n);
            p3[t] = vt, p3[(size_t) n + t] = vh, p3[(size_t) 2 * n + t] = vw;
            mx = std::max(mx, std::max(vt, std::max(vh, vw)));
        }
        delta = mx + 1 - n;   // HF: max(position_ids) + 1 - prompt length
        rope_delta_[slot] = delta;
    }
    auto extra = [&](int s, int T) {   // this chunk's image rows and 3-D positions
        ChunkExtra x;
        x.pos3 = p3.empty() ? nullptr : p3.data(), x.pos3_stride = n, x.pos3_row0 = s, x.rope_delta = delta;
        if (flat) {
            x.flat = img, x.n_flat = n_img, x.flat_row0 = s;
        } else if (n_img > 0) {
            const int a = std::max(s, img->row0), b = std::min(s + T, img->row0 + img->n);
            if (b > a) {
                x.img = img->embeds + (size_t) (a - img->row0) * img->cols;
                x.img_row0 = a - s, x.img_n = b - a, x.img_cols = img->cols;
            }
        }
        return x;
    };
    auto last_chunk = [&](int s, int T, long ticket) {   // the last stage's part of a chunk; the head after the last
        const int32_t * nx;
        int TT, nT;
        chunk(s, TT, nx, nT);
        Held<Stage> h(*m_, ticket);
        m_->select(slot);
        m_->settle();
        m_->run_chunk(tokens + s, pos0 + s, T, nullptr, false, nx, nT, extra(s, T));
        inject_fault("prompt-last", slot);   // this chunk's rows are taken; its head and later chunks never run
        if (s + T == n) {
            if (logits) m_->head(T - 1, 1, logits);
            if (after) after();
        }
    };
    if (front_.empty()) {
        for (int s = 0; s < n; s += step) {
            const int T = std::min(step, n - s);
            last_chunk(s, T, 0);
            if (done) done(s + T);
        }
        seq_pos_[slot] += n;
        return;
    }
    // Layer split: every stage before the last runs on its own thread, a chunk ahead of the one after it (the
    // hand-off's handshake orders them), so all devices compute at once. Every row is what run() computes: the same
    // output. Stage i > 0 takes its hand-offs in the order stage i - 1 queued them (tickets[i - 1]).
    const int F = (int) front_.size();
    std::mutex qm;
    std::condition_variable qcv;
    std::vector<std::deque<long>> tickets((size_t) F);   // hand-off ticket of each chunk front stage i enqueued
    bool front_failed = false;
    std::vector<long> front_handed((size_t) F, 0);   // the highest hand-off this request's stage i produced (C1)
    // Strata's decode share: while another slot is decoding, the prompt gives way after each chunk - the first stage
    // waits until the last one has finished the chunks handed to it, then for TRUSS_DECODE_SHARE x that chunk's time
    static const double share = [] { const char * e = std::getenv("TRUSS_DECODE_SHARE"); return e ? std::atof(e) : 0.5; }();
    int main_done = 0;   // chunks the last stage has finished (under qm)
    std::vector<std::chrono::steady_clock::time_point> t_start((size_t) (n + step - 1) / step);
    double chunk_s = 0;  // the last chunk's time on the stages (start to last stage done)
    auto others_decoding = [&] {
        const long long now = std::chrono::steady_clock::now().time_since_epoch().count();
        for (size_t i = 0; i < seq_pos_.size(); ++i)
            if ((int) i != slot) {
                const long long t = round_ns_[i].load();
                if (t < 0 || (t > 0 && now - t < 300'000'000LL)) return true;   // in a round, or one ended < 0.3 s ago
            }
        return false;
    };
    std::vector<std::exception_ptr> errs((size_t) F);
    const auto t0 = std::chrono::steady_clock::now();
    double front_s = 0;
    std::vector<std::thread> threads;
    for (int i = 0; i < F; ++i)
        threads.emplace_back([&, i] {
            Stage & f = *front_[(size_t) i];
            DeviceGuard g(f.device());
            rt::ctx().slot = slot;   // thread-local: the front thread's records are this slot's front-stage work
            try {
                int k = 0;
                for (int s = 0; s < n; s += step, ++k) {
                    int T, nT;
                    const int32_t * nx;
                    chunk(s, T, nx, nT);
                    if (i == 0 && share > 0 && k > 0 && others_decoding()) {   // give way: drain, then the decode window
                        TRUSS_RT(rt::kGiveWay);
                        double w;
                        {
                            std::unique_lock<std::mutex> l(qm);
                            qcv.wait(l, [&] { return main_done >= k || front_failed; });
                            w = share * chunk_s;
                        }
                        std::this_thread::sleep_for(std::chrono::duration<double>(std::min(w, 5.0)));
                    }
                    long tin = 0;
                    {
                        std::unique_lock<std::mutex> l(qm);
                        if (i == 0) t_start[(size_t) k] = std::chrono::steady_clock::now();
                        else {
                            qcv.wait(l, [&] { return !tickets[(size_t) i - 1].empty() || front_failed; });
                            require(!tickets[(size_t) i - 1].empty(), "the previous layer-split stage failed");
                            tin = tickets[(size_t) i - 1].front();
                            tickets[(size_t) i - 1].pop_front();
                        }
                    }
                    long tk;
                    {
                        Held<Stage> h(f, tin);
                        f.select(slot);
                        f.settle();
                        f.run_chunk(tokens + s, pos0 + s, T, nullptr, false, nx, nT, extra(s, T));
                        tk = f.handed.load();
                        {
                            std::lock_guard<std::mutex> l(qm);
                            front_handed[(size_t) i] = tk;   // the hand-off exists: the next stage must take it
                        }
                        inject_fault("prompt-front", slot);
                    }
                    {
                        std::lock_guard<std::mutex> l(qm);
                        tickets[(size_t) i].push_back(tk);
                    }
                    qcv.notify_all();
                }
                TRUSS_RT(rt::kFrontDrain);
                TRUSS_CUDA(cudaStreamSynchronize(f.stream()));
                if (i == F - 1)
                    front_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            } catch (...) {
                errs[(size_t) i] = std::current_exception();
                f.failures.fetch_add(1, std::memory_order_relaxed);   // the next stage's wait for a hand-off stops (H1)
                {
                    std::lock_guard<std::mutex> l(qm);
                    front_failed = true;
                }
                qcv.notify_all();
            }
        });
    std::exception_ptr err2;
    try {
        for (int s = 0; s < n; s += step) {
            const int T = std::min(step, n - s);
            long tk;
            {
                std::unique_lock<std::mutex> l(qm);
                qcv.wait(l, [&] { return !tickets[(size_t) F - 1].empty() || front_failed; });
                require(!tickets[(size_t) F - 1].empty(), "the previous layer-split stage failed");
                tk = tickets[(size_t) F - 1].front();
                tickets[(size_t) F - 1].pop_front();
            }
            last_chunk(s, T, tk);
            {
                std::lock_guard<std::mutex> l(qm);
                chunk_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start[(size_t) main_done]).count();
                ++main_done;
            }
            qcv.notify_all();
            if (done) done(s + T);
        }
    } catch (...) {
        err2 = std::current_exception();
        m_->failures.fetch_add(1, std::memory_order_relaxed);   // the first stage's wait for a take stops (H1)
    }
    for (auto & t : threads) t.join();
    // C1: hand-offs a stage wrote that the next one never read; drained up to the highest hand-off THIS request
    // produced (one belonging to another request would let its chunk read stale rows)
    for (int i = 0; i < F; ++i) {
        long up_to = 0;
        {
            std::lock_guard<std::mutex> l(qm);
            up_to = front_handed[(size_t) i];
        }
        Stage & nx = i + 1 < F ? *front_[(size_t) i + 1] : *m_;
        if (up_to > 0)
            try {
                DeviceGuard g(nx.device());
                Held<Stage> h(nx, 0);
                nx.drain_handoffs(up_to);
            } catch (...) {}   // the request's own error is the one the caller has to see
    }
    std::exception_ptr err;
    for (auto & e : errs)
        if (e && !err) err = e;
    if (m_->prof().on() && !err && !err2)   // how balanced the stages are
        std::fprintf(stderr, "layer split prompt: %d tokens, first stage done at %.2f s, last at %.2f s\n", n, front_s,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (err) std::rethrow_exception(err);
    if (err2) std::rethrow_exception(err2);
    seq_pos_[slot] += n;
}

int Pipeline::round(int slot, int32_t next, int nd, int32_t * win, float * logits, const Drafter::Sampling & ds,
                    const std::function<int(int)> & pick)
{
    require(slot >= 0 && slot < (int) seq_pos_.size(), "round(): no such slot");
    rt::ctx().slot = slot, ++rt::ctx().round;
    TRUSS_RT(rt::kRound);
    const int pos = seq_pos_[slot];
    const int rope_delta = rope_delta_[slot];
    round_ns_[slot] = -1;   // inside a round (the prompt of another slot gives way to it)
    struct Out {
        std::atomic<long long> & t;
        ~Out() { t = std::chrono::steady_clock::now().time_since_epoch().count(); }
    } out{ round_ns_[slot] };
    long handed = 0;      // the last hand-off this round produced on a front stage (0: none)
    int front_open = 0;   // front stages now holding this slot's verify window
    bool last_open = false;
    auto unwind = [&] {
        if (!robust_on()) return;   // C1/C2 fixes off
        if (last_open) {   // first, and before any select(): select() inside the last stage's window is forbidden
            try {
                Held<Stage> h(*m_, 0);
                m_->select(slot);
                if (m_->window_open()) m_->accept(0);
            } catch (...) {}
        }
        for (size_t i = 0; i < front_.size() && (int) i < front_open; ++i) {
            Stage * p = front_[i].get();
            try {
                DeviceGuard g(p->device());
                Held<Stage> h(*p, 0);
                p->select(slot);
                if (p->window_open()) p->accept(0);
            } catch (...) {}
        }
        if (handed > 0) {
            try {
                Held<Stage> h(*m_, 0);
                m_->drain_handoffs(handed);
            } catch (...) {}
        }
    };
    int nw = 0, keep = 0;
    long ticket = 0;
    int nw_sel_ = -1, nw_drafted_ = -1;
    bool logged_ = false;
    try {
        {   // the drafts (the drafter lives on the last stage)
            Held<Stage> h(*m_, 0);
            m_->select(slot);
            inject_fault("round-draft", slot);
            nw = m_->draft(next, pos, nd, win + 1, ds, rope_delta);
        }
        if (varlen_on()) {   // the confidence-scheduled verify: only the prefix the schedule pays for
            int cn = 0, nw_drafted = nw;
            const float * p = m_->draft_confidence(&cn);
            const int nw_sel = varlen_pick(p, cn, nw, nw_drafted);
            nw_sel_ = nw_sel;
            nw_drafted_ = nw_drafted;
            logged_ = true;
            nw = nw_sel;
        }
        win[0] = next;
        const int T = nw + 1;
        require(m_->spec_rows() && T <= m_->spec_rows(),
                "verify() of " + std::to_string(T) + " rows (spec_rows " + std::to_string(m_->spec_rows()) + ")");
        for (auto & p : front_) {   // the window's rows on the earlier stages, held only while they are enqueued
            DeviceGuard g(p->device());
            Held<Stage> h(*p, ticket);
            p->select(slot);
            p->settle();
            p->run_chunk(win, pos, T, nullptr, true, nullptr, 0,
                         ChunkExtra{ nullptr, 0, 0, 0, nullptr, 0, 0, rope_delta });
            ++front_open;
            ticket = p->handed.load();
            handed = ticket;
            TRUSS_CUDA(cudaEventRecord(p->slot_done(slot), p->stream()));
            inject_fault("round-front", slot);
        }
        // the rows reach the last stage's input when the earlier GPUs are done: wait for that holding nothing
        static const bool wait_free = [] { const char * e = std::getenv("TRUSS_ROUND_WAIT_FREE"); return !e || std::atoi(e); }();
        if (wait_free)
            for (auto & p : front_) {
                DeviceGuard g(p->device());
                const int rt_saved = rt::ctx().stage;
                rt::ctx().stage = 0;
                {
                    TRUSS_RT(rt::kFrontWait);
                    TRUSS_CUDA(cudaEventSynchronize(p->slot_done(slot)));
                }
                rt::ctx().stage = rt_saved;
            }
        {   // the last stage: its rows, the head, the caller's pick, the accept
            Held<Stage> h(*m_, ticket);
            m_->select(slot);
            m_->run_chunk(win, pos, T, nullptr, true, nullptr, 0,
                          ChunkExtra{ nullptr, 0, 0, 0, nullptr, 0, 0, rope_delta });
            last_open = true;
            m_->head(0, T, logits);
            inject_fault("round-pick", slot);
            rt::stamp_mid(m_->rt_st);
            {
                TRUSS_RT(rt::kPick);
                keep = pick(nw);
            }
            require(keep >= 1 && keep <= T, "round(): pick() kept " + std::to_string(keep) + " of " + std::to_string(T));
            m_->accept(keep);
            last_open = false;
            if (logged_) {   // the calibration record: pos, the accepted rows, the verified prefix, the drafts
                static FILE * f = std::getenv("TRUSS_ACCEPT_LOG") ? std::fopen(std::getenv("TRUSS_ACCEPT_LOG"), "ab") : nullptr;
                if (f) {
                    const int32_t rec[4] = { pos, keep, nw_sel_, nw_drafted_ };
                    std::fwrite(rec, 4, 4, f);
                    std::fflush(f);
                }
            }
        }
        for (auto & p : front_) p->set_defer(slot, keep);   // read under that stage's lock by this slot's next call
    } catch (...) {
        const std::exception_ptr first = std::current_exception();
        unwind();
        std::rethrow_exception(first);
    }
    seq_pos_[slot] += keep;
    return keep;
}

}  // namespace truss::engine
