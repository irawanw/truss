// engine::Pipeline::prompt and ::round: the concurrent slot calls. With a layer split the first stage runs a chunk
// ahead of the last one on its own thread; a round holds each stage only while its rows are enqueued there, and any
// throw part way through is unwound (accept(0) of each open window, drain of this round's hand-off) so no slot is
// wedged behind it (review C1/C2).
#include "engine/pipeline.h"

#include <condition_variable>
#include <deque>
#include <thread>

namespace truss::engine {

void Pipeline::prompt(int slot, const int32_t * tokens, int n, int step, float * logits,
                      const std::function<void()> & after, const std::function<void(int)> & done)
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
    auto last_chunk = [&](int s, int T, long ticket) {   // the last stage's part of a chunk; the head after the last
        const int32_t * nx;
        int TT, nT;
        chunk(s, TT, nx, nT);
        Held<Stage> h(*m_, ticket);
        m_->select(slot);
        m_->settle();
        m_->run_chunk(tokens + s, pos0 + s, T, nullptr, false, nx, nT);
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
    // Layer split: the first stage runs on its own thread, a chunk ahead of the last one (the hand-off's handshake
    // orders them), so both devices compute at once. Every row is what run() computes: the same output.
    require(front_.size() == 1, "prompt(): more than two stages");
    Stage & f = *front_[0];
    std::mutex qm;
    std::condition_variable qcv;
    std::deque<long> tickets;   // the hand-off ticket of each chunk the first stage has enqueued
    bool front_failed = false;
    long front_handed = 0;      // the highest hand-off this request's first stage produced (its own to drain, C1)
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
    std::exception_ptr err;
    const auto t0 = std::chrono::steady_clock::now();
    double front_s = 0;
    std::thread t([&] {
        DeviceGuard g(f.device());
        rt::ctx().slot = slot;   // thread-local: the front thread's records are this slot's front-stage work
        try {
            int k = 0;
            for (int s = 0; s < n; s += step, ++k) {
                int T, nT;
                const int32_t * nx;
                chunk(s, T, nx, nT);
                if (share > 0 && k > 0 && others_decoding()) {   // give way: drain, then the decode window
                    TRUSS_RT(rt::kGiveWay);
                    double w;
                    {
                        std::unique_lock<std::mutex> l(qm);
                        qcv.wait(l, [&] { return main_done >= k || front_failed; });
                        w = share * chunk_s;
                    }
                    std::this_thread::sleep_for(std::chrono::duration<double>(std::min(w, 5.0)));
                }
                {
                    std::lock_guard<std::mutex> l(qm);
                    t_start[(size_t) k] = std::chrono::steady_clock::now();
                }
                long tk;
                {
                    Held<Stage> h(f, 0);
                    f.select(slot);
                    f.settle();
                    f.run_chunk(tokens + s, pos0 + s, T, nullptr, false, nx, nT);
                    tk = f.handed.load();
                    {
                        std::lock_guard<std::mutex> l(qm);
                        front_handed = tk;   // the hand-off exists: the last stage must take it even if we die now
                    }
                    inject_fault("prompt-front", slot);
                }
                {
                    std::lock_guard<std::mutex> l(qm);
                    tickets.push_back(tk);
                }
                qcv.notify_all();
            }
            TRUSS_RT(rt::kFrontDrain);
            TRUSS_CUDA(cudaStreamSynchronize(f.stream()));
            front_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        } catch (...) {
            err = std::current_exception();
            f.failures.fetch_add(1, std::memory_order_relaxed);   // the last stage's wait for a hand-off stops (H1)
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
                qcv.wait(l, [&] { return !tickets.empty() || front_failed; });
                require(!tickets.empty(), "the previous layer-split stage failed");
                tk = tickets.front();
                tickets.pop_front();
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
    t.join();
    // C1: hand-offs the first stage wrote that the last one never read; drained up to the highest hand-off THIS
    // request produced (one belonging to another request would let its chunk read stale rows)
    {
        long up_to = 0;
        {
            std::lock_guard<std::mutex> l(qm);
            up_to = front_handed;
        }
        if (up_to > 0)
            try {
                Held<Stage> h(*m_, 0);
                m_->drain_handoffs(up_to);
            } catch (...) {}   // the request's own error is the one the caller has to see
    }
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
    try {
        {   // the drafts (the drafter lives on the last stage)
            Held<Stage> h(*m_, 0);
            m_->select(slot);
            inject_fault("round-draft", slot);
            nw = m_->draft(next, pos, nd, win + 1, ds);
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
            p->run_chunk(win, pos, T, nullptr, true);
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
            m_->run_chunk(win, pos, T, nullptr, true);
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
