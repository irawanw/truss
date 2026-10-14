// A stage's host-side lock for concurrent sequence slots (engine::Pipeline::prompt / round).
//
// FIFO among waiters, except that work which takes a hand-off from the previous stage carries that hand-off's ticket
// and is only granted in ticket order: the stage's input buffer holds one hand-off at a time, taken in the order they
// were handed on. Kernels are only enqueued under it, so two stages hold different slots' work on their GPUs at the
// same time. Waits are bounded (spin_timeout_s): a ticket whose hand-off was abandoned is never granted (review C1),
// and the timeout is an error, never a grant.
#pragma once
#include "engine/util.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>

namespace truss::engine {

struct StageLock {
    std::mutex mu;
    std::condition_variable cv;
    bool held = false;
    uint64_t next_id = 0;
    std::map<uint64_t, long> waiting;   // arrival order -> the hand-off ticket it takes (0: none)
    std::atomic<long long> held_ns{ 0 }, wait_ns{ 0 };   // time held / waited, ns (Pipeline::lock_ms)
    std::chrono::steady_clock::time_point t_held;

    // turn(): the ticket that may go now (the stage's taken hand-offs + 1)
    template <class Turn> void lock(long ticket, Turn turn)
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> l(mu);
        const uint64_t me = next_id++;
        waiting.emplace(me, ticket);
        const double limit = spin_timeout_s();
        try {
            while (!cv.wait_for(l, std::chrono::seconds(1), [&] {
                       if (held) return false;
                       const long t = turn();
                       for (const auto & [id, tk] : waiting)
                           if (tk == 0 || tk == t) return id == me;   // the earliest waiter that may go now
                       return false;
                   }))
                require(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < limit,
                        "stage lock: hand-off ticket " + std::to_string(ticket) + " ungranted for " +
                            std::to_string((int) limit) + " s (TRUSS_SPIN_TIMEOUT_S)");
        } catch (...) {
            waiting.erase(me);   // the queue is how the others order themselves: a leaver must really leave it
            cv.notify_all();
            throw;
        }
        waiting.erase(me);
        held = true;
        t_held = std::chrono::steady_clock::now();
        wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_held - t0).count();
    }
    void unlock()
    {
        {
            std::lock_guard<std::mutex> l(mu);
            held = false;
            held_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_held).count();
        }
        cv.notify_all();
    }
};

template <class M> struct Held {   // a stage held for one piece of work (M::acquire / M::release)
    M & m;
    Held(M & mm, long ticket) : m(mm) { m.acquire(ticket); }
    ~Held() { m.release(); }
    Held(const Held &) = delete;
    Held & operator=(const Held &) = delete;
};

}  // namespace truss::engine
