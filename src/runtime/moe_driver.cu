// MoeExec's doorbell driver thread (TRACKER #61): for each queued decode FFN, wait for its doorbell, split the routed
// experts between the CPU tier and the GPU (static set, or Strata's dynamic split, TRACKER #73), start the CPU work,
// issue the expert copies and the go word, write the plan, publish the CPU result. Plus the hint prefetch and the
// idle-link admission (TRACKER #113), which run here because this thread owns the copy stream's order.
#include "runtime/moe_exec_state.h"

#include <algorithm>
#include <cstring>
#include <immintrin.h>

namespace truss::runtime {

// /proc/self/task/<tid>/stat field 39: the logical CPU the thread last ran on (-2: not readable). Field 2 (comm) can
// hold spaces and parentheses, so parse from after the last ')'.
int MoeExec::last_cpu_of(pid_t tid) const
{
    char path[64], buf[2048];
    std::snprintf(path, sizeof path, "/proc/self/task/%ld/stat", (long) tid);
    FILE * f = std::fopen(path, "r");
    if (!f) return -2;
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    buf[n] = 0;
    char * p = buf + n;
    while (p > buf && *(p - 1) != ')') --p;
    for (int i = 0; i < 36; ++i) {   // fields 3..38 precede the processor field
        while (p < buf + n && *p == ' ') ++p;
        while (p < buf + n && *p != ' ') ++p;
    }
    while (p < buf + n && *p == ' ') ++p;
    return p < buf + n ? (int) std::strtol(p, nullptr, 10) : -2;
}

void MoeExec::setup_doorbells()
{
    const int R = cpu::MAX_ROWS, d = k_.d_model;
    bells_.resize(n_store());
    auto mapped = [](void ** p, size_t bytes) {
        TRUSS_CUDA(cudaHostAlloc(p, bytes, cudaHostAllocMapped));
        std::memset(*p, 0, bytes);
    };
    for (Bell & b : bells_) {
        mapped((void **) &b.ids, sizeof(int) * R * K_);
        mapped((void **) &b.pred, sizeof(int) * R * K_);
        mapped((void **) &b.mids, sizeof(int) * R * K_);
        mapped((void **) &b.w, sizeof(float) * R * K_);
        mapped((void **) &b.x, sizeof(float) * R * d);
        mapped((void **) &b.y, sizeof(float) * R * d);
        mapped((void **) &b.doorbell, sizeof(int));
        mapped((void **) &b.cpu_done, sizeof(int));
        mapped((void **) &b.plan, sizeof(int));
        mapped((void **) &b.need_copy, sizeof(int));
        b.go = g_.alloc<int>(1);
        b.arrive = g_.alloc<unsigned>(1);   // TRUSS_ROUTE_PUBLISH; the kernel zeroes it again after every launch
        b.d_mids = g_.alloc<int>((size_t) R * K_);
    }
    drv_ts_ = std::getenv("TRUSS_DRIVER_TS") && std::atoi(std::getenv("TRUSS_DRIVER_TS"));
    driver_ = std::thread([this] { drive(); });
    // TRUSS_DRIVER_DEBUG=1: a watchdog that prints the driver's phase with the stuck layer's doorbell words when no
    // job is served for 10 s (phases: 1 job spin, 2 job sleep, 3 doorbell spin, 4 serve, 5 fetch, 6 claim, 7 pool
    // wait, 8 admit_step)
    dbg_on_ = std::getenv("TRUSS_DRIVER_DEBUG") && std::atoi(std::getenv("TRUSS_DRIVER_DEBUG"));
    if (dbg_on_)
        watchdog_ = std::thread([this] {
            long last = -1;
            int still = 0;
            while (!driver_stop_) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                const long s = dbg_served_.load();
                still = s == last && jobs_pushed_.load() > s ? still + 1 : 0;
                last = s;
                if (still == 10) {
                    const int l = dbg_layer_.load();
                    std::fprintf(stderr, "driver watchdog: served %ld, pushed %ld, phase %d, layer %d seq %d", s,
                                 jobs_pushed_.load(), dbg_phase_.load(), l, dbg_seq_.load());
                    if (l >= 0) {
                        const Bell & b = bells_[l];
                        std::fprintf(stderr, " | bell seq %d doorbell %d plan %d need_copy %d cpu_done %d", b.seq,
                                     *(volatile int *) b.doorbell, *(volatile int *) b.plan,
                                     *(volatile int *) b.need_copy, *(volatile int *) b.cpu_done);
                    }
                    std::fprintf(stderr, " | %s\n", store_->admit_debug().c_str());
                }
            }
        });
}

void MoeExec::drive()
{
    if (g_.device >= 0) cudaSetDevice(g_.device);   // its copies and events are this stage's device's
    if (aux_.on) cpu::pin_self(aux_.driver_cpu);
    driver_tid_ = cpu::this_tid();
    long taken = 0;
    for (;;) {
        Job j;
        // spin first: the next layer's job arrives within ~0.5 ms during a pass, and a condvar wake costs more
        dbg_phase_ = 1;
        for (int i = 0; i < 200000 && jobs_pushed_.load(std::memory_order_acquire) <= taken; ++i) {
            _mm_pause();
            if ((admit_idle_ > 0 || store_->pin_pending()) && i % 64 == 0)   // the link is ours (and E7's refill)
                dbg_phase_ = 8, store_->pin_step(), store_->admit_step(), dbg_phase_ = 1;
            if (prof_ && prof_->on() && i == 4096 && (dbg_served_ & 127) == 0) drv_cpu_.store(last_cpu_of(driver_tid_.load()));
        }
        {
            std::unique_lock<std::mutex> g(dq_mu_);
            dbg_phase_ = 2;
            dq_cv_.wait(g, [&] { return driver_stop_ || !jobs_.empty(); });
            ++taken;
            if (driver_stop_ && jobs_.empty()) return;
            j = jobs_.front();
            jobs_.pop_front();
        }
        Bell & b = bells_[j.sl];
        dbg_layer_ = j.sl, dbg_seq_ = j.seq, dbg_phase_ = 3;
        for (int i = 0; *(volatile int *) b.doorbell < j.seq; ++i) {
            _mm_pause();
            if ((admit_idle_ > 0 || store_->pin_pending()) && i % 64 == 0)
                dbg_phase_ = 8, store_->pin_step(), store_->admit_step(), dbg_phase_ = 3;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        drv_t_[0] = std::chrono::steady_clock::now();
        if (drv_ts_ && ts_prev_.time_since_epoch().count())
            ts_ms_[0] += std::chrono::duration<double, std::milli>(drv_t_[0] - ts_prev_).count();
        try {
            dbg_phase_ = 4;
            serve(j, b);
            ++dbg_served_;
        } catch (...) {   // keep the GPU moving (its output is garbage now); the caller rethrows at its next sync
            try {
                throw;
            } catch (const std::exception & e) {
                std::fprintf(stderr, "driver: layer %d seq %d T %d threw: %s\n", j.sl, j.seq, j.T, e.what());
            } catch (...) {
                std::fprintf(stderr, "driver: layer %d seq %d T %d threw a non-std exception\n", j.sl, j.seq, j.T);
            }
            if (!driver_error_) driver_error_ = std::current_exception();
            store_->signal(j.sl, b.go, j.seq);
            *(volatile int *) b.need_copy = j.seq;
            std::atomic_thread_fence(std::memory_order_release);
            *(volatile int *) b.plan = j.seq;
            std::atomic_thread_fence(std::memory_order_release);
            *(volatile int *) b.cpu_done = j.seq;
        }
        jobs_done_.fetch_add(1, std::memory_order_release);
    }
}

}  // namespace truss::runtime
