// MoeExec::serve: the driver's work for one doorbell decode FFN (see moe_driver.cu), and the CPU cost fit that
// Strata's dynamic split uses.
#include "runtime/moe_exec_state.h"

#include <algorithm>

namespace truss::runtime {

void MoeExec::serve(const Job & j, Bell & b)
{
    const int K = K_, l = j.sl, n = j.T * K;
    const int ml = layers_[l].model_layer;
    const bool prof = prof_ && prof_->on();
    CpuTier * ct = j.cpu ? cpu_tier_.get() : nullptr;
    if (!route_usage_.empty())
        for (int i = 0; i < n; ++i) route_usage_[(size_t) l * E_ + b.ids[i]] += 1.f;
    if (admit_idle_ > 0 && layers_[l].cpu_ok) {
        if (ml == 0) {   // a new pass (model layer 0: on a split's last stage this never fires - kept as it was)
            for (float & u : admit_u_) u *= o_.admit_decay;
            admit_pass_ = 0;
        }
        for (int i = 0; i < n; ++i) admit_u_[(size_t) l * E_ + b.ids[i]] += 1.f;
    }
    if (route_trace_) {
        const int32_t h[2] = { ml, j.T };
        std::fwrite(h, 4, 2, route_trace_);
        std::fwrite(b.ids, 4, (size_t) n, route_trace_);
    }
    // TRUSS_TAIL_SKIP=R (Strata's route tail skip, #192): an expert that is not on the device and that every row of the
    // window ranks R or lower by weight (0-based; ties by position) is neither copied nor computed - its term is 0, the
    // other weights stay as routed (no renormalization, as Strata). CPU tier only: a skipped entry is masked (-1) like
    // a CPU one but gets no CPU slot, so the joined CPU rows hold nothing for it.
    uint8_t skip[8 * 16] = {};
    if (ct && o_.tail_skip > 0)
        for (int i = 0; i < n; ++i) {
            const int e = b.ids[i];
            if (skip[i] || store_->on_device(l, e)) continue;
            bool low = true;
            for (int i2 = 0; i2 < n && low; ++i2) {
                if (b.ids[i2] != e) continue;
                const int r0 = i2 / K * K;
                int rank = 0;
                for (int q = r0; q < r0 + K; ++q) rank += b.w[q] > b.w[i2] || (b.w[q] == b.w[i2] && q < i2);
                low = rank >= o_.tail_skip;
            }
            if (!low) continue;
            for (int i2 = i; i2 < n; ++i2) skip[i2] |= b.ids[i2] == e;
            tail_skipped_.fetch_add(1, std::memory_order_relaxed);
        }
    std::vector<cpu::Slot> slots;
    std::vector<int> gpu_ids;
    gpu_ids.reserve(n);
    // which routed experts the CPU computes: static (the CPU set always), or dynamic (Strata's split)
    uint8_t to_cpu[8 * 16] = {};   // per entry
    int n_cpu = 0;
    if (ct && cpu_dynamic_) {
        // distinct experts in routing order; the GPU takes the resident ones and the non-eligible misses
        int dist[8 * 16], nd = 0;
        std::vector<int> elig;          // eligible misses (positions in dist), routing order
        double pcie_bytes = 0;          // copies the layer needs anyway
        for (int i = 0; i < n; ++i) {
            const int e = b.ids[i];
            bool seen = false;
            for (int q = 0; q < nd; ++q) seen |= b.ids[dist[q]] == e;
            if (seen) continue;
            dist[nd] = i;
            if (!skip[i] && !store_->on_device(l, e)) {
                if (ct->is_cpu[l][e]) elig.push_back(nd);
                else pcie_bytes += (double) store_->bytes_of(l, e);
            }
            ++nd;
        }
        // m = how many eligible misses (the last ones) go over PCIe: minimize max(CPU time, copy time)
        const int ne = (int) elig.size();
        int best_m = 0;
        double best = 1e30, tail = 0;
        for (int m = 0; m <= ne && o_.pcie_frac < 0; ++m) {
            if (m > 0) tail += (double) store_->bytes_of(l, b.ids[dist[elig[ne - m]]]);
            const double tc = ne - m > 0 ? cpu_ms_call_ + cpu_ms_expert_ * (ne - m) : 0;
            const double t = std::max(tc, pcie_ms_byte_ * (pcie_bytes + tail));
            if (t < best - 1e-9) best = t, best_m = m;
        }
        if (o_.pcie_frac >= 0) best_m = std::min(ne, (int) std::lround(o_.pcie_frac * ne));
        for (int q = 0; q < ne - best_m; ++q) {
            const int e = b.ids[dist[elig[q]]];
            for (int i = 0; i < n; ++i) to_cpu[i] |= b.ids[i] == e;
        }
        n_cpu = ne - best_m;
        dyn_cpu_ += n_cpu;
        dyn_pcie_ += best_m;
    } else if (ct) {
        for (int i = 0; i < n; ++i) to_cpu[i] = !skip[i] && ct->is_cpu[l][b.ids[i]];
    }
    for (int i = 0; i < n; ++i) {
        const int e = b.ids[i];
        b.mids[i] = to_cpu[i] || skip[i] ? -1 : e;
        if (skip[i]) continue;
        if (to_cpu[i]) slots.push_back(ct->slot(i / K, l, e, b.w[i]));
        else gpu_ids.push_back(e);
    }
    drv_t_[1] = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> pool_lock;   // a split's stages share the pool (pipelined windows overlap)
    if (ct && ct->pool_mu) pool_lock = std::unique_lock<std::mutex>(*ct->pool_mu);
    const long long item0 = prof && ct ? ct->pool->item_us() : 0;   // cumulative; the call's own is the diff
    if (ct) ct->pool->start(b.x, j.T, slots, b.y);   // CPU first: it runs while the copies are issued
    drv_t_[2] = std::chrono::steady_clock::now();
    // the plan (masked ids in mapped b.mids) goes to the GPU through mapped words, as Strata's: a layer that needs no
    // copies is not ordered behind the copy stream; one that copies also waits for go, which the copy stream writes
    // behind its copies
    dbg_phase_ = 5;
    const bool copied = store_->fetch(l, gpu_ids.data(), (int) gpu_ids.size(), nullptr);
    if (copied) store_->signal(l, b.go, j.seq);   // go before the hints: this layer must not wait for the next one's
    *(volatile int *) b.need_copy = copied ? j.seq : 0;
    std::atomic_thread_fence(std::memory_order_release);
    *(volatile int *) b.plan = j.seq;
    auto ms = [](auto a, auto c) { return std::chrono::duration<double, std::milli>(c - a).count(); };
    auto us = [](auto a, auto c) { return std::chrono::duration<double, std::micro>(c - a).count(); };
    {   // driver timing (driver_ms): split, CPU start, copies + plan
        const auto t3 = std::chrono::steady_clock::now();
        drv_ms_[0] += ms(drv_t_[0], drv_t_[1]), drv_ms_[1] += ms(drv_t_[1], drv_t_[2]), drv_ms_[2] += ms(drv_t_[2], t3);
        ++drv_n_;
        if (drv_ts_) {
            ts_ms_[1] += ms(drv_t_[0], drv_t_[1]), ts_ms_[2] += ms(drv_t_[1], drv_t_[2]), ts_ms_[3] += ms(drv_t_[2], t3);
            ts_t3_ = t3;
        }
    }
    engine::Profile::Host * hp = prof ? &prof_->host(ml) : nullptr;
    if (hp) {   // this layer's host chain, us after the driver saw the doorbell (-> the section CSV)
        hp->t[0] = us(drv_t_[0], drv_t_[1]);
        hp->t[1] = us(drv_t_[0], drv_t_[2]);
        hp->t[2] = us(drv_t_[0], std::chrono::steady_clock::now());
        hp->ncpu = n_cpu;
    }
    if (j.hint) {
        std::vector<int> h;
        // static: the next layer's CPU set never goes to the GPU, so it is not hinted; dynamic: a hinted expert that
        // arrives in time is simply a GPU hit
        const CpuTier * nt = layers_[l].next_in_stage && !cpu_dynamic_ ? cpu_tier_.get() : nullptr;
        for (int t = 0; t < j.T; ++t)
            for (int i = 0; i < std::min(K, o_.hint_k); ++i) {
                const int e = b.pred[t * K + i];
                if (!nt || !nt->is_cpu[l + 1][e]) h.push_back(e);
            }
        store_->prefetch_hint(l, h.data(), (int) h.size());
    }
    if (admit_idle_ > 0 && n_cpu > 0 && layers_[l].cpu_ok && admit_pass_ < admit_idle_) {
        // claim slots for the CPU's experts most worth keeping (after fetch(): its experts are protected)
        std::vector<std::pair<float, int>> cand;
        for (int i = 0; i < n; ++i)
            if (to_cpu[i]) {
                const int e = b.ids[i];
                const float u = admit_u_[(size_t) l * E_ + e];
                bool dup = false;
                for (const auto & x : cand) dup |= x.second == e;
                if (!dup && u >= o_.admit_min) cand.push_back({ u, e });
            }
        std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & c) { return a.first > c.first; });
        dbg_phase_ = 6;
        for (size_t i = 0; i < cand.size() && i < 2 && admit_pass_ < admit_idle_; ++i)
            admit_pass_ += store_->claim(l, cand[i].second);
    }
    if (drv_ts_ || prof) ts_t3b_ = std::chrono::steady_clock::now();
    if (hp) hp->t[3] = us(drv_t_[0], ts_t3b_);   // hint + admit
    if (ct) {
        dbg_phase_ = 7;
        ct->pool->wait();
        if (cpu_dynamic_ && n_cpu > 0) fit_cpu_cost(n_cpu, ct->pool->last_call_ms());   // the pool's own time
        if (hp) {   // the pool's own split, and the instant cpu_done is about to be raised
            hp->t[4] = us(drv_t_[0], std::chrono::steady_clock::now());
            hp->pool = 1000.0 * ct->pool->last_call_ms();
            hp->wake = (double) ct->pool->last_wake_us();
            hp->item = (double) (ct->pool->item_us() - item0);   // item_us is cumulative over the run
            hp->longest = (double) ct->pool->last_longest_us();
        }
        std::atomic_thread_fence(std::memory_order_release);
        *(volatile int *) b.cpu_done = j.seq;
        if (drv_ts_) {
            ts_t4_ = std::chrono::steady_clock::now();
            ts_pool_ += ct->pool->last_call_ms();
        }
    }
    if (drv_ts_) {
        if (!ct) ts_t4_ = ts_t3b_;
        const auto t5 = std::chrono::steady_clock::now();
        ts_ms_[4] += ms(ts_t3_, ts_t3b_), ts_ms_[5] += ms(ts_t3b_, ts_t4_), ts_ms_[6] += ms(ts_t4_, t5);
        ts_prev_ = t5;
        if (++ts_n_ % 480 == 0) {   // ~10 decode passes
            std::fprintf(stderr, "DRIVER_TS n=%ld pre %.3f split %.3f start %.3f copies %.3f hint_admit %.3f wait %.3f "
                                 "post %.3f pool_done %.3f ms/call\n",
                         ts_n_, ts_ms_[0] / 480, ts_ms_[1] / 480, ts_ms_[2] / 480, ts_ms_[3] / 480, ts_ms_[4] / 480,
                         ts_ms_[5] / 480, ts_ms_[6] / 480, ts_pool_ / 480);
            for (double & v : ts_ms_) v = 0;
            ts_pool_ = 0;
        }
    }
}

// add one call (n experts, t ms) to the weighted sums and refit cpu_ms_call + cpu_ms_expert * n; the slope keeps its
// last value while the calls are too alike in n to fit one (and stays within sane bounds)
void MoeExec::fit_cpu_cost(int n, double t)
{
    const double d = 0.98;
    dyn_s1_ = d * dyn_s1_ + 1, dyn_sn_ = d * dyn_sn_ + n, dyn_st_ = d * dyn_st_ + t;
    dyn_snn_ = d * dyn_snn_ + (double) n * n, dyn_snt_ = d * dyn_snt_ + n * t;
    const double var = dyn_snn_ * dyn_s1_ - dyn_sn_ * dyn_sn_;
    if (dyn_s1_ > 8 && var > 0.25 * dyn_s1_ * dyn_s1_)
        cpu_ms_expert_ = std::min(2.0, std::max(0.02, (dyn_snt_ * dyn_s1_ - dyn_sn_ * dyn_st_) / var));
    cpu_ms_call_ = std::max(0.0, (dyn_st_ - cpu_ms_expert_ * dyn_sn_) / dyn_s1_);
}

}  // namespace truss::runtime
