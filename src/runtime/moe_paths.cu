// MoeExec's three host-involved chunk paths: doorbell (decode windows, no host sync), fetch (other fetch-sized
// chunks: one sync for the routing) and split (E7: short prompt chunks share their cold experts between the CPU tier
// and PCIe). The stream path and the joins are in moe_exec.cu.
#include "runtime/moe_exec_state.h"

#include "kernels/ffn/ffn_ops.cuh"
#include "runtime/doorbell.cuh"

#include <algorithm>

namespace truss::runtime {

// TRUSS_PUB_X_LAZY publishes the rows only for a layer whose CPU tier reads them (serve() touches b.x only then):
// with no CPU tier the copy is dead work on the link (18 us of every publish on device 0)
static bool pub_x_lazy()
{
    static const bool v = [] { const char * e = std::getenv("TRUSS_PUB_X_LAZY"); return e && std::atoi(e); }();
    return v;
}

void MoeExec::run_doorbell(const Call & c, bool cpu_on, bool fused, bool hint)
{
    const int l = c.sl, T = c.T, ml = layers_[l].model_layer, dm = k_.d_model;
    engine::Profile * const pf = layers_[l].profiled ? prof_ : nullptr;
    Bell & b = bells_[l];
    const int seq = ++b.seq;
    const int n_x = pub_x_lazy() && !cpu_on ? 0 : T * dm;
    if (fused) {
        // the hint's prediction goes along in the same launch; its weights were never read, so it needs no device
        // buffer at all
        ffn::route_publish(c.logits, hint ? c.pred_logits : nullptr, T, E_, K_, c.ids, c.wts, c.x, n_x, b.ids, b.pred,
                           b.w, b.x, b.doorbell, seq, b.arrive, ffn::publish_ctas(n_x), g_.s);
    } else {
        int * pred = nullptr;
        if (hint) {
            float * pw = g_.sc->alloc((size_t) T * K_);
            pred = g_.sc->alloc<int>((size_t) T * K_);
            c.route(c.pred_logits, c.pred_bias, T, E_, K_, pred, pw, g_.s);
        }
        runtime::publish(c.ids, pred, c.wts, T * K_, c.x, n_x, b.ids, b.pred, b.w, b.x, b.doorbell, seq, g_.s);
    }
    if (pf) pf->record(ml, 7, g_.s), pf->mark_bell(ml);
    {
        std::lock_guard<std::mutex> g(dq_mu_);
        jobs_.push_back({ l, T, seq, hint, cpu_on });
        jobs_pushed_.fetch_add(1, std::memory_order_release);
    }
    dq_cv_.notify_one();
    // the shared expert does not need the plan or the copies: it runs while the driver splits and the copies land
    // (TRACKER #112; same math, so the same output)
    if (c.shared) c.shared();
    if (pf && pf->on()) {   // profiling: the plan and the copies waited for apart (two launches instead of one)
        runtime::spin_until(b.plan, seq, g_.s);
        pf->record(ml, 9, g_.s);
    }
    runtime::wait_plan(b.plan, b.need_copy, b.go, seq, b.mids, b.d_mids, cpu_on ? T * K_ : 0, g_.s);
    if (pf) pf->record(ml, 8, g_.s);
    k_.window(store_->weights(l), c.x, cpu_on ? b.d_mids : c.ids, c.wts, T, c.routed, window_ws_, g_.s);
    if (pf) pf->record(ml, 10, g_.s);   // the window kernel ends here; the joins wait for other workers
    if (cpu_on) cpu_rows_ = -T;                 // the CPU rows join after the shared expert (mapped)
}

void MoeExec::run_fetch(const Call & c, bool hint)
{
    const int l = c.sl, T = c.T, dm = k_.d_model;
    // pre-gating: the next layer's router on this layer's input predicts 72% of its experts (TRACKER #58); their
    // copies start behind this layer's, one sync for both id sets
    if (hint) {
        float * pw = g_.sc->alloc((size_t) T * K_);
        int * pred = g_.sc->alloc<int>((size_t) T * K_);
        c.route(c.pred_logits, c.pred_bias, T, E_, K_, pred, pw, g_.s);
        TRUSS_CUDA(cudaMemcpyAsync(ids_host_ + T * K_, pred, sizeof(int) * T * K_, cudaMemcpyDeviceToHost, g_.s));
    }
    // the CPU tier serves windows of up to cpu::MAX_ROWS rows: decode steps, verify windows, drafter commits
    CpuTier * ct = layers_[l].cpu_ok && T <= cpu::MAX_ROWS ? cpu_tier_.get() : nullptr;
    if (ct) {   // the rows and weights the CPU experts need, in the same sync
        TRUSS_CUDA(cudaMemcpyAsync(ct->x, c.x, sizeof(float) * T * dm, cudaMemcpyDeviceToHost, g_.s));
        TRUSS_CUDA(cudaMemcpyAsync(ct->w, c.wts, sizeof(float) * T * K_, cudaMemcpyDeviceToHost, g_.s));
    }
    TRUSS_CUDA(cudaMemcpyAsync(ids_host_, c.ids, sizeof(int) * T * K_, cudaMemcpyDeviceToHost, g_.s));
    TRUSS_CUDA(cudaStreamSynchronize(g_.s));
    // split the slots: CPU-tier experts go to the pool; the GPU kernel sees them as id -1 (skipped)
    std::vector<cpu::Slot> slots;
    std::vector<int> gpu_ids;
    const int * k_ids = c.ids;
    if (ct) {
        for (int j = 0; j < T * K_; ++j) {
            const int e = ids_host_[j];
            const bool on_cpu = ct->is_cpu[l][e];
            ct->ids[j] = on_cpu ? -1 : e;
            if (on_cpu) slots.push_back(ct->slot(j / K_, l, e, ct->w[j]));
            else gpu_ids.push_back(e);
        }
        if (!slots.empty()) {
            int * m_ids = g_.sc->alloc<int>((size_t) T * K_);
            TRUSS_CUDA(cudaMemcpyAsync(m_ids, ct->ids, sizeof(int) * T * K_, cudaMemcpyHostToDevice, g_.s));
            k_ids = m_ids;
            lock_pool();
            ct->pool->start(ct->x, T, slots, ct->y);
        }
    } else {
        gpu_ids.assign(ids_host_, ids_host_ + T * K_);
    }
    store_->fetch(l, gpu_ids.data(), (int) gpu_ids.size(), g_.s);
    if (hint) {   // each row's first hint_k guesses (ids are in descending probability), GPU experts only
        std::vector<int> h;
        const CpuTier * nt = layers_[l].next_in_stage ? cpu_tier_.get() : nullptr;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < std::min(K_, o_.hint_k); ++i) {
                const int e = ids_host_[T * K_ + t * K_ + i];
                if (!nt || !nt->is_cpu[l + 1][e]) h.push_back(e);
            }
        store_->prefetch_hint(l, h.data(), (int) h.size(), o_.hint_far_only);
    }
    store_->acquire(l, g_.s);
    if (T <= moe::MAX_ROWS) k_.window(store_->weights(l), c.x, k_ids, c.wts, T, c.routed, window_ws_, g_.s);
    else k_.prefill(store_->weights(l), c.x, c.ids, c.wts, T, c.routed, c.moe_ws, c.moe_rows, g_.s);
    store_->release(l, g_.s);
    cpu_rows_ = slots.empty() ? 0 : T;
}

// E7 (TRACKER #117): the routing first, then the CPU's experts and the GPU's copies
void MoeExec::run_split(const Call & c)
{
    const int l = c.sl, T = c.T, dm = k_.d_model;
    CpuTier * ct = layers_[l].cpu_ok ? cpu_tier_.get() : nullptr;   // a drafter's layer: copies only
    if (ct) {
        TRUSS_CUDA(cudaMemcpyAsync(ct->x, c.x, sizeof(float) * T * dm, cudaMemcpyDeviceToHost, g_.s));
        TRUSS_CUDA(cudaMemcpyAsync(ct->w, c.wts, sizeof(float) * T * K_, cudaMemcpyDeviceToHost, g_.s));
    }
    TRUSS_CUDA(cudaMemcpyAsync(ids_host_, c.ids, sizeof(int) * T * K_, cudaMemcpyDeviceToHost, g_.s));
    TRUSS_CUDA(cudaStreamSynchronize(g_.s));
    std::vector<cpu::Slot> slots;
    std::vector<int> gpu_ids;
    const int * k_ids = c.ids;
    if (ct) {
        // balanced: the routed cold experts (hot ones have is_cpu 0) by ascending row count go to the CPU while its
        // estimated time (the decode split's fit, plus split_ms_row per extra row) stays within what the copies left
        // on PCIe would take; at most split_t rows each
        std::vector<int> cnt(E_, 0), cand;
        std::vector<uint8_t> to_cpu(E_, 0);
        for (int j = 0; j < T * K_; ++j) ++cnt[ids_host_[j]];
        double pcie = 0, cpu = cpu_ms_call_;
        for (int e = 0; e < E_; ++e)
            if (cnt[e] > 0 && ct->is_cpu[l][e]) {
                cand.push_back(e);
                pcie += pcie_ms_byte_ * store_->bytes_of(l, e);
            }
        std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return cnt[a] < cnt[b]; });
        for (int e : cand) {
            const double ce = cpu_ms_expert_ + o_.split_ms_row * (cnt[e] - 1), q = pcie_ms_byte_ * store_->bytes_of(l, e);
            if (cnt[e] > o_.split_t || cpu + ce > pcie - q) break;
            cpu += ce, pcie -= q, to_cpu[e] = 1;
        }
        for (int e : cand) to_cpu[e] ? ++split_cpu_experts_ : ++split_gpu_experts_;
        for (int j = 0; j < T * K_; ++j) {
            const int e = ids_host_[j];
            const bool on_cpu = to_cpu[e];
            ct->ids[j] = on_cpu ? -1 : e;
            if (on_cpu) slots.push_back(ct->slot(j / K_, l, e, ct->w[j]));
            else gpu_ids.push_back(e);
        }
        if (!slots.empty()) {
            int * m_ids = g_.sc->alloc<int>((size_t) T * K_);
            TRUSS_CUDA(cudaMemcpyAsync(m_ids, ct->ids, sizeof(int) * T * K_, cudaMemcpyHostToDevice, g_.s));
            k_ids = m_ids;
            lock_pool();
            ct->pool->start(ct->x, T, slots, ct->y);   // CPU first: it runs while the copies land
        }
    } else {
        gpu_ids.assign(ids_host_, ids_host_ + T * K_);
    }
    split_host_bytes_ += (long long) store_->prefetch_some(l, gpu_ids.data(), (int) gpu_ids.size());
    store_->acquire(l, g_.s);
    k_.prefill(store_->weights(l), c.x, k_ids, c.wts, T, c.routed, c.moe_ws, c.moe_rows, g_.s);
    store_->release(l, g_.s);
    cpu_rows_ = slots.empty() ? 0 : T;
}

}  // namespace truss::runtime
