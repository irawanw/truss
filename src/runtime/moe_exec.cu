// MoeExec: construction, the per-call dispatch, the stream path and the between-pass policies. The doorbell driver
// is in moe_driver.cu, the fetch / doorbell / split paths in moe_paths.cu, the CPU tier loaders in cpu_tier.cc.
#include "runtime/moe_exec_state.h"

#include "engine/limits.h"
#include "kernels/ffn/ffn_ops.cuh"
#include "runtime/doorbell.cuh"

#include <immintrin.h>

#include <algorithm>
#include <tuple>

namespace truss::runtime {

using engine::require;

MoeExec::MoeExec(engine::Gpu & g, const MoeKernels & k, std::vector<MoeLayer> layers, int n_expert, int topk,
                 const std::vector<float> & usage, size_t stream_extra, const MoeOptions & o, int max_rows,
                 std::shared_ptr<cpu::ExpertPool> pool, std::shared_ptr<std::mutex> pool_mu, engine::Profile * prof)
    : g_(g), k_(k), layers_(std::move(layers)), E_(n_expert), K_(topk), o_(o), prof_(prof)
{
    require(k_.topk == K_, "expert kernels' top-k differs from the model's");
    (void) max_rows;
    window_ws_ = g_.alloc<unsigned char>(k_.window_ws());
    k_.window_init(window_ws_, g_.s);
    TRUSS_CUDA(cudaMallocHost(&ids_host_, sizeof(int) * 2 * std::max(engine::FETCH_ROWS, o_.split_rows) * K_));
    std::vector<ExpertLayer> tables;
    for (const MoeLayer & L : layers_) tables.push_back(L.tables);
    size_t budget = o_.expert_budget;
    if (!budget) {
        size_t free_b, total_b;
        TRUSS_CUDA(cudaMemGetInfo(&free_b, &total_b));
        constexpr size_t MARGIN = 768ull << 20;   // cuBLAS workspaces, the CUDA context's growth
        require(free_b > MARGIN, "no device memory left for the routed experts");
        budget = (size_t) ((double) (free_b - MARGIN) * std::min(1.0, std::max(0.05, o_.budget_frac)));
        const double gb = 1073741824.0;
        std::fprintf(stderr, "VRAM: %.2f GiB total, %.2f GiB in use before the experts; routed experts get %.2f GiB "
                             "(margin %.2f)\n", total_b / gb, (total_b - free_b) / gb, budget / gb, MARGIN / gb);
    }
    ExpertStore::Sizes z;
    z.ring_bytes = o_.ring_bytes;
    z.plan_alpha = o_.plan_alpha;
    z.stream_extra = stream_extra;
    const ExpertStore::HotSet hot = ExpertStore::plan(tables, budget, z, usage);
    for (const auto & h : hot) n_hot_ += (int) std::count(h.begin(), h.end(), 1);
    engine::load_mark("expert plan");
    store_ = std::make_unique<ExpertStore>(tables, hot, z);
    engine::load_mark("expert store (hot upload + cold copy; page-locking continues in the background)");
    if (!o_.cpu_dir.empty()) load_cpu_tier(hot, usage);
    else if (o_.cpu_trellis) load_cpu_tier_trellis(hot, usage);
    if (cpu_tier_) {
        if (pool) cpu_tier_->pool = pool;
        else cpu_tier_->pool = std::make_shared<cpu::ExpertPool>(o_.cpu_threads);
        cpu_tier_->pool_mu = pool_mu;
    }
    engine::load_mark("CPU tier");
    cpu_dynamic_ = o_.cpu_dynamic && cpu_tier_ && o_.doorbell;
    pcie_ms_byte_ = 1.0 / (o_.pcie_gbps * 1e6);
    if (o_.adapt_every > 0) route_usage_.assign((size_t) n_store() * E_, 0.f);
    admit_idle_ = cpu_tier_ && o_.doorbell ? o_.admit_idle : 0;
    if (admit_idle_ > 0) admit_u_.assign((size_t) n_store() * E_, 0.f);
    if (cpu_tier_ && o_.last_stage) aux_ = cpu::plan_aux(cpu_tier_->pool->worker_cpus());   // a split: the last stage places
    if (const char * rt = o_.last_stage ? std::getenv("TRUSS_ROUTE_TRACE") : nullptr) {   // one stage writes it
        route_trace_ = std::fopen(rt, "wb");
        require(route_trace_ != nullptr, std::string("cannot write ") + rt);
        const int32_t hdr[3] = { n_store(), E_, K_ };
        std::fwrite(hdr, 4, 3, route_trace_);
        for (int l = 0; l < n_store(); ++l)
            for (int e = 0; e < E_; ++e) {
                const int32_t b = (int32_t) store_->bytes_of(l, e);
                const uint8_t h = store_->host_part(l, e, 0) == nullptr;
                std::fwrite(&b, 4, 1, route_trace_), std::fwrite(&h, 1, 1, route_trace_);
            }
    }
    setup_doorbells();
}

MoeExec::~MoeExec()
{
    if (g_.device >= 0) cudaSetDevice(g_.device);
    if (route_trace_) std::fclose(route_trace_);
    if (driver_.joinable()) {
        {
            std::lock_guard<std::mutex> l(dq_mu_);
            driver_stop_ = true;
        }
        dq_cv_.notify_all();
        driver_.join();
        if (watchdog_.joinable()) watchdog_.join();
    }
    if (prof_ && prof_->on())   // where the two threads that matter ran: the caller launches, the driver plans and copies
        std::fprintf(stderr, "threads: caller %ld, driver %ld on cpu %d; aux %s\n", (long) caller_tid_,
                     (long) driver_tid_.load(), drv_cpu_.load(), aux_.on ? aux_.text.c_str() : "off");
    for (Bell & b : bells_)
        for (void * p : { (void *) b.ids, (void *) b.pred, (void *) b.mids, (void *) b.w, (void *) b.x, (void *) b.y,
                          (void *) b.doorbell, (void *) b.cpu_done, (void *) b.plan, (void *) b.need_copy })
            if (p) cudaFreeHost(p);
    if (ids_host_) cudaFreeHost(ids_host_);
    if (cpu_tier_)
        for (void * p : { (void *) cpu_tier_->x, (void *) cpu_tier_->w, (void *) cpu_tier_->y, (void *) cpu_tier_->ids })
            if (p) cudaFreeHost(p);
}

void MoeExec::start_pinning()
{
    store_->start_pinning();   // last: it holds the driver lock for ~3 s
    if (!aux_.text.empty()) std::fprintf(stderr, "%s\n", aux_.text.c_str());
    if (aux_.on) {   // after start_pinning: its thread moves too
        while (!driver_tid_.load()) std::this_thread::yield();
        place_threads();
    }
}

// TRUSS_AUX_CPUS: the caller (this thread, which launches the passes) on its own CPU, every thread but the workers,
// the driver and the caller on the aux set (Strata's --aux-cpus). At start, at each prompt chunk, when another thread
// starts calling and every 512 passes: the server's threads and CUDA's come and go.
void MoeExec::place_threads()
{
    const pid_t me = cpu::this_tid();
    if (me != caller_tid_ && aux_.caller_cpu >= 0) cpu::pin_self(aux_.caller_cpu);
    caller_tid_ = me;
    std::vector<pid_t> keep = cpu_tier_->pool->worker_tids();
    keep.push_back(driver_tid_.load());
    if (aux_.caller_cpu >= 0) keep.push_back(me);
    const int n = cpu::sweep_aux(aux_, keep);
    if (!aux_placed_) std::fprintf(stderr, "TRUSS_AUX_CPUS: %d threads moved to the aux set\n", n);
    aux_placed_ = true;
}

void MoeExec::after_chunk_aux(bool stream)
{
    if (aux_.on && (stream || cpu::this_tid() != caller_tid_ || ++aux_passes_ % 512 == 0)) place_threads();
}

void MoeExec::run(const Call & c)
{
    const int T = c.T, l = c.sl;
    const bool hint = c.pred_logits != nullptr;   // the block asked wants_hint() and computed them
    const bool door = !c.stream && o_.doorbell && T <= cpu::MAX_ROWS;   // no host sync: the driver serves this FFN
    const bool cpu_on = cpu_tier_ && layers_[l].cpu_ok;
    static const bool route_publish = [] { const char * e = std::getenv("TRUSS_ROUTE_PUBLISH"); return e && std::atoi(e); }();
    const bool fused = door && route_publish && c.fused_route_ok;
    if (!fused) c.route(c.logits, c.bias, T, E_, K_, c.ids, c.wts, g_.s);   // the fused launch does the top-k itself
    if (route_counts_) ffn::count(c.ids, T * K_, route_counts_ + (size_t) l * E_, g_.s);
    bool shared_done = false;
    auto shared = [&] {
        if (c.shared) c.shared();
        shared_done = true;
    };
    const int ml = layers_[l].model_layer;
    engine::Profile * const pf = layers_[l].profiled ? prof_ : nullptr;
    if (door) {
        run_doorbell(c, cpu_on, fused, hint);
        shared_done = true;   // run_doorbell ran it between the publish and the plan wait
    } else if (!c.stream) {
        run_fetch(c, hint);
    } else if (split_chunk_) {
        run_split(c);
    } else {   // whole layers stream ahead (prefetch(l + 2) while this layer computes)
        store_->acquire(l, g_.s);
        k_.prefill(store_->weights(l), c.x, c.ids, c.wts, T, c.routed, c.moe_ws, c.moe_rows, g_.s);
        store_->release(l, g_.s);
        if (l + 2 < n_store()) store_->prefetch(l + 2);   // a drafter's layer streams after the last layer
    }
    if (pf) pf->record(ml, 4, g_.s);   // shared expert (the doorbell path ran it before its wait) + the CPU join
    if (!shared_done) shared();
    if (pf) pf->record(ml, 5, g_.s);   // the CPU tier's join (its spin is the section)
    if (cpu_rows_ < 0) {   // doorbell path: wait for the CPU tier's rows (mapped) and add them
        runtime::spin_until(bells_[l].cpu_done, bells_[l].seq, g_.s);
        runtime::add_mapped(c.routed, bells_[l].y, -cpu_rows_ * k_.d_model, g_.s);
        cpu_rows_ = 0;
    }
    // recorded on every path: elapsed time between a recorded and a never-recorded event reads garbage
    if (pf) pf->record(ml, 11, g_.s), pf->record(ml, 12, g_.s);
    if (cpu_rows_) {   // the CPU experts' sum joins the routed output
        float * yc = g_.sc->alloc((size_t) cpu_rows_ * k_.d_model);
        cpu_tier_->pool->wait();
        TRUSS_CUDA(cudaMemcpyAsync(yc, cpu_tier_->y, sizeof(float) * cpu_rows_ * k_.d_model, cudaMemcpyHostToDevice, g_.s));
        ffn::add(c.routed, yc, cpu_rows_ * k_.d_model, g_.s);
        cpu_rows_ = 0;
    }
    if (pool_lock_.owns_lock()) pool_lock_.unlock();
}

void MoeExec::lock_pool()
{
    if (cpu_tier_->pool_mu && !pool_lock_.owns_lock()) pool_lock_ = std::unique_lock<std::mutex>(*cpu_tier_->pool_mu);
}

// Strata's adapt() (generate.cpp): the most-routed missing experts (decayed count >= adapt_min) of every store layer,
// at most adapt_swaps, into the ring; then counts x adapt_decay. Between passes: the driver thread is idle (the
// caller synced the last pass for its logits), so route_usage is not being written.
void MoeExec::adapt_tick()
{
    if (o_.adapt_every <= 0 || ++decode_passes_ % o_.adapt_every != 0) return;
    store_->poll_admitted();
    if (store_->admitting()) return;   // the previous batch is still being copied
    std::vector<std::tuple<float, int, int>> cand;
    for (int l = 0; l < n_store(); ++l)
        for (int e = 0; e < E_; ++e) {
            const float u = route_usage_[(size_t) l * E_ + e];
            if (u >= o_.adapt_min && !store_->on_device(l, e)) cand.emplace_back(u, l, e);
        }
    const size_t n = std::min(cand.size(), (size_t) o_.adapt_swaps);
    std::partial_sort(cand.begin(), cand.begin() + (ptrdiff_t) n, cand.end(),
                      [](const auto & a, const auto & b) { return std::get<0>(a) > std::get<0>(b); });
    std::vector<std::pair<int, int>> le;
    for (size_t i = 0; i < n; ++i) le.push_back({ std::get<1>(cand[i]), std::get<2>(cand[i]) });
    store_->admit(le, g_.s);
    for (float & u : route_usage_) u *= o_.adapt_decay;
}

// Wait until the driver has finished every queued decode FFN. begin_stream() switches the store's mode on the host at
// once, while a decode job queued just before may not have been served yet: its fetch() then threw "outside ring
// mode" (the served hangs of 2026-10-12). The GPU reaches the doorbells on its own, so this wait ends.
void MoeExec::drain_jobs()
{
    while (jobs_done_.load(std::memory_order_acquire) < jobs_pushed_.load(std::memory_order_acquire)) _mm_pause();
}

void MoeExec::check_driver()
{
    if (driver_error_) {
        std::exception_ptr e = driver_error_;
        driver_error_ = nullptr;
        std::rethrow_exception(e);
    }
}

void MoeExec::sequence_reset()
{
    if (route_trace_) {   // the driver is idle between passes; the marker separates sequences
        const int32_t h[2] = { -1, 0 };
        std::fwrite(h, 4, 2, route_trace_);
    }
}

void MoeExec::profile_routes(bool on)
{
    if (on && !route_counts_) route_counts_ = g_.alloc<float>((size_t) n_store() * E_);   // zeroed
    if (!on && route_counts_) {
        g_.sync();
        g_.free(route_counts_);
        route_counts_ = nullptr;
    }
}

std::vector<float> MoeExec::route_counts()
{
    std::vector<float> out((size_t) n_store() * E_, 0.f);
    if (route_counts_) {
        g_.sync();
        TRUSS_CUDA(cudaMemcpy(out.data(), route_counts_, out.size() * 4, cudaMemcpyDeviceToHost));
    }
    return out;
}

cpu::ExpertPool * MoeExec::pool() const { return cpu_tier_ ? cpu_tier_->pool.get() : nullptr; }
std::shared_ptr<cpu::ExpertPool> MoeExec::shared_pool() const { return cpu_tier_ ? cpu_tier_->pool : nullptr; }

void MoeExec::driver_ms(double out[3], long & n, bool reset)
{
    for (int i = 0; i < 3; ++i) out[i] = drv_ms_[i];
    n = drv_n_;
    if (reset) {
        for (double & v : drv_ms_) v = 0;
        drv_n_ = 0;
    }
}

void MoeExec::dyn_stats(long & cpu, long & pcie, double & ms_expert, double & ms_call) const
{
    cpu = dyn_cpu_, pcie = dyn_pcie_, ms_expert = cpu_ms_expert_, ms_call = cpu_ms_call_;
}

void MoeExec::split_stats(long long out[3]) const
{
    out[0] = split_host_bytes_, out[1] = split_cpu_experts_, out[2] = split_gpu_experts_;
}

void MoeExec::lend(void * base, size_t bytes, size_t row_bytes)
{
    const int j = store_->add_segment(static_cast<uint8_t *>(base), bytes);
    if (j >= 0) kv_seg_.push_back({ j, row_bytes });
}

// E6: positions [0, upto) (and one 4,096 step ahead) belong to the caches before anything writes them; with shrink,
// positions past that are lent back (a new sequence, a restore)
void MoeExec::kv_reserve(int upto, bool shrink, int n_ctx)
{
    if (kv_seg_.empty()) return;
    constexpr int STEP = 4096;
    const int R = std::min(n_ctx, (upto + STEP - 1) / STEP * STEP + STEP);
    if (R == kv_reserved_ || (R < kv_reserved_ && !shrink)) return;
    for (const auto & [j, row] : kv_seg_) store_->reclaim(j, (size_t) R * row);
    if (R > kv_reserved_) store_->fence(g_.s);   // copies into the reclaimed rows land before the caches are written
    kv_reserved_ = R;
}

}  // namespace truss::runtime
