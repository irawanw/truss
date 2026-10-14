// Routed experts of one stage: residency, the doorbell driver, the CPU tier and the residency policies, behind one
// call per MoE layer. Model-agnostic: it sees expert tables, a kernel table and router outputs, never a model.
//
// A stage's MoE layers (and a drafter's, e.g. Flash-Next's MTP block) are "store layers" 0 .. n_store-1 of one
// runtime::ExpertStore. Per call, by chunk size:
//   doorbell  decode windows (T <= cpu::MAX_ROWS, Options::doorbell): the GPU publishes the routing into mapped host
//             memory and continues; the driver thread (moe_driver.cu) splits the slots between the GPU and the CPU
//             tier, issues the copies and writes the plan; the GPU waits for the plan on the device (TRACKER #61)
//   fetch     other chunks up to FETCH_ROWS rows: one sync for the routing, then the copies
//   split     short prompt chunks with the CPU tier (Options::split_rows, E7, TRACKER #117)
//   stream    longer prompt chunks: whole layers stream ahead of the compute (prefetch two layers ahead)
// The block passes its shared expert (or nothing) as `shared`, so the doorbell path can run it while the driver
// plans (TRACKER #112). Moved from the old qwen4exp::Forward unchanged (bit-identical, tests/golden).
#pragma once
#include "cpu/expert_q4.h"
#include "cpu/expert_trellis.h"
#include "cpu/placement.h"
#include "engine/gpu.h"
#include "engine/profile.h"
#include "runtime/expert_store.h"
#include "runtime/moe_kernels.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace truss::runtime {

// the knobs of the routed-expert runtime (engine::Options carries them; meaning and TRACKER rows there)
struct MoeOptions {
    size_t expert_budget = 0;
    size_t ring_bytes = 4ull << 30;
    double plan_alpha = 0.0;
    std::string cpu_dir;
    float cpu_share = 0.5f;
    int cpu_threads = 12;
    bool cpu_dynamic = false, cpu_trellis = false;
    float pcie_gbps = 13.5f, pcie_frac = -1.f;
    int adapt_every = 0, adapt_swaps = 96;
    float adapt_min = 2.f, adapt_decay = 0.7f;
    int admit_idle = 0;
    float admit_min = 1.f, admit_decay = 0.9f;
    int split_rows = 0, split_t = 16;
    float split_ms_row = 0.01f;
    bool doorbell = true;
    int hint_k = 4, hint_min_rows = 0;
    double budget_frac = 1.0;   // share of the free VRAM its experts take (layer-split stages sharing a device)
    bool last_stage = true;     // places the aux threads and writes TRUSS_ROUTE_TRACE (one stage does)
};

// one store layer as the stage sees it
struct MoeLayer {
    ExpertLayer tables;        // gate, up, down trellis tables
    int model_layer = 0;       // for traces and the section profile (a drafter's layer: n_layer)
    bool cpu_ok = true;        // the CPU tier may take its experts (not a drafter's)
    bool next_in_stage = false;   // store layer + 1 is the next model layer on this stage (pre-gating hint target)
    bool profiled = true;         // its sections go to the stage's Profile (a drafter's layer: not)
};

class MoeExec {
public:
    // usage: routed counts [n_store][n_expert] (may be empty); stream_extra: bytes of the stage's prompt buffers to
    // carve from the ring's spare region; pool/pool_mu: a CPU pool shared with an earlier split stage
    MoeExec(engine::Gpu & g, const MoeKernels & k, std::vector<MoeLayer> layers, int n_expert, int topk,
            const std::vector<float> & usage, size_t stream_extra, const MoeOptions & o, int max_rows,
            std::shared_ptr<cpu::ExpertPool> pool, std::shared_ptr<std::mutex> pool_mu, engine::Profile * prof);
    ~MoeExec();

    // the stage's memory plan: VRAM figures for the startup log, and where a stage carves its prompt buffers
    ExpertStore & store() { return *store_; }
    const ExpertStore & store() const { return *store_; }
    int n_hot() const { return n_hot_; }
    void * spare() { return store_->spare(); }

    // One MoE layer. logits [T][E] router output (the block computed it); pred_logits [T][E] the next store layer's
    // router on the same input, or nullptr (pre-gating hint, TRACKER #58); ids/wts [T][K] scratch the routing goes
    // to; routed [T][d] gets the expert sum; shared() runs the block's shared expert (may be empty); moe_ws/moe_rows:
    // the prefill workspace of the buffers in use; route(): the router's top-k (the block's routing rule).
    // the routing rule: logits [T][E] (+ a per-expert selection bias, may be nullptr) -> ids, weights [T][K]
    using RouteFn = void (*)(const float * logits, const float * bias, int T, int E, int K, int * ids, float * wts,
                             cudaStream_t);
    struct Call {
        int sl;                                 // store layer
        const float * x;                        // [T][d] FFN input rows
        int T;
        const float * logits, * pred_logits;
        const float * bias, * pred_bias;        // the router's selection bias (and the next layer's), or nullptr
        int * ids;
        float * wts;
        float * routed;
        bool stream;                            // a prompt chunk in stream mode
        void * moe_ws;
        int moe_rows;
        RouteFn route;
        bool fused_route_ok;                    // route == softmax top-k: the fused route_publish may replace it
        std::function<void()> shared;
    };
    void run(const Call & c);
    // whether the block should compute pred_logits (the next layer's router on this input) for this call
    bool wants_hint(int sl, int T, bool stream) const
    {
        return !stream && o_.hint_k > 0 && layers_[sl].next_in_stage && T >= o_.hint_min_rows;
    }

    // between passes
    void begin_ring() { store_->begin_ring(); }
    void begin_stream(cudaStream_t s) { store_->begin_stream(s); }
    void prefetch(int sl) { store_->prefetch(sl); }
    int n_store() const { return (int) layers_.size(); }
    void adapt_tick();                          // Options::adapt_every: Strata's adaptive tier, every n decode passes
    void drain_jobs();                          // the driver has served every queued decode FFN
    void check_driver();                        // rethrow a driver error at the caller's next sync point
    void place_threads();                       // TRUSS_AUX_CPUS
    bool aux_on() const { return aux_.on; }
    pid_t caller_tid() const { return caller_tid_; }
    void after_chunk_aux(bool stream);          // the thread placement check run_chunk does per chunk
    bool split_chunk(bool stream, int T) const { return stream && cpu_tier_ && T <= o_.split_rows; }
    void set_split(bool on) { split_chunk_ = on; }
    void sequence_reset();                      // TRUSS_ROUTE_TRACE marker
    void start_pinning();
    void wait_pinned() { store_->wait_pinned(); }

    // routing profile (Pipeline::profile_routes)
    void profile_routes(bool on);
    std::vector<float> route_counts();

    // statistics (Pipeline's diagnostics)
    cpu::ExpertPool * pool() const;
    void driver_ms(double out[3], long & n, bool reset);
    void dyn_stats(long & cpu, long & pcie, double & ms_expert, double & ms_call) const;
    void split_stats(long long out[3]) const;
    std::shared_ptr<cpu::ExpertPool> shared_pool() const;

    // E6: lend a position-indexed array's unused tail to the ring (positions in 4,096 steps); reserve() before writes
    void lend(void * base, size_t bytes, size_t row_bytes);
    void kv_reserve(int upto, bool shrink, int n_ctx);
    size_t lent_bytes() const { return store_->lent_bytes(); }

private:
    friend struct Driver;
    struct CpuTier;
    struct Bell;
    struct Job {
        int sl, T, seq;
        bool hint, cpu;
    };

    void load_cpu_tier(const ExpertStore::HotSet & hot, const std::vector<float> & usage);           // cpu_tier.cc
    void load_cpu_tier_trellis(const ExpertStore::HotSet & hot, const std::vector<float> & usage);   // cpu_tier.cc
    void setup_doorbells();                     // moe_driver.cu
    void drive();                               // moe_driver.cu: the driver thread
    void serve(const Job & j, Bell & b);        // moe_driver.cu
    void fit_cpu_cost(int n, double t);         // moe_driver.cu
    void run_doorbell(const Call & c, bool cpu_on, bool fused, bool hint);   // moe_paths.cu
    void run_fetch(const Call & c, bool hint);                                // moe_paths.cu
    void run_split(const Call & c);                                           // moe_paths.cu
    int last_cpu_of(pid_t tid) const;

    engine::Gpu & g_;
    const MoeKernels k_;
    std::vector<MoeLayer> layers_;
    const int E_, K_;
    const MoeOptions o_;
    engine::Profile * prof_;
    std::unique_ptr<ExpertStore> store_;
    int n_hot_ = 0;
    void * window_ws_ = nullptr;
    int * ids_host_ = nullptr;                  // pinned: routing + prediction of a fetch chunk
    int cpu_rows_ = 0;                          // rows of the CPU tier call in flight
    // a split's stages share the CPU pool and may run prompt chunks at once: one call at a time, held from start()
    // to the join's wait() at the end of run()
    std::unique_lock<std::mutex> pool_lock_;
    void lock_pool();
    bool split_chunk_ = false;
    float * route_counts_ = nullptr;            // [n_store][E] while profiling

    // the CPU tier and the doorbell driver (definitions in moe_exec_state.h)
    std::unique_ptr<CpuTier> cpu_tier_;
    std::vector<Bell> bells_;                   // [store layer]
    std::thread driver_, watchdog_;
    std::mutex dq_mu_;
    std::condition_variable dq_cv_;
    std::deque<Job> jobs_;
    std::atomic<long> jobs_pushed_{ 0 }, jobs_done_{ 0 };
    bool driver_stop_ = false;
    std::exception_ptr driver_error_;
    cpu::AuxPlan aux_;
    std::atomic<pid_t> driver_tid_{ 0 };
    pid_t caller_tid_ = 0;
    long aux_passes_ = 0;
    bool aux_placed_ = false;
    std::chrono::steady_clock::time_point drv_t_[3];
    double drv_ms_[3] = { 0, 0, 0 };
    long drv_n_ = 0;
    bool drv_ts_ = false;
    double ts_ms_[7] = {}, ts_pool_ = 0;
    long ts_n_ = 0;
    std::chrono::steady_clock::time_point ts_prev_{}, ts_t3_{}, ts_t3b_{}, ts_t4_{};
    std::atomic<int> dbg_phase_{ 0 }, dbg_layer_{ -1 }, dbg_seq_{ 0 };
    std::atomic<long> dbg_served_{ 0 };
    bool dbg_on_ = false;
    std::atomic<int> drv_cpu_{ -1 };
    FILE * route_trace_ = nullptr;

    // residency policies: Strata's dynamic split (cost fit), adaptive tier, idle-link admission, E7 split stats
    double cpu_ms_call_ = 0.12, cpu_ms_expert_ = 0.085, pcie_ms_byte_ = 1.0 / 13.5e6;
    double dyn_s1_ = 0, dyn_sn_ = 0, dyn_st_ = 0, dyn_snn_ = 0, dyn_snt_ = 0;
    long dyn_cpu_ = 0, dyn_pcie_ = 0;
    bool cpu_dynamic_ = false;
    long decode_passes_ = 0;
    std::vector<float> route_usage_;            // [n_store][E] decayed (adaptive tier, driver thread)
    int admit_idle_ = 0, admit_pass_ = 0;
    std::vector<float> admit_u_;                // [n_store][E] decayed (idle-link admission, driver thread)
    long long split_host_bytes_ = 0, split_cpu_experts_ = 0, split_gpu_experts_ = 0;
    std::vector<std::pair<int, size_t>> kv_seg_;   // E6: (ring segment, bytes per position)
    int kv_reserved_ = 0;
};

}  // namespace truss::runtime
