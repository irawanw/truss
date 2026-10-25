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

#include <array>
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
    size_t pin_fill = 0;        // E7 (E6 option c, D8): bytes of the ring kept FIFO; the rest is pinned-refilled after a prompt
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
    bool hint_far_only = false;   // the hints start far experts' NVMe reads only (ExpertStore::prefetch_hint)
    int tail_skip = 0;          // Strata's route tail skip: a missed expert ranked >= this by weight in every row -> 0 (#192)
    float skip_weight = 0.f;    // PolyStrata #34 (D10): a missed expert under this share of its row's K routed weights in every row -> 0
    double budget_frac = 1.0;   // share of the free VRAM its experts take (layer-split stages sharing a device)
    bool last_stage = true;     // places the aux threads and writes TRUSS_ROUTE_TRACE (one stage does)
    // the far tier (ExpertStore far experts): far_path = n_layer x n_expert uint8 file, "" = off. far_cache_bytes is the
    // run's slot budget; a stage takes the share of it that its layers hold of the file's far experts. A decode layer
    // waits up to far_wait_us for its routed far experts' reads, then skips the ones still missing (term 0, as
    // tail_skip); prompt chunks route around them (their router bias is masked).
    // far_drop: the far experts are routed around everywhere (decode too) and never read: the pack without them
    // (E2's file benches as E1: the same experts at the same rates, top-6 over the rest)
    std::string far_path;
    size_t far_cache_bytes = 6000000000ull;
    int far_threads = 8, far_wait_us = 2000;
    int far_piece_kb = 1024;   // a far read cut in pieces of this many KiB for all readers (0: whole, one reader)
    bool far_drop = false;
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
    bool all_resident() const { return all_resident_; }   // B1: every expert on the device, no CPU tier
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
        bool fused_route_ok;                    // route == softmax top-k: the fused route+publish may replace it
        bool pre_routed = false;                // the block already routed (a fused projection+route): skip route()
        std::function<void()> shared;
        // DS41 image rows: rows [vl[2i], vl[2i+1]) (host, n_vl runs) are routed again with bias_vl, the router bias
        // of image-span tokens (the reference's Gate with image_mask); the far mask applies to it as to bias
        const float * bias_vl = nullptr;
        const int * vl = nullptr;
        int n_vl = 0;
    };
    void run(const Call & c);
    // whether the block should compute pred_logits (the next layer's router on this input) for this call
    bool wants_hint(int sl, int T, bool stream) const
    {
        return !stream && !all_resident_ && o_.hint_k > 0 && layers_[sl].next_in_stage && T >= o_.hint_min_rows;
    }
    // B2: one zeroed device word for the fused lone-router kernel's CTA counter (same-stream launches share it by
    // ordering; the kernel resets it, so it is graph-safe)
    unsigned * route_arrive() { return route_arrive_; }

    // between passes
    void begin_ring() { store_->begin_ring(); }
    void begin_stream(cudaStream_t s) { store_->begin_stream(s); }
    void prefetch(int sl) { store_->prefetch(sl); }
    int n_store() const { return (int) layers_.size(); }
    void adapt_tick();                          // Options::adapt_every: Strata's adaptive tier, every n decode passes
    void drain_jobs();                          // the driver has served every queued decode FFN
    // a decode knob changed between passes (truss_set_option: the bench's arms on one loaded model). Names are the
    // MoeOptions fields: skip_weight, tail_skip, pcie_frac, hint_k, hint_min_rows, adapt_every, adapt_swaps,
    // adapt_min, adapt_decay. Waits for the driver first. False: not a runtime knob (load-time ones need a reload).
    bool set_option(const std::string & name, double v);
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
    long tail_skipped() const { return tail_skipped_.load(std::memory_order_relaxed); }   // experts, since construction
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
    MoeOptions o_;   // fixed after construction except the set_option() knobs
    engine::Profile * prof_;
    std::unique_ptr<ExpertStore> store_;
    ExpertStore::HotSet far_;                   // [store layer][expert] far (o_.far_path); empty: no far tier
    void load_far();                            // far_ from o_.far_path (model layers -> store layers)
    // prompt chunks: the router's bias with the far experts at -1e30, per store layer (made on first use from the
    // block's bias; nullptr: none yet). Selection = score + bias, so a chunk never routes to a far expert and picks
    // the next one instead (E1's routing); a layer without a bias gets its far ids masked after routing.
    std::vector<std::array<float *, 2>> far_bias_;              // per layer: the far-masked copies of up to two
    std::vector<std::array<const float *, 2>> far_bias_src_;    // biases (text, image rows) and their sources
    std::vector<uint8_t *> far_dev_;            // [store layer] far_ on the device (mask_far_ids)
    double far_share_ = 0;                      // this stage's share of the far experts (its slot budget)
    const float * far_masked_bias(int l, const float * bias);
    static void mask_far_ids(int * ids, int n, const uint8_t * far_dev, cudaStream_t s);
    int n_hot_ = 0;
    bool all_resident_ = false;   // every expert of the stage on the device, no CPU tier: decode needs no copy plan
    void * window_ws_ = nullptr;
    unsigned * route_arrive_ = nullptr;   // B2: the fused lone-router kernel's self-resetting CTA counter
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
    // the decode log line (serve(): every 10 s of decode, far tier or CPU tier): counters at the last line
    std::chrono::steady_clock::time_point dlog_t_{}, dlog_last_{};
    ExpertStore::FarStats dlog_far_{};
    long dlog_calls_ = 0, dlog_cpu_ = 0, dlog_pcie_ = 0, dlog_n_ = 0;
    void decode_log();

    // residency policies: Strata's dynamic split (cost fit), adaptive tier, idle-link admission, E7 split stats
    double cpu_ms_call_ = 0.12, cpu_ms_expert_ = 0.085, pcie_ms_byte_ = 1.0 / 13.5e6;
    double dyn_s1_ = 0, dyn_sn_ = 0, dyn_st_ = 0, dyn_snn_ = 0, dyn_snt_ = 0;
    long dyn_cpu_ = 0, dyn_pcie_ = 0;
    std::atomic<long> tail_skipped_{ 0 };
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
