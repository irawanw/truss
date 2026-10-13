#include "model/qwen4exp/forward.h"

#include "core/cublas_check.h"
#include "core/cuda_check.h"
#include "core/device_tensors.h"
#include "core/scratch.h"
#include "cpu/expert_q4.h"
#include "cpu/placement.h"
#include "kernels/dense/q8_gemm.cuh"
#include "kernels/dsa/dsa_prefill.cuh"
#include "kernels/dsa/dsa_prepare.cuh"
#include "kernels/ffn/ffn_ops.cuh"
#include "kernels/gdn/gdn_prefill.cuh"
#include "kernels/gdn/gdn_prepare.cuh"
#include "kernels/hc/hc_prefill.cuh"
#include "kernels/moe/moe_prefill.cuh"
#include "kernels/moe/moe_window.cuh"
#include "kernels/mtp/mtp_ops.cuh"
#include "kernels/ple/ple_prefill.cuh"
#include "kernels/sampling/argmax.cuh"
#include "kernels/sampling/spec.cuh"
#include "kernels/spec/rollback.cuh"
#include "model/qwen4exp/ple.h"
#include "model/qwen4exp/ple_reader.h"
#include "runtime/doorbell.cuh"
#include "runtime/expert_store.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <tuple>
#include <type_traits>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <immintrin.h>
#include <mutex>
#include <numeric>
#include <thread>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>

namespace truss::qwen4exp {

using DsaShape = dsa::FlashNext;
using MoeShape = moe::FlashNext;

namespace {

// TRUSS_LOAD_TIMES=1: seconds since the library was loaded at each startup step (stderr)
const auto load_t0 = std::chrono::steady_clock::now();
void load_mark(const char * what)
{
    static const bool on = std::getenv("TRUSS_LOAD_TIMES") != nullptr;
    if (on)
        std::fprintf(stderr, "load %7.2f s  %s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - load_t0).count(), what);
}

void require(bool ok, const std::string & what)
{
    if (!ok) throw std::runtime_error("qwen4exp::Forward: " + what);
}

template <class X> X * dmalloc(size_t n, bool zero = true)
{
    X * p;
    TRUSS_CUDA(cudaMalloc(&p, n * sizeof(X)));
    if (zero) TRUSS_CUDA(cudaMemset(p, 0, n * sizeof(X)));
    return p;
}

// the device current for one scope (a split's stages live on different devices)
struct DeviceGuard {
    int prev = -1;
    explicit DeviceGuard(int dev)
    {
        TRUSS_CUDA(cudaGetDevice(&prev));
        if (dev >= 0 && dev != prev) TRUSS_CUDA(cudaSetDevice(dev));
        else prev = -1;
    }
    ~DeviceGuard()
    {
        if (prev >= 0) cudaSetDevice(prev);
    }
};

}  // namespace

// One stage of a layer split (Options::split_layer): layers [lb, le) on `device`; the first stage embeds the tokens,
// the last one (le == n_layer) has the head and the MTP block. One stage over every layer is the unsplit engine.
struct StageSpec {
    int lb = 0, le = 0;
    int device = -1;                           // -1: the current device
    double budget_frac = 1.0;                  // share of the free VRAM its experts take (stages sharing a device)
    std::shared_ptr<cpu::ExpertPool> pool;     // the CPU pool of an earlier stage, shared (one set of workers)
    std::shared_ptr<std::mutex> pool_mu;
};

struct Forward::Impl {
    const Config & c;
    const Weights & w;
    const int n_ctx, max_chunk, prefill_rows;
    const int lb, le, device;                            // StageSpec: layers [lb, le) on device
    const bool first, last;                              // embeds the tokens / has the head and the MTP block
    int sl(int l) const { return l - lb; }               // ExpertStore layer of model layer l (MTP: n_layer)
    // layer split hand-off: the residual rows after le - 1 go to the next stage through pinned host memory
    // (out_host, written on this stream, out_ready recorded behind it); the next stage records in_taken once its
    // copy out of them is issued, and this stage waits for that before it writes them again
    float * out_host = nullptr;
    cudaEvent_t out_ready = nullptr, in_taken = nullptr;
    Impl * prev = nullptr, * next = nullptr;             // the stages before / after (prev's out_host is the input)
    // host-side order of the hand-off when the stages run on their own threads (Forward::run_prompt): the events are
    // only waited for once recorded, so a stage enqueues its write of out_host after the next stage has enqueued its
    // take of the previous rows (handed == next->taken), and its take after the previous stage handed them on
    std::atomic<long> handed{ 0 }, taken{ 0 };
    std::atomic<bool> halted{ false };                   // this stage's prompt thread failed: the other stops waiting
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    std::unique_ptr<DeviceTensors> dev;                  // every tensor except Q8_0 matrices and the PLE table
    std::unordered_map<T, dense::Q8Matrix> q8;           // Q8_0 matrices, repacked
    std::vector<void *> owned;                           // cudaFree at exit
    // Per-chunk buffers come in two sizes. Chunks of up to FETCH_ROWS rows (decode steps, verify windows) use small
    // resident ones; longer prompt chunks borrow the ExpertStore ring's spare region (stream mode), so the VRAM a
    // prompt needs for scratch, MoE workspace and residual holds cached experts during decode (TRACKER #57).
    struct Buffers {
        Scratch * scratch;                               // temporaries, reset per layer
        void * moe_ws;                                   // moe::prefill
        int moe_rows;                                    // rows moe_ws was sized for
        float * res;                                     // [rows][hc][d]
        float * mh, * mres;                              // MTP block: input hidden rows, its residual [rows][hc][d]
    };
    Scratch small_scratch;
    Buffers small{}, big{}, cur{};
    std::unique_ptr<Scratch> big_scratch;
    Scratch * sc = nullptr;                              // cur.scratch
    float * res = nullptr;                               // cur.res (persists after run() for head())
    half * w16 = nullptr;                                // q8_gemm_a16's dequantized weight
    size_t w16_elems = 0;
    void * window_ws = nullptr;                          // moe::window (decode steps)
    int * ids_host = nullptr;                            // pinned: a decode step's routing, for the expert fetch
    // CPU tier (Options::cpu_dir): which experts of each layer run on the host, their 4-bit copies, the pool, and
    // pinned staging for one layer's rows (x, routing weights in; y out) and the GPU kernel's masked routing
    struct CpuTier {
        std::vector<std::vector<uint8_t>> is_cpu;         // [layer][expert]
        std::vector<std::vector<uint8_t>> data;           // [layer] the CPU experts' q4s bytes
        std::vector<std::vector<float>> fscales;          // [layer] preconverted fp32 weight scales (one per
                                                          // q4s block, gate+up+down order per expert; the per-block
                                                          // _cvtsh_ss in the hot loop cost ~0.9 ms/expert)
        std::vector<std::vector<cpu::Q4Expert>> view;     // [layer][expert]
        bool trellis = false;                             // Options::cpu_trellis: tview, not view
        std::vector<std::vector<cpu::TrellisExpert>> tview;   // [layer][expert] views of the pinned pack bytes
        cpu::Slot slot(int row, int l, int e, float w)
        {
            return trellis ? cpu::Slot{ row, nullptr, w, &tview[l][e] } : cpu::Slot{ row, &view[l][e], w };
        }
        std::shared_ptr<cpu::ExpertPool> pool;
        std::shared_ptr<std::mutex> pool_mu;              // a split's stages share the pool: one call at a time
        float * x = nullptr, * w = nullptr, * y = nullptr;   // pinned [rows][d], [rows][k], [rows][d]
        int * ids = nullptr;                                 // pinned [rows][k] routing with CPU slots as -1
        size_t bytes = 0;
    };
    std::unique_ptr<CpuTier> cpu_tier;

    // Doorbell decode (TRACKER #61): FFNs of windows up to cpu::MAX_ROWS rows run without a host sync. Per store
    // layer: mapped host buffers the GPU publishes the routing and FFN input into, the doorbell, the CPU tier's
    // result and done flag (mapped), a device "go" word the copy stream writes after the layer's copies, and the
    // masked ids (CPU slots = -1) the copy stream uploads. A driver thread services the layers in enqueue order.
    struct Bell {
        int * ids = nullptr, * pred = nullptr, * mids = nullptr;     // mapped [8][k]; mids: pinned source
        float * w = nullptr, * x = nullptr, * y = nullptr;           // mapped [8][k], [8][d], [8][d]
        int * doorbell = nullptr, * cpu_done = nullptr;             // mapped words
        int * plan = nullptr, * need_copy = nullptr;                // mapped words the driver writes (wait_plan)
        int * go = nullptr, * d_mids = nullptr;                       // device
        unsigned * arrive = nullptr;      // device: the fused route+publish's CTA counter (zeroed once, resets itself)
        int seq = 0;                                                  // host: the last seq enqueued
        // TRUSS_SECT_CSV: the host side of this layer's chain, microseconds after the driver saw the doorbell
        // (drv_t[0]) - the same origin as the CSV's GPU columns, which start at the publish event a few us earlier.
        // All of it is written before the release fence in front of cpu_done, and the GPU only reaches its last
        // event after that, so sect_flush reads these without racing the driver.
        double h_t[5] = {};      // split done, pool started, plan written, hint + admit done, pool wait returned
        double h_pool = 0;       // the pool's own span for this call (start -> last item)
        double h_wake = 0;       // ... its start -> first item started (-1: no item ran)
        double h_item = 0;       // ... its sum of item time, over every thread that ran one
        double h_longest = 0;    // ... its longest single item (the straggler the call ends with)
        int h_ncpu = 0;          // slots the CPU tier took for this layer (0: no CPU tier)
    };
    std::vector<Bell> bells;
    struct Job {
        int layer, T, seq;
        bool hint, cpu;
    };
    std::thread driver;
    // TRUSS_AUX_CPUS (cpu/placement.h): the driver pins itself to aux.driver_cpu, place_threads() moves the rest
    cpu::AuxPlan aux;
    std::atomic<pid_t> driver_tid{ 0 };
    pid_t caller_tid = 0;
    long aux_passes = 0;
    bool aux_placed = false;
    std::mutex dq_mu;
    std::condition_variable dq_cv;
    std::deque<Job> jobs;
    std::chrono::steady_clock::time_point drv_t[3];   // driver thread only: doorbell seen, split done, CPU started
    double drv_ms[3] = { 0, 0, 0 };                    // summed: split, CPU start, copies + plan (driver_ms)
    // TRUSS_DRIVER_TS=1: fine-grained driver segments (steady_clock only, no syncs; driver thread only).
    // [0] pre (exit -> next doorbell: job spin + admit_step), [1] split, [2] pool start, [3] copies + plan,
    // [4] hint + admit, [5] pool->wait() + publish, [6] exit tail. pool_done = pool's own start->last item.
    bool drv_ts = false;
    double ts_ms[7] = {}, ts_pool = 0; long ts_n = 0;
    std::chrono::steady_clock::time_point ts_prev{}, ts_t3{}, ts_t3b{}, ts_t4{};
    double ple_host_ms = 0;                            // host time of the PLE row hash + gather + fp16 convert
    // TRUSS_ROUTE_TRACE=<file>: every decode layer's routing (driver thread), for offline cache-policy replays.
    // Header: int32 n_store, n_expert, K, then per (layer, expert) int32 bytes and uint8 hot. Records: int32 layer, T,
    // ids [T * K]; layer -1 (T 0) marks a sequence reset.
    FILE * route_trace = nullptr;
    static double vram_used()   // GiB in use on the device (TRACKER #93 breakdown)
    {
        size_t f, t;
        TRUSS_CUDA(cudaMemGetInfo(&f, &t));
        return (t - f) / 1073741824.0;
    }
    std::unique_ptr<PleReader> ple_reader;           // Options::ple_file: rows read with O_DIRECT, not page faults
    half * ple_stage = nullptr;                        // pinned [rows][E]: the gathered rows for the device copy
    size_t ple_stage_n = 0;
    cudaEvent_t ple_copied = nullptr;                  // the last copy out of ple_stage has been read
    long ple_calls = 0, ple_rows_n = 0;                // (ple_host_ms(); the rows are table rows read)
    // PLE lookahead: run() may describe the prompt chunk after the current one; its rows are hashed and read
    // from disk while the current chunk computes, so the next layer-1 gather does not stall the host (E8).
    const int32_t * ple_next = nullptr; int ple_next_T = 0;
    bool ple_tk_valid = false; int ple_tk = -1, ple_tk_pos0 = -1;   // the prefetched ticket and its chunk's pos0
    long drv_n = 0;
    std::atomic<long> jobs_pushed{ 0 };   // the driver spins on this before it sleeps on dq_cv (Strata's host spins)
    std::atomic<long> jobs_done{ 0 };     // jobs the driver has finished (served or failed): see drain_jobs()
    bool driver_stop = false;
    // TRUSS_DRIVER_DEBUG=1: the driver's phase, and a watchdog that prints it with the stuck layer's doorbell words
    // when no job is served for 10 s (phases: 1 job spin, 2 job sleep, 3 doorbell spin, 4 serve, 5 fetch, 6 claim,
    // 7 pool wait, 8 admit_step)
    std::atomic<int> dbg_phase{ 0 }, dbg_layer{ -1 }, dbg_seq{ 0 };
    std::atomic<long> dbg_served{ 0 };
    std::thread watchdog;
    bool dbg_on = false;
    std::exception_ptr driver_error;
    bool use_doorbell = true;
    // Options::cpu_dynamic (Strata's split, TRACKER #73): per layer, missed experts with a host copy go to the CPU or
    // over PCIe so that both finish together. CPU time of a call with n experts = cpu_ms_call + cpu_ms_expert * n
    // (n > 0), a least-squares line through the pool's own call times (exponentially weighted sums dyn_s*; a flat
    // per-expert mean charged the ~0.1 ms wake/join of every call to its ~1 expert, TRACKER #75); pcie_ms_byte:
    // 1 / Options::pcie_gbps.
    bool cpu_dynamic = false;
    double cpu_ms_call = 0.12, cpu_ms_expert = 0.085, pcie_ms_byte = 1.0 / 13.5e6;
    double pcie_frac = -1;   // Options::pcie_frac
    // adaptive tier (Options::adapt_every): decayed routing counts per (store layer, expert), written by the driver
    // thread in serve(), read between passes by adapt()
    int adapt_every = 0, adapt_swaps = 96;
    float adapt_min = 2.f, adapt_decay = 0.7f;
    long decode_passes = 0;
    std::vector<float> route_usage;
    int admit_idle = 0, admit_pass = 0;                  // Options::admit_idle; claims this pass
    std::vector<std::pair<int, size_t>> kv_seg;          // E6: (ring segment, bytes per position)
    int kv_reserved = 0;                                 // positions of the K/V caches not lent

    // E6: positions [0, upto) (and one 4,096 step ahead) belong to the caches before anything writes them; with
    // shrink, positions past that are lent back (a new sequence, a restore)
    void kv_reserve(int upto, bool shrink)
    {
        if (kv_seg.empty()) return;
        constexpr int STEP = 4096;
        const int R = std::min(n_ctx, (upto + STEP - 1) / STEP * STEP + STEP);
        if (R == kv_reserved || (R < kv_reserved && !shrink)) return;
        for (const auto & [j, row] : kv_seg) experts->reclaim(j, (size_t) R * row);
        if (R > kv_reserved) experts->fence(s);   // copies into the reclaimed rows land before the caches are written
        kv_reserved = R;
    }
    float admit_min = 1.f, admit_decay = 0.9f;
    std::vector<float> admit_u;                          // decayed routing counts [layer][expert] (driver thread)
    double dyn_s1 = 0, dyn_sn = 0, dyn_st = 0, dyn_snn = 0, dyn_snt = 0;
    long dyn_cpu = 0, dyn_pcie = 0;                      // eligible misses sent each way (stats)
    int8_t * embd_q = nullptr;                           // token embedding in pinned host memory (upload())
    half * embd_d = nullptr;

    struct LayerState {
        dsa::KvCache kv;                                        // DSA K/V cache (fp16 or int8, Options::kv_int8)
        half * idx_k = nullptr;                                 // DSA indexer block cache
        float * idx_partial = nullptr;                          // raw indexer keys of the open block
        float * state_buf[2] = {};                              // GDN recurrence (two with speculation: a verify
        int cur = 0;                                            // window writes the other one)
        float * conv = nullptr;                                 // GDN conv rows
        float * ple_hist = nullptr;
        // verify-window rollback (Options::spec_rows): snapshots before the window and the window's raw rows
        float * conv_snap = nullptr, * raw_qkv = nullptr, * raw_alpha = nullptr, * raw_beta = nullptr;
        float * partial_snap = nullptr, * raw_ik = nullptr;
        float * hist_snap = nullptr, * normed_rows = nullptr;
        float * state() const { return state_buf[cur]; }
    };
    std::vector<LayerState> st;
    LayerState mst;                                      // the MTP block's DSA caches
    const Mtp * mtp = nullptr;
    const int spec_rows;
    float * pending_h = nullptr;                         // MTP: hidden state [hc][d] of the last committed row
    bool has_pending = false;
    float * mtp_partial_snap = nullptr;                  // MTP indexer open block before a draft chain
    float * mtp_logits = nullptr;                        // [n_vocab]
    int * draft_ids = nullptr;                           // device [spec_rows]
    int32_t * draft_host = nullptr;                      // pinned
    int32_t * commit_host = nullptr;                     // pinned [2][MAX_ROWS + 1]: a decode commit's tokens (mtp_commit)
    int commit_flip = 0;
    // accept() defers the MTP commit to draft(), which runs it together with the first draft row (pending_h = the
    // last accepted row's state, the bonus token): one MTP pass fewer per decode step. That row is the next commit's
    // first row, so it is remembered (ahead_*) and the next commit skips it when its token matches; otherwise the
    // commit recomputes it, which rewrites every position-indexed cache row it touched.
    bool dc_on = false;
    int dc_pos0 = 0, dc_T = 0;
    std::vector<int32_t> dc_tokens;
    bool mtp_ahead = false;
    int32_t ahead_tok = -1;
    int ahead_pos = -1;
    void flush_commit()
    {
        if (!dc_on) return;
        dc_on = false;
        mtp_commit(dc_tokens.data(), dc_pos0, dc_T, false);
    }
    struct Verify {                                      // the window verify() left uncommitted
        bool open = false;
        int pos0 = 0, T = 0;
        std::vector<int32_t> tokens;
    } vw;
    std::unique_ptr<runtime::ExpertStore> experts;
    int n_hot = 0;                                       // resident experts, all layers
    std::vector<int32_t> tail;                           // the last ple_ngram - 1 tokens seen (PLE window)

    const Activations act;
    const int hint_k;                                    // Options::hint_k
    const int hint_min_rows;                             // Options::hint_min_rows
    const int split_rows, split_t;                       // Options::split_rows / split_t (E7)
    const double split_ms_row;                           // Options::split_ms_row
    bool split_chunk = false;                            // this prompt chunk runs E7's split (run_chunk)
    long long split_host_bytes = 0, split_cpu_experts = 0, split_gpu_experts = 0;   // E7 stats, since construction
    dense::Q8Matrix draft_head{};                        // the output matrix's draft_vocab rows (Options::draft_vocab)
    int * draft_map = nullptr;                           // device [draft_head.out]: row -> token id
    float draft_min_p = 0.f;
    const sampling::SampleParams * ds_p = nullptr;        // set_draft_sampling(): sampled guesses (spec. sampling)
    uint64_t ds_counter = 0;
    float * ds_q = nullptr;                              // device [draft][n_vocab]
    float * draft_prob = nullptr;                        // device [spec_rows], pinned mirror below
    float * draft_prob_host = nullptr;
    int n_store = 0;                                     // ExpertStore layers: n_layer (+ 1 with the MTP block)

    // The prompt path's buffers are sized by prefill_rows, not max_chunk: they are carved from the ring's spare
    // region, so every row above the prompt's real length holds cached experts out of VRAM for the whole run
    // (TRACKER #70). 0 (or >= max_chunk) keeps the old behaviour.
    static int prefill_rows_of(int mc, int rows)
    {
        if (rows <= 0 || rows >= mc) return mc;
        const int r = (rows + DsaShape::RATIO - 1) / DsaShape::RATIO * DsaShape::RATIO;   // whole DSA blocks
        return std::min(std::max(r, FETCH_ROWS + 1), mc);
    }

    Impl(const Config & cc, const Weights & ww, int nc, int mc, const Options & o, const StageSpec & sg)
        : c(cc), w(ww), n_ctx(nc), max_chunk(mc),
          prefill_rows(prefill_rows_of(mc, o.prefill_rows)),
          lb(sg.lb), le(sg.le), device(sg.device), first(sg.lb == 0), last(sg.le == cc.n_layer),
          small_scratch(scratch_bytes(cc, std::min(mc, FETCH_ROWS), nc, o.mtp)),
          spec_rows(o.spec_rows), act(o.act), hint_k(o.hint_k), hint_min_rows(o.hint_min_rows), split_rows(o.split_rows), split_t(o.split_t), split_ms_row(o.split_ms_row)
    {
        load_mark("Forward: start");
        if (device >= 0) TRUSS_CUDA(cudaSetDevice(device));   // the caller's guard restores its device
        check_shapes();
        require(lb >= 0 && lb < le && le <= c.n_layer, "stage layer range");
        mtp = last ? o.mtp : nullptr;
        draft_min_p = o.draft_min_p;
        TRUSS_CUDA(cudaStreamCreate(&s));
        TRUSS_CUBLAS(cublasCreate(&blas));
        const double vram0 = vram_used();   // the CUDA context and anything before this Forward
        upload();
        load_mark("dense weights uploaded");
        const double vram_w = vram_used();
        if (o.after_upload) o.after_upload();
        const int small_rows = std::min(max_chunk, FETCH_ROWS);
        require(spec_rows >= 0 && spec_rows <= moe::MAX_ROWS, "spec_rows must be 0.." + std::to_string(moe::MAX_ROWS));
        small = { &small_scratch, alloc<unsigned char>(moe::prefill_workspace_bytes<MoeShape>(small_rows)), small_rows,
                  alloc<float>((size_t) small_rows * c.hc_dim()), nullptr, nullptr };
        if (mtp) {
            small.mh = alloc<float>((size_t) small_rows * c.hc_dim());
            small.mres = alloc<float>((size_t) small_rows * c.hc_dim());
            pending_h = alloc<float>(c.hc_dim());
            mtp_partial_snap = alloc<float>((size_t) (DsaShape::RATIO - 1) * c.idx_head_dim);
            mtp_logits = alloc<float>(c.n_vocab);
            draft_ids = alloc<int>(moe::MAX_ROWS + 1);
            TRUSS_CUDA(cudaMallocHost(&draft_host, sizeof(int32_t) * (moe::MAX_ROWS + 1)));
            TRUSS_CUDA(cudaMallocHost(&commit_host, sizeof(int32_t) * 2 * (moe::MAX_ROWS + 1)));
            draft_prob = alloc<float>(moe::MAX_ROWS + 1);
            TRUSS_CUDA(cudaMallocHost(&draft_prob_host, sizeof(float) * (moe::MAX_ROWS + 1)));
            if (!o.draft_vocab.empty()) {   // gather the subset's output rows once
                const dense::Q8Matrix & O = q8.at(w.output);
                const int n = (int) o.draft_vocab.size();
                for (int32_t t : o.draft_vocab) require(t >= 0 && t < O.out, "draft_vocab token id out of range");
                draft_map = alloc<int>(n);
                TRUSS_CUDA(cudaMemcpy(draft_map, o.draft_vocab.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
                int8_t * q = alloc<int8_t>((size_t) n * O.in);
                half * dd = alloc<half>((size_t) n * O.in / 32);
                dense::q8_gather(O, draft_map, n, q, dd, s);
                draft_head = { q, dd, O.in, n };
            }
        }
        use(small);
        window_ws = alloc<unsigned char>(moe::workspace_bytes<MoeShape>());
        moe::workspace_init<MoeShape>(window_ws, s);
        TRUSS_CUDA(cudaMallocHost(&ids_host, sizeof(int) * 2 * std::max(FETCH_ROWS, split_rows) * MoeShape::TOPK));   // routing + prediction
        const double vram_st0 = vram_used();
        st.resize(c.n_layer);   // indexed by model layer; only [lb, le) allocated
        if (!last) {
            TRUSS_CUDA(cudaMallocHost(&out_host, sizeof(float) * (size_t) std::max(max_chunk, prefill_rows) * c.hc_dim()));
            TRUSS_CUDA(cudaEventCreateWithFlags(&out_ready, cudaEventDisableTiming));
        }
        if (!first) TRUSS_CUDA(cudaEventCreateWithFlags(&in_taken, cudaEventDisableTiming));
        const int R = DsaShape::RATIO;
        const int W = spec_rows;
        auto dsa_state = [&](LayerState & L) {
            const size_t kvn = (size_t) n_ctx * c.n_head_kv * c.head_dim;
            if (o.kv_int8) {   // Strata's int8 KV: 1,056 B per cell per layer instead of 2,048 (TRACKER #87)
                L.kv.kq = alloc<int8_t>(kvn), L.kv.vq = alloc<int8_t>(kvn);
                L.kv.ks = alloc<half>(kvn / dsa::KV_GROUP), L.kv.vs = alloc<half>(kvn / dsa::KV_GROUP);
            } else {
                L.kv.k16 = alloc<half>(kvn), L.kv.v16 = alloc<half>(kvn);
            }
            L.idx_k = alloc<half>((size_t) (n_ctx / R) * c.idx_head_dim);
            L.idx_partial = alloc<float>((size_t) (R - 1) * c.idx_head_dim);
        };
        for (int l = lb; l < le; ++l) {
            LayerState & L = st[l];
            if (c.mixer[l] == Mixer::DSA) {
                dsa_state(L);
                if (W) L.partial_snap = alloc<float>((size_t) (R - 1) * c.idx_head_dim), L.raw_ik = alloc<float>((size_t) W * c.idx_head_dim);
            } else {
                const size_t sz = (size_t) c.ssm_v_heads * c.ssm_state * c.ssm_state;
                L.state_buf[0] = alloc<float>(sz);
                if (W) L.state_buf[1] = alloc<float>(sz);
                L.conv = alloc<float>((size_t) (c.ssm_conv - 1) * c.conv_dim());
                if (W) {
                    L.conv_snap = alloc<float>((size_t) (c.ssm_conv - 1) * c.conv_dim());
                    L.raw_qkv = alloc<float>((size_t) W * c.conv_dim());
                    L.raw_alpha = alloc<float>((size_t) W * c.ssm_v_heads);
                    L.raw_beta = alloc<float>((size_t) W * c.ssm_v_heads);
                }
            }
            if (c.is_ple(l)) {
                const size_t hist = (size_t) (c.ple_conv - 1) * c.ple_ngram * c.hc_dim();
                L.ple_hist = alloc<float>(hist);
                if (W) L.hist_snap = alloc<float>(hist), L.normed_rows = alloc<float>((size_t) W * c.hc_dim());
            }
        }
        if (mtp) dsa_state(mst);
        const double vram_st = vram_used();
        std::vector<runtime::ExpertLayer> tables;
        for (int l = lb; l < le; ++l) {
            const Layer & L = w.layers[l];
            tables.push_back({ &L.moe.gate, &L.moe.up, &L.moe.down });
        }
        if (mtp) tables.push_back({ &mtp->layer.moe.gate, &mtp->layer.moe.up, &mtp->layer.moe.down });
        n_store = (int) tables.size();
        std::vector<float> usage;   // this stage's rows of [layer][expert] (and the MTP block's when given)
        if (!o.expert_usage.empty()) {
            const size_t E = c.n_expert;
            require(o.expert_usage.size() >= (size_t) c.n_layer * E, "expert_usage has fewer layers than the model");
            usage.assign(o.expert_usage.begin() + (size_t) lb * E, o.expert_usage.begin() + (size_t) le * E);
            if (mtp && o.expert_usage.size() >= (size_t) (c.n_layer + 1) * E)
                usage.insert(usage.end(), o.expert_usage.begin() + (size_t) c.n_layer * E,
                             o.expert_usage.begin() + (size_t) (c.n_layer + 1) * E);
        }
        if (mtp && usage.size() == (size_t) (le - lb) * c.n_expert) {
            // no MTP profile: the MTP block runs once per draft (3-4 times per verify pass), so its experts rank
            // above every layer's (all resident when they fit; TRACKER #58)
            const float top = *std::max_element(usage.begin(), usage.end());
            usage.insert(usage.end(), c.n_expert, 4.f * top + 1.f);
        }
        size_t expert_budget = o.expert_budget;
        if (!expert_budget) {
            size_t free_b, total_b;
            TRUSS_CUDA(cudaMemGetInfo(&free_b, &total_b));
            constexpr size_t MARGIN = 768ull << 20;   // cuBLAS workspaces, the CUDA context's growth
            require(free_b > MARGIN, "no device memory left for the routed experts");
            expert_budget = (size_t) ((double) (free_b - MARGIN) * std::min(1.0, std::max(0.05, sg.budget_frac)));
            const double gb = 1073741824.0;
            std::fprintf(stderr, "VRAM: %.2f GiB total, %.2f GiB in use before the experts = context and before %.2f + "
                                 "weights %.2f + buffers %.2f + KV/state %.2f + more buffers %.2f; routed experts get "
                                 "%.2f GiB (margin %.2f)\n",
                         total_b / gb, (total_b - free_b) / gb, vram0, vram_w - vram0, vram_st0 - vram_w,
                         vram_st - vram_st0, (total_b - free_b) / gb - vram_st, expert_budget / gb, MARGIN / gb);
        }
        runtime::ExpertStore::Sizes z;
        z.ring_bytes = o.ring_bytes_override ? o.ring_bytes_override : o.ring_bytes;
        z.plan_alpha = o.plan_alpha;
        if (prefill_rows > FETCH_ROWS) z.stream_extra = big_bytes();
        const runtime::ExpertStore::HotSet hot = runtime::ExpertStore::plan(tables, expert_budget, z, usage);
        for (const auto & h : hot) n_hot += (int) std::count(h.begin(), h.end(), 1);
        load_mark("expert plan");
        experts = std::make_unique<runtime::ExpertStore>(tables, hot, z);
        load_mark("expert store (hot upload + cold copy; page-locking continues in the background)");
        if (o.kv_lend) {   // E6: every DSA layer's K and V position arrays (MTP block included)
            auto lend = [&](const LayerState & L) {
                const size_t row = (size_t) c.n_head_kv * c.head_dim * (L.kv.int8() ? 1 : 2);
                for (void * p : { L.kv.int8() ? (void *) L.kv.kq : (void *) L.kv.k16,
                                  L.kv.int8() ? (void *) L.kv.vq : (void *) L.kv.v16 }) {
                    const int j = experts->add_segment(static_cast<uint8_t *>(p), (size_t) n_ctx * row);
                    if (j >= 0) kv_seg.push_back({ j, row });
                }
            };
            for (int l = lb; l < le; ++l)
                if (c.mixer[l] == Mixer::DSA) lend(st[l]);
            if (mtp) lend(mst);
            kv_reserve(0, true);
        }
        // The store's constructor is the last reader of the shards: it memcpy'd every cold expert into pinned host
        // memory and uploaded the hot set, so from here the mapping is dead weight. Releasing before load_cpu_tier
        // keeps a ~24 GB resident peak off the books while that vector allocates (TRACKER #72).
        if (o.after_upload) o.after_upload();
        if (o.ple_file && w.ple_table) {   // PLE rows by O_DIRECT reads (TRACKER #88)
            ple_reader = std::make_unique<PleReader>(*o.ple_file, *w.ple_table, *w.ple_scale, c.ple_head_dim);
            ple_stage_n = (size_t) std::max(max_chunk, prefill_rows) * c.ple_heads() * c.ple_head_dim;
            TRUSS_CUDA(cudaMallocHost(&ple_stage, ple_stage_n * sizeof(half)));
            TRUSS_CUDA(cudaEventCreateWithFlags(&ple_copied, cudaEventDisableTiming));
        }
        load_mark("after_upload + PLE reader");
        if (!o.cpu_dir.empty()) load_cpu_tier(o, hot, sg);
        else if (o.cpu_trellis) load_cpu_tier_trellis(o, hot, sg);
        load_mark("CPU tier");
        use_doorbell = o.doorbell;
        cpu_dynamic = o.cpu_dynamic && cpu_tier && use_doorbell;
        pcie_ms_byte = 1.0 / (o.pcie_gbps * 1e6);
        pcie_frac = o.pcie_frac;
        adapt_every = o.adapt_every, adapt_swaps = o.adapt_swaps;
        adapt_min = o.adapt_min, adapt_decay = o.adapt_decay;
        if (adapt_every > 0) route_usage.assign((size_t) n_store * c.n_expert, 0.f);
        admit_idle = cpu_tier && use_doorbell ? o.admit_idle : 0;
        admit_min = o.admit_min, admit_decay = o.admit_decay;
        if (admit_idle > 0) admit_u.assign((size_t) c.n_layer * c.n_expert, 0.f);
        if (cpu_tier && last) aux = cpu::plan_aux(cpu_tier->pool->worker_cpus());   // a split: the last stage places
        if (const char * rt = last ? std::getenv("TRUSS_ROUTE_TRACE") : nullptr) {   // one stage writes it
            route_trace = std::fopen(rt, "wb");
            require(route_trace != nullptr, std::string("cannot write ") + rt);
            const int32_t hdr[3] = { n_store, c.n_expert, c.n_expert_used };
            std::fwrite(hdr, 4, 3, route_trace);
            for (int l = 0; l < n_store; ++l)
                for (int e = 0; e < c.n_expert; ++e) {
                    const int32_t b = (int32_t) experts->bytes_of(l, e);
                    const uint8_t hot = experts->host_part(l, e, 0) == nullptr;
                    std::fwrite(&b, 4, 1, route_trace), std::fwrite(&hot, 1, 1, route_trace);
                }
        }
        setup_doorbells();
        if (prefill_rows > FETCH_ROWS) {   // the prompt path's buffers, carved from the ring's spare region
            auto * p = static_cast<unsigned char *>(experts->spare());
            const size_t sb = scratch_bytes(c, prefill_rows, n_ctx, mtp), wb = align(moe::prefill_workspace_bytes<MoeShape>(prefill_rows));
            big_scratch = std::make_unique<Scratch>(p, sb);
            const size_t rb = align(sizeof(float) * prefill_rows * c.hc_dim());
            big = { big_scratch.get(), p + align(sb), prefill_rows, reinterpret_cast<float *>(p + align(sb) + wb), nullptr,
                    nullptr };
            if (mtp) {
                big.mh = reinterpret_cast<float *>(p + align(sb) + wb + rb);
                big.mres = reinterpret_cast<float *>(p + align(sb) + wb + 2 * rb);
            }
        }
        if (experts) experts->start_pinning();   // last: it holds the driver lock for ~3 s
        if (!aux.text.empty()) std::fprintf(stderr, "%s\n", aux.text.c_str());
        if (aux.on) {   // after start_pinning: its thread moves too
            while (!driver_tid.load()) std::this_thread::yield();
            place_threads();
        }
        load_mark("Forward: done");
        if (mtp) mtp_calib_open(prefill_rows);
        if (getenv("TRUSS_PROFILE_SECTIONS") || getenv("TRUSS_SECT_CSV")) {
            sect_on = true;
            sect_ev.resize(c.n_layer + 1);   // by model layer (MTP: n_layer)
            sect_rec.assign(c.n_layer + 1, 0);
            sect_bell.assign(c.n_layer + 1, 0);
            for (auto & a : sect_ev)
                for (cudaEvent_t & e : a) TRUSS_CUDA(cudaEventCreate(&e));
            if (const char * p = getenv("TRUSS_SECT_CSV")) {   // one line per layer per pass (the pass breakdown)
                char path[1024];
                // One FILE per stage: two stages appending to one path interleave their half-written lines.
                std::snprintf(path, sizeof path, "%s.d%d", p, device);
                sect_csv = std::fopen(path, "ab");
                if (sect_csv)
                    std::fprintf(sect_csv,
                                 "# pass rows layer route_publish wait_plan wait_copies window cpu_join helper_join "
                                 "tail ple_attn_mix mixer hc_mix moe_total shared combine host_split host_pool_start "
                                 "host_plan host_hint_admit host_publish pool_span pool_wake pool_compute "
                                 "pool_longest ncpu\n");
            }
        }
    }

    ~Impl()
    {
        if (device >= 0) cudaSetDevice(device);   // the owner's guard restores its device
        if (route_trace) std::fclose(route_trace);
        if (out_host) cudaFreeHost(out_host);
        if (out_ready) cudaEventDestroy(out_ready);
        if (in_taken) cudaEventDestroy(in_taken);
        if (ple_stage) cudaFreeHost(ple_stage);
        if (ple_copied) cudaEventDestroy(ple_copied);
        if (driver.joinable()) {
            {
                std::lock_guard<std::mutex> g(dq_mu);
                driver_stop = true;
            }
            dq_cv.notify_all();
            driver.join();
            if (watchdog.joinable()) watchdog.join();
        }
        if (sect_on)   // where the two threads that matter ran: the caller launches, the driver plans and copies
            std::fprintf(stderr, "threads: caller %ld on cpu %d, driver %ld on cpu %d; aux %s\n",
                         (long) cpu::this_tid(), caller_cpu, (long) driver_tid.load(), drv_cpu.load(),
                         aux.on ? aux.text.c_str() : "off");
        for (Bell & b : bells)
            for (void * p : { (void *) b.ids, (void *) b.pred, (void *) b.mids, (void *) b.w, (void *) b.x, (void *) b.y,
                              (void *) b.doorbell, (void *) b.cpu_done, (void *) b.plan, (void *) b.need_copy })
                if (p) cudaFreeHost(p);
        for (void * p : owned) cudaFree(p);
        if (ids_host) cudaFreeHost(ids_host);
        if (draft_host) cudaFreeHost(draft_host);
        if (commit_host) cudaFreeHost(commit_host);
        if (draft_prob_host) cudaFreeHost(draft_prob_host);
        if (embd_q) cudaFreeHost(embd_q);
        if (cpu_tier)
            for (void * p : { (void *) cpu_tier->x, (void *) cpu_tier->w, (void *) cpu_tier->y, (void *) cpu_tier->ids })
                if (p) cudaFreeHost(p);
        if (embd_d) cudaFreeHost(embd_d);
        if (calib_host) cudaFreeHost(calib_host);
        if (ck.host) cudaFreeHost(ck.host);
        if (mtp_calib_f) std::fclose(mtp_calib_f);
        for (auto & a : sect_ev)
            for (cudaEvent_t e : a)
                if (e) cudaEventDestroy(e);
        if (sect_csv) std::fclose(sect_csv);
        if (blas) cublasDestroy(blas);
        if (s) cudaStreamDestroy(s);
    }

    static size_t align(size_t x) { return (x + 255) / 256 * 256; }

    // TRUSS_AUX_CPUS: the caller (this thread, which launches the passes) on its own CPU, every thread but the workers,
    // the driver and the caller on the aux set (Strata's --aux-cpus). At start, at each prompt chunk, when another
    // thread starts calling and every 512 passes: the server's threads and CUDA's come and go.
    void place_threads()
    {
        const pid_t me = cpu::this_tid();
        if (me != caller_tid && aux.caller_cpu >= 0) cpu::pin_self(aux.caller_cpu);
        caller_tid = me;
        std::vector<pid_t> keep = cpu_tier->pool->worker_tids();
        keep.push_back(driver_tid.load());
        if (aux.caller_cpu >= 0) keep.push_back(me);
        const int n = cpu::sweep_aux(aux, keep);
        if (!aux_placed) std::fprintf(stderr, "TRUSS_AUX_CPUS: %d threads moved to the aux set\n", n);
        aux_placed = true;
    }

    void setup_doorbells()
    {
        const int K = c.n_expert_used, R = cpu::MAX_ROWS;
        bells.resize(c.n_layer + 1);   // by model layer (MTP: n_layer); only this stage's are set up
        auto mapped = [](void ** p, size_t bytes) {
            TRUSS_CUDA(cudaHostAlloc(p, bytes, cudaHostAllocMapped));
            std::memset(*p, 0, bytes);
        };
        for (int l = 0; l <= c.n_layer; ++l) {
            if ((l < lb || l >= le) && !(l == c.n_layer && mtp)) continue;
            Bell & b = bells[l];
            mapped((void **) &b.ids, sizeof(int) * R * K);
            mapped((void **) &b.pred, sizeof(int) * R * K);
            mapped((void **) &b.mids, sizeof(int) * R * K);
            mapped((void **) &b.w, sizeof(float) * R * K);
            mapped((void **) &b.x, sizeof(float) * R * c.d_model);
            mapped((void **) &b.y, sizeof(float) * R * c.d_model);
            mapped((void **) &b.doorbell, sizeof(int));
            mapped((void **) &b.cpu_done, sizeof(int));
            mapped((void **) &b.plan, sizeof(int));
            mapped((void **) &b.need_copy, sizeof(int));
            b.go = alloc<int>(1);
            b.arrive = alloc<unsigned>(1);   // TRUSS_ROUTE_PUBLISH; the kernel zeroes it again after every launch
            TRUSS_CUDA(cudaMemset(b.arrive, 0, sizeof(unsigned)));
            b.d_mids = alloc<int>((size_t) R * K);
        }
        drv_ts = std::getenv("TRUSS_DRIVER_TS") && std::atoi(std::getenv("TRUSS_DRIVER_TS"));
        driver = std::thread([this] { drive(); });
        dbg_on = std::getenv("TRUSS_DRIVER_DEBUG") && std::atoi(std::getenv("TRUSS_DRIVER_DEBUG"));
        if (dbg_on)
            watchdog = std::thread([this] {
                long last = -1;
                int still = 0;
                while (!driver_stop) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    const long s = dbg_served.load();
                    still = s == last && jobs_pushed.load() > s ? still + 1 : 0;
                    last = s;
                    if (still == 10) {
                        const int l = dbg_layer.load();
                        std::fprintf(stderr, "driver watchdog: served %ld, pushed %ld, phase %d, layer %d seq %d", s,
                                     jobs_pushed.load(), dbg_phase.load(), l, dbg_seq.load());
                        if (l >= 0) {
                            const Bell & b = bells[l];
                            std::fprintf(stderr, " | bell seq %d doorbell %d plan %d need_copy %d cpu_done %d",
                                         b.seq, *(volatile int *) b.doorbell, *(volatile int *) b.plan,
                                         *(volatile int *) b.need_copy, *(volatile int *) b.cpu_done);
                        }
                        std::fprintf(stderr, " | %s\n", experts->admit_debug().c_str());
                    }
                }
            });
    }

    // The driver: for each queued decode FFN, wait for its doorbell, split the routed experts (CPU tier / GPU),
    // start the CPU work, issue the expert copies and the go word, then publish the CPU result.
    void drive()
    {
        if (device >= 0) cudaSetDevice(device);   // its copies and events are this stage's device's
        if (aux.on) cpu::pin_self(aux.driver_cpu);
        driver_tid = cpu::this_tid();
        long taken = 0;
        for (;;) {
            Job j;
            // spin first: the next layer's job arrives within ~0.5 ms during a pass, and a condvar wake costs more
            dbg_phase = 1;
            for (int i = 0; i < 200000 && jobs_pushed.load(std::memory_order_acquire) <= taken; ++i) {
                _mm_pause();
                if (admit_idle > 0 && i % 64 == 0) dbg_phase = 8, experts->admit_step(), dbg_phase = 1;   // the link is ours while we wait
                if (sect_on && i == 4096 && (dbg_served & 127) == 0) drv_cpu.store(last_cpu(driver_tid.load()));
            }
            {
                std::unique_lock<std::mutex> g(dq_mu);
                dbg_phase = 2;
                dq_cv.wait(g, [&] { return driver_stop || !jobs.empty(); });
                ++taken;
                if (driver_stop && jobs.empty()) return;
                j = jobs.front();
                jobs.pop_front();
            }
            Bell & b = bells[j.layer];
            dbg_layer = j.layer, dbg_seq = j.seq, dbg_phase = 3;
            for (int i = 0; *(volatile int *) b.doorbell < j.seq; ++i) {
                _mm_pause();
                if (admit_idle > 0 && i % 64 == 0) dbg_phase = 8, experts->admit_step(), dbg_phase = 3;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            drv_t[0] = std::chrono::steady_clock::now();
            if (drv_ts && ts_prev.time_since_epoch().count())
                ts_ms[0] += std::chrono::duration<double, std::milli>(drv_t[0] - ts_prev).count();
            try {
                dbg_phase = 4;
                serve(j, b);
                ++dbg_served;
            } catch (...) {   // keep the GPU moving (its output is garbage now); the caller rethrows at its next sync
                try {
                    throw;
                } catch (const std::exception & e) {
                    std::fprintf(stderr, "driver: layer %d seq %d T %d threw: %s\n", j.layer, j.seq, j.T, e.what());
                } catch (...) {
                    std::fprintf(stderr, "driver: layer %d seq %d T %d threw a non-std exception\n", j.layer, j.seq, j.T);
                }
                if (!driver_error) driver_error = std::current_exception();
                experts->signal(sl(j.layer), b.go, j.seq);
                *(volatile int *) b.need_copy = j.seq;
                std::atomic_thread_fence(std::memory_order_release);
                *(volatile int *) b.plan = j.seq;
                std::atomic_thread_fence(std::memory_order_release);
                *(volatile int *) b.cpu_done = j.seq;
            }
            jobs_done.fetch_add(1, std::memory_order_release);
        }
    }

    void serve(const Job & j, Bell & b)
    {
        const int K = c.n_expert_used, l = j.layer, n = j.T * K;
        CpuTier * ct = j.cpu ? cpu_tier.get() : nullptr;
        if (!route_usage.empty())
            for (int i = 0; i < n; ++i) route_usage[(size_t) sl(l) * c.n_expert + b.ids[i]] += 1.f;
        if (admit_idle > 0 && l < c.n_layer) {
            if (l == 0) {   // a new pass
                for (float & u : admit_u) u *= admit_decay;
                admit_pass = 0;
            }
            for (int i = 0; i < n; ++i) admit_u[(size_t) l * c.n_expert + b.ids[i]] += 1.f;
        }
        if (route_trace) {
            const int32_t h[2] = { l, j.T };
            std::fwrite(h, 4, 2, route_trace);
            std::fwrite(b.ids, 4, (size_t) n, route_trace);
        }
        std::vector<cpu::Slot> slots;
        std::vector<int> gpu_ids;
        gpu_ids.reserve(n);
        // which routed experts the CPU computes: static (the CPU set always), or dynamic (Strata's split)
        uint8_t to_cpu[8 * 16] = {};   // per entry
        int n_cpu = 0;
        if (ct && cpu_dynamic) {
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
                if (!experts->on_device(sl(l), e)) {
                    if (ct->is_cpu[l][e]) elig.push_back(nd);
                    else pcie_bytes += (double) experts->bytes_of(sl(l), e);
                }
                ++nd;
            }
            // m = how many eligible misses (the last ones) go over PCIe: minimize max(CPU time, copy time)
            const int ne = (int) elig.size();
            int best_m = 0;
            double best = 1e30, tail = 0;
            for (int m = 0; m <= ne && pcie_frac < 0; ++m) {
                if (m > 0) tail += (double) experts->bytes_of(sl(l), b.ids[dist[elig[ne - m]]]);
                const double tc = ne - m > 0 ? cpu_ms_call + cpu_ms_expert * (ne - m) : 0;
                const double t = std::max(tc, pcie_ms_byte * (pcie_bytes + tail));
                if (t < best - 1e-9) best = t, best_m = m;
            }
            if (pcie_frac >= 0) best_m = std::min(ne, (int) std::lround(pcie_frac * ne));
            for (int q = 0; q < ne - best_m; ++q) {
                const int e = b.ids[dist[elig[q]]];
                for (int i = 0; i < n; ++i) to_cpu[i] |= b.ids[i] == e;
            }
            n_cpu = ne - best_m;
            dyn_cpu += n_cpu;
            dyn_pcie += best_m;
        } else if (ct) {
            for (int i = 0; i < n; ++i) to_cpu[i] = ct->is_cpu[l][b.ids[i]];
        }
        for (int i = 0; i < n; ++i) {
            const int e = b.ids[i];
            b.mids[i] = to_cpu[i] ? -1 : e;
            if (to_cpu[i]) slots.push_back(ct->slot(i / K, l, e, b.w[i]));
            else gpu_ids.push_back(e);
        }
        drv_t[1] = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> pool_lock;   // a split's stages share the pool (pipelined windows overlap)
        if (ct && ct->pool_mu) pool_lock = std::unique_lock<std::mutex>(*ct->pool_mu);
        const long long item0 = sect_on && ct ? ct->pool->item_us() : 0;   // cumulative; the call's own is the diff
        if (ct) ct->pool->start(b.x, j.T, slots, b.y);   // CPU first: it runs while the copies are issued
        drv_t[2] = std::chrono::steady_clock::now();
        // the plan (masked ids in mapped b.mids) goes to the GPU through mapped words, as Strata's: a layer that needs
        // no copies is not ordered behind the copy stream (earlier layers' hints and meta uploads); one that copies
        // also waits for go, which the copy stream writes behind its copies
        dbg_phase = 5;
        const bool copied = experts->fetch(sl(l), gpu_ids.data(), (int) gpu_ids.size(), nullptr);
        if (copied) experts->signal(sl(l), b.go, j.seq);   // go before the hints: this layer must not wait for the next one's copies
        *(volatile int *) b.need_copy = copied ? j.seq : 0;
        std::atomic_thread_fence(std::memory_order_release);
        *(volatile int *) b.plan = j.seq;
        {   // driver timing (Forward::driver_ms): split, CPU start, copies + plan
            const auto t3 = std::chrono::steady_clock::now();
            auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            drv_ms[0] += ms(drv_t[0], drv_t[1]), drv_ms[1] += ms(drv_t[1], drv_t[2]), drv_ms[2] += ms(drv_t[2], t3);
            ++drv_n;
            if (drv_ts)
            {
                ts_ms[1] += ms(drv_t[0], drv_t[1]), ts_ms[2] += ms(drv_t[1], drv_t[2]), ts_ms[3] += ms(drv_t[2], t3);
                ts_t3 = t3;
            }
        }
        if (sect_on)   // this layer's host chain, us after the driver saw the doorbell (Bell::h_t -> the CSV)
        {
            auto us = [](auto a, auto t) { return std::chrono::duration<double, std::micro>(t - a).count(); };
            b.h_t[0] = us(drv_t[0], drv_t[1]);
            b.h_t[1] = us(drv_t[0], drv_t[2]);
            b.h_t[2] = us(drv_t[0], std::chrono::steady_clock::now());
            b.h_ncpu = n_cpu;
        }
        if (j.hint) {
            std::vector<int> h;
            // static: the next layer's CPU set never goes to the GPU, so it is not hinted; dynamic: a hinted expert
            // that arrives in time is simply a GPU hit
            const CpuTier * nt = l + 1 < c.n_layer && !cpu_dynamic ? cpu_tier.get() : nullptr;
            for (int t = 0; t < j.T; ++t)
                for (int i = 0; i < std::min(K, hint_k); ++i) {
                    const int e = b.pred[t * K + i];
                    if (!nt || !nt->is_cpu[l + 1][e]) h.push_back(e);
                }
            experts->prefetch_hint(sl(l), h.data(), (int) h.size());
        }
        if (admit_idle > 0 && n_cpu > 0 && l < c.n_layer && admit_pass < admit_idle) {
            // claim slots for the CPU's experts most worth keeping (after fetch(): its experts are protected)
            std::vector<std::pair<float, int>> cand;
            for (int i = 0; i < n; ++i)
                if (to_cpu[i]) {
                    const int e = b.ids[i];
                    const float u = admit_u[(size_t) l * c.n_expert + e];
                    bool dup = false;
                    for (const auto & x : cand) dup |= x.second == e;
                    if (!dup && u >= admit_min) cand.push_back({ u, e });
                }
            std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            dbg_phase = 6;
            for (size_t i = 0; i < cand.size() && i < 2 && admit_pass < admit_idle; ++i)
                admit_pass += experts->claim(sl(l), cand[i].second);
        }
        if (drv_ts || sect_on) ts_t3b = std::chrono::steady_clock::now();
        if (sect_on) b.h_t[3] = std::chrono::duration<double, std::micro>(ts_t3b - drv_t[0]).count();   // hint + admit
        if (ct) {
            dbg_phase = 7;
            ct->pool->wait();
            if (cpu_dynamic && n_cpu > 0)   // the pool's own time (start to last item), not this thread's
                fit_cpu_cost(n_cpu, ct->pool->last_call_ms());
            if (sect_on)   // the pool's own split, and the instant cpu_done is about to be raised (the GPU sees it
            {              // a round trip later: see the residuals sect.py prints)
                b.h_t[4] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - drv_t[0]).count();
                b.h_pool = 1000.0 * ct->pool->last_call_ms();
                b.h_wake = (double) ct->pool->last_wake_us();
                b.h_item = (double) (ct->pool->item_us() - item0);   // item_us_ is cumulative over the run
                b.h_longest = (double) ct->pool->last_longest_us();
            }
            std::atomic_thread_fence(std::memory_order_release);
            *(volatile int *) b.cpu_done = j.seq;
            if (drv_ts)
            {
                ts_t4 = std::chrono::steady_clock::now();
                ts_pool += ct->pool->last_call_ms();
            }
        }
        if (drv_ts)
        {
            const auto dms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            if (!ct) ts_t4 = ts_t3b;
            const auto t5 = std::chrono::steady_clock::now();
            ts_ms[4] += dms(ts_t3, ts_t3b), ts_ms[5] += dms(ts_t3b, ts_t4), ts_ms[6] += dms(ts_t4, t5);
            ts_prev = t5;
            if (++ts_n % 480 == 0)   // ~10 decode passes
            {
                std::fprintf(stderr, "DRIVER_TS n=%ld pre %.3f split %.3f start %.3f copies %.3f hint_admit %.3f wait %.3f post %.3f pool_done %.3f ms/call\n",
                             ts_n, ts_ms[0] / 480, ts_ms[1] / 480, ts_ms[2] / 480,
                             ts_ms[3] / 480, ts_ms[4] / 480, ts_ms[5] / 480,
                             ts_ms[6] / 480, ts_pool / 480);
                for (double & v : ts_ms) v = 0;
                ts_pool = 0;
            }
        }
    }

    // add one call (n experts, t ms) to the weighted sums and refit cpu_ms_call + cpu_ms_expert * n; the slope
    // keeps its last value while the calls are too alike in n to fit one (and stays within sane bounds)
    void fit_cpu_cost(int n, double t)
    {
        const double d = 0.98;
        dyn_s1 = d * dyn_s1 + 1, dyn_sn = d * dyn_sn + n, dyn_st = d * dyn_st + t;
        dyn_snn = d * dyn_snn + (double) n * n, dyn_snt = d * dyn_snt + n * t;
        const double var = dyn_snn * dyn_s1 - dyn_sn * dyn_sn;
        if (dyn_s1 > 8 && var > 0.25 * dyn_s1 * dyn_s1)
            cpu_ms_expert = std::min(2.0, std::max(0.02, (dyn_snt * dyn_s1 - dyn_sn * dyn_st) / var));
        cpu_ms_call = std::max(0.0, (dyn_st - cpu_ms_expert * dyn_sn) / dyn_s1);
    }

    // Wait until the driver has finished every queued decode FFN. begin_stream() switches the store's mode on the host
    // at once, while a decode job queued just before (run_chunk's flush_commit: the deferred MTP commit) may not have
    // been served yet: its fetch() then threw "outside ring mode", and the failure path's go word queued on the copy
    // stream behind a wait for the very compute that spins on it (the served hangs of 2026-10-12). The GPU reaches the
    // doorbells on its own, so this wait ends.
    void drain_jobs()
    {
        while (jobs_done.load(std::memory_order_acquire) < jobs_pushed.load(std::memory_order_acquire)) _mm_pause();
    }

    void check_driver()
    {
        if (driver_error) {
            std::exception_ptr e = driver_error;
            driver_error = nullptr;
            std::rethrow_exception(e);
        }
    }

    // Options::cpu_trellis: per layer, the cold experts in ascending usage until they hold cpu_share of the layer's
    // cold routing mass form the static CPU set (cpu_dynamic: all cold experts are eligible); views point into the
    // ExpertStore's pinned copies and the pack's suh/svh rows. Nothing is read or allocated per expert.
    void load_cpu_tier_trellis(const Options & o, const runtime::ExpertStore::HotSet & hot, const StageSpec & sg)
    {
        require(!o.expert_usage.empty() || o.cpu_dynamic, "the static CPU tier needs expert_usage (it picks the rarest)");
        auto t = std::make_unique<CpuTier>();
        t->trellis = true;
        const int E = c.n_expert;
        require(c.d_model == cpu::D_MODEL && c.d_ff_exp == cpu::D_FF, "CPU tier shape");
        t->is_cpu.assign(c.n_layer, std::vector<uint8_t>(E, 0));
        t->tview.resize(c.n_layer);
        size_t n_cpu = 0;
        for (int l = lb; l < le; ++l) {
            const Moe & mo = w.layers[l].moe;
            const formats::ExpertTable * tab[3] = { &mo.gate, &mo.up, &mo.down };
            t->tview[l].resize(E);
            std::vector<int> cold;
            double mass = 0;
            for (int e = 0; e < E; ++e)
                if (!hot[sl(l)][e]) {
                    cold.push_back(e);
                    if (!o.expert_usage.empty()) mass += o.expert_usage[(size_t) l * E + e];
                    cpu::TrellisMat m[3];
                    for (int p = 0; p < 3; ++p) {
                        const formats::ExpertTable & T = *tab[p];
                        require((T.k[e] >= 1 && T.k[e] <= 6) || T.k[e] == 25 || T.k[e] == 35, "CPU tier: trellis K out of 1..6, 25, 35");
                        m[p] = { reinterpret_cast<const uint32_t *>(experts->host_part(sl(l), e, p)), T.k[e], (int) T.in,
                                 (int) T.out,
                                 reinterpret_cast<const uint16_t *>(T.suh->data) + (size_t) e * T.in,
                                 reinterpret_cast<const uint16_t *>(T.svh->data) + (size_t) e * T.out };
                    }
                    t->tview[l][e] = { m[0], m[1], m[2] };
                }
            if (o.cpu_dynamic) {
                for (int e : cold) t->is_cpu[l][e] = 1;
                n_cpu += cold.size();
                continue;
            }
            std::stable_sort(cold.begin(), cold.end(), [&](int a, int b) {
                return o.expert_usage[(size_t) l * E + a] < o.expert_usage[(size_t) l * E + b];
            });
            double acc = 0;
            for (int e : cold) {
                const double u = o.expert_usage[(size_t) l * E + e];
                if (acc + u > o.cpu_share * mass) break;
                acc += u;
                t->is_cpu[l][e] = 1;
                ++n_cpu;
            }
        }
        std::fprintf(stderr, "CPU tier (trellis, pinned pack bytes): %zu experts eligible (%.1f per layer), %s\n", n_cpu,
                     (double) n_cpu / (le - lb), o.cpu_dynamic ? "dynamic split" : "static set");
        t->pool = sg.pool ? sg.pool : std::make_shared<cpu::ExpertPool>(o.cpu_threads);
        t->pool_mu = sg.pool_mu;
        const size_t rows = std::max({ FETCH_ROWS, cpu::MAX_ROWS, split_rows });   // E7: a split chunk's rows
        TRUSS_CUDA(cudaMallocHost(&t->x, sizeof(float) * rows * c.d_model));
        TRUSS_CUDA(cudaMallocHost(&t->w, sizeof(float) * rows * c.n_expert_used));
        TRUSS_CUDA(cudaMallocHost(&t->y, sizeof(float) * rows * c.d_model));
        TRUSS_CUDA(cudaMallocHost(&t->ids, sizeof(int) * rows * c.n_expert_used));
        cpu_tier = std::move(t);
    }

    // Per layer, the non-resident experts in ascending usage until they hold cpu_share of the layer's non-resident
    // routing mass; their q4s copies are read from cpu_dir/L<nn>.q4s. With cpu_dir/L<nn>.ids (int32 expert ids in
    // file order) the file holds only those experts (an NVMe subset of the full archive, flashnext_truss_cpu_subset.py)
    // and the CPU set is chosen among them.
    void load_cpu_tier(const Options & o, const runtime::ExpertStore::HotSet & hot, const StageSpec & sg)
    {
        require(!o.expert_usage.empty(), "the CPU tier needs expert_usage (it picks the rarest experts)");
        auto t = std::make_unique<CpuTier>();
        const int E = c.n_expert, K = c.n_expert_used;
        require(c.d_model == cpu::D_MODEL && c.d_ff_exp == cpu::D_FF, "CPU tier shape");
        t->is_cpu.assign(c.n_layer, std::vector<uint8_t>(E, 0));
        t->data.resize(c.n_layer);
        t->fscales.resize(c.n_layer);
        t->view.resize(c.n_layer);
        std::vector<std::vector<int>> pick_of(c.n_layer);
        std::vector<std::vector<int64_t>> slot_of(c.n_layer);
        for (int l = lb; l < le; ++l) {
            const std::string base = o.cpu_dir + "/L" + (l < 10 ? "0" : "") + std::to_string(l);
            std::vector<int64_t> & slot_of_l = slot_of[l];
            slot_of_l.assign(E, -1);   // expert -> position in the q4s file
            if (FILE * fi = std::fopen((base + ".ids").c_str(), "rb")) {
                int32_t e;
                for (int64_t k = 0; std::fread(&e, 4, 1, fi) == 1; ++k) {
                    require(e >= 0 && e < E, base + ".ids: expert id out of range");
                    slot_of_l[e] = k;
                }
                std::fclose(fi);
            } else {
                for (int e = 0; e < E; ++e) slot_of_l[e] = e;
            }
            std::vector<int> cold;
            double mass = 0;
            for (int e = 0; e < E; ++e)
                if (!hot[sl(l)][e]) {
                    mass += o.expert_usage[(size_t) l * E + e];
                    if (slot_of_l[e] >= 0) cold.push_back(e);
                }
            std::stable_sort(cold.begin(), cold.end(), [&](int a, int b) {
                return o.expert_usage[(size_t) l * E + a] < o.expert_usage[(size_t) l * E + b];
            });
            std::vector<int> pick;
            double acc = 0;
            for (int e : cold) {
                const double u = o.expert_usage[(size_t) l * E + e];
                if (acc + u > o.cpu_share * mass) break;
                acc += u;
                pick.push_back(e);
            }
            std::sort(pick.begin(), pick.end());
            pick_of[l] = pick;
            t->data[l].resize(pick.size() * cpu::EXPERT_BYTES);
            t->fscales[l].resize(pick.size() * cpu::SCALES_PER_EXPERT);
            t->view[l].resize(E);
        }
        // Read the experts in .q4s slot order, not pick order. pick is sorted by expert id while the file is
        // ordered rarest-first, so the slots scatter and each fseek+fread is an independent random 2.7 MB read:
        // one at a time that is latency-bound, ~15000 of them cost ~18 min on a 2.8 GB/s NVMe. In slot order the
        // reads run forward, and the layers are independent, so they go in parallel too (TRACKER #71).
        {
            const int nth = std::max(1, std::min((int) c.n_layer, (int) std::thread::hardware_concurrency()));
            std::vector<std::thread> workers;
            std::mutex mu;
            std::string err;
            std::atomic<size_t> bytes{0};
            for (int w = 0; w < nth; ++w)
                workers.emplace_back([&, w] {
                    for (int l = lb + w; l < le; l += nth) {
                        const std::vector<int> & pick = pick_of[l];
                        const std::vector<int64_t> & slot_of_l = slot_of[l];
                        const std::string path =
                            o.cpu_dir + "/L" + (l < 10 ? "0" : "") + std::to_string(l) + ".q4s";
                        FILE * f = std::fopen(path.c_str(), "rb");
                        if (!f) {
                            std::lock_guard<std::mutex> g(mu);
                            err = "cannot open " + path;
                            return;
                        }
                        std::vector<int> order(pick.size());
                        std::iota(order.begin(), order.end(), 0);
                        std::sort(order.begin(), order.end(),
                                  [&](int a, int b) { return slot_of_l[pick[a]] < slot_of_l[pick[b]]; });
                        for (int i : order) {
                            uint8_t * dst = t->data[l].data() + (size_t) i * cpu::EXPERT_BYTES;
                            const bool ok =
                                std::fseek(f, (long) ((size_t) slot_of_l[pick[i]] * cpu::EXPERT_BYTES), SEEK_SET) == 0 &&
                                std::fread(dst, 1, cpu::EXPERT_BYTES, f) == cpu::EXPERT_BYTES;
                            if (!ok) {
                                std::lock_guard<std::mutex> g(mu);
                                err = "short read of expert " + std::to_string(pick[i]) + " from " + path;
                                std::fclose(f);
                                return;
                            }
                            t->view[l][pick[i]] = cpu::expert_view(dst);
                            cpu::preconvert_scales(t->view[l][pick[i]],
                                                    t->fscales[l].data() + (size_t) i * cpu::SCALES_PER_EXPERT);
                            t->is_cpu[l][pick[i]] = 1;
                        }
                        std::fclose(f);
                        bytes += t->data[l].size();
                    }
                });
            for (auto & th : workers) th.join();
            require(err.empty(), err);
            t->bytes = bytes;
            size_t n_cpu = 0;
            for (const auto & pk : pick_of) n_cpu += pk.size();
            std::fprintf(stderr, "CPU tier: %zu experts (%.1f per layer), %.2f GB of q4s in RAM\n", n_cpu,
                         (double) n_cpu / (le - lb), t->bytes / 1e9);
        }
        t->pool = sg.pool ? sg.pool : std::make_shared<cpu::ExpertPool>(o.cpu_threads);
        t->pool_mu = sg.pool_mu;
        const size_t rows = std::max({ FETCH_ROWS, cpu::MAX_ROWS, split_rows });   // E7: a split chunk's rows
        TRUSS_CUDA(cudaMallocHost(&t->x, sizeof(float) * rows * c.d_model));
        TRUSS_CUDA(cudaMallocHost(&t->w, sizeof(float) * rows * K));
        TRUSS_CUDA(cudaMallocHost(&t->y, sizeof(float) * rows * c.d_model));
        TRUSS_CUDA(cudaMallocHost(&t->ids, sizeof(int) * rows * K));
        cpu_tier = std::move(t);
    }

    // bytes of the prompt path's buffers (scratch, moe::prefill workspace, residual) at prefill_rows rows
    size_t big_bytes() const
    {
        // the same scratch size the carve uses (with the MTP join when the block is present): sized without it, the
        // carve ran ~T x 83 KB past the stream area, over the ring's tail and the hot experts after it (lead 10-07)
        return align(scratch_bytes(c, prefill_rows, n_ctx, mtp != nullptr)) + align(moe::prefill_workspace_bytes<MoeShape>(prefill_rows)) +
               (mtp ? 3 : 1) * align(sizeof(float) * prefill_rows * c.hc_dim());
    }

    void copy(float * dst, const float * src, size_t n)
    {
        TRUSS_CUDA(cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, s));
    }

    void use(const Buffers & b)
    {
        cur = b;
        sc = b.scratch;
        res = b.res;
    }

    template <class X> X * alloc(size_t n)
    {
        X * p = dmalloc<X>(n);
        owned.push_back(p);
        return p;
    }

    // peak per-layer temporaries: the layer's own buffers plus the largest of hc mix / PLE / GDN / DSA / FFN
    static size_t scratch_bytes(const Config & c, int T, int n_ctx, bool mtp_run = false)
    {
        const size_t hcd = c.hc_dim(), dm = c.d_model, qd = (size_t) c.n_head * c.head_dim;
        const size_t base = dm * 4 + 4 + dm * (4 + 4 + 2) + c.hc * 4;   // embedding + ids, then the layer's own
        const size_t mix = hcd * (2 + 4) + c.hc * 4 + c.hc_rank * (4 + 2);
        const size_t ple = (size_t) c.ple_heads() * c.ple_head_dim * 2 + hcd * 8 + dm * 4 + c.hc * 4;
        const size_t gdn = (size_t) c.conv_dim() * 4 + (size_t) c.value_dim() * (4 + 4 + 4 + 2) + (size_t) c.key_dim() * 8 +
                           (size_t) c.ssm_v_heads * 16;
        const size_t dsa = qd * (8 + 2 + 4 + 2) + (size_t) c.n_head_kv * c.head_dim * 8 +
                           (size_t) c.idx_heads * c.idx_head_dim * 6 + c.idx_head_dim * 4 + DsaShape::ROPE_DIMS * 4 +
                           DsaShape::TOP_BLOCKS * 4 + 4;
        const size_t ffn = (size_t) c.n_expert * 4 + c.n_expert_used * 12 + dm * 12 + (size_t) c.d_ff_shexp * 10 + 4;   // E7: + masked ids, CPU rows
        // MTP join: hn16, rstd, [e | hn] fp16 and its Q8_1 form (the widest GEMM input when present)
        const size_t mtp = hcd * 2 + c.hc * 4 + (size_t) c.hc * 2 * dm * (2 + 1) + (size_t) c.hc * 2 * dm / 16;
        const size_t q8_act = 2 * (hcd + hcd / 16);   // Q8_1: a shared quantized input (Act) + one per-call input
        // mtp_rows holds its join buffers while it runs the layer's mixer and FFN, so with the MTP block the join
        // and the layer's own peak are live together: add them. Taking the max of the two only fit because the
        // prompt path's cap used to be max_chunk rows, well above the chunk actually run (TRACKER #70).
        const size_t per_token = base + q8_act + std::max({ mix, ple, gdn, dsa, ffn }) + (mtp_run ? mtp : 0);
        const size_t fixed = dsa::select_workspace_bytes<DsaShape>(T, n_ctx) + dsa::attention_workspace_bytes<DsaShape>(T) +
                             (64ull << 20);   // + alignment slack
        return (size_t) T * per_token + fixed;
    }

    void check_shapes() const
    {
        require(c.d_model == MoeShape::D_MODEL && c.d_ff_exp == MoeShape::D_FF && c.n_expert_used == MoeShape::TOPK,
                "routed-expert shape differs from moe::FlashNext");
        require(c.n_head == DsaShape::H && c.n_head_kv == DsaShape::HKV && c.head_dim == DsaShape::D &&
                    c.idx_heads == DsaShape::IH && c.idx_head_dim == DsaShape::ID && c.rope_dims == DsaShape::ROPE_DIMS &&
                    c.idx_top_k == DsaShape::TOP_BLOCKS * DsaShape::RATIO,
                "DSA shape differs from dsa::FlashNext");
        for (int l = 0; l < c.n_layer; ++l)
            require(c.mixer[l] != Mixer::DSA || c.compress_ratio[l] == DsaShape::RATIO, "DSA ratio differs");
        require(max_chunk % DsaShape::RATIO == 0, "max_chunk must be a multiple of the DSA ratio");
    }

    // Q8_0 matrices are repacked (one staging buffer); everything else is uploaded as stored
    // A tensor's file pages are dead weight once it has been copied to the device. MADV_DONTNEED only drops them:
    // the mapping stays PROT_READ|MAP_SHARED, so a later touch re-faults from disk and nothing can break. Doing this
    // per tensor instead of once after upload() keeps the peak RSS down (upload used to fault ~48 GB at once, which
    // is more than this box has free beside the renter) (TRACKER #72).
    static void release_tensor(const gguf::Tensor * t)
    {
        long ps = sysconf(_SC_PAGESIZE);
        if (!t || !t->data || !t->bytes || ps <= 0) return;
        const uintptr_t a = (uintptr_t) t->data, m = (uintptr_t) ps - 1;
        const uintptr_t lo = a & ~m, hi = (a + (uintptr_t) t->bytes + m) & ~m;
        madvise((void *) lo, (size_t) (hi - lo), MADV_DONTNEED);
    }

    // The dense tensors' file pages, faulted in by 16 threads (MADV_POPULATE_READ, 64 MB pieces) before the uploads
    // read them: the pageable copies below otherwise fault every 4 KB page on one thread (~2.6 s for ~5 GB).
    static void prefault(const std::vector<const gguf::Tensor *> & a, const std::vector<const gguf::Tensor *> & b)
    {
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
        const long ps = sysconf(_SC_PAGESIZE);
        if (ps <= 0) return;
        constexpr size_t PIECE = 64u << 20;
        std::vector<std::pair<uintptr_t, size_t>> pieces;
        for (const auto * v : { &a, &b })
            for (const gguf::Tensor * t : *v) {
                if (!t || !t->data || !t->bytes) continue;
                const uintptr_t m = (uintptr_t) ps - 1, lo = (uintptr_t) t->data & ~m,
                                hi = ((uintptr_t) t->data + t->bytes + m) & ~m;
                for (uintptr_t x = lo; x < hi; x += PIECE) pieces.push_back({ x, std::min<size_t>(PIECE, hi - x) });
            }
        std::atomic<size_t> next{ 0 };
        std::vector<std::thread> th;
        for (int i = 0; i < 16; ++i)
            th.emplace_back([&] {
                for (size_t j; (j = next.fetch_add(1)) < pieces.size();)
                    madvise((void *) pieces[j].first, pieces[j].second, MADV_POPULATE_READ);   // best effort
            });
        for (auto & t : th) t.join();
    }

    void upload()
    {
        std::vector<const gguf::Tensor *> plain, q8_list;
        auto add = [&](T t) {
            if (!t) return;
            (t->type == gguf::Type::Q8_0 ? q8_list : plain).push_back(t);
        };
        auto add_hc = [&](const HyperConnection & h) { add(h.norm), add(h.down), add(h.up), add(h.inject); };
        // a split stage: its own layers; the token embedding where it is read (the first stage, the MTP block), the
        // head on the last stage
        if (first || mtp) add(w.token_embd);
        if (last) add(w.output), add_hc(w.hc_head);
        std::vector<const Layer *> layers;
        for (int l = lb; l < le; ++l) layers.push_back(&w.layers[l]);
        if (mtp) {
            layers.push_back(&mtp->layer);
            add(mtp->eh_proj), add(mtp->enorm), add(mtp->hnorm);
        }
        for (const Layer * Lp : layers) {
            const Layer & L = *Lp;
            add_hc(L.hc_attn), add_hc(L.hc_ffn);
            for (T t : { L.gdn.qkv, L.gdn.gate, L.gdn.conv1d, L.gdn.dt_bias, L.gdn.a, L.gdn.beta, L.gdn.alpha,
                         L.gdn.norm, L.gdn.out })
                add(t);
            for (T t : { L.dsa.q, L.dsa.k, L.dsa.v, L.dsa.out, L.dsa.q_norm, L.dsa.k_norm, L.dsa.idx_q, L.dsa.idx_k,
                         L.dsa.idx_q_norm, L.dsa.idx_k_norm })
                add(t);
            for (T t : { L.ple.key, L.ple.value, L.ple.norm_key, L.ple.norm_query, L.ple.norm_conv, L.ple.conv1d })
                add(t);
            add(L.moe.router), add(L.moe.shexp_gate_inp), add(L.moe.shexp_gate), add(L.moe.shexp_up),
                add(L.moe.shexp_down);
        }
        prefault(plain, q8_list);
        load_mark("dense weights: file pages faulted in");
        dev = std::make_unique<DeviceTensors>(plain);
        for (T t : plain) release_tensor(t);

        // The token embedding is read one row per token: it lives in pinned host memory and the gather reads it over
        // PCIe (2.5 KB per token), leaving its 0.68 GB of VRAM to the expert cache (TRACKER #60). Tied: stays.
        const bool host_embd = w.token_embd != w.output && (first || mtp);
        size_t biggest = 0, total_q = 0, total_d = 0;
        for (T t : q8_list) {
            biggest = std::max<size_t>(biggest, t->bytes);
            if (host_embd && t == w.token_embd) continue;
            total_q += (size_t) t->elements();
            total_d += (size_t) t->elements() / 32;
        }
        int8_t * qs = alloc<int8_t>(total_q);
        half * ds = alloc<half>(total_d);
        void * stage = dmalloc<unsigned char>(biggest, false);
        size_t max_w = 0;
        for (T t : q8_list) {
            if (q8.count(t) || (host_embd && t == w.token_embd)) continue;
            const int in = (int) t->shape[0], out = (int) t->elements() / in;
            TRUSS_CUDA(cudaMemcpy(stage, t->data, t->bytes, cudaMemcpyHostToDevice));
            release_tensor(t);
            dense::q8_repack(stage, in, out, qs, ds, s);
            q8[t] = { qs, ds, in, out };
            qs += (size_t) in * out;
            ds += (size_t) in * out / 32;
            if (t != w.token_embd && t != w.output) max_w = std::max(max_w, (size_t) in * out);
        }
        if (host_embd) {   // repack on the device through temporaries, keep the result in pinned host memory
            const T t = w.token_embd;
            const int in = (int) t->shape[0], out = (int) t->elements() / in;
            const size_t nq = (size_t) in * out, nd = nq / 32;
            int8_t * tq = dmalloc<int8_t>(nq, false);
            half * td = dmalloc<half>(nd, false);
            TRUSS_CUDA(cudaMemcpy(stage, t->data, t->bytes, cudaMemcpyHostToDevice));
            release_tensor(t);
            dense::q8_repack(stage, in, out, tq, td, s);
            TRUSS_CUDA(cudaStreamSynchronize(s));
            TRUSS_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&embd_q), nq, cudaHostAllocMapped));
            TRUSS_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&embd_d), nd * sizeof(half), cudaHostAllocMapped));
            TRUSS_CUDA(cudaMemcpy(embd_q, tq, nq, cudaMemcpyDeviceToHost));
            TRUSS_CUDA(cudaMemcpy(embd_d, td, nd * sizeof(half), cudaMemcpyDeviceToHost));
            cudaFree(tq);
            cudaFree(td);
            q8[t] = { embd_q, embd_d, in, out };   // unified addressing: the kernels read the host pointers directly
        }
        TRUSS_CUDA(cudaStreamSynchronize(s));
        cudaFree(stage);
        w16_elems = max_w;
        w16 = alloc<half>(max_w);
    }

    const DTensor & d(T t) const { return (*dev)(t); }
    const float * f32(T t) const { return d(t).as<float>(); }

    // y [rows][out] = W x, fp16 activations
    void lin(T t, const half * x, int rows, float * y)
    {
        const size_t m = sc->mark();
        lin(t, quant(x, rows, q8.at(t).in), rows, y);
        sc->release(m);
    }

    // An fp16 GEMM input and, with Q8_1 activations, its quantized form: quantized once for every projection that
    // reads it (GDN qkv / gate / alpha / beta, DSA q / k / v / indexer, hc down + inject, shared gate + up), in the
    // caller's scratch scope (TRACKER #60)
    struct Act {
        const half * x;
        int8_t * q = nullptr;
        half * d = nullptr;
    };
    Act quant(const half * x, int rows, int in)
    {
        Act a{ x };
        if (act == Activations::Q8_1) {
            a.q = sc->alloc<int8_t>((size_t) rows * in);
            a.d = sc->alloc<half>((size_t) rows * in / 32);
            dense::q8_quantize_act(x, rows, in, a.q, a.d, s);
        }
        return a;
    }
    void lin(T t, const Act & a, int rows, float * y)
    {
        const dense::Q8Matrix & W = q8.at(t);
        if (act == Activations::FP16 && rows <= dense::GEMV_ROWS) dense::q8_gemv_a16(W, a.x, rows, y, s);
        else if (act == Activations::FP16) dense::q8_gemm_a16(W, a.x, rows, y, w16, blas, s);
        else if (rows <= dense::GEMV_ROWS) dense::q8_gemv(W, a.q, a.d, rows, y, s);
        else dense::q8_gemm(W, a.q, a.d, rows, y, s);
    }

    // y [rows][out] = W x, fp32 weights and activations (router, shared-expert gate)
    void lin32(T t, const float * x, int rows, float * y)
    {
        const DTensor & W = d(t);
        const int in = (int) W.ne[0], out = (int) W.ne[1];
        if (rows <= FETCH_ROWS) {   // decode sizes: row-invariant (routing must not depend on the window size)
            dense::f32_gemv(W.as<float>(), in, out, x, rows, y, s);
            return;
        }
        const float one = 1.f, zero = 0.f;
        TRUSS_CUBLAS(cublasSetStream(blas, s));
        TRUSS_CUBLAS(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, &one, W.as<float>(), in, x, in, &zero,
                                 y, out));
    }

    // Several projections of the same activations. Decode rows: one launch (dense::q8_gemv_multi / f32_gemv_multi,
    // TRACKER #83: D0 counted 2,672 launches per pass, the small ones far below bandwidth); each matrix's result is
    // bit-identical to its own lin()/lin32() at those rows. Larger row counts: one lin()/lin32() each.
    using Proj = std::pair<T, float *>;
    void lin_multi(const std::vector<Proj> & ps, const Act & a, int rows)
    {
        if (rows <= dense::GEMV_ROWS && ps.size() > 1 && ps.size() <= (size_t) dense::MULTI_MAX) {
            const dense::Q8Matrix * W[dense::MULTI_MAX];
            float * y[dense::MULTI_MAX];
            for (size_t i = 0; i < ps.size(); ++i) W[i] = &q8.at(ps[i].first), y[i] = ps[i].second;
            if (act == Activations::FP16) dense::q8_gemv_multi_a16(W, y, (int) ps.size(), a.x, rows, s);
            else dense::q8_gemv_multi(W, y, (int) ps.size(), a.q, a.d, rows, s);
            return;
        }
        for (const Proj & p : ps) lin(p.first, a, rows, p.second);
    }
    // lin_multi with the first matrix's output written as silu(row * scale) in fp16 by the gemv itself: the same
    // launch, the same arithmetic, and one kernel fewer after it (TRUSS_HC_SILU).
    void lin_multi_silu(const std::vector<Proj> & ps, half * y16, float scale, const Act & a, int rows)
    {
        const int n = (int) ps.size();
        const dense::Q8Matrix * W[dense::MULTI_MAX];
        float * y[dense::MULTI_MAX];
        half * o16[dense::MULTI_MAX];
        float s16[dense::MULTI_MAX];
        for (int i = 0; i < n; ++i) W[i] = &q8.at(ps[i].first), y[i] = ps[i].second, o16[i] = nullptr, s16[i] = 0.f;
        o16[0] = y16, s16[0] = scale;
        dense::q8_gemv_multi_a16_silu(W, y, o16, s16, n, a.x, rows, s);
    }

    void lin32_multi(const std::vector<Proj> & ps, const float * x, int rows)
    {
        if (rows <= FETCH_ROWS && ps.size() > 1 && ps.size() <= (size_t) dense::MULTI_MAX) {
            const float * W[dense::MULTI_MAX];
            float * y[dense::MULTI_MAX];
            int out[dense::MULTI_MAX];
            const int in = (int) d(ps[0].first).ne[0];
            for (size_t i = 0; i < ps.size(); ++i) {
                const DTensor & t = d(ps[i].first);
                if ((int) t.ne[0] != in) throw std::runtime_error("lin32_multi: inputs differ");
                W[i] = t.as<float>(), out[i] = (int) t.ne[1], y[i] = ps[i].second;
            }
            dense::f32_gemv_multi(W, out, y, (int) ps.size(), in, x, rows, s);
            return;
        }
        for (const Proj & p : ps) lin32(p.first, x, rows, p.second);
    }

    // Phase 1's two hyper-connection fusions, each off until its gate: TRUSS_HC_SILU folds the down projection's
    // silu into the gemv that produces its input, TRUSS_HC_COLLAPSE folds the stream collapse into the up
    // projection's launch. Neither changes an operation; they remove launches and the round trips between them.
    static bool hc_silu()
    {
        static const bool v = [] { const char * e = std::getenv("TRUSS_HC_SILU"); return e && std::atoi(e); }();
        return v;
    }
    static bool hc_collapse()
    {
        static const bool v = [] { const char * e = std::getenv("TRUSS_HC_COLLAPSE"); return e && std::atoi(e); }();
        return v;
    }

    // hyper-connection mix: mixed [T][d] (fp32 and fp16), inject [T][hc] when h.inject.
    // L3 (Order 11): with Q8_1 activations the norm fuses the activation quantize (and, when cout/cinj are
    // given, the preceding hc::combine) into one launch; bit-identical arithmetic, see hc_prefill.cuh.
    void hc_mix(const HyperConnection & h, const float * res, int T, float * mixed, half * mixed16, float * inject,
                const float * cout = nullptr, const float * cinj = nullptr)
    {
        const size_t m = sc->mark();
        // Both fusions live in the fp16-activation gemv at decode rows; anything else (q8_1 activations, prefill
        const bool gemv_ok = act == Activations::FP16 && T <= dense::GEMV_ROWS && dense::mma_on();
        const bool silu_epi = gemv_ok && hc_silu();
        // The shapes come from the q8 table because that is what the fused launch reads. It matters: the head's
        // hyper-connection (w.hc_head, output_hc_*) has its up matrix there but not as an uploaded fp16 tensor.
        const dense::Q8Matrix & wup = q8.at(h.up);
        const bool collapse_epi =
            gemv_ok && hc_collapse() && dense::up_collapse_ok(wup.in, wup.out, c.hc, c.d_model);
        float * rstd = sc->alloc((size_t) T * c.hc);
        float * lo = silu_epi ? nullptr : sc->alloc((size_t) T * c.hc_rank);   // the epilogue makes it dead
        half * lo16 = sc->alloc<half>((size_t) T * c.hc_rank);
        float * gate = collapse_epi ? nullptr : sc->alloc((size_t) T * c.hc_dim());   // ditto the gate
        Act xa;
        if (act == Activations::Q8_1)
        {
            xa.q = sc->alloc<int8_t>((size_t) T * c.hc_dim());
            xa.d = sc->alloc<half>((size_t) T * c.hc_dim() / 32);
            if (cout) hc::combine_norm_q8(const_cast<float *>(res), cout, cinj, f32(h.norm), T, c.hc, c.d_model, c.rms_eps, rstd, xa.q, xa.d, s);
            else      hc::norm_q8(res, f32(h.norm), T, c.hc, c.d_model, c.rms_eps, rstd, xa.q, xa.d, s);
        }
        else
        {
            half * xn16 = sc->alloc<half>((size_t) T * c.hc_dim());
            if (cout) hc::combine_norm(const_cast<float *>(res), cout, cinj, f32(h.norm), T, c.hc, c.d_model, c.rms_eps, xn16, rstd, s);
            else hc::norm(res, f32(h.norm), T, c.hc, c.d_model, c.rms_eps, xn16, rstd, s);
            xa = quant(xn16, T, c.hc_dim());
        }
        const float inv_hc = 1.f / c.hc;
        if (silu_epi) {   // the down projection writes silu(lo) as fp16 itself; inject shares the launch
            std::vector<Proj> ps = { { h.down, nullptr } };
            if (h.inject) ps.push_back({ h.inject, inject });   // inject reads the same xa
            lin_multi_silu(ps, lo16, inv_hc, xa, T);
        } else {
            if (h.inject) lin_multi({ { h.down, lo }, { h.inject, inject } }, xa, T);   // inject reads the same xa
            else lin(h.down, xa, T, lo);
            hc::silu(lo, T * c.hc_rank, inv_hc, lo16, s);
        }
        if (collapse_epi)   // gate stays in the CTA that computed it; mixed and mixed16 come out of the gemv
            dense::q8_gemv_a16_up_collapse(wup, lo16, T, res, rstd, f32(h.norm), mixed, mixed16, c.hc, c.d_model, s);
        else {
            lin(h.up, lo16, T, gate);
            hc::collapse(res, rstd, f32(h.norm), gate, T, c.hc, c.d_model, mixed, mixed16, s);
        }
        sc->release(m);
    }

    void ple(const Ple & p, LayerState & L, const int32_t * tokens, int T, int pos0, bool tentative)
    {
        // n-gram window across chunks: hash [tail | chunk] and keep the chunk's rows
        const auto th0 = std::chrono::steady_clock::now();
        const int H = c.ple_heads(), E = H * c.ple_head_dim, n_prev = (int) tail.size();
        std::vector<int32_t> seq(tail);
        seq.insert(seq.end(), tokens, tokens + T);
        std::vector<int32_t> rows(seq.size() * H);
        ple_rows(c, seq.data(), (int) seq.size(), rows.data());
        std::vector<half> emb16;
        const half * host16;
        if (ple_reader) {   // O_DIRECT reads into the pinned stage (the previous copy out of it has been read)
            require((size_t) T * E <= ple_stage_n, "PLE stage too small for the chunk");
            int tk;
            if (ple_tk_valid && ple_tk_pos0 == pos0) { tk = ple_tk; ple_tk_valid = false; }   // already in flight
            else { ple_tk_valid = false; tk = ple_reader->issue(rows.data() + (size_t) n_prev * H, (size_t) T * H,
                                                                 T <= 1); }   // a chunk's rows are used once: only
                                                                                // decode tickets populate the cache
            TRUSS_CUDA(cudaEventSynchronize(ple_copied));
            ple_reader->collect(tk, ple_stage);
            host16 = ple_stage;
        } else {
            std::vector<float> emb((size_t) T * E);
            ple_gather(c, w, rows.data() + (size_t) n_prev * H, T, emb.data());
            emb16.resize(emb.size());
            for (size_t i = 0; i < emb.size(); ++i) emb16[i] = __float2half(emb[i]);
            host16 = emb16.data();
        }
        ple_host_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - th0).count();
        ++ple_calls, ple_rows_n += (long) T * H;

        const size_t m = sc->mark();
        half * e16 = sc->alloc<half>((size_t) T * E);
        float * key = sc->alloc((size_t) T * c.hc_dim()), * value = sc->alloc((size_t) T * c.d_model);
        float * gate = sc->alloc((size_t) T * c.hc), * normed = sc->alloc((size_t) T * c.hc_dim());
        TRUSS_CUDA(cudaMemcpyAsync(e16, host16, (size_t) T * E * 2, cudaMemcpyHostToDevice, s));
        if (ple_reader) TRUSS_CUDA(cudaEventRecord(ple_copied, s));
        if (ple_reader && ple_next && ple_next_T > 0) {   // read the next chunk's rows while this chunk computes
            const size_t keep = std::min<size_t>(seq.size(), (size_t) c.ple_ngram - 1);
            std::vector<int32_t> nseq(seq.end() - keep, seq.end());
            nseq.insert(nseq.end(), ple_next, ple_next + ple_next_T);
            std::vector<int32_t> nrows(nseq.size() * H);
            ple_rows(c, nseq.data(), (int) nseq.size(), nrows.data());
            ple_tk = ple_reader->issue(nrows.data() + keep * H, (size_t) ple_next_T * H, false);
            ple_tk_valid = true; ple_tk_pos0 = pos0 + T;
        }
        ple_next = nullptr; ple_next_T = 0;
        const Act ea = quant(e16, T, c.ple_heads() * c.ple_head_dim);
        lin_multi({ { p.key, key }, { p.value, value } }, ea, T);
        ple::gate(key, res, f32(p.norm_key), f32(p.norm_query), T, c.hc, c.d_model, c.rms_eps, gate, s);
        const size_t hist = (size_t) (c.ple_conv - 1) * c.ple_ngram * c.hc_dim();
        if (tentative) copy(L.hist_snap, L.ple_hist, hist);
        ple::apply(value, gate, f32(p.norm_conv), d(p.conv1d).as<half>(), L.ple_hist, T, c.hc, c.d_model, c.ple_conv,
                   c.ple_ngram, c.rms_eps, normed, res, s);
        if (tentative) copy(L.normed_rows, normed, (size_t) T * c.hc_dim());
        if (!ple_reader) TRUSS_CUDA(cudaStreamSynchronize(s));   // emb16 is a host temporary (the stage is not)
        sc->release(m);
    }

    void gdn(const Gdn & g, LayerState & L, const half * in16, int T, float * out, bool tentative)
    {
        const size_t m = sc->mark();
        const gdn::Dims dm{ c.ssm_groups, c.ssm_v_heads, c.ssm_state, c.ssm_conv };
        const int kd = c.key_dim(), vd = c.value_dim(), Hv = c.ssm_v_heads;
        float * qkv = sc->alloc((size_t) T * c.conv_dim()), * z = sc->alloc((size_t) T * vd);
        float * alpha = sc->alloc((size_t) T * Hv), * beta_raw = sc->alloc((size_t) T * Hv);
        float * q = sc->alloc((size_t) T * kd), * k = sc->alloc((size_t) T * kd), * v = sc->alloc((size_t) T * vd);
        float * gt = sc->alloc((size_t) T * Hv), * beta = sc->alloc((size_t) T * Hv);
        float * core = sc->alloc((size_t) T * vd);
        half * o16 = sc->alloc<half>((size_t) T * vd);
        const Act ia = quant(in16, T, c.d_model);
        lin_multi({ { g.qkv, qkv }, { g.gate, z }, { g.alpha, alpha }, { g.beta, beta_raw } }, ia, T);
        if (tentative) {   // accept() may redo the accepted rows from these
            copy(L.conv_snap, L.conv, (size_t) (c.ssm_conv - 1) * c.conv_dim());
            copy(L.raw_qkv, qkv, (size_t) T * c.conv_dim());
            copy(L.raw_alpha, alpha, (size_t) T * Hv);
            copy(L.raw_beta, beta_raw, (size_t) T * Hv);
        }
        gdn::prepare(dm, qkv, f32(g.conv1d), L.conv, alpha, beta_raw, f32(g.dt_bias), f32(g.a), T, c.rms_eps, q, k, v,
                     gt, beta, s);
        gdn::delta_rule(q, k, v, gt, beta, L.state(), tentative ? L.state_buf[1 - L.cur] : L.state(), core, T,
                        c.ssm_groups, Hv, s);
        gdn::output_norm(dm, core, z, f32(g.norm), T, c.rms_eps, o16, s);
        lin(g.out, o16, T, out);
        sc->release(m);
    }

    void dsa(const Dsa & a, LayerState & L, const half * in16, int pos0, int T, float * out, bool tentative)
    {
        const size_t m = sc->mark();
        const int H = c.n_head, D = c.head_dim, Hkv = c.n_head_kv, IH = c.idx_heads, ID = c.idx_head_dim;
        float * qfull = sc->alloc((size_t) T * H * 2 * D);
        float * k = sc->alloc((size_t) T * Hkv * D), * v = sc->alloc((size_t) T * Hkv * D);
        float * iq = sc->alloc((size_t) T * IH * ID), * ik = sc->alloc((size_t) T * ID);
        float2 * cs = sc->alloc<float2>((size_t) T * DsaShape::ROPE_DIMS / 2);
        half * q16 = sc->alloc<half>((size_t) T * H * D), * iq16 = sc->alloc<half>((size_t) T * IH * ID);
        float * gate = sc->alloc((size_t) T * H * D);
        int * blocks = sc->alloc<int>((size_t) T * DsaShape::TOP_BLOCKS), * n_blocks = sc->alloc<int>(T);
        half * att16 = sc->alloc<half>((size_t) T * H * D);
        const Act ia = quant(in16, T, c.d_model);
        lin_multi({ { a.q, qfull }, { a.k, k }, { a.v, v }, { a.idx_q, iq }, { a.idx_k, ik } }, ia, T);
        if (tentative) {
            copy(L.partial_snap, L.idx_partial, (size_t) (DsaShape::RATIO - 1) * c.idx_head_dim);
            copy(L.raw_ik, ik, (size_t) T * c.idx_head_dim);
        }
        dsa::rope_table<DsaShape>(pos0, T, c.rope_base, cs, s);
        dsa::prepare_qkv<DsaShape>(qfull, k, v, f32(a.q_norm), f32(a.k_norm), cs, pos0, T, c.rms_eps, q16, gate, L.kv,
                                   s);
        dsa::prepare_index<DsaShape>(iq, ik, f32(a.idx_q_norm), f32(a.idx_k_norm), cs, c.rope_base, pos0, T, c.rms_eps,
                                     L.idx_partial, iq16, L.idx_k, s);
        const size_t ws_bytes = dsa::select_workspace_bytes<DsaShape>(T, pos0 + T);
        void * ws = sc->alloc<unsigned char>(ws_bytes);
        dsa::select<DsaShape>(iq16, L.idx_k, pos0, T, blocks, n_blocks, ws, ws_bytes, s);
        const size_t aws_bytes = dsa::attention_workspace_bytes<DsaShape>(T);
        void * aws = aws_bytes ? sc->alloc<unsigned char>(aws_bytes) : nullptr;
        dsa::attention<DsaShape>(q16, gate, L.kv, blocks, n_blocks, pos0, T, att16, aws, aws_bytes, s);
        lin(a.out, att16, T, out);
        sc->release(m);
    }

    // Chunks of up to FETCH_ROWS rows (decode steps, verify windows, short prompts) fetch only the cold experts
    // their routing uses; longer chunks use most experts, so whole layers stream ahead of the compute instead.
    // A 67-token prompt took 2.1 s streaming ~24 GB (TRACKER #55). The window kernel serves up to moe::MAX_ROWS rows.
    static constexpr int FETCH_ROWS = 32;
    static bool fetch_mode(int T) { return T <= FETCH_ROWS; }

    // Phase 2's two doorbell changes, each off until its gate. TRUSS_ROUTE_PUBLISH folds the publish into the
    // routing: one launch instead of route + route(pred) + publish, and the FFN rows stream out of several CTAs
    // next to the top-k instead of out of one 1024-thread CTA after it. TRUSS_PUB_X_LAZY publishes the rows only
    // for a layer whose CPU tier reads them - serve() touches b.x behind `if (ct)`, so with no CPU tier the copy
    // is dead work on the link (measured: 18 us of every publish on device 0, 24 layers a pass).
    static bool route_publish_on()
    {
        static const bool v = [] { const char * e = std::getenv("TRUSS_ROUTE_PUBLISH"); return e && std::atoi(e); }();
        return v;
    }
    static bool pub_x_lazy()
    {
        static const bool v = [] { const char * e = std::getenv("TRUSS_PUB_X_LAZY"); return e && std::atoi(e); }();
        return v;
    }

    // stream: this chunk streams whole layers (prompt chunk, stream mode); else it fetches (decode, windows)
    void ffn(int l, const Moe & mo, const float * in, const half * in16, int T, float * out, bool stream)
    {
        const size_t m = sc->mark();
        // a split's stages share the CPU pool and may run prompt chunks at once (run_prompt): one call at a time,
        // held from start() to the join's wait() below
        std::unique_lock<std::mutex> pool_lock;
        auto lock_pool = [&](CpuTier * ct) {
            if (ct->pool_mu && !pool_lock.owns_lock()) pool_lock = std::unique_lock<std::mutex>(*ct->pool_mu);
        };
        // only a run_chunk() layer brackets all five sections; the MTP block's FFN (draft steps, commits) does not
        const bool sect = sect_on && l < c.n_layer;
        if (sect) sect_record(l, 3);   // router + routed MoE (its copies and the CPU tier's start are inside)
        const int E = c.n_expert, K = c.n_expert_used, F = c.d_ff_shexp, dm = c.d_model;
        float * logits = sc->alloc((size_t) T * E), * wts = sc->alloc((size_t) T * K);
        int * ids = sc->alloc<int>((size_t) T * K);
        float * routed = sc->alloc((size_t) T * dm), * y = sc->alloc((size_t) T * dm);
        float * g = sc->alloc((size_t) T * F), * u = sc->alloc((size_t) T * F), * sg = sc->alloc(T);
        half * mid16 = sc->alloc<half>((size_t) T * F);
        bool shared_done = false;
        auto shared = [&] {   // y = shared expert (sg, its gate, comes with the router)
            const size_t m2 = sc->mark();
            const Act ia = quant(in16, T, c.d_model);
            lin_multi({ { mo.shexp_gate, g }, { mo.shexp_up, u } }, ia, T);
            ffn::swiglu(g, u, T * F, mid16, s);
            lin(mo.shexp_down, mid16, T, y);
            sc->release(m2);
            shared_done = true;
        };
        // the router, the next layer's router on this input (pre-gating hint, TRACKER #58) and the shared-expert
        // gate read the same fp32 rows: one launch at decode sizes
        const bool hint = !stream && hint_k > 0 && l + 1 < le && T >= hint_min_rows;   // not across a split stage
        float * pl = nullptr;
        if (hint) pl = sc->alloc((size_t) T * E);
        {
            std::vector<Proj> ps{ { mo.router, logits }, { mo.shexp_gate_inp, sg } };
            if (hint) ps.push_back({ w.layers[l + 1].moe.router, pl });
            lin32_multi(ps, in, T);
        }
        const bool door = !stream && use_doorbell && T <= cpu::MAX_ROWS;   // no host sync: the driver serves this FFN
        const bool cpu_on = cpu_tier && l < c.n_layer;
        const bool fused_pub = door && route_publish_on();
        if (!fused_pub) ffn::route(logits, T, E, K, ids, wts, s);   // the fused launch does the top-k itself
        if (route_counts) ffn::count(ids, T * K, route_counts + (size_t) sl(l) * E, s);
        if (door) {
            Bell & b = bells[l];
            const int seq = ++b.seq;
            const int n_x = pub_x_lazy() && !cpu_on ? 0 : T * dm;
            if (fused_pub) {
                // the hint's prediction goes along in the same launch; its weights were never read, so it needs no
                // device buffer at all
                ffn::route_publish(logits, hint ? pl : nullptr, T, E, K, ids, wts, in, n_x, b.ids, b.pred, b.w, b.x,
                                   b.doorbell, seq, b.arrive, ffn::publish_ctas(n_x), s);
            } else {
                int * pred = nullptr;
                if (hint) {
                    float * pw = sc->alloc((size_t) T * K);
                    pred = sc->alloc<int>((size_t) T * K);
                    ffn::route(pl, T, E, K, pred, pw, s);
                }
                runtime::publish(ids, pred, wts, T * K, in, n_x, b.ids, b.pred, b.w, b.x, b.doorbell, seq, s);
            }
            if (sect) sect_record(l, 7), sect_bell[l] = 1;
            {
                std::lock_guard<std::mutex> g(dq_mu);
                jobs.push_back({ l, T, seq, hint, cpu_on });
                jobs_pushed.fetch_add(1, std::memory_order_release);
            }
            dq_cv.notify_one();
            // the shared expert does not need the plan or the copies: it runs while the driver splits and the
            // copies land (TRACKER #112; same math, so the same output)
            shared();
            // the driver's plan (mapped), the layer's copies when it queued any, the masked ids into d_mids
            if (sect) {   // profiling: the plan and the copies waited for apart (two launches instead of one)
                runtime::spin_until(b.plan, seq, s);
                sect_record(l, 9);
            }
            runtime::wait_plan(b.plan, b.need_copy, b.go, seq, b.mids, b.d_mids, cpu_on ? T * K : 0, s);
            if (sect) sect_record(l, 8);
            moe::window<MoeShape>(experts->weights(sl(l)), in, cpu_on ? b.d_mids : ids, wts, T, routed, window_ws, s);
            if (sect) sect_record(l, 10);   // the window kernel ends here; the joins below wait for other workers
            if (cpu_on) cpu_rows = -T;            // the CPU rows join after the shared expert (mapped)
        } else if (!stream) {   // fetch the few cold experts this chunk routes to
            // pre-gating: the next layer's router on this layer's input predicts 72% of its experts (TRACKER #58);
            // their copies start behind this layer's, one sync for both id sets
            int * pred = nullptr;
            if (hint) {
                float * pw = sc->alloc((size_t) T * K);
                pred = sc->alloc<int>((size_t) T * K);
                ffn::route(pl, T, E, K, pred, pw, s);
                TRUSS_CUDA(cudaMemcpyAsync(ids_host + T * K, pred, sizeof(int) * T * K, cudaMemcpyDeviceToHost, s));
            }
            // the CPU tier serves windows of up to cpu::MAX_ROWS (= moe::MAX_ROWS) rows: decode steps, verify
            // windows, MTP commits; the MTP block has none
            CpuTier * ct = l < c.n_layer && T <= cpu::MAX_ROWS ? cpu_tier.get() : nullptr;
            if (ct) {   // the rows and weights the CPU experts need, in the same sync
                TRUSS_CUDA(cudaMemcpyAsync(ct->x, in, sizeof(float) * T * dm, cudaMemcpyDeviceToHost, s));
                TRUSS_CUDA(cudaMemcpyAsync(ct->w, wts, sizeof(float) * T * K, cudaMemcpyDeviceToHost, s));
            }
            TRUSS_CUDA(cudaMemcpyAsync(ids_host, ids, sizeof(int) * T * K, cudaMemcpyDeviceToHost, s));
            TRUSS_CUDA(cudaStreamSynchronize(s));
            // split the slots: CPU-tier experts go to the pool; the GPU kernel sees them as id -1 (skipped)
            std::vector<cpu::Slot> slots;
            std::vector<int> gpu_ids;
            const int * k_ids = ids;
            if (ct) {
                for (int j = 0; j < T * K; ++j) {
                    const int e = ids_host[j];
                    const bool on_cpu = ct->is_cpu[l][e];
                    ct->ids[j] = on_cpu ? -1 : e;
                    if (on_cpu) slots.push_back(ct->slot(j / K, l, e, ct->w[j]));
                    else gpu_ids.push_back(e);
                }
                if (!slots.empty()) {
                    int * m_ids = sc->alloc<int>((size_t) T * K);
                    TRUSS_CUDA(cudaMemcpyAsync(m_ids, ct->ids, sizeof(int) * T * K, cudaMemcpyHostToDevice, s));
                    k_ids = m_ids;
                    lock_pool(ct);
                    ct->pool->start(ct->x, T, slots, ct->y);
                }
            } else {
                gpu_ids.assign(ids_host, ids_host + T * K);
            }
            experts->fetch(sl(l), gpu_ids.data(), (int) gpu_ids.size(), s);
            if (hint) {   // each row's first hint_k guesses (ids are in descending probability), GPU experts only
                std::vector<int> h;
                const CpuTier * nt = l + 1 < le ? cpu_tier.get() : nullptr;
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < std::min(K, hint_k); ++i) {
                        const int e = ids_host[T * K + t * K + i];
                        if (!nt || !nt->is_cpu[l + 1][e]) h.push_back(e);
                    }
                experts->prefetch_hint(sl(l), h.data(), (int) h.size());
            }
            experts->acquire(sl(l), s);
            if (T <= moe::MAX_ROWS) moe::window<MoeShape>(experts->weights(sl(l)), in, k_ids, wts, T, routed, window_ws, s);
            else moe::prefill<MoeShape>(experts->weights(sl(l)), in, ids, wts, T, routed, cur.moe_ws, cur.moe_rows, s);
            experts->release(sl(l), s);
            cpu_rows = slots.empty() ? 0 : T;
        } else if (split_chunk) {   // E7 (TRACKER #117): the routing first, then the CPU's experts and the GPU's copies
            CpuTier * ct = l < c.n_layer ? cpu_tier.get() : nullptr;   // the MTP block: copies only
            if (ct) {
                TRUSS_CUDA(cudaMemcpyAsync(ct->x, in, sizeof(float) * T * dm, cudaMemcpyDeviceToHost, s));
                TRUSS_CUDA(cudaMemcpyAsync(ct->w, wts, sizeof(float) * T * K, cudaMemcpyDeviceToHost, s));
            }
            TRUSS_CUDA(cudaMemcpyAsync(ids_host, ids, sizeof(int) * T * K, cudaMemcpyDeviceToHost, s));
            TRUSS_CUDA(cudaStreamSynchronize(s));
            std::vector<cpu::Slot> slots;
            std::vector<int> gpu_ids;
            const int * k_ids = ids;
            if (ct) {
                // balanced: the routed cold experts (hot ones have is_cpu 0) by ascending row count go to the CPU
                // while its estimated time (the decode split's fit, plus split_ms_row per extra row) stays within
                // what the copies left on PCIe would take; at most split_t rows each
                std::vector<int> cnt(E, 0), cand;
                std::vector<uint8_t> to_cpu(E, 0);
                for (int j = 0; j < T * K; ++j) ++cnt[ids_host[j]];
                double pcie = 0, cpu = cpu_ms_call;
                for (int e = 0; e < E; ++e)
                    if (cnt[e] > 0 && ct->is_cpu[l][e]) {
                        cand.push_back(e);
                        pcie += pcie_ms_byte * experts->bytes_of(sl(l), e);
                    }
                std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return cnt[a] < cnt[b]; });
                for (int e : cand) {
                    const double c = cpu_ms_expert + split_ms_row * (cnt[e] - 1), q = pcie_ms_byte * experts->bytes_of(sl(l), e);
                    if (cnt[e] > split_t || cpu + c > pcie - q) break;
                    cpu += c, pcie -= q, to_cpu[e] = 1;
                }
                for (int e : cand) to_cpu[e] ? ++split_cpu_experts : ++split_gpu_experts;
                for (int j = 0; j < T * K; ++j) {
                    const int e = ids_host[j];
                    const bool on_cpu = to_cpu[e];
                    ct->ids[j] = on_cpu ? -1 : e;
                    if (on_cpu) slots.push_back(ct->slot(j / K, l, e, ct->w[j]));
                    else gpu_ids.push_back(e);
                }
                if (!slots.empty()) {
                    int * m_ids = sc->alloc<int>((size_t) T * K);
                    TRUSS_CUDA(cudaMemcpyAsync(m_ids, ct->ids, sizeof(int) * T * K, cudaMemcpyHostToDevice, s));
                    k_ids = m_ids;
                    lock_pool(ct);
                    ct->pool->start(ct->x, T, slots, ct->y);   // CPU first: it runs while the copies land
                }
            } else {
                gpu_ids.assign(ids_host, ids_host + T * K);
            }
            split_host_bytes += (long long) experts->prefetch_some(sl(l), gpu_ids.data(), (int) gpu_ids.size());
            experts->acquire(sl(l), s);
            moe::prefill<MoeShape>(experts->weights(sl(l)), in, k_ids, wts, T, routed, cur.moe_ws, cur.moe_rows, s);
            experts->release(sl(l), s);
            cpu_rows = slots.empty() ? 0 : T;
        } else {                // whole layers stream ahead (prefetch(l + 2) while this layer computes)
            experts->acquire(sl(l), s);
            moe::prefill<MoeShape>(experts->weights(sl(l)), in, ids, wts, T, routed, cur.moe_ws, cur.moe_rows, s);
            experts->release(sl(l), s);
            if (sl(l) + 2 < n_store) experts->prefetch(sl(l) + 2);   // the MTP block streams after the last layer
        }
        if (sect) sect_record(l, 4);   // shared expert (the doorbell path ran it before its wait) + the CPU join
        if (!shared_done) shared();
        if (sect) sect_record(l, 5);   // section 5: the CPU tier's join (its spin is the section)
        if (cpu_rows < 0) {   // doorbell path: wait for the CPU tier's rows (mapped) and add them
            runtime::spin_until(bells[l].cpu_done, bells[l].seq, s);
            runtime::add_mapped(routed, bells[l].y, -cpu_rows * dm, s);
            cpu_rows = 0;
        }
        // The join boundary is recorded on every path: cudaEventElapsedTime between a recorded and a never-recorded
        // event reads garbage, and the CSV is read per layer whether or not this layer had CPU rows to join.
        if (sect) sect_record(l, 11);   // the CPU tier's rows are in, or this layer had none
        if (sect) sect_record(l, 12);   // no peer join on this branch: the segment between 11 and 12 is empty
        if (cpu_rows) {   // the CPU experts' sum joins the routed output
            CpuTier & ct = *cpu_tier;
            float * yc = sc->alloc((size_t) cpu_rows * dm);
            ct.pool->wait();
            TRUSS_CUDA(cudaMemcpyAsync(yc, ct.y, sizeof(float) * cpu_rows * dm, cudaMemcpyHostToDevice, s));
            ffn::add(routed, yc, cpu_rows * dm, s);
            cpu_rows = 0;
        }
        ffn::shared_add(routed, y, sg, T, dm, out, s);
        sc->release(m);
    }

    int last_T = 0;                                      // rows of res from the last run()
    float * route_counts = nullptr;                      // [layer][expert] while profiling (Forward::profile_routes)
    int cpu_rows = 0;                                    // rows of the CPU tier call in flight (ffn)

    void head(int first, int n, float * logits)
    {
        require(first >= 0 && n > 0 && first + n <= last_T, "head rows outside the last chunk");
        check_driver();
        sc->reset();   // run()'s temporaries are dead; the residual is not in the scratch
        head_rows(res + (size_t) first * c.hc_dim(), n, logits);
    }

    // logits [n][vocab] of residual rows [n][hc][d]: the head hc mix, then the output projection
    void head_rows(const float * rows, int n, float * logits, bool draft = false)
    {
        const size_t m = sc->mark();
        float * mixed = sc->alloc((size_t) n * c.d_model);
        half * mixed16 = sc->alloc<half>((size_t) n * c.d_model);
        hc_mix(w.hc_head, rows, n, mixed, mixed16, nullptr);
        if (act == Activations::Q8_1) {   // q8_gemv / q8_gemm write the whole vocab (decode: 0.9 ms vs 4 ms, #56)
            if (draft && draft_head.q) {   // the draft vocabulary's rows only
                const Act a = quant(mixed16, n, draft_head.in);
                if (n <= dense::GEMV_ROWS) dense::q8_gemv(draft_head, a.q, a.d, n, logits, s);
                else dense::q8_gemm(draft_head, a.q, a.d, n, logits, s);
                sc->release(m);
                return;
            }
            lin(w.output, mixed16, n, logits);
            sc->release(m);
            return;
        }
        if (n <= dense::GEMV_ROWS) {   // FP16 decode: the fused gemv reads the int8 output matrix once (no dequantized copy)
            dense::q8_gemv_a16(draft && draft_head.q ? draft_head : q8.at(w.output), mixed16, n, logits, s);
            sc->release(m);
            return;
        }
        // FP16: the output matrix in vocab tiles that fit the dequantized-weight scratch
        const dense::Q8Matrix & O = q8.at(w.output);
        const int tile = (int) std::min<size_t>(O.out, w16_elems / O.in) / 64 * 64;
        float * part = sc->alloc((size_t) n * tile);
        for (int v0 = 0; v0 < O.out; v0 += tile) {
            const int nv = std::min(tile, O.out - v0);
            const dense::Q8Matrix sub{ O.q + (size_t) v0 * O.in, O.d + (size_t) v0 * (O.in / 32), O.in, nv };
            dense::q8_gemm_a16(sub, mixed16, n, part, w16, blas, s);
            TRUSS_CUDA(cudaMemcpy2DAsync(logits + v0, (size_t) O.out * 4, part, (size_t) nv * 4, (size_t) nv * 4, n,
                                         cudaMemcpyDeviceToDevice, s));
        }
        sc->release(m);
    }

    void reset()
    {
        for (int l = lb; l < le; ++l) {
            const LayerState & L = st[l];
            if (L.state()) TRUSS_CUDA(cudaMemsetAsync(L.state(), 0, sizeof(float) * c.ssm_v_heads * c.ssm_state * c.ssm_state, s));
            if (L.conv) TRUSS_CUDA(cudaMemsetAsync(L.conv, 0, sizeof(float) * (c.ssm_conv - 1) * c.conv_dim(), s));
            if (L.ple_hist)
                TRUSS_CUDA(cudaMemsetAsync(L.ple_hist, 0, sizeof(float) * (c.ple_conv - 1) * c.ple_ngram * c.hc_dim(), s));
        }
        tail.clear();
        ple_next = nullptr; ple_next_T = 0; ple_tk_valid = false;   // a dangling ticket is replaced by the next issue
        last_T = 0;
        has_pending = false;
        dc_on = false, mtp_ahead = false;
        vw.open = false;
        kv_reserve(0, true);
    }

    // One chunk of the target model at positions pos0 .. pos0 + T - 1. tentative: a verify window (accept() commits).
    void run_chunk(const int32_t * tokens, int pos0, int T, const LayerHook & hook, bool tentative)
    {
        require(T > 0 && T <= max_chunk, "chunk of " + std::to_string(T) + " tokens (max " + std::to_string(max_chunk) + ")");
        require(pos0 + T <= n_ctx, "sequence longer than n_ctx");
        require(!vw.open, "run()/verify() before accept() of the previous verify()");
        flush_commit();   // it reads res, which this chunk rewrites
        check_driver();
        kv_reserve(pos0 + T, false);
        const bool stream = !fetch_mode(T);
        if (aux.on && (stream || cpu::this_tid() != caller_tid || ++aux_passes % 512 == 0)) place_threads();
        if (stream)
            require(T <= prefill_rows, "prompt chunk of " + std::to_string(T) + " tokens over prefill_rows (" +
                                            std::to_string(prefill_rows) + "); raise Options::prefill_rows");
        if (stream) drain_jobs();   // before begin_stream: a queued decode FFN (the flushed MTP commit) must fetch in ring mode
        if (!stream) {
            use(small);
            experts->begin_ring();
            if (adapt_every > 0 && ++decode_passes % adapt_every == 0) adapt();
        } else {   // the ring becomes the stream slots and this chunk's buffers
            experts->begin_stream(s);
            use(big);
        }
        split_chunk = stream && cpu_tier && T <= split_rows;   // E7: per-layer copies after the routing, no prefetch

        sc->reset();
        const size_t row_bytes = sizeof(float) * (size_t) T * c.hc_dim();
        if (first) {
            int * ids = sc->alloc<int>(T);
            float * emb = sc->alloc((size_t) T * c.d_model);
            TRUSS_CUDA(cudaMemcpyAsync(ids, tokens, (size_t) T * 4, cudaMemcpyHostToDevice, s));
            dense::q8_rows(q8.at(w.token_embd), ids, T, emb, s);
            hc::expand(emb, T, c.hc, c.d_model, res, s);
        } else {   // the previous stage's residual rows (its out_host, behind its out_ready)
            while (prev->handed.load(std::memory_order_acquire) <= taken.load()) {
                require(!prev->halted, "the previous layer-split stage failed");
                std::this_thread::yield();
            }
            TRUSS_CUDA(cudaStreamWaitEvent(s, prev->out_ready, 0));
            TRUSS_CUDA(cudaMemcpyAsync(res, prev->out_host, row_bytes, cudaMemcpyHostToDevice, s));
            TRUSS_CUDA(cudaEventRecord(in_taken, s));
            taken.fetch_add(1, std::memory_order_release);
        }
        if (stream && !split_chunk)
            for (int l = 0; l < std::min(2, n_store); ++l) experts->prefetch(l);
        const size_t base = sc->mark();
        // a layer's closing combine is folded into the next layer's first norm (one launch fewer per layer): out and
        // inject sit at the same scratch offsets every layer and are read by that first kernel before anything
        // rewrites them. Not before a PLE layer (ple() reads res), with a hook, or at the stage's end.
        bool pend_combine = false;
        for (int l = lb; l < le; ++l) {
            sc->release(base);
            const Layer & L = w.layers[l];
            float * mixed = sc->alloc((size_t) T * c.d_model), * out = sc->alloc((size_t) T * c.d_model);
            float * inject = sc->alloc((size_t) T * c.hc);
            half * mixed16 = sc->alloc<half>((size_t) T * c.d_model);
            sect_record(l, 0);
            if (c.is_ple(l)) ple(L.ple, st[l], tokens, T, pos0, tentative);
            hc_mix(L.hc_attn, res, T, mixed, mixed16, inject, pend_combine ? out : nullptr, pend_combine ? inject : nullptr);
            pend_combine = false;
            sect_record(l, 1);
            if (L.mixer == Mixer::GDN) gdn(L.gdn, st[l], mixed16, T, out, tentative);
            else dsa(L.dsa, st[l], mixed16, pos0, T, out, tentative);
            sect_record(l, 2);
            hc_mix(L.hc_ffn, res, T, mixed, mixed16, inject, out, inject);   // combine #1 fused into the norm (L3)
            ffn(l, L.moe, mixed, mixed16, T, out, stream);   // records 3 (routed MoE), 4 (shared), 5 (cpu join)
            sect_record(l, 6);
            if (!hook && l + 1 < le && !c.is_ple(l + 1)) pend_combine = true;
            else hc::combine(res, out, inject, T, c.hc, c.d_model, s);
            if (hook) {
                TRUSS_CUDA(cudaStreamSynchronize(s));
                hook(l, res, T);
            }
        }
        if (!last) {   // hand the residual rows on, once the next stage has taken the previous ones
            if (next) {
                while (next->taken.load(std::memory_order_acquire) < handed.load()) {
                    require(!next->halted, "the next layer-split stage failed");
                    std::this_thread::yield();
                }
                TRUSS_CUDA(cudaStreamWaitEvent(s, next->in_taken, 0));
            }
            TRUSS_CUDA(cudaMemcpyAsync(out_host, res, row_bytes, cudaMemcpyDeviceToHost, s));
            TRUSS_CUDA(cudaEventRecord(out_ready, s));
            handed.fetch_add(1, std::memory_order_release);
        }
        sect_flush(T);
        ple_next = nullptr; ple_next_T = 0;   // consumed by this chunk's ple(), or there was no lookahead
        last_T = T;
        if (tentative) {
            vw = { true, pos0, T, std::vector<int32_t>(tokens, tokens + T) };
            return;
        }
        commit_tail(tokens, T);
        if (mtp) mtp_commit(tokens, pos0, T, stream);
    }

    // Strata's adapt() (generate.cpp): the most-routed missing experts (decayed count >= 2) of every store layer,
    // at most adapt_swaps, into the ring; then counts x0.7. Between passes: the driver thread is idle (the caller
    // synced the last pass for its logits), so route_usage is not being written.
    void adapt()
    {
        experts->poll_admitted();
        if (experts->admitting()) return;   // the previous batch is still being copied
        std::vector<std::tuple<float, int, int>> cand;
        for (int l = 0; l < n_store; ++l)
            for (int e = 0; e < c.n_expert; ++e) {
                const float u = route_usage[(size_t) l * c.n_expert + e];
                if (u >= adapt_min && !experts->on_device(l, e)) cand.emplace_back(u, l, e);
            }
        const size_t n = std::min(cand.size(), (size_t) adapt_swaps);
        std::partial_sort(cand.begin(), cand.begin() + (ptrdiff_t) n, cand.end(),
                          [](const auto & a, const auto & b) { return std::get<0>(a) > std::get<0>(b); });
        std::vector<std::pair<int, int>> le;
        for (size_t i = 0; i < n; ++i) le.push_back({ std::get<1>(cand[i]), std::get<2>(cand[i]) });
        experts->admit(le, s);
        for (float & u : route_usage) u *= adapt_decay;
    }

    void commit_tail(const int32_t * tokens, int T)   // the PLE window's predecessors for the next chunk
    {
        if (c.ple_layers.empty()) return;
        std::vector<int32_t> seq(tail);
        seq.insert(seq.end(), tokens, tokens + T);
        const size_t keep = std::min<size_t>(seq.size(), c.ple_ngram - 1);
        tail.assign(seq.end() - keep, seq.end());
    }

    // Commit the first n rows of the open verify window: every recurrent state as if only those rows had run.
    void accept(int n)
    {
        require(vw.open, "accept() without verify()");
        require(n >= 1 && n <= vw.T, "accept(" + std::to_string(n) + ") of a " + std::to_string(vw.T) + "-row window");
        vw.open = false;
        const bool all = n == vw.T;
        const size_t m = sc->mark();
        const gdn::Dims dm{ c.ssm_groups, c.ssm_v_heads, c.ssm_state, c.ssm_conv };
        for (int l = lb; l < le; ++l) {
            LayerState & L = st[l];
            const Layer & W = w.layers[l];
            if (W.mixer == Mixer::GDN) {
                if (all) {   // the window's final state is the other buffer
                    L.cur ^= 1;
                } else {     // conv rows and state from before the window, then the accepted rows again
                    const int kd = c.key_dim(), vd = c.value_dim(), Hv = c.ssm_v_heads;
                    float * q = sc->alloc((size_t) n * kd), * k = sc->alloc((size_t) n * kd), * v = sc->alloc((size_t) n * vd);
                    float * gt = sc->alloc((size_t) n * Hv), * beta = sc->alloc((size_t) n * Hv), * core = sc->alloc((size_t) n * vd);
                    copy(L.conv, L.conv_snap, (size_t) (c.ssm_conv - 1) * c.conv_dim());
                    gdn::prepare(dm, L.raw_qkv, f32(W.gdn.conv1d), L.conv, L.raw_alpha, L.raw_beta, f32(W.gdn.dt_bias),
                                 f32(W.gdn.a), n, c.rms_eps, q, k, v, gt, beta, s);
                    gdn::delta_rule(q, k, v, gt, beta, L.state(), L.state(), core, n, c.ssm_groups, Hv, s);
                    sc->release(m);
                }
            } else if (!all) {   // the indexer's open block: the snapshot plus the accepted rows' raw keys
                copy(L.idx_partial, L.partial_snap, (size_t) (DsaShape::RATIO - 1) * c.idx_head_dim);
                dsa::carry_partial<DsaShape>(L.raw_ik, vw.pos0, n, L.idx_partial, s);
            }
            if (c.is_ple(l) && !all)
                spec::tail_rows(L.hist_snap, (c.ple_conv - 1) * c.ple_ngram, L.normed_rows, n, c.hc_dim(), L.ple_hist, s);
        }
        commit_tail(vw.tokens.data(), n);
        last_T = n;
        if (mtp) dc_on = true, dc_pos0 = vw.pos0, dc_T = n, dc_tokens.assign(vw.tokens.begin(), vw.tokens.begin() + n);
    }

    // Sequence checkpoint (Forward::checkpoint / restore): every recurrent piece of the committed sequence - GDN state
    // and conv rows, the DSA indexer's open block, PLE history and token window, the MTP block's open block and
    // pending row - copied to pinned host memory (~115 MB here, no VRAM taken from the experts). Position-indexed
    // caches (K/V, indexer blocks, MTP rows) need no copy: rows past the checkpoint are rewritten before any query
    // reads them, as after a verify window.
    struct Checkpoint {
        bool valid = false;
        int pos = 0;
        bool has_pending = false;
        std::vector<int32_t> tail;
        char * host = nullptr;   // pinned
        size_t bytes = 0;
    } ck;

    template <class F> void each_ck_buffer(F && f)   // f(device pointer, floats) in a fixed order
    {
        for (int l = lb; l < le; ++l) {
            LayerState & L = st[l];
            if (w.layers[l].mixer == Mixer::GDN) {
                f(L.state(), (size_t) c.ssm_v_heads * c.ssm_state * c.ssm_state);
                f(L.conv, (size_t) (c.ssm_conv - 1) * c.conv_dim());
            } else if (L.idx_partial) {
                f(L.idx_partial, (size_t) (DsaShape::RATIO - 1) * c.idx_head_dim);
            }
            if (L.ple_hist) f(L.ple_hist, (size_t) (c.ple_conv - 1) * c.ple_ngram * c.hc_dim());
        }
        if (mtp) {
            f(mst.idx_partial, (size_t) (DsaShape::RATIO - 1) * c.idx_head_dim);
            f(pending_h, (size_t) c.hc_dim());
        }
    }

    void checkpoint(int pos)
    {
        require(!vw.open, "checkpoint() inside a verify window");
        flush_commit();
        if (!ck.host) {
            each_ck_buffer([&](float *, size_t n) { ck.bytes += n * sizeof(float); });
            TRUSS_CUDA(cudaMallocHost(&ck.host, ck.bytes));
        }
        size_t at = 0;
        each_ck_buffer([&](float * d, size_t n) {
            TRUSS_CUDA(cudaMemcpyAsync(ck.host + at, d, n * sizeof(float), cudaMemcpyDeviceToHost, s));
            at += n * sizeof(float);
        });
        TRUSS_CUDA(cudaStreamSynchronize(s));
        ck.valid = true, ck.pos = pos, ck.has_pending = has_pending, ck.tail = tail;
    }

    int restore()
    {
        require(!vw.open, "restore() inside a verify window");
        if (!ck.valid) return -1;
        size_t at = 0;
        each_ck_buffer([&](float * d, size_t n) {
            TRUSS_CUDA(cudaMemcpyAsync(d, ck.host + at, n * sizeof(float), cudaMemcpyHostToDevice, s));
            at += n * sizeof(float);
        });
        TRUSS_CUDA(cudaStreamSynchronize(s));
        has_pending = ck.has_pending, tail = ck.tail, last_T = 0;
        dc_on = false, mtp_ahead = false;   // the next commit recomputes its first row
        kv_reserve(ck.pos, true);
        return ck.pos;
    }

    // MTP calibration capture (TRUSS_MTP_CALIB=/path/file.f32): the MTP block's FFN input rows (fp32 [d_model],
    // one per true-path row) appended for an offline Hessian (gate/up share this input; the down Hessian is derived
    // from the BF16 gate/up weights). Only true-path commits (mtp_commit), never draft chains.
    FILE * mtp_calib_f = nullptr;
    float * calib_host = nullptr;   // pinned [max_chunk][d_model] staging

    // Layer-chain section profile (TRUSS_PROFILE_SECTIONS=1). Five events reused for every layer: they complete in
    // stream order, so elapsed(e[i-1], e[i]) is one section's device time including any host stall inside it.
    // Sums accumulate per run()/verify(); section_ms() reads them back.
    bool sect_on = false;
    std::vector<std::array<cudaEvent_t, 13>> sect_ev;   // [layer][0..6] sections; 7, 9, 8, 10, 11, 12: inside the MoE
    mutable std::vector<uint8_t> sect_rec;             // layers recorded since the last flush
    mutable double sect_ms[6] = { 0, 0, 0, 0, 0, 0 };
    // routed MoE on the doorbell path split at events 7 (doorbell raised), 9 (plan received) and 8 (copies landed,
    // TRACKER #77): router + publish, wait for the host's plan, wait for the layer's copies, expert kernel
    mutable std::vector<uint8_t> sect_bell;
    mutable double sect_moe[4] = { 0, 0, 0, 0 };
    // /proc/self/task/<tid>/stat field 39: the logical CPU the thread last ran on. Field 2 (comm) can hold spaces
    // and parentheses, so parse from after the last ')'. -2: not readable.
    static int last_cpu(pid_t tid)
    {
        char path[64], buf[2048];
        std::snprintf(path, sizeof path, "/proc/self/task/%ld/stat", (long) tid);
        FILE * f = std::fopen(path, "r");
        if (!f) return -2;
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        std::fclose(f);
        buf[n] = 0;
        char * p = buf + n;
        while (p > buf && *(p - 1) != ')') --p;   // just past the last ')'
        for (int i = 0; i < 36; ++i) {            // fields 3..38 precede the processor field
            while (p < buf + n && *p == ' ') ++p;
            while (p < buf + n && *p != ' ') ++p;
        }
        while (p < buf + n && *p == ' ') ++p;
        return p < buf + n ? (int) std::strtol(p, nullptr, 10) : -2;
    }
    int caller_cpu = -1;                                // sampled by sect_flush, on the thread that launches
    std::atomic<int> drv_cpu{ -1 };                     // the driver's CPU, sampled by the driver itself
    FILE * sect_csv = nullptr;                          // TRUSS_SECT_CSV: per layer per pass, the doorbell path
    long sect_pass = 0;                                 // chunks flushed (one per verify pass)
    void sect_record(int layer, int i)
    {
        if (!sect_on) return;
        TRUSS_CUDA(cudaEventRecord(sect_ev[layer][i], s));
        sect_rec[layer] = 1;
    }
    // Read the events back after the whole chunk (one sync, no per-layer stall) and add the five sections. Only
    // the layers that ran are read: draft()'s MTP steps record only some.
    void sect_flush(int T = 0)
    {
        if (!sect_on) return;
        ++sect_pass;
        caller_cpu = last_cpu(cpu::this_tid());
        // The events were recorded on THIS stage's device; the thread reading them runs both stages, so its current
        // device is whichever one ran last - reading another device's event is "invalid resource handle".
        DeviceGuard g(device);
        for (int l = 0; l <= c.n_layer; ++l) {
            if (!sect_rec[l]) continue;
            TRUSS_CUDA(cudaEventSynchronize(sect_ev[l][6]));   // the last one recorded for the layer, stream order
            float sec[6];   // the layer chain around the MoE: see Forward::section_ms
            for (int i = 1; i <= 6; ++i) {
                float ms = 0.f;
                TRUSS_CUDA(cudaEventElapsedTime(&ms, sect_ev[l][i - 1], sect_ev[l][i]));
                sect_ms[i - 1] += sec[i - 1] = ms;
            }
            if (sect_bell[l]) {
                const int at[5] = { 3, 7, 9, 8, 4 };
                for (int i = 0; i < 4; ++i) {
                    float ms = 0.f;
                    TRUSS_CUDA(cudaEventElapsedTime(&ms, sect_ev[l][at[i]], sect_ev[l][at[i + 1]]));
                    sect_moe[i] += ms;
                }
                if (sect_csv) {   // the doorbell FFN to its last boundary, plus the whole layer chain around it
                    const int b[8] = { 3, 7, 9, 8, 10, 11, 12, 6 };
                    std::fprintf(sect_csv, "%ld %d %d", sect_pass, T, l);
                    for (int i = 0; i < 7; ++i) {
                        float ms = 0.f;
                        TRUSS_CUDA(cudaEventElapsedTime(&ms, sect_ev[l][b[i]], sect_ev[l][b[i + 1]]));
                        std::fprintf(sect_csv, " %.3f", ms);
                    }
                    for (int i = 0; i < 6; ++i) std::fprintf(sect_csv, " %.3f", sec[i]);
                    // the host chain of this layer, us after the driver saw the doorbell (see Bell::h_t). The GPU
                    // waits for the plan and the copies, so the driver had served this seq when its last event
                    // completed: no race with the writer.
                    const Bell & hb = bells[l];
                    for (int i = 0; i < 5; ++i) std::fprintf(sect_csv, " %.1f", hb.h_t[i]);
                    std::fprintf(sect_csv, " %.1f %.1f %.1f %.1f %d\n", hb.h_pool, hb.h_wake, hb.h_item, hb.h_longest, hb.h_ncpu);
                }
            }
            sect_rec[l] = 0, sect_bell[l] = 0;
        }
    }
    void mtp_calib_open(int max_rows)
    {
        if (const char * p = getenv("TRUSS_MTP_CALIB")) {
            mtp_calib_f = std::fopen(p, "ab");
            require(mtp_calib_f != nullptr, "cannot open TRUSS_MTP_CALIB file");
            TRUSS_CUDA(cudaMallocHost(&calib_host, sizeof(float) * (size_t) max_rows * c.d_model));
        }
    }
    // The MTP block on T rows: hidden states h [T][hc][d] (device), next tokens d_ids [T] (device) at MTP positions
    // pos0 .. pos0 + T - 1; its residual in cur.mres; logits of the last row into mtp_logits when asked.
    // calib: also append the FFN input rows to the TRUSS_MTP_CALIB capture (true-path commits only).
    void mtp_rows(const float * h, const int * d_ids, int pos0, int T, bool stream, bool logits, bool calib = false)
    {
        const Layer & L = mtp->layer;
        const int hc = c.hc, dm = c.d_model;
        float * mres = cur.mres;
        const size_t m = sc->mark();
        float * emb = sc->alloc((size_t) T * dm), * rstd = sc->alloc((size_t) T * hc);
        half * hn16 = sc->alloc<half>((size_t) T * c.hc_dim()), * cat16 = sc->alloc<half>((size_t) T * hc * 2 * dm);
        dense::q8_rows(q8.at(w.token_embd), d_ids, T, emb, s);
        hc::norm(h, f32(mtp->hnorm), T, hc, dm, c.rms_eps, hn16, rstd, s);
        mtp::join(emb, f32(mtp->enorm), hn16, T, hc, dm, c.rms_eps, cat16, s);
        lin(mtp->eh_proj, cat16, T * hc, mres);   // per stream: [2d] -> [d]
        const size_t base = sc->mark();
        float * mixed = sc->alloc((size_t) T * dm), * out = sc->alloc((size_t) T * dm);
        float * inject = sc->alloc((size_t) T * hc);
        half * mixed16 = sc->alloc<half>((size_t) T * dm);
        hc_mix(L.hc_attn, mres, T, mixed, mixed16, inject);
        dsa(L.dsa, mst, mixed16, pos0, T, out, false);
        hc::combine(mres, out, inject, T, hc, dm, s);
        hc_mix(L.hc_ffn, mres, T, mixed, mixed16, inject);
        if (calib && mtp_calib_f) {   // the routed experts' input rows, true path only
            TRUSS_CUDA(cudaMemcpyAsync(calib_host, mixed, sizeof(float) * (size_t) T * dm, cudaMemcpyDeviceToHost, s));
            TRUSS_CUDA(cudaStreamSynchronize(s));
            require(std::fwrite(calib_host, sizeof(float) * dm, T, mtp_calib_f) == (size_t) T, "TRUSS_MTP_CALIB write");
        }
        ffn(c.n_layer, L.moe, mixed, mixed16, T, out, stream);
        hc::combine(mres, out, inject, T, hc, dm, s);
        (void) base;
        if (logits) head_rows(mres + (size_t) (T - 1) * c.hc_dim(), 1, mtp_logits, true);
        sc->release(m);
    }

    // After committing rows at positions pos0 .. pos0 + T - 1 (tokens, host): the MTP block at those positions, each
    // from the previous position's hidden state (pending for the first, the target's residual rows after), so its
    // cache matches a run over the true hidden states. Position 0 has no previous state and is skipped.
    void mtp_commit(const int32_t * tokens, int pos0, int T, bool stream)
    {
        const size_t row = c.hc_dim();
        // only position 0 lacks a pending state; draft() may have computed the first row already (ahead)
        const bool ahead_hit = mtp_ahead && T > 0 && tokens[0] == ahead_tok && pos0 == ahead_pos;
        mtp_ahead = false;
        const int skip = has_pending && !ahead_hit ? 0 : 1;
        const int n = T - skip;
        if (n > 0) {
            float * mh = cur.mh;
            if (!skip) copy(mh, pending_h, row);
            if (T > 1) copy(mh + (size_t) (1 - skip) * row, res, (size_t) (T - 1) * row);
            const size_t m = sc->mark();
            int * d_ids = sc->alloc<int>(n);
            // a decode commit's tokens go through pinned staging (alternating halves: the previous commit's copy ran
            // a pass ago), so accept() returns without waiting for the GPU; a prompt chunk's tokens may be a host
            // temporary: that copy is waited for
            const bool staged = n <= moe::MAX_ROWS + 1;
            const int32_t * src = tokens + skip;
            if (staged) {
                int32_t * h = commit_host + (size_t) commit_flip * (moe::MAX_ROWS + 1);
                commit_flip ^= 1;
                std::copy(src, src + n, h);
                src = h;
            }
            TRUSS_CUDA(cudaMemcpyAsync(d_ids, src, sizeof(int32_t) * n, cudaMemcpyHostToDevice, s));
            mtp_rows(mh, d_ids, pos0 + skip, n, stream, false, true);
            if (!staged) TRUSS_CUDA(cudaStreamSynchronize(s));
            sc->release(m);
        }
        copy(pending_h, res + (size_t) (T - 1) * row, row);
        has_pending = true;
    }

    // n greedy drafts after `next` (the token at position pos): a chain of single-row MTP steps, each from the
    // previous step's residual and argmax. The chain's cache rows and indexer block are tentative: the snapshot is
    // restored at the end, and mtp_commit rewrites those positions from the target's states.
    int draft(int32_t next, int pos, int n, int32_t * out)
    {
        require(mtp && has_pending, "draft() needs an MTP block and a committed position");
        require(n >= 1 && n <= moe::MAX_ROWS, "draft count");
        require(!vw.open, "draft() before accept()");
        kv_reserve(pos + n + 1, false);
        use(small);
        experts->begin_ring();
        sc->reset();
        const size_t row = c.hc_dim();
        // the deferred commit and the first draft row (pending state = the last accepted row, token = next) as one
        // MTP pass when they fit a window; that row is true, not tentative, so it is kept (ahead) below
        int merged = 0;   // rows of the merged pass (0: none)
        if (dc_on) {
            dc_on = false;
            const int T = dc_T;
            const bool ahead_hit = mtp_ahead && T > 0 && dc_tokens[0] == ahead_tok && dc_pos0 == ahead_pos;
            const int skip = has_pending && !ahead_hit ? 0 : 1, nc = T - skip;
            static const bool merge_on = [] { const char * e = std::getenv("TRUSS_MTP_MERGE"); return !e || std::atoi(e); }();
            if (merge_on && nc + 1 <= moe::MAX_ROWS && dc_pos0 + T == pos && T >= 1) {
                mtp_ahead = false;
                float * mh = cur.mh;
                if (!skip) copy(mh, pending_h, row);
                copy(mh + (size_t) (1 - skip) * row, res, (size_t) T * row);
                int32_t * h = commit_host + (size_t) commit_flip * (moe::MAX_ROWS + 1);
                commit_flip ^= 1;
                std::copy(dc_tokens.begin() + skip, dc_tokens.begin() + T, h);
                h[nc] = next;
                int * d_ids = sc->alloc<int>(nc + 1);
                TRUSS_CUDA(cudaMemcpyAsync(d_ids, h, sizeof(int32_t) * (nc + 1), cudaMemcpyHostToDevice, s));
                mtp_rows(mh, d_ids, dc_pos0 + skip, nc + 1, false, true, true);
                copy(pending_h, res + (size_t) (T - 1) * row, row);
                has_pending = true;
                mtp_ahead = true, ahead_tok = next, ahead_pos = pos;
                merged = nc + 1;
            } else {
                mtp_commit(dc_tokens.data(), dc_pos0, T, false);
            }
        }
        copy(mtp_partial_snap, mst.idx_partial, (size_t) (DsaShape::RATIO - 1) * c.idx_head_dim);
        draft_host[0] = next;
        TRUSS_CUDA(cudaMemcpyAsync(draft_ids, draft_host, sizeof(int32_t), cudaMemcpyHostToDevice, s));
        if (!merged) copy(cur.mh, pending_h, row);
        const int nv = draft_head.q ? draft_head.out : c.n_vocab;
        int made = 0;
        for (int i = 0; i < n; ++i) {
            if (i > 0 || !merged) mtp_rows(cur.mh, draft_ids + i, pos + i, 1, false, true);
            sampling::argmax_prob(mtp_logits, nv, draft_map, draft_ids + i + 1, draft_prob + i, s);
            if (ds_p)   // the guess becomes a sample of q (the argmax probability above still decides the stop)
                sampling::draft_sample(mtp_logits, nv, draft_map, c.n_vocab, *ds_p, ds_counter + i,
                                       ds_q + (size_t) i * c.n_vocab, draft_ids + i + 1, s);
            if (draft_min_p > 0.f) {   // an unsure guess does not enter the window, and ends the chain
                TRUSS_CUDA(cudaMemcpyAsync(draft_prob_host + i, draft_prob + i, sizeof(float), cudaMemcpyDeviceToHost, s));
                TRUSS_CUDA(cudaStreamSynchronize(s));
                if (draft_prob_host[i] < draft_min_p) break;
            }
            ++made;
            if (i + 1 < n) copy(cur.mh, cur.mres + (size_t) (i == 0 && merged ? merged - 1 : 0) * row, row);
        }
        copy(mst.idx_partial, mtp_partial_snap, (size_t) (DsaShape::RATIO - 1) * c.idx_head_dim);
        if (made) TRUSS_CUDA(cudaMemcpyAsync(draft_host, draft_ids + 1, sizeof(int32_t) * made, cudaMemcpyDeviceToHost, s));
        TRUSS_CUDA(cudaStreamSynchronize(s));
        std::copy(draft_host, draft_host + made, out);
        return made;
    }
};

Forward::Forward(const Config & c, const Weights & w, int n_ctx, int max_chunk, const Options & o)
{
    const int S = o.layer_split;
    if (S <= 0 || S >= c.n_layer) {
        m_ = std::make_unique<Impl>(c, w, n_ctx, max_chunk, o, StageSpec{ 0, c.n_layer, -1, 1.0, nullptr, nullptr });
        return;
    }
    int cur = 0;
    TRUSS_CUDA(cudaGetDevice(&cur));
    const int d0 = o.layer_split_device >= 0 ? o.layer_split_device : cur;
    StageSpec a{ 0, S, d0, d0 == cur ? (double) o.layer_split_frac : 1.0, nullptr, std::make_shared<std::mutex>() };
    {
        DeviceGuard g(d0);
        front_.push_back(std::make_unique<Impl>(c, w, n_ctx, max_chunk, o, a));
    }
    Impl & f = *front_.back();
    StageSpec b{ S, c.n_layer, cur, 1.0, f.cpu_tier ? f.cpu_tier->pool : nullptr, a.pool_mu };
    m_ = std::make_unique<Impl>(c, w, n_ctx, max_chunk, o, b);
    m_->prev = &f, f.next = m_.get();
    std::fprintf(stderr, "layer split: layers 0-%d on device %d (%d experts resident), layers %d-%d, head and MTP on "
                         "device %d (%d resident)\n",
                 S - 1, d0, f.n_hot, S, c.n_layer - 1, cur, m_->n_hot);
}

Forward::~Forward()
{
    int d = 0;
    cudaGetDevice(&d);
    m_.reset();   // the last stage first: its driver may share the first stage's CPU pool
    front_.clear();
    cudaSetDevice(d);
}

// every stage in layer order, each with its device current
template <class F> void Forward::each_stage(F && f)
{
    for (auto & p : front_) {
        DeviceGuard g(p->device);
        f(*p);
    }
    f(*m_);
}

void Forward::wait_pinned()
{
    each_stage([](Impl & m) { if (m.experts) m.experts->wait_pinned(); });
}

cudaStream_t Forward::stream() const { return m_->s; }

int Forward::hot_experts() const
{
    int n = m_->n_hot;
    for (auto & p : front_) n += p->n_hot;
    return n;
}
int Forward::fetch_rows() { return Impl::FETCH_ROWS; }

size_t Forward::cold_bytes() const
{
    size_t n = m_->experts->cold_bytes();
    for (auto & p : front_) n += p->experts->cold_bytes();
    return n;
}

std::pair<long long, long> Forward::cpu_stats(bool reset)
{
    if (!m_->cpu_tier || !m_->cpu_tier->pool) return { 0, 0 };
    auto * p = m_->cpu_tier->pool.get();   // a split's stages share one pool
    const auto out = std::make_pair(p->wait_us(), p->waits());
    if (reset) p->reset_stats();
    return out;
}

long long Forward::cpu_item_us() const
{
    if (!m_->cpu_tier || !m_->cpu_tier->pool) return 0;
    return m_->cpu_tier->pool.get()->item_us();
}

void Forward::section_ms(double out[6], bool reset) const
{
    for (int i = 0; i < 6; ++i) out[i] = 0;
    auto add = [&](Impl & m) {
        for (int i = 0; i < 6; ++i) out[i] += m.sect_ms[i];
        if (reset) for (int i = 0; i < 6; ++i) m.sect_ms[i] = 0;
    };
    for (auto & p : front_) add(*p);
    add(*m_);
}

void Forward::driver_ms(double out[3], long & n, bool reset) const
{
    for (int i = 0; i < 3; ++i) out[i] = 0;
    n = 0;
    auto add = [&](Impl & m) {
        for (int i = 0; i < 3; ++i) out[i] += m.drv_ms[i];
        n += m.drv_n;
        if (reset) {
            for (int i = 0; i < 3; ++i) m.drv_ms[i] = 0;
            m.drv_n = 0;
        }
    };
    for (auto & p : front_) add(*p);
    add(*m_);
}

long Forward::adapt_admitted() const
{
    long n = m_->experts->admitted();
    for (auto & p : front_) n += p->experts->admitted();
    return n;
}

void Forward::section_moe_ms(double out[4], bool reset) const
{
    for (int i = 0; i < 4; ++i) out[i] = 0;
    auto add = [&](Impl & m) {
        for (int i = 0; i < 4; ++i) out[i] += m.sect_moe[i];
        if (reset) for (int i = 0; i < 4; ++i) m.sect_moe[i] = 0;
    };
    for (auto & p : front_) add(*p);
    add(*m_);
}

void Forward::ple_host_ms(double & ms, long & calls, long & rows, bool reset, double * wait_ms, double * phase_ms) const
{
    ms = 0, calls = 0, rows = 0;
    if (wait_ms) *wait_ms = 0;
    if (phase_ms) phase_ms[0] = phase_ms[1] = phase_ms[2] = 0;
    auto add = [&](Impl & m) {
        ms += m.ple_host_ms, calls += m.ple_calls, rows += m.ple_rows_n;
        if (wait_ms && m.ple_reader) *wait_ms += m.ple_reader->take_wait_ms();
        if (phase_ms && m.ple_reader) {
            double ph[3] = { 0, 0, 0 };
            m.ple_reader->take_phase_ms(ph);
            for (int i = 0; i < 3; ++i) phase_ms[i] += ph[i];
        }
        if (reset) m.ple_host_ms = 0, m.ple_calls = 0, m.ple_rows_n = 0;
    };
    for (auto & p : front_) add(*p);
    add(*m_);
}

void Forward::cpu_shape(long long out[4]) const
{
    if (!m_->cpu_tier || !m_->cpu_tier->pool) {
        out[0] = out[1] = out[2] = out[3] = 0;
        return;
    }
    m_->cpu_tier->pool.get()->shape(out);
}

void Forward::dyn_stats(long & cpu, long & pcie, double & cpu_ms_expert, double & cpu_ms_call) const
{
    cpu = m_->dyn_cpu;
    pcie = m_->dyn_pcie;
    for (auto & p : front_) cpu += p->dyn_cpu, pcie += p->dyn_pcie;
    cpu_ms_expert = m_->cpu_ms_expert;
    cpu_ms_call = m_->cpu_ms_call;
}

void Forward::cpu_shape_reset()
{
    if (m_->cpu_tier && m_->cpu_tier->pool) m_->cpu_tier->pool.get()->reset_stats();
}

void Forward::cpu_phase_us(long long out[4]) const
{
    if (!m_->cpu_tier || !m_->cpu_tier->pool) {
        out[0] = out[1] = out[2] = out[3] = 0;
        return;
    }
    auto * p = m_->cpu_tier->pool.get();
    out[0] = p->gate_us();
    out[1] = p->silu_us();
    out[2] = p->hq_us();
    out[3] = p->down_us();
}

const runtime::ExpertStore & Forward::experts() const { return *m_->experts; }   // a split: the last stage's
void Forward::split_stats(long long out[3]) const
{
    out[0] = m_->split_host_bytes, out[1] = m_->split_cpu_experts, out[2] = m_->split_gpu_experts;
    for (auto & p : front_) out[0] += p->split_host_bytes, out[1] += p->split_cpu_experts, out[2] += p->split_gpu_experts;
}

size_t Forward::kv_lent() const
{
    size_t n = m_->experts->lent_bytes();
    for (auto & p : front_) n += p->experts->lent_bytes();
    return n;
}

void Forward::profile_routes(bool on)
{
    each_stage([on](Impl & m) {
        if (on && !m.route_counts) m.route_counts = m.alloc<float>((size_t) m.n_store * m.c.n_expert);   // zeroed
        if (!on && m.route_counts) {
            TRUSS_CUDA(cudaStreamSynchronize(m.s));
            TRUSS_CUDA(cudaFree(m.route_counts));
            m.owned.erase(std::find(m.owned.begin(), m.owned.end(), m.route_counts));
            m.route_counts = nullptr;
        }
    });
}

std::vector<float> Forward::route_counts() const   // the stages' store layers in order: [layer][expert], MTP last
{
    std::vector<float> out;
    auto add = [&](const Impl & m) {
        DeviceGuard g(m.device);
        const size_t at = out.size(), n = (size_t) m.n_store * m.c.n_expert;
        out.resize(at + n, 0.f);
        if (m.route_counts) {
            TRUSS_CUDA(cudaStreamSynchronize(m.s));
            TRUSS_CUDA(cudaMemcpy(out.data() + at, m.route_counts, n * 4, cudaMemcpyDeviceToHost));
        }
    };
    for (auto & p : front_) add(*p);
    add(*m_);
    return out;
}

void Forward::head(int first, int n, float * logits) { m_->head(first, n, logits); }

void Forward::reset()
{
    if (m_->route_trace) {   // the driver is idle between passes; the marker separates sequences
        const int32_t h[2] = { -1, 0 };
        std::fwrite(h, 4, 2, m_->route_trace);
    }
    each_stage([](Impl & m) { m.reset(); });
    pos_ = 0;
}

void Forward::run(const int32_t * tokens, int T, const LayerHook & hook, const int32_t * next, int next_T)
{
    if (next) require(next == tokens + T && next_T > 0, "PLE lookahead must follow the chunk");
    each_stage([&](Impl & m) {
        m.ple_next = next; m.ple_next_T = next_T;
        m.run_chunk(tokens, pos_, T, hook, false);
    });
    pos_ += T;
}

void Forward::run_prompt(const int32_t * tokens, int n, int step, const std::function<void(int)> & done)
{
    require(n > 0 && step > 0, "run_prompt: no tokens");
    auto chunk = [&](int s, int & T, const int32_t *& nxt, int & nxt_T) {   // the run() loop's chunks and lookahead
        T = std::min(step, n - s);
        nxt_T = std::max(0, std::min(step, n - s - T));
        nxt = nxt_T > 0 ? tokens + s + T : nullptr;
    };
    if (front_.empty()) {
        for (int s = 0; s < n; s += step) {
            int T, nT;
            const int32_t * nx;
            chunk(s, T, nx, nT);
            run(tokens + s, T, nullptr, nx, nT);
            if (done) done(s + T);
        }
        return;
    }
    // Layer split: the first stage runs on its own thread, a chunk ahead of the last one (the hand-off's handshake
    // orders them), so both devices compute at once. Every row is what run() computes: the same output.
    Impl & f = *front_[0];
    std::exception_ptr err;
    const int pos0 = pos_;
    const auto t0 = std::chrono::steady_clock::now();
    double front_s = 0;
    std::thread t([&] {
        DeviceGuard g(f.device);
        try {
            for (int s = 0; s < n; s += step) {
                int T, nT;
                const int32_t * nx;
                chunk(s, T, nx, nT);
                f.ple_next = nx, f.ple_next_T = nT;
                f.run_chunk(tokens + s, pos0 + s, T, nullptr, false);
            }
            TRUSS_CUDA(cudaStreamSynchronize(f.s));
            front_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        } catch (...) {
            err = std::current_exception();
            f.halted = true;
        }
    });
    std::exception_ptr err2;
    try {
        for (int s = 0; s < n; s += step) {
            int T, nT;
            const int32_t * nx;
            chunk(s, T, nx, nT);
            m_->ple_next = nx, m_->ple_next_T = nT;
            m_->run_chunk(tokens + s, pos0 + s, T, nullptr, false);
            if (done) done(s + T);
        }
    } catch (...) {
        err2 = std::current_exception();
        m_->halted = true;
    }
    t.join();
    f.halted = false, m_->halted = false;
    if (m_->sect_on && !err && !err2)   // TRUSS_PROFILE_SECTIONS: how balanced the stages are
        std::fprintf(stderr, "layer split prompt: %d tokens, first stage done at %.2f s, last at %.2f s\n", n, front_s,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (err) std::rethrow_exception(err);
    if (err2) std::rethrow_exception(err2);
    pos_ += n;
}

void Forward::verify(const int32_t * tokens, int T)
{
    if (!m_->spec_rows || T > m_->spec_rows)
        throw std::runtime_error("qwen4exp::Forward: verify() of " + std::to_string(T) + " rows (spec_rows " +
                                 std::to_string(m_->spec_rows) + ")");
    each_stage([&](Impl & m) { m.run_chunk(tokens, pos_, T, nullptr, true); });
}

void Forward::accept(int n)
{
    // the last stage first: its rollback and MTP commit are queued (no host wait) and run on its device while this
    // thread launches the other stages' rollbacks (~100 small launches each), off the draft's critical path
    m_->accept(n);
    for (auto & p : front_) {
        DeviceGuard g(p->device);
        p->accept(n);
    }
    pos_ += n;
}

int Forward::draft(int32_t next, int n, int32_t * out) { return m_->draft(next, pos_, n, out); }

void Forward::set_draft_sampling(const sampling::SampleParams * p, uint64_t counter, float * q_full)
{
    m_->ds_p = p, m_->ds_counter = counter, m_->ds_q = q_full;
}

void Forward::checkpoint()
{
    each_stage([this](Impl & m) { m.checkpoint(pos_); });
}

int Forward::restore()
{
    int p = -1;
    each_stage([&](Impl & m) { p = m.restore(); });
    if (p >= 0) pos_ = p;
    return p;
}

void apply_env(ForwardOptions & o)
{
    auto f = [](const char * k, auto & v) {
        if (const char * e = std::getenv(k)) v = (std::remove_reference_t<decltype(v)>) std::atof(e);
    };
    f("TRUSS_CPU_SHARE", o.cpu_share);
    f("TRUSS_CPU_THREADS", o.cpu_threads);
    f("TRUSS_CPU_DYNAMIC", o.cpu_dynamic);     // Strata's split
    f("TRUSS_CPU_TRELLIS", o.cpu_trellis);     // CPU tier from the pack's own bytes
    f("TRUSS_PCIE_GBPS", o.pcie_gbps);
    f("TRUSS_PCIE_FRAC", o.pcie_frac);         // fixed PCIe share of the misses
    f("TRUSS_ADAPT_EVERY", o.adapt_every);     // adaptive tier
    f("TRUSS_ADAPT_SWAPS", o.adapt_swaps);
    f("TRUSS_ADAPT_MIN", o.adapt_min);
    f("TRUSS_ADAPT_DECAY", o.adapt_decay);
    f("TRUSS_HINT_K", o.hint_k);               // pre-gated prefetch width
    f("TRUSS_HINT_MIN_ROWS", o.hint_min_rows);
    f("TRUSS_ADMIT_IDLE", o.admit_idle);       // admission on the idle link, experts per pass
    f("TRUSS_KV_LEND", o.kv_lend);             // E6: the KV caches' unused tail lent to the expert ring
    f("TRUSS_ADMIT_MIN", o.admit_min);
    f("TRUSS_ADMIT_DECAY", o.admit_decay);
    f("TRUSS_PREFILL_ROWS", o.prefill_rows);
    f("TRUSS_KV_INT8", o.kv_int8);             // Strata's int8 KV
    if (const char * e = std::getenv("TRUSS_ACT"))   // dense activations: fp16 (W8A16, KL 0.040) or anything else = q8_1 (0.065). The server defaults to fp16 (truss_c.cu)
        o.act = std::string(e) == "fp16" ? Activations::FP16 : Activations::Q8_1;
    f("TRUSS_PLAN_ALPHA", o.plan_alpha);       // hot-set ranking: usage / bytes^alpha
    f("TRUSS_LAYER_SPLIT", o.layer_split);             // two devices: layers [0, N) on TRUSS_LAYER_SPLIT_DEVICE
    f("TRUSS_LAYER_SPLIT_DEVICE", o.layer_split_device);
    f("TRUSS_LAYER_SPLIT_FRAC", o.layer_split_frac);
    f("TRUSS_SPLIT_ROWS", o.split_rows);       // E7: short prompt chunks split cold experts CPU / PCIe
    f("TRUSS_SPLIT_T", o.split_t);
    f("TRUSS_SPLIT_MS_ROW", o.split_ms_row);
    if (const char * e = std::getenv("TRUSS_PLE_DIRECT"); e && !std::atoi(e)) o.ple_file = nullptr;   // A/B: the mapping
    if (const char * e = std::getenv("TRUSS_RING_GB")) o.ring_bytes_override = (size_t) (std::atof(e) * (1ull << 30));
    // a fixed VRAM budget for the routed experts instead of "all free memory": a test beside another GPU job (#118)
    if (const char * e = std::getenv("TRUSS_EXPERT_GB")) o.expert_budget = (size_t) (std::atof(e) * (1ull << 30));
}

}  // namespace truss::qwen4exp
