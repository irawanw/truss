#include "model/qwen4exp/forward.h"

#include "core/cublas_check.h"
#include "core/cuda_check.h"
#include "core/device_tensors.h"
#include "core/scratch.h"
#include "cpu/expert_q4.h"
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

}  // namespace

struct Forward::Impl {
    const Config & c;
    const Weights & w;
    const int n_ctx, max_chunk, prefill_rows;
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
        std::unique_ptr<cpu::ExpertPool> pool;
        float * x = nullptr, * w = nullptr, * y = nullptr;   // pinned [8][d], [8][k], [8][d]
        int * ids = nullptr;                                 // pinned [8][k] routing with CPU slots as -1
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
        int seq = 0;                                                  // host: the last seq enqueued
    };
    std::vector<Bell> bells;
    struct Job {
        int layer, T, seq;
        bool hint, cpu;
    };
    std::thread driver;
    std::mutex dq_mu;
    std::condition_variable dq_cv;
    std::deque<Job> jobs;
    std::chrono::steady_clock::time_point drv_t[3];   // driver thread only: doorbell seen, split done, CPU started
    double drv_ms[3] = { 0, 0, 0 };                    // summed: split, CPU start, copies + plan (driver_ms)
    double ple_host_ms = 0;                            // host time of the PLE row hash + gather + fp16 convert
    // TRUSS_ROUTE_TRACE=<file>: every decode layer's routing (driver thread), for offline cache-policy replays.
    // Header: int32 n_store, n_expert, K, then per (layer, expert) int32 bytes and uint8 hot. Records: int32 layer, T,
    // ids [T * K]; layer -1 (T 0) marks a sequence reset.
    FILE * route_trace = nullptr;
    std::unique_ptr<PleReader> ple_reader;           // Options::ple_file: rows read with O_DIRECT, not page faults
    half * ple_stage = nullptr;                        // pinned [rows][E]: the gathered rows for the device copy
    size_t ple_stage_n = 0;
    cudaEvent_t ple_copied = nullptr;                  // the last copy out of ple_stage has been read
    long ple_calls = 0, ple_rows_n = 0;                // (ple_host_ms(); the rows are table rows read)
    long drv_n = 0;
    std::atomic<long> jobs_pushed{ 0 };   // the driver spins on this before it sleeps on dq_cv (Strata's host spins)
    bool driver_stop = false;
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
    dense::Q8Matrix draft_head{};                        // the output matrix's draft_vocab rows (Options::draft_vocab)
    int * draft_map = nullptr;                           // device [draft_head.out]: row -> token id
    float draft_min_p = 0.f;
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

    Impl(const Config & cc, const Weights & ww, int nc, int mc, const Options & o)
        : c(cc), w(ww), n_ctx(nc), max_chunk(mc),
          prefill_rows(prefill_rows_of(mc, o.prefill_rows)),
          small_scratch(scratch_bytes(cc, std::min(mc, FETCH_ROWS), nc, o.mtp)),
          spec_rows(o.spec_rows), act(o.act), hint_k(o.hint_k)
    {
        check_shapes();
        mtp = o.mtp;
        draft_min_p = o.draft_min_p;
        TRUSS_CUDA(cudaStreamCreate(&s));
        TRUSS_CUBLAS(cublasCreate(&blas));
        upload();
        if (o.after_upload) o.after_upload();
        const int small_rows = std::min(max_chunk, FETCH_ROWS);
        mtp = o.mtp;
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
        TRUSS_CUDA(cudaMallocHost(&ids_host, sizeof(int) * 2 * FETCH_ROWS * MoeShape::TOPK));   // routing + prediction
        st.resize(c.n_layer);
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
        for (int l = 0; l < c.n_layer; ++l) {
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
        std::vector<runtime::ExpertLayer> tables;
        for (const Layer & L : w.layers) tables.push_back({ &L.moe.gate, &L.moe.up, &L.moe.down });
        if (mtp) tables.push_back({ &mtp->layer.moe.gate, &mtp->layer.moe.up, &mtp->layer.moe.down });
        n_store = (int) tables.size();
        std::vector<float> usage = o.expert_usage;
        if (mtp && usage.size() == (size_t) c.n_layer * c.n_expert) {
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
            expert_budget = free_b - MARGIN;
        }
        runtime::ExpertStore::Sizes z;
        z.ring_bytes = o.ring_bytes_override ? o.ring_bytes_override : o.ring_bytes;
        z.plan_alpha = o.plan_alpha;
        if (prefill_rows > FETCH_ROWS) z.stream_extra = big_bytes();
        const runtime::ExpertStore::HotSet hot = runtime::ExpertStore::plan(tables, expert_budget, z, usage);
        for (const auto & h : hot) n_hot += (int) std::count(h.begin(), h.end(), 1);
        experts = std::make_unique<runtime::ExpertStore>(tables, hot, z);
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
        if (!o.cpu_dir.empty()) load_cpu_tier(o, hot);
        else if (o.cpu_trellis) load_cpu_tier_trellis(o, hot);
        use_doorbell = o.doorbell;
        cpu_dynamic = o.cpu_dynamic && cpu_tier && use_doorbell;
        pcie_ms_byte = 1.0 / (o.pcie_gbps * 1e6);
        pcie_frac = o.pcie_frac;
        adapt_every = o.adapt_every, adapt_swaps = o.adapt_swaps;
        adapt_min = o.adapt_min, adapt_decay = o.adapt_decay;
        if (adapt_every > 0) route_usage.assign((size_t) n_store * c.n_expert, 0.f);
        if (const char * rt = std::getenv("TRUSS_ROUTE_TRACE")) {
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
        if (mtp) mtp_calib_open(prefill_rows);
        if (getenv("TRUSS_PROFILE_SECTIONS")) {
            sect_on = true;
            sect_ev.resize(n_store);
            sect_rec.assign(n_store, 0);
            sect_bell.assign(n_store, 0);
            for (auto & a : sect_ev)
                for (cudaEvent_t & e : a) TRUSS_CUDA(cudaEventCreate(&e));
        }
    }

    ~Impl()
    {
        if (route_trace) std::fclose(route_trace);
        if (ple_stage) cudaFreeHost(ple_stage);
        if (ple_copied) cudaEventDestroy(ple_copied);
        if (driver.joinable()) {
            {
                std::lock_guard<std::mutex> g(dq_mu);
                driver_stop = true;
            }
            dq_cv.notify_all();
            driver.join();
        }
        for (Bell & b : bells)
            for (void * p : { (void *) b.ids, (void *) b.pred, (void *) b.mids, (void *) b.w, (void *) b.x, (void *) b.y,
                              (void *) b.doorbell, (void *) b.cpu_done, (void *) b.plan, (void *) b.need_copy })
                if (p) cudaFreeHost(p);
        for (void * p : owned) cudaFree(p);
        if (ids_host) cudaFreeHost(ids_host);
        if (draft_host) cudaFreeHost(draft_host);
        if (draft_prob_host) cudaFreeHost(draft_prob_host);
        if (embd_q) cudaFreeHost(embd_q);
        if (cpu_tier)
            for (void * p : { (void *) cpu_tier->x, (void *) cpu_tier->w, (void *) cpu_tier->y, (void *) cpu_tier->ids })
                if (p) cudaFreeHost(p);
        if (embd_d) cudaFreeHost(embd_d);
        if (calib_host) cudaFreeHost(calib_host);
        if (mtp_calib_f) std::fclose(mtp_calib_f);
        for (auto & a : sect_ev)
            for (cudaEvent_t e : a)
                if (e) cudaEventDestroy(e);
        if (blas) cublasDestroy(blas);
        if (s) cudaStreamDestroy(s);
    }

    static size_t align(size_t x) { return (x + 255) / 256 * 256; }

    void setup_doorbells()
    {
        const int K = c.n_expert_used, R = cpu::MAX_ROWS;
        bells.resize(n_store);
        auto mapped = [](void ** p, size_t bytes) {
            TRUSS_CUDA(cudaHostAlloc(p, bytes, cudaHostAllocMapped));
            std::memset(*p, 0, bytes);
        };
        for (Bell & b : bells) {
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
            b.d_mids = alloc<int>((size_t) R * K);
        }
        driver = std::thread([this] { drive(); });
    }

    // The driver: for each queued decode FFN, wait for its doorbell, split the routed experts (CPU tier / GPU),
    // start the CPU work, issue the expert copies and the go word, then publish the CPU result.
    void drive()
    {
        long taken = 0;
        for (;;) {
            Job j;
            // spin first: the next layer's job arrives within ~0.5 ms during a pass, and a condvar wake costs more
            for (int i = 0; i < 200000 && jobs_pushed.load(std::memory_order_acquire) <= taken; ++i) _mm_pause();
            {
                std::unique_lock<std::mutex> g(dq_mu);
                dq_cv.wait(g, [&] { return driver_stop || !jobs.empty(); });
                ++taken;
                if (driver_stop && jobs.empty()) return;
                j = jobs.front();
                jobs.pop_front();
            }
            Bell & b = bells[j.layer];
            while (*(volatile int *) b.doorbell < j.seq) _mm_pause();
            std::atomic_thread_fence(std::memory_order_acquire);
            drv_t[0] = std::chrono::steady_clock::now();
            try {
                serve(j, b);
            } catch (...) {   // keep the GPU moving (its output is garbage now); the caller rethrows at its next sync
                if (!driver_error) driver_error = std::current_exception();
                experts->signal(j.layer, b.go, j.seq);
                *(volatile int *) b.need_copy = j.seq;
                std::atomic_thread_fence(std::memory_order_release);
                *(volatile int *) b.plan = j.seq;
                std::atomic_thread_fence(std::memory_order_release);
                *(volatile int *) b.cpu_done = j.seq;
            }
        }
    }

    void serve(const Job & j, Bell & b)
    {
        const int K = c.n_expert_used, l = j.layer, n = j.T * K;
        CpuTier * ct = j.cpu ? cpu_tier.get() : nullptr;
        if (!route_usage.empty())
            for (int i = 0; i < n; ++i) route_usage[(size_t) l * c.n_expert + b.ids[i]] += 1.f;
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
                if (!experts->on_device(l, e)) {
                    if (ct->is_cpu[l][e]) elig.push_back(nd);
                    else pcie_bytes += (double) experts->bytes_of(l, e);
                }
                ++nd;
            }
            // m = how many eligible misses (the last ones) go over PCIe: minimize max(CPU time, copy time)
            const int ne = (int) elig.size();
            int best_m = 0;
            double best = 1e30, tail = 0;
            for (int m = 0; m <= ne && pcie_frac < 0; ++m) {
                if (m > 0) tail += (double) experts->bytes_of(l, b.ids[dist[elig[ne - m]]]);
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
        if (ct) ct->pool->start(b.x, j.T, slots, b.y);   // CPU first: it runs while the copies are issued
        drv_t[2] = std::chrono::steady_clock::now();
        // the plan (masked ids in mapped b.mids) goes to the GPU through mapped words, as Strata's: a layer that needs
        // no copies is not ordered behind the copy stream (earlier layers' hints and meta uploads); one that copies
        // also waits for go, which the copy stream writes behind its copies
        const bool copied = experts->fetch(l, gpu_ids.data(), (int) gpu_ids.size(), nullptr);
        if (copied) experts->signal(l, b.go, j.seq);   // go before the hints: this layer must not wait for the next one's copies
        *(volatile int *) b.need_copy = copied ? j.seq : 0;
        std::atomic_thread_fence(std::memory_order_release);
        *(volatile int *) b.plan = j.seq;
        {   // driver timing (Forward::driver_ms): split, CPU start, copies + plan
            const auto t3 = std::chrono::steady_clock::now();
            auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            drv_ms[0] += ms(drv_t[0], drv_t[1]), drv_ms[1] += ms(drv_t[1], drv_t[2]), drv_ms[2] += ms(drv_t[2], t3);
            ++drv_n;
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
            experts->prefetch_hint(l, h.data(), (int) h.size());
        }
        if (ct) {
            ct->pool->wait();
            if (cpu_dynamic && n_cpu > 0)   // the pool's own time (start to last item), not this thread's
                fit_cpu_cost(n_cpu, ct->pool->last_call_ms());
            std::atomic_thread_fence(std::memory_order_release);
            *(volatile int *) b.cpu_done = j.seq;
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
    void load_cpu_tier_trellis(const Options & o, const runtime::ExpertStore::HotSet & hot)
    {
        require(!o.expert_usage.empty() || o.cpu_dynamic, "the static CPU tier needs expert_usage (it picks the rarest)");
        auto t = std::make_unique<CpuTier>();
        t->trellis = true;
        const int E = c.n_expert;
        require(c.d_model == cpu::D_MODEL && c.d_ff_exp == cpu::D_FF, "CPU tier shape");
        t->is_cpu.assign(c.n_layer, std::vector<uint8_t>(E, 0));
        t->tview.resize(c.n_layer);
        size_t n_cpu = 0;
        for (int l = 0; l < c.n_layer; ++l) {
            const Moe & mo = w.layers[l].moe;
            const formats::ExpertTable * tab[3] = { &mo.gate, &mo.up, &mo.down };
            t->tview[l].resize(E);
            std::vector<int> cold;
            double mass = 0;
            for (int e = 0; e < E; ++e)
                if (!hot[l][e]) {
                    cold.push_back(e);
                    if (!o.expert_usage.empty()) mass += o.expert_usage[(size_t) l * E + e];
                    cpu::TrellisMat m[3];
                    for (int p = 0; p < 3; ++p) {
                        const formats::ExpertTable & T = *tab[p];
                        require(T.k[e] >= 1 && T.k[e] <= 6, "CPU tier: trellis K out of 1..6");
                        m[p] = { reinterpret_cast<const uint32_t *>(experts->host_part(l, e, p)), T.k[e], (int) T.in,
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
                     (double) n_cpu / c.n_layer, o.cpu_dynamic ? "dynamic split" : "static set");
        t->pool = std::make_unique<cpu::ExpertPool>(o.cpu_threads);
        TRUSS_CUDA(cudaMallocHost(&t->x, sizeof(float) * FETCH_ROWS * c.d_model));
        TRUSS_CUDA(cudaMallocHost(&t->w, sizeof(float) * FETCH_ROWS * c.n_expert_used));
        TRUSS_CUDA(cudaMallocHost(&t->y, sizeof(float) * cpu::MAX_ROWS * c.d_model));
        TRUSS_CUDA(cudaMallocHost(&t->ids, sizeof(int) * cpu::MAX_ROWS * c.n_expert_used));
        cpu_tier = std::move(t);
    }

    // Per layer, the non-resident experts in ascending usage until they hold cpu_share of the layer's non-resident
    // routing mass; their q4s copies are read from cpu_dir/L<nn>.q4s. With cpu_dir/L<nn>.ids (int32 expert ids in
    // file order) the file holds only those experts (an NVMe subset of the full archive, flashnext_truss_cpu_subset.py)
    // and the CPU set is chosen among them.
    void load_cpu_tier(const Options & o, const runtime::ExpertStore::HotSet & hot)
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
        for (int l = 0; l < c.n_layer; ++l) {
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
                if (!hot[l][e]) {
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
                    for (int l = w; l < c.n_layer; l += nth) {
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
                         (double) n_cpu / c.n_layer, t->bytes / 1e9);
        }
        t->pool = std::make_unique<cpu::ExpertPool>(o.cpu_threads);
        TRUSS_CUDA(cudaMallocHost(&t->x, sizeof(float) * FETCH_ROWS * c.d_model));
        TRUSS_CUDA(cudaMallocHost(&t->w, sizeof(float) * FETCH_ROWS * K));
        TRUSS_CUDA(cudaMallocHost(&t->y, sizeof(float) * cpu::MAX_ROWS * c.d_model));
        TRUSS_CUDA(cudaMallocHost(&t->ids, sizeof(int) * cpu::MAX_ROWS * K));
        cpu_tier = std::move(t);
    }

    // bytes of the prompt path's buffers (scratch, moe::prefill workspace, residual) at prefill_rows rows
    size_t big_bytes() const
    {
        return align(scratch_bytes(c, prefill_rows, n_ctx)) + align(moe::prefill_workspace_bytes<MoeShape>(prefill_rows)) +
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
        const size_t ffn = (size_t) c.n_expert * 4 + c.n_expert_used * 8 + dm * 8 + (size_t) c.d_ff_shexp * 10 + 4;
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

    void upload()
    {
        std::vector<const gguf::Tensor *> plain, q8_list;
        auto add = [&](T t) {
            if (!t) return;
            (t->type == gguf::Type::Q8_0 ? q8_list : plain).push_back(t);
        };
        auto add_hc = [&](const HyperConnection & h) { add(h.norm), add(h.down), add(h.up), add(h.inject); };
        add(w.token_embd), add(w.output);
        add_hc(w.hc_head);
        std::vector<const Layer *> layers;
        for (const Layer & L : w.layers) layers.push_back(&L);
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
        dev = std::make_unique<DeviceTensors>(plain);
        for (T t : plain) release_tensor(t);

        // The token embedding is read one row per token: it lives in pinned host memory and the gather reads it over
        // PCIe (2.5 KB per token), leaving its 0.68 GB of VRAM to the expert cache (TRACKER #60). Tied: stays.
        const bool host_embd = w.token_embd != w.output;
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
        if (act == Activations::FP16) dense::q8_gemm_a16(W, a.x, rows, y, w16, blas, s);
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
        if (act == Activations::Q8_1 && rows <= dense::GEMV_ROWS && ps.size() > 1 && ps.size() <= (size_t) dense::MULTI_MAX) {
            const dense::Q8Matrix * W[dense::MULTI_MAX];
            float * y[dense::MULTI_MAX];
            for (size_t i = 0; i < ps.size(); ++i) W[i] = &q8.at(ps[i].first), y[i] = ps[i].second;
            dense::q8_gemv_multi(W, y, (int) ps.size(), a.q, a.d, rows, s);
            return;
        }
        for (const Proj & p : ps) lin(p.first, a, rows, p.second);
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

    // hyper-connection mix: mixed [T][d] (fp32 and fp16), inject [T][hc] when h.inject
    void hc_mix(const HyperConnection & h, const float * res, int T, float * mixed, half * mixed16, float * inject)
    {
        const size_t m = sc->mark();
        half * xn16 = sc->alloc<half>((size_t) T * c.hc_dim());
        float * rstd = sc->alloc((size_t) T * c.hc);
        float * lo = sc->alloc((size_t) T * c.hc_rank);
        half * lo16 = sc->alloc<half>((size_t) T * c.hc_rank);
        float * gate = sc->alloc((size_t) T * c.hc_dim());
        hc::norm(res, f32(h.norm), T, c.hc, c.d_model, c.rms_eps, xn16, rstd, s);
        const Act xa = quant(xn16, T, c.hc_dim());
        if (h.inject) lin_multi({ { h.down, lo }, { h.inject, inject } }, xa, T);   // inject reads the same xa
        else lin(h.down, xa, T, lo);
        hc::silu(lo, T * c.hc_rank, 1.f / c.hc, lo16, s);
        lin(h.up, lo16, T, gate);
        hc::collapse(res, rstd, f32(h.norm), gate, T, c.hc, c.d_model, mixed, mixed16, s);
        sc->release(m);
    }

    void ple(const Ple & p, LayerState & L, const int32_t * tokens, int T, bool tentative)
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
            const int tk = ple_reader->issue(rows.data() + (size_t) n_prev * H, (size_t) T * H);
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

    // stream: this chunk streams whole layers (prompt chunk, stream mode); else it fetches (decode, windows)
    void ffn(int l, const Moe & mo, const float * in, const half * in16, int T, float * out, bool stream)
    {
        const size_t m = sc->mark();
        // only a run_chunk() layer brackets all five sections; the MTP block's FFN (draft steps, commits) does not
        const bool sect = sect_on && l < c.n_layer;
        if (sect) sect_record(l, 3);   // router + routed MoE (its copies and the CPU tier's start are inside)
        const int E = c.n_expert, K = c.n_expert_used, F = c.d_ff_shexp, dm = c.d_model;
        float * logits = sc->alloc((size_t) T * E), * wts = sc->alloc((size_t) T * K);
        int * ids = sc->alloc<int>((size_t) T * K);
        float * routed = sc->alloc((size_t) T * dm), * y = sc->alloc((size_t) T * dm);
        float * g = sc->alloc((size_t) T * F), * u = sc->alloc((size_t) T * F), * sg = sc->alloc(T);
        half * mid16 = sc->alloc<half>((size_t) T * F);
        // the router, the next layer's router on this input (pre-gating hint, TRACKER #58) and the shared-expert
        // gate read the same fp32 rows: one launch at decode sizes
        const bool hint = !stream && hint_k > 0 && l + 1 < c.n_layer;
        float * pl = nullptr;
        if (hint) pl = sc->alloc((size_t) T * E);
        {
            std::vector<Proj> ps{ { mo.router, logits }, { mo.shexp_gate_inp, sg } };
            if (hint) ps.push_back({ w.layers[l + 1].moe.router, pl });
            lin32_multi(ps, in, T);
        }
        ffn::route(logits, T, E, K, ids, wts, s);
        if (route_counts) ffn::count(ids, T * K, route_counts + (size_t) l * E, s);
        if (!stream && use_doorbell && T <= cpu::MAX_ROWS) {   // no host sync: the driver thread serves this FFN
            int * pred = nullptr;
            if (hint) {
                float * pw = sc->alloc((size_t) T * K);
                pred = sc->alloc<int>((size_t) T * K);
                ffn::route(pl, T, E, K, pred, pw, s);
            }
            const bool cpu_on = cpu_tier && l < c.n_layer;
            Bell & b = bells[l];
            const int seq = ++b.seq;
            runtime::publish(ids, pred, wts, T * K, in, T * dm, b.ids, b.pred, b.w, b.x, b.doorbell, seq, s);
            if (sect) sect_record(l, 7), sect_bell[l] = 1;
            {
                std::lock_guard<std::mutex> g(dq_mu);
                jobs.push_back({ l, T, seq, hint, cpu_on });
                jobs_pushed.fetch_add(1, std::memory_order_release);
            }
            dq_cv.notify_one();
            // the driver's plan (mapped), the layer's copies when it queued any, the masked ids into d_mids
            if (sect) {   // profiling: the plan and the copies waited for apart (two launches instead of one)
                runtime::spin_until(b.plan, seq, s);
                sect_record(l, 9);
            }
            runtime::wait_plan(b.plan, b.need_copy, b.go, seq, b.mids, b.d_mids, cpu_on ? T * K : 0, s);
            if (sect) sect_record(l, 8);
            moe::window<MoeShape>(experts->weights(l), in, cpu_on ? b.d_mids : ids, wts, T, routed, window_ws, s);
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
                    ct->pool->start(ct->x, T, slots, ct->y);
                }
            } else {
                gpu_ids.assign(ids_host, ids_host + T * K);
            }
            experts->fetch(l, gpu_ids.data(), (int) gpu_ids.size(), s);
            if (hint) {   // each row's first hint_k guesses (ids are in descending probability), GPU experts only
                std::vector<int> h;
                const CpuTier * nt = l + 1 < c.n_layer ? cpu_tier.get() : nullptr;
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < std::min(K, hint_k); ++i) {
                        const int e = ids_host[T * K + t * K + i];
                        if (!nt || !nt->is_cpu[l + 1][e]) h.push_back(e);
                    }
                experts->prefetch_hint(l, h.data(), (int) h.size());
            }
            experts->acquire(l, s);
            if (T <= moe::MAX_ROWS) moe::window<MoeShape>(experts->weights(l), in, k_ids, wts, T, routed, window_ws, s);
            else moe::prefill<MoeShape>(experts->weights(l), in, ids, wts, T, routed, cur.moe_ws, cur.moe_rows, s);
            experts->release(l, s);
            cpu_rows = slots.empty() ? 0 : T;
        } else {                // whole layers stream ahead (prefetch(l + 2) while this layer computes)
            experts->acquire(l, s);
            moe::prefill<MoeShape>(experts->weights(l), in, ids, wts, T, routed, cur.moe_ws, cur.moe_rows, s);
            experts->release(l, s);
            if (l + 2 < n_store) experts->prefetch(l + 2);   // the MTP block streams after the last layer
        }
        if (sect) sect_record(l, 4);   // shared expert + the CPU tier's join (its spin lands in this section)
        const Act ia = quant(in16, T, c.d_model);
        lin_multi({ { mo.shexp_gate, g }, { mo.shexp_up, u } }, ia, T);
        ffn::swiglu(g, u, T * F, mid16, s);
        lin(mo.shexp_down, mid16, T, y);   // sg (shared-expert gate) came with the router
        if (sect) sect_record(l, 5);   // section 5: the CPU tier's join (its spin is the section)
        if (cpu_rows < 0) {   // doorbell path: wait for the CPU tier's rows (mapped) and add them
            runtime::spin_until(bells[l].cpu_done, bells[l].seq, s);
            runtime::add_mapped(routed, bells[l].y, -cpu_rows * dm, s);
            cpu_rows = 0;
        }
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
        for (int l = 0; l < c.n_layer; ++l) {
            const LayerState & L = st[l];
            if (L.state()) TRUSS_CUDA(cudaMemsetAsync(L.state(), 0, sizeof(float) * c.ssm_v_heads * c.ssm_state * c.ssm_state, s));
            if (L.conv) TRUSS_CUDA(cudaMemsetAsync(L.conv, 0, sizeof(float) * (c.ssm_conv - 1) * c.conv_dim(), s));
            if (L.ple_hist)
                TRUSS_CUDA(cudaMemsetAsync(L.ple_hist, 0, sizeof(float) * (c.ple_conv - 1) * c.ple_ngram * c.hc_dim(), s));
        }
        tail.clear();
        last_T = 0;
        has_pending = false;
        vw.open = false;
    }

    // One chunk of the target model at positions pos0 .. pos0 + T - 1. tentative: a verify window (accept() commits).
    void run_chunk(const int32_t * tokens, int pos0, int T, const LayerHook & hook, bool tentative)
    {
        require(T > 0 && T <= max_chunk, "chunk of " + std::to_string(T) + " tokens (max " + std::to_string(max_chunk) + ")");
        require(pos0 + T <= n_ctx, "sequence longer than n_ctx");
        require(!vw.open, "run()/verify() before accept() of the previous verify()");
        check_driver();
        const bool stream = !fetch_mode(T);
        if (stream)
            require(T <= prefill_rows, "prompt chunk of " + std::to_string(T) + " tokens over prefill_rows (" +
                                            std::to_string(prefill_rows) + "); raise Options::prefill_rows");
        if (!stream) {
            use(small);
            experts->begin_ring();
            if (adapt_every > 0 && ++decode_passes % adapt_every == 0) adapt();
        } else {   // the ring becomes the stream slots and this chunk's buffers
            experts->begin_stream(s);
            use(big);
        }
        sc->reset();
        int * ids = sc->alloc<int>(T);
        float * emb = sc->alloc((size_t) T * c.d_model);
        TRUSS_CUDA(cudaMemcpyAsync(ids, tokens, (size_t) T * 4, cudaMemcpyHostToDevice, s));
        dense::q8_rows(q8.at(w.token_embd), ids, T, emb, s);
        hc::expand(emb, T, c.hc, c.d_model, res, s);
        if (stream)
            for (int l = 0; l < std::min(2, n_store); ++l) experts->prefetch(l);
        const size_t base = sc->mark();
        for (int l = 0; l < c.n_layer; ++l) {
            sc->release(base);
            const Layer & L = w.layers[l];
            float * mixed = sc->alloc((size_t) T * c.d_model), * out = sc->alloc((size_t) T * c.d_model);
            float * inject = sc->alloc((size_t) T * c.hc);
            half * mixed16 = sc->alloc<half>((size_t) T * c.d_model);
            sect_record(l, 0);
            if (c.is_ple(l)) ple(L.ple, st[l], tokens, T, tentative);
            hc_mix(L.hc_attn, res, T, mixed, mixed16, inject);
            sect_record(l, 1);
            if (L.mixer == Mixer::GDN) gdn(L.gdn, st[l], mixed16, T, out, tentative);
            else dsa(L.dsa, st[l], mixed16, pos0, T, out, tentative);
            sect_record(l, 2);
            hc::combine(res, out, inject, T, c.hc, c.d_model, s);
            hc_mix(L.hc_ffn, res, T, mixed, mixed16, inject);
            ffn(l, L.moe, mixed, mixed16, T, out, stream);   // records 3 (routed MoE), 4 (shared), 5 (cpu join)
            sect_record(l, 6);
            hc::combine(res, out, inject, T, c.hc, c.d_model, s);
            if (hook) {
                TRUSS_CUDA(cudaStreamSynchronize(s));
                hook(l, res, T);
            }
        }
        sect_flush();
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
        for (int l = 0; l < c.n_layer; ++l) {
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
        if (mtp) mtp_commit(vw.tokens.data(), vw.pos0, n, false);
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
    std::vector<std::array<cudaEvent_t, 10>> sect_ev;   // [layer][0..6] section bounds; 7, 9, 8: inside routed MoE (doorbell)
    mutable std::vector<uint8_t> sect_rec;             // layers recorded since the last flush
    mutable double sect_ms[6] = { 0, 0, 0, 0, 0, 0 };
    // routed MoE on the doorbell path split at events 7 (doorbell raised), 9 (plan received) and 8 (copies landed,
    // TRACKER #77): router + publish, wait for the host's plan, wait for the layer's copies, expert kernel
    mutable std::vector<uint8_t> sect_bell;
    mutable double sect_moe[4] = { 0, 0, 0, 0 };
    void sect_record(int layer, int i)
    {
        if (!sect_on) return;
        TRUSS_CUDA(cudaEventRecord(sect_ev[layer][i], s));
        sect_rec[layer] = 1;
    }
    // Read the events back after the whole chunk (one sync, no per-layer stall) and add the five sections. Only
    // the layers that ran are read: draft()'s MTP steps record only some.
    void sect_flush()
    {
        if (!sect_on) return;
        for (int l = 0; l < n_store; ++l) {
            if (!sect_rec[l]) continue;
            TRUSS_CUDA(cudaEventSynchronize(sect_ev[l][6]));
            for (int i = 1; i <= 6; ++i) {
                float ms = 0.f;
                TRUSS_CUDA(cudaEventElapsedTime(&ms, sect_ev[l][i - 1], sect_ev[l][i]));
                sect_ms[i - 1] += ms;
            }
            if (sect_bell[l]) {
                const int at[5] = { 3, 7, 9, 8, 4 };
                for (int i = 0; i < 4; ++i) {
                    float ms = 0.f;
                    TRUSS_CUDA(cudaEventElapsedTime(&ms, sect_ev[l][at[i]], sect_ev[l][at[i + 1]]));
                    sect_moe[i] += ms;
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
        const int skip = has_pending ? 0 : 1;   // only position 0 lacks a pending state
        const int n = T - skip;
        if (n > 0) {
            float * mh = cur.mh;
            if (!skip) copy(mh, pending_h, row);
            if (T > 1) copy(mh + (size_t) (1 - skip) * row, res, (size_t) (T - 1) * row);
            const size_t m = sc->mark();
            int * d_ids = sc->alloc<int>(n);
            TRUSS_CUDA(cudaMemcpyAsync(d_ids, tokens + skip, sizeof(int32_t) * n, cudaMemcpyHostToDevice, s));
            mtp_rows(mh, d_ids, pos0 + skip, n, stream, false, true);
            TRUSS_CUDA(cudaStreamSynchronize(s));   // tokens may be a host temporary
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
        use(small);
        experts->begin_ring();
        sc->reset();
        const size_t row = c.hc_dim();
        copy(mtp_partial_snap, mst.idx_partial, (size_t) (DsaShape::RATIO - 1) * c.idx_head_dim);
        draft_host[0] = next;
        TRUSS_CUDA(cudaMemcpyAsync(draft_ids, draft_host, sizeof(int32_t), cudaMemcpyHostToDevice, s));
        copy(cur.mh, pending_h, row);
        const int nv = draft_head.q ? draft_head.out : c.n_vocab;
        int made = 0;
        for (int i = 0; i < n; ++i) {
            mtp_rows(cur.mh, draft_ids + i, pos + i, 1, false, true);
            sampling::argmax_prob(mtp_logits, nv, draft_map, draft_ids + i + 1, draft_prob + i, s);
            if (draft_min_p > 0.f) {   // an unsure guess does not enter the window, and ends the chain
                TRUSS_CUDA(cudaMemcpyAsync(draft_prob_host + i, draft_prob + i, sizeof(float), cudaMemcpyDeviceToHost, s));
                TRUSS_CUDA(cudaStreamSynchronize(s));
                if (draft_prob_host[i] < draft_min_p) break;
            }
            ++made;
            if (i + 1 < n) copy(cur.mh, cur.mres, row);
        }
        copy(mst.idx_partial, mtp_partial_snap, (size_t) (DsaShape::RATIO - 1) * c.idx_head_dim);
        if (made) TRUSS_CUDA(cudaMemcpyAsync(draft_host, draft_ids + 1, sizeof(int32_t) * made, cudaMemcpyDeviceToHost, s));
        TRUSS_CUDA(cudaStreamSynchronize(s));
        std::copy(draft_host, draft_host + made, out);
        return made;
    }
};

Forward::Forward(const Config & c, const Weights & w, int n_ctx, int max_chunk, const Options & o)
    : m_(std::make_unique<Impl>(c, w, n_ctx, max_chunk, o))
{
}

Forward::~Forward() = default;

cudaStream_t Forward::stream() const { return m_->s; }

int Forward::hot_experts() const { return m_->n_hot; }
int Forward::fetch_rows() { return Impl::FETCH_ROWS; }

size_t Forward::cold_bytes() const { return m_->experts->cold_bytes(); }

std::pair<long long, long> Forward::cpu_stats(bool reset)
{
    if (!m_->cpu_tier || !m_->cpu_tier->pool) return { 0, 0 };
    auto * p = m_->cpu_tier->pool.get();
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
    for (int i = 0; i < 6; ++i) out[i] = m_->sect_ms[i];
    if (reset) for (int i = 0; i < 6; ++i) m_->sect_ms[i] = 0;
}

void Forward::driver_ms(double out[3], long & n, bool reset) const
{
    for (int i = 0; i < 3; ++i) out[i] = m_->drv_ms[i];
    n = m_->drv_n;
    if (reset) {
        for (int i = 0; i < 3; ++i) m_->drv_ms[i] = 0;
        m_->drv_n = 0;
    }
}

long Forward::adapt_admitted() const { return m_->experts->admitted(); }

void Forward::ple_host_ms(double & ms, long & calls, long & rows, bool reset) const
{
    ms = m_->ple_host_ms, calls = m_->ple_calls, rows = m_->ple_rows_n;
    if (reset) m_->ple_host_ms = 0, m_->ple_calls = 0, m_->ple_rows_n = 0;
}

void Forward::section_moe_ms(double out[4], bool reset) const
{
    for (int i = 0; i < 4; ++i) out[i] = m_->sect_moe[i];
    if (reset) for (int i = 0; i < 4; ++i) m_->sect_moe[i] = 0;
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

const runtime::ExpertStore & Forward::experts() const { return *m_->experts; }

void Forward::profile_routes(bool on)
{
    Impl & m = *m_;
    if (on && !m.route_counts) m.route_counts = m.alloc<float>((size_t) m.n_store * m.c.n_expert);   // zeroed
    if (!on && m.route_counts) {
        TRUSS_CUDA(cudaStreamSynchronize(m.s));
        TRUSS_CUDA(cudaFree(m.route_counts));
        m.owned.erase(std::find(m.owned.begin(), m.owned.end(), m.route_counts));
        m.route_counts = nullptr;
    }
}

std::vector<float> Forward::route_counts() const
{
    const Impl & m = *m_;
    std::vector<float> out((size_t) m.n_store * m.c.n_expert, 0.f);
    if (m.route_counts) {
        TRUSS_CUDA(cudaStreamSynchronize(m.s));
        TRUSS_CUDA(cudaMemcpy(out.data(), m.route_counts, out.size() * 4, cudaMemcpyDeviceToHost));
    }
    return out;
}

void Forward::head(int first, int n, float * logits) { m_->head(first, n, logits); }

void Forward::reset()
{
    if (m_->route_trace) {   // the driver is idle between passes; the marker separates sequences
        const int32_t h[2] = { -1, 0 };
        std::fwrite(h, 4, 2, m_->route_trace);
    }
    m_->reset();
    pos_ = 0;
}

void Forward::run(const int32_t * tokens, int T, const LayerHook & hook)
{
    m_->run_chunk(tokens, pos_, T, hook, false);
    pos_ += T;
}

void Forward::verify(const int32_t * tokens, int T)
{
    if (!m_->spec_rows || T > m_->spec_rows)
        throw std::runtime_error("qwen4exp::Forward: verify() of " + std::to_string(T) + " rows (spec_rows " +
                                 std::to_string(m_->spec_rows) + ")");
    m_->run_chunk(tokens, pos_, T, nullptr, true);
}

void Forward::accept(int n)
{
    m_->accept(n);
    pos_ += n;
}

int Forward::draft(int32_t next, int n, int32_t * out) { return m_->draft(next, pos_, n, out); }

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
    f("TRUSS_PREFILL_ROWS", o.prefill_rows);
    f("TRUSS_KV_INT8", o.kv_int8);             // Strata's int8 KV
    f("TRUSS_PLAN_ALPHA", o.plan_alpha);       // hot-set ranking: usage / bytes^alpha
    if (const char * e = std::getenv("TRUSS_PLE_DIRECT"); e && !std::atoi(e)) o.ple_file = nullptr;   // A/B: the mapping
    if (const char * e = std::getenv("TRUSS_RING_GB")) o.ring_bytes_override = (size_t) (std::atof(e) * (1ull << 30));
}

}  // namespace truss::qwen4exp
