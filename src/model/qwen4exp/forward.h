// qwen4exp prompt prefill on one GPU: the fast ops chained layer by layer over chunks of a sequence.
//
// Per chunk (positions pos .. pos + T - 1): token embedding -> per layer [PLE] -> hc mix -> GDN or DSA mixer -> hc
// combine -> hc mix -> routed experts (trellis, moe::prefill) + gated shared expert -> hc combine. Dense projections
// run on Q8_0 weights with fp16 activations (dense::q8_gemm_a16), the router and shared-expert gate in fp32
// (cuBLAS SGEMM, so routing keeps full precision). State carried across chunks: DSA K/V and indexer-block caches,
// GDN recurrent state and conv rows, PLE conv history, the PLE n-gram window.
//
// Routed experts live in a runtime::ExpertStore: as many as fit stay resident, the rest stream over PCIe per layer,
// overlapped with the previous layer's compute. The math is qwen4exp::reference's; tests/layer/qwen4exp_forward
// checks it layer by layer.
#pragma once
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/weights.h"
#include "kernels/sampling/sample.cuh"
#include "runtime/expert_store.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace truss::qwen4exp {
namespace rt { struct StageState; }   // round_trace.h: TRUSS_ROUND_TRACE's per-stage device-work accounting

// Activations into the dense Q8_0 GEMMs. Q8_1 (default): llama-paw's rounding on the int8 GEMM; full-model KL to
// llama-paw's Q8 logits 0.0131 = its own run-to-run floor, and the lower perplexity of the two on chat text (TRACKER
// #53). FP16: closer to exact math per matmul (TRACKER #33) but +3% perplexity on chat-format text vs Q8_1; kept until
// an unquantized anchor decides which is closer to the real model.
enum class Activations { FP16, Q8_1 };

struct ForwardOptions {
    size_t expert_budget = 0;                 // device bytes for routed experts (0: all memory left, minus a margin)
    Activations act = Activations::Q8_1;
    std::vector<float> expert_usage;          // routed counts [layer][expert] (ExpertStore::load_usage); empty: the
                                              // same hot count per layer in index order. Decode speed depends on it.
    size_t ring_bytes = 4ull << 30;           // ExpertStore ring: decode's FIFO of fetched experts (~2,500 at 4 GiB),
                                              // prefill's two stream slots (grown to fit them if smaller)
    const Mtp * mtp = nullptr;                // MTP draft block (bind_mtp; must outlive the Forward): draft() works
    int spec_rows = 0;                        // > 0: verify() / accept() for windows of up to spec_rows (<= 8) rows
    // CPU tier (decode): the rarest non-resident experts of each layer, together cpu_share of the layer's
    // non-resident routing mass (by expert_usage), are always computed on the host from 4-bit copies (cpu_dir:
    // L<nn>.q4s from flashnext_truss_cpu_q4.py) by cpu_threads threads, beside the GPU. Empty dir: off.
    std::string cpu_dir;
    float cpu_share = 0.5f;
    int cpu_threads = 12;
    // Strata's split (TRACKER #73): the CPU tier set (cpu_share of the loaded copies) becomes the experts the CPU MAY
    // compute. Per decode layer, the routed experts already on the GPU (hot or in the ring) run there; of the
    // missed ones with a host copy, the last m in routing order go over PCIe and the rest to the CPU, m chosen so
    // the CPU's time (measured per expert, running mean) and the copies' time (bytes / pcie_gbps) are balanced.
    // Faster, but the result then depends on the cache state (the CPU's q4s copy and the pack differ): verify
    // windows no longer reproduce plain steps bit for bit. Doorbell path only (the sync path keeps the static set).
    bool cpu_dynamic = false;
    // CPU tier from the pack itself (TRACKER #73): the CPU decodes the cold experts' trellis tiles straight from the
    // ExpertStore's pinned copies (cpu/expert_trellis.h) instead of q4s files: no extra RAM (the q4s tier needed
    // 2.76 MB per CPU expert beside the pinned 1.77, 28 GB at share 0.15, which swapped beside the renters), and the
    // CPU computes the GPU's own weights. Every cold expert is eligible; cpu_share picks the static set as with
    // cpu_dir (ignored when cpu_dir is set).
    bool cpu_trellis = false;
    float pcie_gbps = 13.5f;                  // host -> device copy rate the dynamic split assumes (PCIe 4.0 x8)
    // cpu_dynamic with a fixed share, as Strata's --pcie-frac (TRACKER #76): per layer round(pcie_frac * eligible
    // misses) go over PCIe (the last ones in routing order), the rest to the CPU. < 0: the fitted cost model instead,
    // which can starve the CPU when its fitted per-call cost drifts up (few experts per call make calls look fixed-cost)
    float pcie_frac = -1.f;
    // Strata's adaptive tier (TRACKER #78): every adapt_every decode passes (0: off), before the pass, the cold
    // experts not on the device whose decayed routing count is >= 2 (most-routed first, at most adapt_swaps) are
    // copied into the ring off the critical path (ExpertStore::admit); counts decay x0.7 after each admission round.
    // Meant with pcie_frac 0 and hint_k 0: no copies inside a pass at all.
    int adapt_every = 0;
    int adapt_swaps = 96;
    // admission threshold on the decayed count; decay per round. TRACKER #92: admitting 64/pass at >= 1 cut CPU misses
    // 20% but its copies delayed the demand fetches (shared copy stream): off by default (adapt_every 0)
    float adapt_min = 2.f, adapt_decay = 0.7f;
    // Admission on the idle link (TRACKER #113; doorbell decode with the CPU tier): after serving a layer the driver
    // claims ring slots for up to 2 of the experts the CPU just computed (highest decayed routing count first, count
    // >= admit_min, counts x admit_decay per pass), at most admit_idle per pass, and copies them in pieces only while
    // the link has no demand copy (runtime::ExpertStore::claim / admit_step). 0: off. Replay (R3): 32/pass at C2
    // size cuts misses 153 -> 109 per pass.
    int admit_idle = 0;
    // E6 (TRACKER #114): the K/V caches' unused tail (positions past the sequence, in 4,096-position steps with one
    // step ahead) is lent to the expert ring as extra segments and reclaimed before those positions are written
    // (runtime::ExpertStore::add_segment / reclaim). ~14 KB per position over the 13 DSA layers: an agent turn at
    // 150K lends ~1.6 GB of the 3.4 GB sized for 262K.
    bool kv_lend = false;
    float admit_min = 1.f, admit_decay = 0.9f;
    // E7 (TRACKER #117): prompt chunks of more than 32 and at most split_rows tokens, with the CPU tier: per layer the
    // routing comes first, then the routed cold experts with the fewest rows (at most split_t) run on the CPU pool
    // until its estimated time (the decode split's per-expert fit + split_ms_row per extra row) meets the time of the
    // copies left, and only those others are copied into the layer's stream slot (an agent step's tool result of
    // 100-2K tokens streamed every cold expert, ~24 GB in ~2.5 s, TRACKER #116). 0: off.
    int split_rows = 0;
    int split_t = 16;
    float split_ms_row = 0.01f;
    std::vector<int32_t> draft_vocab;         // MTP drafts score only these token ids (empty: all), e.g.
                                              // data/draft_vocab_en.bin: the draft head reads 16% of the output
                                              // matrix; verify keeps the output exact
    float draft_min_p = 0.f;                  // a draft chain stops once the MTP head's top probability drops below
    bool doorbell = true;                     // decode FFNs without a host sync (driver thread; TRACKER #61)
    size_t ring_bytes_override = 0;         // ring size (0: Options::ring_bytes) — the ring competes with the
                                              // static hot set for VRAM; a sweep knob for the decode trade-off
    int prefill_rows = 0;                  // rows the prompt path's buffers are sized for (0: max_chunk). They are
                                              // carved from the ring's spare region, so rows past the prompt's real
                                              // length evict cached experts for the whole run; pass the prompt's
                                              // length and step run() by it (TRACKER #70)
    // DSA K/V caches as int8 codes + an fp16 scale per 64 values (Strata's served KV format, dsa::KvCache): half the
    // VRAM of fp16 (6.4 -> 3.3 GB at 256K for the 12 layers + the MTP layer: ~2,000 more resident experts)
    bool kv_int8 = false;
    double plan_alpha = 0.0;                  // hot set ranked by usage / bytes^plan_alpha (ExpertStore::Sizes)
    // the model file the weights are bound from: with it the PLE table rows are read by PleReader (O_DIRECT, many in
    // flight, row cache) instead of page faults through the mapping (TRACKER #88). nullptr: the mapping.
    const gguf::File * ple_file = nullptr;
    int hint_k = 4;                           // pre-gated prefetch: per row, the next layer's top hint_k predicted
                                              // experts start copying early (0: off; 3-6 measured equal, TRACKER #59)
    int hint_min_rows = 0;                    // ... only for windows of at least this many rows: the next layer's router
                                              // costs ~0.6 ms a pass at 1-5 rows and repays it only at 7-8 (TRUSS_HINT_MIN_ROWS)
    // Layer split over two devices (TRUSS_LAYER_SPLIT, TRUSS_LAYER_SPLIT_DEVICE): layers [0, layer_split) run on
    // layer_split_device, the rest, the head and the MTP block on the current device (so the logits, stream() and
    // the caller's buffers stay there). Each stage has its own expert store, K/V, recurrent state and driver; the
    // CPU pool is shared; the residual rows go from one to the other through pinned host memory. Every row is
    // computed as unsplit, so the output is the same up to which experts each device holds. 0: one device.
    // layer_split_device -1: both stages on the current device (a test of the split), the first taking
    // layer_split_frac of the free VRAM for its experts.
    int layer_split = 0;
    int layer_split_device = -1;
    float layer_split_frac = 0.5f;
    // Sequence slots (TRUSS_SLOTS): the engine holds `slots` independent sequences. Each slot's position-indexed
    // caches (K/V, indexer blocks, MTP rows) are a sub-range of the same n_ctx-row arrays (n_ctx / slots positions
    // each, so the VRAM is unchanged), its recurrent state (GDN state and conv rows, the indexer's open block, PLE
    // history, the MTP pending row) has its own copy. select() switches between them between calls. > 1 turns
    // kv_lend off (it lends one sequence's unused tail).
    int slots = 1;
    // Called once upload() has the weights on the device but before the constructor does anything else expensive
    // (the expert plan, the CPU tier). The owner releases the model's file-backed pages there — the shards fault to a
    // ~47 GB resident peak during upload, and the CPU tier's anonymous vector then allocates on top of it and the
    // kernel OOM-kills the run. The mapping stays valid, so anything still read just re-faults (TRACKER #72).
    std::function<void()> after_upload;
};

// The decode-tier sweep knobs from the environment, applied on top of `o` (each only when set): TRUSS_CPU_SHARE,
// TRUSS_CPU_THREADS, TRUSS_CPU_DYNAMIC, TRUSS_CPU_TRELLIS, TRUSS_PCIE_GBPS, TRUSS_PCIE_FRAC, TRUSS_ADAPT_EVERY,
// TRUSS_ADAPT_SWAPS, TRUSS_RING_GB, TRUSS_HINT_K, TRUSS_PREFILL_ROWS, TRUSS_KV_INT8, TRUSS_PLAN_ALPHA, TRUSS_ADAPT_MIN, TRUSS_ADAPT_DECAY, TRUSS_SPLIT_ROWS, TRUSS_SPLIT_T, TRUSS_SPLIT_MS_ROW. tk-bench-spec and the C API (so the server)
// both call it, so a config benched is the config served.
void apply_env(ForwardOptions & o);

class Forward {
public:
    using Options = ForwardOptions;
    // n_ctx: the longest sequence; max_chunk: the most tokens per run() call (a multiple of the DSA block ratio)
    // c and w must outlive the Forward (w references the file mapping the weights are uploaded from)
    Forward(const Config & c, const Weights & w, int n_ctx, int max_chunk, const Options & o = {});
    ~Forward();
    // blocks until the expert store's cold copy is page-locked (it finishes ~3 s after the constructor returns;
    // until then cold copies are pageable and slower): benches call it before timing
    void wait_pinned();
    Forward(const Forward &) = delete;
    Forward & operator=(const Forward &) = delete;

    // called after each layer with the device residual [T][hc][d_model] of the current chunk
    using LayerHook = std::function<void(int layer, const float * res, int T)>;

    // The next T tokens of the sequence (any T <= max_chunk: a prompt chunk, or one decoded token).
    // For a prompt chunk, next / next_T may describe the chunk that follows it (contiguous, tokens + T): the
    // PLE rows that chunk needs are then hashed and read from disk while this chunk computes, so the next
    // chunk's layer-1 gather does not stall the host. Without the hint the gather still blocks its own chunk.
    void run(const int32_t * tokens, int T, const LayerHook & hook = nullptr,
             const int32_t * next = nullptr, int next_T = 0);

    // A prompt of n tokens in chunks of step, as run() per chunk with the PLE lookahead; done(tokens so far) after
    // each chunk. With a layer split the stages run on two threads, a chunk apart, so both devices compute at once
    // (the same output as run()).
    void run_prompt(const int32_t * tokens, int n, int step, const std::function<void(int)> & done = nullptr);

    // Speculative decoding (Options::spec_rows > 0). verify() runs a window of T <= spec_rows tokens (the next token
    // and the drafts after it) without committing it: head() then gives every row's logits. accept(n) keeps the
    // first n rows (1 <= n <= T) and rolls the recurrent state (GDN state and conv rows, PLE history, the DSA
    // indexer's open block) back to them; caches indexed by position need no rollback. Exactly one accept() per
    // verify(); position() advances by n.
    void verify(const int32_t * tokens, int T);
    void accept(int n);

    // MTP drafts (Options::mtp): up to n greedy guesses for the tokens after `next`, the token at position(); returns
    // how many (0 .. n: the chain stops before a guess whose probability < Options::draft_min_p). The MTP block reads the last
    // committed row's hidden state; every run() / accept() keeps the MTP layer's own cache in step.
    int draft(int32_t next, int n, int32_t * out);

    // speculative sampling: while set, draft() SAMPLES each guess from the draft head's distribution under p (stream
    // counter + i; the stop rule still reads the argmax probability) and writes its kept probabilities to
    // q_full[i][n_vocab] (device); nullptr: greedy guesses
    void set_draft_sampling(const sampling::SampleParams * p, uint64_t counter, float * q_full);

    // logits [n][n_vocab] (device, fp32) of rows first .. first + n - 1 of the last chunk: the head hc mix, then
    // the output projection
    void head(int first, int n, float * logits);

    // start a new sequence (zero recurrent state, conv histories, the PLE window; caches are overwritten by position)
    void reset();

    // Sequence checkpoint: checkpoint() saves the recurrent state at position() (pinned host memory, one slot,
    // ~115 MB for Flash-Next); restore() returns the sequence to it (position() back to that point, -1 if none was
    // taken) so a prompt that shares only the checkpoint's prefix with the engine's sequence reads only the rest.
    // The server checkpoints the end of each prompt (its old reasoning can then be dropped without losing reuse).
    void checkpoint();
    int restore();

    // Options::slots: make slot k (0 .. slots() - 1) the sequence the calls without a slot act on. Each slot keeps
    // its position, caches, recurrent state, verify window and checkpoint; slot 0 is selected at start.
    void select(int k);
    int slots() const { return (int) seq_pos_.size(); }
    int slot() const { return active_; }
    int seq_ctx() const { return seq_ctx_; }   // positions per slot (n_ctx with one slot)

    // Concurrent slots. These name their slot and may run on different threads at once (one thread per slot at a
    // time): each stage is held only while a slot's work is enqueued on it, so with a layer split one slot's rows run
    // on one GPU while another slot's run on the other (Strata's pipelined batch groups, with every slot's own MTP
    // drafts and verify window).
    // prompt(): tokens [0, n) appended to `slot` in chunks of `step` (the stages a chunk apart, as run_prompt); with
    // logits, the last row's logits go there and after() runs, both while the last stage is still held for this slot.
    void prompt(int slot, const int32_t * tokens, int n, int step, float * logits, const std::function<void()> & after,
                const std::function<void(int)> & done = nullptr);
    // round(): one speculative round of `slot` after token `next` (the token at its position): up to nd MTP drafts
    // into win[1..] (win[0] = next; draft sampling with ds_p as set_draft_sampling, nullptr: greedy), the verify window
    // over win, its logits [nw + 1][n_vocab] into `logits`, then pick(nw) - run with the last stage held - returns
    // how many window rows to keep (1 .. nw + 1). Returns that count; position(slot) advances by it.
    int round(int slot, int32_t next, int nd, int32_t * win, float * logits, const sampling::SampleParams * ds_p,
              uint64_t ds_counter, float * ds_q, const std::function<int(int)> & pick);
    void reset(int slot);
    void checkpoint(int slot);
    int restore(int slot);
    int position(int slot) const;
    // host time each stage was held and waited for (ms since the last reset): [front held, front wait, last held, last wait]
    void lock_ms(double out[4], bool reset = false);
    // TRUSS_ROUND_TRACE: the last stage's accounting, for a host wait on its device made outside the engine (the
    // caller's pick). Harmless when the trace is off.
    rt::StageState * rt_last_stage();

    int position() const { return seq_pos_[active_]; }   // tokens consumed so far (the selected slot)
    static int fetch_rows();                // run() chunks of at most this many tokens take the fetch (decode) path
    int hot_experts() const;                // resident experts, all layers
    size_t cold_bytes() const;              // streamed per chunk
    // CPU tier benchmark stats (0, 0 without Options::cpu_dir): total microseconds in ExpertPool::wait() and its
    // calls since the last reset (reset with cpu_stats(true)).
    std::pair<long long, long> cpu_stats(bool reset = false);
    long long cpu_item_us() const;   // sum of ExpertPool::item() compute time, all threads
    void cpu_phase_us(long long out[4]) const;   // gate/up gemv, silu, h-quantize, down gemv sums
    void cpu_shape(long long out[4]) const;    // calls, distinct experts, slots, rows-served sum (the gemv's R)
    void cpu_shape_reset();
    // Options::cpu_dynamic: eligible misses sent to the CPU / over PCIe since construction, and the CPU's current
    // cost model of one CPU call, cpu_ms_call + cpu_ms_expert * n (ms, fitted to the pool's call times)
    void dyn_stats(long & cpu, long & pcie, double & cpu_ms_expert, double & cpu_ms_call) const;
    // Per-section device time of the layer chain, in ms, accumulated over every run()/verify() since the last
    // reset: [0] PLE + attn hyper-connection mix, [1] GDN/DSA mixer, [2] hc combine + ffn-side mix,
    // [3] router + routed MoE (PCIe copies and the CPU tier's start inside), [4] shared expert,
    // [5] CPU-tier join (its spin) + routed/shared combine. Device events on the engine stream, so host stalls
    // inside a section are included; the gaps between sections are not. TRUSS_PROFILE_SECTIONS=1 turns it on.
    void section_ms(double out[6], bool reset = false) const;
    // TRUSS_PROFILE_SECTIONS, doorbell decode: routed MoE split into router + publish, the wait for the driver's plan
    // (host split, CPU start, copies issued), the wait for the layer's copies, and the GPU expert kernel (ms summed
    // since the last reset)
    void section_moe_ms(double out[4], bool reset = false) const;
    // doorbell driver thread, ms summed over n served layers: the split (from seeing the doorbell), starting the CPU
    // pool, issuing the copies and writing the plan (read between passes only)
    void driver_ms(double out[3], long & n, bool reset = false) const;
    long adapt_admitted() const;              // experts admitted by the adaptive tier since construction
    // host time of the PLE layers' n-gram row hash, table gather (the mmap of the GGUF: page faults for rows not in
    // the page cache) and fp16 conversion, ms summed over calls since the last reset, with the table rows read.
    // Always counted (two clock reads per PLE layer call).
    void ple_host_ms(double & ms, long & calls, long & rows, bool reset = false, double * wait_ms = nullptr,
                     double * phase_ms = nullptr) const;   // phase_ms[3]: collect's hits / copy+put / map+insert
    size_t kv_lent() const;                         // Options::kv_lend: KV bytes lent to the expert ring now
    // E7 since construction: host bytes copied by split chunks, cold experts their layers gave the CPU / the GPU
    void split_stats(long long out[3]) const;
    const runtime::ExpertStore & experts() const;   // residency and fetch statistics (the MTP block, when present,
                                                    // is store layer n_layer)

    // Routing profile: while on, every run() adds each routed (layer, expert) to a device table; route_counts()
    // returns it as [layer][expert] float (the usage file format, ExpertStore::load_usage; n_layer + 1 layers with an
    // MTP block) and profiling stays on.
    void profile_routes(bool on);
    std::vector<float> route_counts() const;
    cudaStream_t stream() const;

private:
    struct Impl;
    std::vector<std::unique_ptr<Impl>> front_;   // a layer split's stages before the last (destroyed after m_)
    std::unique_ptr<Impl> m_;                    // the last stage (the whole model without a split)
    template <class F> void each_stage(F && f);  // every stage in layer order, its device current
    // The same, with each stage held while f runs on it: for anything touching a buffer a chunk's kernels may still
    // be writing (profile_routes() frees one, route_counts() reads one). Without the lock another slot's forward can
    // be enqueued on that stage while it happens (review H3).
    template <class F> void each_stage_locked(F && f) const;
    void construct(const Config & c, const Weights & w, int n_ctx, int max_chunk, const Options & o);
    std::vector<int> seq_pos_{ 0 };              // each slot's position
    std::unique_ptr<std::atomic<long long>[]> round_ns_;   // each slot's last round() end (steady clock ns)
    // The selected slot. Another thread's accept()/draft() read it while this thread is between select() and its
    // work (review H2), so it is atomic; a call that uses it more than once snapshots it (the hazard is a change
    // mid-call, not a torn read).
    std::atomic<int> active_{ 0 };
    int seq_ctx_ = 0;
};

}  // namespace truss::qwen4exp
