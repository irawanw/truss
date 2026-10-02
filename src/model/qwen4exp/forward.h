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
#include "runtime/expert_store.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace truss::qwen4exp {

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
    int hint_k = 4;                           // pre-gated prefetch: per row, the next layer's top hint_k predicted
                                              // experts start copying early (0: off; 3-6 measured equal, TRACKER #59)
    // Called once upload() has the weights on the device but before the constructor does anything else expensive
    // (the expert plan, the CPU tier). The owner releases the model's file-backed pages there — the shards fault to a
    // ~47 GB resident peak during upload, and the CPU tier's anonymous vector then allocates on top of it and the
    // kernel OOM-kills the run. The mapping stays valid, so anything still read just re-faults (TRACKER #72).
    std::function<void()> after_upload;
};

// The decode-tier sweep knobs from the environment, applied on top of `o` (each only when set): TRUSS_CPU_SHARE,
// TRUSS_CPU_THREADS, TRUSS_CPU_DYNAMIC, TRUSS_CPU_TRELLIS, TRUSS_PCIE_GBPS, TRUSS_PCIE_FRAC, TRUSS_ADAPT_EVERY,
// TRUSS_ADAPT_SWAPS, TRUSS_RING_GB, TRUSS_HINT_K, TRUSS_PREFILL_ROWS. tk-bench-spec and the C API (so the server)
// both call it, so a config benched is the config served.
void apply_env(ForwardOptions & o);

class Forward {
public:
    using Options = ForwardOptions;
    // n_ctx: the longest sequence; max_chunk: the most tokens per run() call (a multiple of the DSA block ratio)
    // c and w must outlive the Forward (w references the file mapping the weights are uploaded from)
    Forward(const Config & c, const Weights & w, int n_ctx, int max_chunk, const Options & o = {});
    ~Forward();
    Forward(const Forward &) = delete;
    Forward & operator=(const Forward &) = delete;

    // called after each layer with the device residual [T][hc][d_model] of the current chunk
    using LayerHook = std::function<void(int layer, const float * res, int T)>;

    // The next T tokens of the sequence (any T <= max_chunk: a prompt chunk, or one decoded token).
    void run(const int32_t * tokens, int T, const LayerHook & hook = nullptr);

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

    // logits [n][n_vocab] (device, fp32) of rows first .. first + n - 1 of the last chunk: the head hc mix, then
    // the output projection
    void head(int first, int n, float * logits);

    // start a new sequence (zero recurrent state, conv histories, the PLE window; caches are overwritten by position)
    void reset();

    int position() const { return pos_; }   // tokens consumed so far
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
    void ple_host_ms(double & ms, long & calls, long & rows, bool reset = false) const;
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
    std::unique_ptr<Impl> m_;
    int pos_ = 0;
};

}  // namespace truss::qwen4exp
