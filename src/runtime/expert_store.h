// Routed-expert weights of every MoE layer on one GPU. Three tiers:
//   hot    a static set (usage-ranked, ExpertStore::plan) resident for the whole run;
//   ring   one device region that holds either recently fetched cold experts (decode: a FIFO cache) or, during a
//          prompt chunk, two whole-layer slots (every cold expert of layer l streams into slot l % 2 while layer l - 1
//          computes) plus the chunk's own large buffers (spare(): scratch, MoE workspace, residual);
//   host   every cold expert in pinned host memory, per layer [expert: gate | up | down] in expert order.
//
// Device layout: one arena [hot experts, first half of the layers][ring][hot experts, second half], each expert's
// three projections contiguous. Every ProjView has the ring start as its base and shift 4: meta offsets count 32-byte
// units, so int32 reaches +-64 GiB. Two meta tables per (layer, projection): the stream table (cold experts at their
// fixed place in slot l % 2, built once) and the ring table (cold experts where the FIFO put them, rewritten by
// fetch()). weights(l) returns the table of the current mode.
//
// Protocol (compute = the caller's stream; copies run on the store's own stream):
//   prompt chunk:  begin_stream(compute); prefetch(0), prefetch(1); per layer acquire(l) -> kernel -> release(l) ->
//                  prefetch(l + 2). spare() is valid until the next fetch().
//   decode step / verify window: per layer, routing to the host -> fetch(l, ids, n, compute) -> acquire(l) -> kernel.
// Switching mode waits for all compute queued so far (the ring and the slots share memory), and a switch to stream
// mode drops the ring's contents. Why a ring: on held-out routing a FIFO of recently fetched experts halves the misses
// of a static set of the same size (10,000 slots: 74 -> 38 per token; FIFO = LRU within 3%, TRACKER #57).
#pragma once
#include "formats/trellis_table.h"
#include "kernels/moe/moe_weights.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace truss::runtime {

using ExpertLayer = std::array<const formats::ExpertTable *, 3>;   // gate, up, down of one layer

class ExpertStore {
public:
    using HotSet = std::vector<std::vector<uint8_t>>;                 // [layer][expert] = 1: resident

    struct Sizes {
        size_t ring_bytes = 4ull << 30;   // decode FIFO (grown to fit the prompt path)
        size_t stream_extra = 0;          // bytes a prompt chunk borrows after the two slots (spare())
    };
    // ring = max(ring_bytes, 2 x the largest cold layer + stream_extra), 256-aligned
    static size_t ring_size(const std::vector<ExpertLayer> & layers, const HotSet & hot, const Sizes & z);
    static size_t device_bytes(const std::vector<ExpertLayer> & layers, const HotSet & hot, const Sizes & z);

    // The hot set whose device_bytes fit `budget`. With `usage` (routed counts, [layer][expert] flattened): every
    // layer first gets its top `floor` experts by count per byte, then greedy by count per byte over all layers; the
    // floor (steps of 8) with the most hot usage wins, since a layer with few hot experts makes the stream slots
    // large. Without usage: the same count per layer, index order.
    static HotSet plan(const std::vector<ExpertLayer> & layers, size_t budget, const Sizes & z,
                       const std::vector<float> & usage = {});
    // usage file: n_layer * n_expert little-endian float32, layer-major (flashnext_truss_usage.py writes it)
    static std::vector<float> load_usage(const std::string & path, int n_layer, int n_expert);

    ExpertStore(const std::vector<ExpertLayer> & layers, const HotSet & hot, const Sizes & z);
    ~ExpertStore();
    ExpertStore(const ExpertStore &) = delete;
    ExpertStore & operator=(const ExpertStore &) = delete;

    moe::Weights weights(int layer) const;                        // meta table of the current mode
    void begin_stream(cudaStream_t compute);                       // prompt chunk starts: stream mode
    // decode starts: ring mode (the ring starts empty over the slots). Called by the thread that enqueues kernels
    // before weights(): with a doorbell driver, fetch() runs on another thread later (TRACKER #61)
    void begin_ring();
    void * spare() const { return spare_; }                        // stream mode: stream_extra bytes after the slots
    void prefetch(int layer);                                      // all cold experts of the layer into slot l % 2
    void fetch(int layer, const int * ids, int n, cudaStream_t compute);   // ring mode: make ids[0 .. n) (host)
                                                                           // resident, copying the missing ones
    // Pre-gated prefetch (after fetch(layer, ...), before acquire): start copying the experts `ids` predicts for
    // layer + 1, behind this layer's copies, so PCIe works while this layer and the next one's attention compute.
    // Never evicts an expert the last fetch() needs (stops instead); wrong guesses only cost bandwidth and ring room.
    void prefetch_hint(int layer, const int * ids, int n);
    // Doorbell decode (qwen4exp::Forward): fetch(l, ids, n, nullptr) skips the wait on compute (the doorbell that
    // gave the ids already orders it), and signal(l, flag, seq) queues, behind every copy issued so far, a write of
    // seq to the device word flag, which a spin kernel on the compute stream waits for instead of acquire().
    void signal(int layer, int * flag, int seq);
    void upload(void * dst, const void * src, size_t bytes);         // a small copy (pinned src) on the copy stream
    void acquire(int layer, cudaStream_t compute);                 // compute waits for layer l's copies
    void release(int layer, cudaStream_t compute);                 // compute is done with slot l % 2 (stream mode)

    size_t device_bytes() const { return device_bytes_; }
    size_t cold_bytes() const { return cold_total_; }                // pinned host bytes, streamed once per chunk
    size_t ring_bytes() const { return ring_; }
    int layers() const { return (int) layers_.size(); }

    struct Stats {                                                  // ring mode, since construction
        long fetch_calls = 0, experts_asked = 0, misses = 0;   // misses: fetched on demand
        size_t bytes = 0;                                      // demand bytes
        long hinted = 0;                                       // copied by prefetch_hint
        size_t hint_bytes = 0;
    };
    const Stats & stats() const { return stats_; }

private:
    struct Layer {
        size_t cold = 0;                                            // cold bytes, all projections
        const uint8_t * host = nullptr;                             // pinned [cold expert: gate | up | down]
        std::vector<int64_t> cold_off;                              // [expert] byte offset in host / slot, -1: hot
        std::vector<size_t> bytes;                                  // [expert] gate + up + down
        std::array<std::vector<size_t>, 3> part;                    // [p][expert] offset of projection p in the expert
        int32_t * stream_meta[3] = {}, * ring_meta[3] = {};         // device (K, offset in 32-byte units)
        int32_t * ring_meta_host = nullptr;                         // pinned mirror of ring_meta, [3][n_expert][2]
        std::vector<int64_t> ring_at;                               // [expert] byte offset in the ring, -1: absent
        const half * suh[3] = {}, * svh[3] = {};
        int n_expert = 0;
    };
    struct RingEntry {
        int layer, expert;
        int64_t off;
    };
    enum class Mode { STREAM, RING };

    void wait_compute(cudaStream_t compute);                        // copy stream waits for all compute queued so far
    bool ring_put(int layer, int expert, bool hint = false);   // hint: refuse (false) to evict a protected expert
    std::vector<int> protect_;                                  // layer, experts the last fetch() needs
    int protect_layer_ = -1;

    std::vector<Layer> layers_;
    uint8_t * base_ = nullptr;                                     // ring start
    size_t ring_ = 0, slot_ = 0;
    void * spare_ = nullptr;
    int64_t head_ = 0;                                             // next free ring byte (FIFO)
    std::deque<RingEntry> fifo_;
    Mode mode_ = Mode::RING;
    std::vector<void *> device_, pinned_;
    cudaStream_t copy_ = nullptr;
    cudaEvent_t copied_[2] = {}, released_[2] = {}, compute_mark_ = nullptr;
    bool released_recorded_[2] = {};
    size_t device_bytes_ = 0, cold_total_ = 0;
    Stats stats_;
    // pinned ring of the words signal() copies: a queued copy reads its source when it runs, so each call gets its
    // own word (the copy stream never lags SIGNAL_RING calls behind)
    static constexpr int SIGNAL_RING = 1024;
    int * signal_src_ = nullptr;
    int signal_next_ = 0;
};

}  // namespace truss::runtime
