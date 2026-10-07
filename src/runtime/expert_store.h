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
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace truss::runtime {

using ExpertLayer = std::array<const formats::ExpertTable *, 3>;   // gate, up, down of one layer

class ExpertStore {
public:
    using HotSet = std::vector<std::vector<uint8_t>>;                 // [layer][expert] = 1: resident

    struct Sizes {
        size_t ring_bytes = 4ull << 30;   // decode FIFO (grown to fit the prompt path)
        size_t stream_extra = 0;          // bytes a prompt chunk borrows after the two slots (spare())
        // plan(): experts ranked by usage / bytes^plan_alpha. 1 = routed mass per byte (the profile's knapsack
        // optimum); 0 = by usage: on real 256K routing it cut LRU-ring misses 234 -> 208/pass and miss bytes
        // 501 -> 388 MB/pass at equal VRAM (the profile's mass on small rarely-routed experts does not show up,
        // and our most-routed experts are the large high-K ones; TRACKER #90)
        double plan_alpha = 0.0;
    };
    // ring = max(ring_bytes, 2 x the largest cold layer + stream_extra), 256-aligned
    static size_t ring_size(const std::vector<ExpertLayer> & layers, const HotSet & hot, const Sizes & z);
    static size_t device_bytes(const std::vector<ExpertLayer> & layers, const HotSet & hot, const Sizes & z);

    // The hot set whose device_bytes fit `budget`. With `usage` (routed counts, [layer][expert] flattened): every
    // layer first gets its top `floor` experts by count / bytes^plan_alpha, then greedy in that order over all layers; the
    // floor (steps of 8) with the most hot usage wins, since a layer with few hot experts makes the stream slots
    // large. Without usage: the same count per layer, index order.
    static HotSet plan(const std::vector<ExpertLayer> & layers, size_t budget, const Sizes & z,
                       const std::vector<float> & usage = {});
    // usage file: n_layer * n_expert little-endian float32, layer-major (flashnext_truss_usage.py writes it)
    static std::vector<float> load_usage(const std::string & path, int n_layer, int n_expert);

    ExpertStore(const std::vector<ExpertLayer> & layers, const HotSet & hot, const Sizes & z);
    ~ExpertStore();
    // The cold copy is page-locked by a background thread that start_pinning() starts (the owner calls it after its
    // own CUDA setup); wait_pinned() starts it if needed and blocks until it is done. Copies before that are pageable.
    void start_pinning();
    void wait_pinned();
    ExpertStore(const ExpertStore &) = delete;
    ExpertStore & operator=(const ExpertStore &) = delete;

    moe::Weights weights(int layer) const;                        // meta table of the current mode
    void begin_stream(cudaStream_t compute);                       // prompt chunk starts: stream mode
    // decode starts: ring mode (the ring starts empty over the slots). Called by the thread that enqueues kernels
    // before weights(): with a doorbell driver, fetch() runs on another thread later (TRACKER #61)
    void begin_ring();
    void * spare() const { return spare_; }                        // stream mode: stream_extra bytes after the slots
    void prefetch(int layer);                                      // all cold experts of the layer into slot l % 2
    // E7 (TRACKER #117): only the cold experts ids[0 .. n) (host, repeats allowed) into their places in slot l % 2,
    // after the routing of a short prompt chunk is known; the kernel must not read the others (masked ids). One still
    // in the ring outside the stream area is copied on the device. Returns the bytes copied from the host.
    size_t prefetch_some(int layer, const int * ids, int n);
    // ring mode: make ids[0 .. n) (host) resident, copying the missing ones; true when it queued any copy (ring
    // entries or meta tables), false when every id was already on the device (nothing for the GPU to wait for)
    bool fetch(int layer, const int * ids, int n, cudaStream_t compute);
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
    // TRUSS_STREAM_PROFILE=1: stream-mode measurements, harvested at every mode switch. busy_ms = PCIe transfer
    // time of the layer prefetches; wait_ms = compute-stream time spent waiting at acquire() (exposed stream).
    void stream_stats(double & busy_ms, double & wait_ms, bool reset = false) const;

    size_t device_bytes() const { return device_bytes_; }
    size_t cold_bytes() const { return cold_total_; }                // pinned host bytes, streamed once per chunk
    size_t ring_bytes() const { return ring_; }
    size_t stream_area() const { return (size_t) stream_end_; }   // what a prompt chunk evicts (slots + buffers)
    int layers() const { return (int) layers_.size(); }
    // expert e of layer l is readable by the GPU now: hot, or in the ring (ring mode; a queued copy counts, since the
    // compute that reads it waits for the copy stream)
    bool on_device(int l, int e) const
    {
        const Layer & Y = layers_[l];
        return Y.cold_off[e] < 0 || (Y.ring_at[e] >= 0 && !Y.pend[e]);
    }

    // Strata's adaptive tier (TRACKER #78), between decode passes: copy the cold experts `le` (layer, expert) into the
    // ring (FIFO eviction) on the copy stream, after the compute queued on `compute` so far (no kernel can read a
    // slot being overwritten). Evicted experts stop being on the device at once; admitted ones count only once
    // their copies have landed (poll_admitted). A no-op while the previous batch is in flight.
    void admit(const std::vector<std::pair<int, int>> & le, cudaStream_t compute);
    void poll_admitted();                                          // promote the batch whose copies have landed
    bool admitting() const { return !adm_.empty(); }
    // Admission on the idle link (TRACKER #113), driver thread only. claim(l, e) takes a ring slot for a cold expert
    // without copying it, never evicting an expert the last fetch() needs (its kernel may be running); the expert is
    // not on the device until its copy lands. admit_step() copies the claimed experts in PIECE-byte pieces, one piece
    // in flight at a time and only while no demand copy is in flight, so a demand fetch queues behind at most one
    // piece. A fetch() that needs a claimed expert copies the rest of it at once. The meta table is no longer copied
    // per mutation: ring_put() marks the layer dirty and fetch()/prefetch_hint()/acquire() upload it once, behind all
    // copies queued so far on copy_ and before the event/flag the compute side waits on (TRACKER 129, 1.A).
    // Returns whether admit_step() issued a copy.
    static constexpr size_t PIECE = 1 << 20;
    bool claim(int l, int e);
    bool admit_step();
    size_t admit_backlog() const { return adm_backlog_; }       // claimed bytes not issued yet
    // E6 (TRACKER #114): device regions lent to the ring (the unused tails of the KV caches). The ring is one virtual
    // offset space: [0, ring_bytes) is its own region, then each segment; an entry never straddles a region and the
    // head skips to the next one as it does at the ring's end. reclaim(j, lo) makes the first lo bytes of segment j
    // unusable again (evicting what lies there) before the owner writes them; lowering lo lends them back. The
    // caller orders its writes after the copy stream (fence()). Driver-thread work (fetch, claim) must be idle.
    int add_segment(uint8_t * ptr, size_t bytes);
    void reclaim(int j, size_t lo);
    void fence(cudaStream_t compute);                           // compute waits for every copy issued so far
    size_t lent_bytes() const;                                  // usable segment bytes now
    std::string admit_debug() const                             // TRUSS_DRIVER_DEBUG watchdog line
    {
        return "claims " + std::to_string(claims_.size()) + " landing " + std::to_string(landing_.size()) +
               " piece_out " + std::to_string(piece_out_) + " demand_out " + std::to_string(demand_out_) +
               " backlog " + std::to_string(adm_backlog_) + " mode " + std::to_string((int) mode_);
    }
    long admit_landed() const { return adm_landed_; }
    long admitted() const { return adm_total_; }
    size_t bytes_of(int l, int e) const { return layers_[l].bytes[e]; }
    // pinned host copy of projection p (gate, up, down) of a cold expert; nullptr for a hot one (it has none). The CPU
    // tier reads the pack's trellis bytes from here (cpu::TrellisExpert): the same bytes PCIe copies.
    const uint8_t * host_part(int l, int e, int p) const
    {
        const Layer & Y = layers_[l];
        return Y.cold_off[e] < 0 ? nullptr : Y.host + Y.cold_off[e] + Y.part[p][e];
    }

    struct Stats {                                                  // ring mode, since construction
        long fetch_calls = 0, experts_asked = 0, misses = 0;   // misses: fetched on demand
        size_t bytes = 0;                                      // demand bytes
        long hinted = 0;                                       // copied by prefetch_hint
        size_t hint_bytes = 0;
        size_t admit_bytes = 0;                                // copied by admit_step() / claim flushes
        long admitted_claims = 0;                              // claims fully issued
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
        bool meta_dirty = false;                                    // ring_meta_host changed; uploaded by the next fetch/hint/acquire (1.A)
        std::vector<int64_t> ring_at;                               // [expert] byte offset in the ring, -1: absent
        std::vector<uint8_t> pend;                                  // [expert] admitted, copy not known to have landed
        std::vector<uint8_t> ref;                                   // [expert] ring entry used since it was copied in
        const half * suh[3] = {}, * svh[3] = {};
        int n_expert = 0;
    };
    struct RingEntry {
        int layer, expert;
        int64_t off;
    };
    enum class Mode { STREAM, RING };

    void wait_compute(cudaStream_t compute);                        // copy stream waits for all compute queued so far
    struct Segment {
        uint8_t * ptr;
        int64_t vbase, size, lo;                                // virtual start, bytes, unusable front bytes
    };
    std::vector<Segment> segs_;
    int64_t stream_end_ = 0;                                    // a prompt chunk's slots + buffers: [0, stream_end_)
    int64_t vend_ = 0;                                          // end of the virtual space (ring_ without segments)
    uint8_t * real(int64_t v) const;                            // device address of virtual offset v
    // the usable region holding v: [*b, *e); false when v lies in no usable region
    bool region(int64_t v, int64_t * b, int64_t * e) const;
    int64_t next_region(int64_t v, int64_t need) const;         // start of the next usable region after v's with room
    void drop(const RingEntry & r);                             // forget an evicted entry
    // hint: refuse (false) to evict a protected expert; copy false: claim the slot only (claim())
    bool ring_put(int layer, int expert, bool hint = false, bool copy = true);
    struct Claim {
        int layer, expert;
        int64_t off;
        size_t done;                                            // bytes issued
    };
    std::deque<Claim> claims_;                                  // claim() order
    std::vector<std::pair<cudaEvent_t, std::vector<std::pair<int, int64_t>>>> landing_;   // last-piece event, claims
    cudaEvent_t piece_ev_ = nullptr, demand_ev_ = nullptr, fence_ev_ = nullptr;
    static constexpr int LANDING_EVENTS = 256;
    bool piece_out_ = false, demand_out_ = false;
    std::vector<cudaEvent_t> ev_pool_;
    size_t adm_backlog_ = 0;
    // admit_step() (driver thread, idle loops) against begin_stream() / begin_ring() (the thread that enqueues): a
    // prompt chunk must not start while a piece is being issued into what becomes its stream slots
    std::mutex adm_mu_;
    long adm_landed_ = 0;
    void claim_issue(Claim & c, size_t upto);                   // copy c's bytes [done, upto)
    void claim_flush(int layer, int expert);                    // fetch(): the rest of a claimed expert now
    std::vector<int> protect_;                                  // layer, experts the last fetch() needs
    // [layer] experts prefetch_hint() queued since that layer's last fetch(): their copies may still be in flight, so
    // a fetch() that needs one reports a copy (the GPU must wait for the copy stream)
    std::vector<std::vector<int>> hinted_;
    std::vector<std::pair<int, int>> adm_;                      // the admit() batch in flight
    cudaEvent_t adm_ev_ = nullptr;
    long adm_total_ = 0;
    int protect_layer_ = -1;

    std::vector<Layer> layers_;
    uint8_t * base_ = nullptr;                                     // ring start
    size_t ring_ = 0, slot_ = 0;
    void * spare_ = nullptr;
    int64_t head_ = 0;                                             // next free ring byte (FIFO)
    std::deque<RingEntry> fifo_;
    Mode mode_ = Mode::RING;
    std::vector<void *> device_, pinned_;
    uint8_t * host_region_ = nullptr;                              // every layer's cold copy: anonymous, registered
    size_t host_region_bytes_ = 0;
    std::unique_ptr<std::atomic<uint8_t>[]> reg_ok_;               // [layer] its cold copy is page-locked
    std::thread reg_thread_;                                       // page-locks the cold copy after the constructor
    std::atomic<bool> reg_stop_{ false };
    // TRUSS_STREAM_PROFILE=1: per-chunk stream measurements. Events are created in the ctor only when enabled.
    bool stream_prof_ = false;
    std::vector<cudaEvent_t> prof_pool_;                          // timing events, taken in order, all returned at harvest
    size_t prof_used_ = 0;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> busy_ev_, wait_ev_;
    mutable double prof_busy_ms_ = 0, prof_wait_ms_ = 0;
    cudaEvent_t prof_take() { return prof_pool_[prof_used_++]; }
    void prof_harvest();                                           // syncs the open pairs (chunk boundary only)
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
