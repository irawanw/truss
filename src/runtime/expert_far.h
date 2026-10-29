// The far tier (docs/PLAN-20261024-ds41-nvme-tier.md): cold experts that have no host copy (pack E2's rare experts,
// K2 on the NVMe) are read on demand into a fixed set of host slots, which the ExpertStore copies to the GPU like a host
// expert. This file is CUDA-free: the owner hands it the slot region (page-locked by the owner) and a read function, so
// tests/unit/expert_far_test.cc runs it on a plain file.
//
// Model: n_slots slots of slot_bytes each; a far expert (key = layer * n_expert + expert) is EMPTY, QUEUED, READING or
// READY in at most one slot. want() queues a read (demand ahead of hints); TRUSS_FAR_THREADS reader threads take the
// queue, pick a victim slot and read the expert's three projections into it (gate | up | down). A read is cut into
// pieces of piece_bytes (TRUSS_FAR_PIECE_KB) that every idle reader takes, so one expert is read at queue depth
// `threads` instead of 1 (10-25: one 4 MiB-at-a-time reader gets ~1.8 GB/s of the drive's 3.5; a decode layer
// usually waits for a single expert). Idle readers help a demand read before they start a new one, then hints.
// Eviction: least recently used among slots nobody holds (F1, 10-24: LRU beat LFU with decay on the API's reasoning
// routing: 6 GB -> 2.94 vs 5.10 misses/token). acquire() returns a READY slot's bytes and holds it (pin count) until
// release(): the owner's PCIe copy reads them, so no reader may overwrite them meanwhile.
//
// Threads: every public call is thread-safe (one mutex; reads run outside it). Hints beyond max_hints queued are
// dropped, oldest first (a stale guess is worth less than a fresh one); a hint never evicts a held slot, one whose
// read is in flight or one a demand read filled that nobody acquired yet; a demand want() moves a queued hint to the
// demand queue.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace truss::runtime {

class FarCache {
public:
    struct Span {                                     // one projection of a far expert in its file
        const void * src = nullptr;                   // its bytes in the mapping (read_direct finds the file from it)
        size_t bytes = 0;
    };
    // read n bytes of the mapping at src into dst (thread-safe; bounce is this thread's DIRECT_ALIGN-aligned buffer
    // of bounce_bytes); false on a failed read. Default (null): gguf::read_direct, then gguf::read_mapped.
    using ReadFn = std::function<bool(const void * src, size_t n, void * dst, void * bounce, size_t bounce_bytes)>;

    struct Config {
        int n_layer = 0, n_expert = 0;
        size_t slot_bytes = 0;                        // >= the largest far expert
        int n_slots = 0;
        int threads = 8;
        int max_hints = 64;
        size_t bounce_bytes = 4u << 20;
        size_t piece_bytes = 1u << 20;                // 0: each read whole, by one reader
    };

    // far[key] = the expert's three spans; an expert whose spans are all empty is not far (want() ignores it).
    // region: n_slots * slot_bytes bytes, owned by the caller (page-locked for the GPU copies), outlives the cache.
    FarCache(const Config & c, std::vector<std::array<Span, 3>> far, uint8_t * region, ReadFn read = nullptr);
    ~FarCache();
    FarCache(const FarCache &) = delete;
    FarCache & operator=(const FarCache &) = delete;

    bool is_far(int l, int e) const { return far_[key(l, e)][0].bytes != 0; }
    size_t bytes_of(int l, int e) const;              // gate + up + down

    // queue a read unless the expert is READY, being read or already queued (a demand want() of a queued hint
    // promotes it). Returns true when the expert is READY now.
    bool want(int l, int e, bool demand);
    // the READY expert's bytes [gate | up | down], held until release(); nullptr when not READY. Counts a hit.
    const uint8_t * acquire(int l, int e);
    void release(int l, int e);
    // block until the expert is READY or its read failed (want(demand) first if it is not queued); nullptr on failure
    // or after timeout_ms (< 0: no limit). Holds it like acquire(). count false: a repeat wait of one demand (a caller
    // waiting in steps), not counted as another hit / miss.
    const uint8_t * wait(int l, int e, int timeout_ms = -1, bool count = true);
    const uint8_t * wait_until(int l, int e, std::chrono::steady_clock::time_point deadline, bool count = true);
    bool failed(int l, int e);                        // its last read failed
    void drop_hints();                                // forget the queued hints (a prompt chunk is about to start)

    struct Stats {
        long hits = 0, misses = 0;                    // acquire()/wait(): READY at the call / not
        long reads = 0, read_fail = 0, hint_reads = 0, hints_dropped = 0, evictions = 0;
        size_t read_bytes = 0;
        double read_ms = 0;                           // summed over reader threads
    };
    Stats stats() const;
    std::string report() const;                       // one line for the bench / server log

private:
    enum State : uint8_t { EMPTY, QUEUED, READING, READY };
    struct Slot {
        int key = -1;
        int held = 0;
        std::list<int>::iterator lru;                 // position in lru_ (only while READY and not held)
        bool in_lru = false;
        bool fresh = false;                           // landed from a demand read, not acquired yet: hints keep off
    };
    int key(int l, int e) const { return l * cfg_.n_expert + e; }
    // a read in flight: its pieces (byte ranges of the spans, in slot order) go to whichever reader is free
    struct Piece {
        const uint8_t * src;
        size_t n, at;                                 // bytes, offset in the slot
    };
    struct Job {
        int key, slot;
        bool hint;
        std::vector<Piece> pieces;
        size_t next = 0, done = 0;                    // pieces handed out / finished
        bool ok = true;
        size_t bytes = 0;
    };
    std::list<Job> jobs_;                             // in start order
    Job * open_job(bool demand_only);                 // the oldest job with a piece left (demand ones only), or null
    Job * start_job(bool hint);                       // the next queued key into a victim slot; null: none / no slot
    void finish_job(std::list<Job>::iterator j);
    void worker();
    int victim(bool hint);                            // a free or least recently used unheld slot (a hint skips
                                                      // fresh ones); -1 if none
    void lru_remove(int s);
    void lru_push(int s);

    Config cfg_;
    std::vector<std::array<Span, 3>> far_;
    uint8_t * region_;
    ReadFn read_;
    std::vector<uint8_t> state_;                      // [key]
    std::vector<int> slot_of_;                        // [key] slot while READING / READY, -1 otherwise
    std::vector<uint8_t> failed_;                     // [key] its last read failed (wait() returns nullptr)
    std::vector<Slot> slots_;
    std::list<int> lru_;                              // READY unheld slots, least recent first
    std::vector<int> free_;                           // slots holding nothing
    std::deque<int> demand_q_, hint_q_;               // keys
    mutable std::mutex mu_;
    std::condition_variable work_cv_, done_cv_;
    std::vector<std::thread> threads_;
    bool stop_ = false;
    Stats st_;
};

}  // namespace truss::runtime
