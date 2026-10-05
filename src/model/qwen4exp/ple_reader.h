// PLE table rows read straight from the SSD (Strata's ngram::PleReader, plan §5 S2).
//
// The int8 table (320M rows of ple_head_dim bytes, 48 GiB) and its fp16 scale tensor are never read through the
// mmap: a page fault per row page and per scale page, one at a time on the host thread, cost 5.0 s of a 14.1 s
// 21.7K-token prompt read and ~3 ms per decode pass (TRACKER #87). Here every row a call needs is served from a
// bounded row cache or from 4 KiB preads (buffered: page-cache hits stay cheap): the call's pages are deduplicated and read by a pool
// of worker threads, many in flight.
//
//     int t = reader.issue(rows, n);   // as soon as the token ids are known
//     ...                              // other work
//     reader.collect(t, emb16);        // the rows dequantized: emb16 [n][row_bytes] = row x its scale
//
// One ticket in flight at a time (the next issue waits for the previous ticket's reads). The cache keeps rows this
// process has read (row bytes + scale, clock eviction); capacity 0 disables it.
#pragma once
#include "formats/gguf.h"

#include <cuda_fp16.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace truss::qwen4exp {

class PleReader {
public:
    // table: int8 [rows][row_bytes]; scale: fp16 [rows]; both in shards of `file`
    PleReader(const gguf::File & file, const gguf::Tensor & table, const gguf::Tensor & scale, int row_bytes,
              int threads = 64, size_t cache_rows = 1u << 20);
    ~PleReader();
    PleReader(const PleReader &) = delete;
    PleReader & operator=(const PleReader &) = delete;

    // cache_misses=false: read the rows but do not populate the cache — a prefill chunk's trigram rows are read
    // once; caching them grows the map/cache working set for no reuse and slows every later lookup.
    int issue(const int32_t * rows, size_t n, bool cache_misses = true);
    void collect(int ticket, half * emb);   // emb [n][row_bytes]

    struct Stats {
        uint64_t rows = 0, cache_hits = 0, pages = 0;   // rows asked for, served by the cache, 4 KiB pages read
        double wait_ms = 0;                             // collect blocked on reads
        double phase_ms[3] = {};                        // collect's host phases: 0 hits put, 1 miss copy+put,
                                                        // 2 miss map+insert
    };
    const Stats & stats() const { return stats_; }
    double take_wait_ms() const;   // collect's cumulative block on the SSD reads since the last take (resets)
    void take_phase_ms(double out[3]) const;   // ditto for the host phases

private:
    struct Page {
        int fd;
        uint64_t off;   // 4 KiB aligned
        uint8_t * buf;
    };
    void worker();
    const uint8_t * page_of(int fd, uint64_t off) const;   // the read page holding byte `off`
    int insert(int32_t row, const int8_t * bytes, half scale);
    // rows [lo,hi) -> emb (hits from the cache, misses copied from the ticket's pages + dequantized). Read-only
    // over the ticket's state, so several ranges run at once; ms receives the thread's own elapsed time.
    void copy_put_rows(size_t lo, size_t hi, half * emb, double & ms) const;

    int row_bytes_;
    int fd_table_ = -1, fd_scale_ = -1;
    uint64_t table_off_, scale_off_;
    // the ticket being read
    std::vector<int32_t> rows_;
    std::vector<int> slot_;                       // per row: cache slot, or -1 (read from pages)
    std::vector<Page> pages_;
    std::unordered_map<uint64_t, uint8_t *> page_at_;   // (fd << 48 | page index) -> buffer
    std::vector<uint8_t *> bufs_;                 // aligned page buffers (grown, reused)
    int ticket_ = 0;
    bool cache_misses_ = true;   // per ticket: misses populate the cache
    // pool
    std::vector<std::thread> pool_;
    mutable std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::atomic<size_t> next_{ 0 };
    size_t n_pages_ = 0, finished_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
    int io_errno_ = 0;
    // row cache
    size_t cap_;
    std::unordered_map<int32_t, int> where_;
    std::vector<int32_t> cache_row_;
    std::vector<int8_t> cache_bytes_;
    std::vector<half> cache_scale_;
    std::vector<uint8_t> ref_;
    size_t hand_ = 0;
    mutable Stats stats_;
};

}  // namespace truss::qwen4exp
