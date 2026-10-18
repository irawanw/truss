// DeepSeek-V4.1 Engram (dsref/model.py Engram + engram.py NgramHashState): at its layers, before the attention, an
// n-gram lookup is written into each residual copy, gated by how well it matches that copy.
//   host    the 2- .. max_ngram-grams ending at each position over the compressed vocab (token_map; pad before the
//           sequence start), hashed per (n, head) into its prime-sized bucket range: buckets = (max_ngram - 1) x heads
//           row ids per token (24); rows read from the int4 table file (tools/tk-pack/ds41_engram.py) by a pool of
//           pread threads into pinned memory
//   device  rows -> fp16 [T][buckets x 256] -> wkv (Q8_0) -> [T][(hc + 1) d] -> kernels/ds41 engram_gate
// Per slot: the last max_ngram - 1 committed tokens (commit(); reset() clears), so a chunk hashes its first rows
// across the chunk boundary; a verify window's rows use the window's own tokens and commit only what is accepted.
#pragma once
#include "blocks/linear.h"
#include "blocks/weights.h"
#include "engine/block.h"
#include "formats/gguf.h"

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace truss::blocks {

struct EngramDims {
    int hc, d_model, heads, head_dim, max_ngram, pad_id;
    float eps;
    const int64_t * multipliers;   // [max_ngram] of this layer
    const int64_t * primes;        // [buckets] of this layer, (n - 2) x heads + head
    const int64_t * offsets;       // [buckets]
    const int64_t * token_map;     // [n_vocab]
    int64_t n_vocab;
    int buckets() const { return (max_ngram - 1) * heads; }
};

class Engram : public engine::Stateful {
public:
    // table: blk.<L>.engram_embd.q4 (I8 [144][rows]) in a mapped GGUF
    Engram(Linear & lin, const EngramWeights & w, const EngramDims & d, const gguf::Tensor & table, int threads = 16);
    ~Engram() override;
    static void add_tensors(Linear & lin, const EngramWeights & w) { lin.add(w.wkv), lin.add(w.q), lin.add(w.k); }
    static size_t scratch_per_row(const EngramDims & d);
    void build(engine::Gpu & g);   // after upload: the q * k product
    // res [T][hc][d] in place, tokens (host) of the chunk at positions pos0 ..
    void forward(engine::Gpu & g, int slot, const int32_t * tokens, int pos0, int T, float * res);
    // the row ids of a chunk (host; exposed for the hash test)
    void hash(int slot, const int32_t * tokens, int pos0, int T, int32_t * ids) const;

    void alloc_slots(engine::Stage & st, int n_slots, int seq_ctx, int spec_rows) override;
    void reset(engine::Stage &, int slot) override;
    void commit(engine::Stage &, int slot, const int32_t * tokens, int n) override;
    void ck_save(int slot, std::vector<int32_t> & host) override;
    void ck_load(int slot, const std::vector<int32_t> & host, size_t & at) override;
    void alloc(int n_slots);   // alloc_slots without a stage (tests)
    void commit_tokens(int slot, const int32_t * tokens, int n);   // commit() without a stage

private:
    void read_rows(const int32_t * ids, size_t n, uint8_t * dst);
    void worker();

    Linear & lin_;
    const EngramWeights w_;
    const EngramDims d_;
    const gguf::Tensor & table_;
    float * qk_ = nullptr;
    std::vector<std::vector<int32_t>> hist_;     // per slot: last max_ngram - 1 committed tokens, oldest first
    uint8_t * pinned_ = nullptr;
    size_t pinned_rows_ = 0;
    // pread pool
    std::vector<std::thread> pool_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::function<void(size_t)> job_;
    size_t n_jobs_ = 0, next_ = 0, done_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
};

}  // namespace truss::blocks
