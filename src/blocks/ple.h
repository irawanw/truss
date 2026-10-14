// PLE: per-layer n-gram embeddings (Flash-Next's PLE layers). Two pieces:
//   PleTable  one per stage: the n-gram window of each slot (the last ngram - 1 committed tokens), the host table
//             reader (O_DIRECT reads with a row cache, TRACKER #88, or the file mapping) and the lookahead that reads
//             the next chunk's rows while this chunk computes (E8). Every PLE layer gathers the same rows.
//   Ple       one per PLE layer: key / value projections of the gathered rows, the gate against the residual and the
//             causal conv over its history (conv - 1 rows of ngram streams), added into the residual in place.
#pragma once
#include "blocks/weights.h"
#include "blocks/linear.h"
#include "blocks/ple_hash.h"
#include "blocks/ple_reader.h"
#include "engine/block.h"

#include <memory>
#include <vector>

namespace truss::blocks {


class PleTable : public engine::Stateful {
public:
    // file: read rows with O_DIRECT from it (nullptr: through the mapping); max_rows: the largest chunk
    PleTable(const PleDims & d, const gguf::Tensor * table, const gguf::Tensor * scale, const gguf::File * file,
             int max_rows);
    ~PleTable();
    const PleDims & dims() const { return d_; }
    void set_next(const int32_t * next, int next_T) { next_ = next, next_T_ = next_T; }   // the chunk's lookahead
    // the rows of tokens [T] at pos0 of `slot`, fp16 [T][heads * head_dim], copied into e16 (device) on g's stream
    void gather(engine::Gpu & g, int slot, const int32_t * tokens, int T, int pos0, half * e16);
    bool direct() const { return reader_ != nullptr; }
    void stats(double & ms, long & calls, long & rows, bool reset, double * wait_ms, double * phase_ms);

    void alloc_slots(engine::Stage & st, int n_slots, int seq_ctx, int spec_rows) override;
    void reset(engine::Stage & st, int slot) override;
    void commit(engine::Stage & st, int slot, const int32_t * tokens, int n) override;
    void ck_save(int slot, std::vector<int32_t> & host) override;
    void ck_load(int slot, const std::vector<int32_t> & host, size_t & at) override;

private:
    const PleDims d_;
    const gguf::Tensor * table_, * scale_;
    std::unique_ptr<PleReader> reader_;
    half * stage_ = nullptr;                   // pinned [rows][E]: the gathered rows for the device copy
    size_t stage_n_ = 0;
    cudaEvent_t copied_ = nullptr;             // the last copy out of stage_ has been read
    std::vector<std::vector<int32_t>> tail_;   // [slot] the last ngram - 1 committed tokens
    const int32_t * next_ = nullptr;
    int next_T_ = 0;
    bool tk_valid_ = false;
    int tk_ = -1, tk_pos0_ = -1, tk_slot_ = -1;
    double host_ms_ = 0;
    long calls_ = 0, rows_n_ = 0;
};

class Ple : public engine::Stateful {
public:
    Ple(Linear & lin, const PleWeights & w, PleTable & table) : lin_(lin), w_(w), t_(table) {}
    static void add_tensors(Linear & lin, const PleWeights & w);
    static size_t scratch_per_row(const PleDims & d);
    // res [T][hc][d] += the layer's PLE term (and its history advances)
    void forward(engine::Gpu & g, int slot, const int32_t * tokens, int T, int pos0, float * res, bool tentative);

    void alloc_slots(engine::Stage & st, int n_slots, int seq_ctx, int spec_rows) override;
    void reset(engine::Stage & st, int slot) override;
    void accept(engine::Stage & st, int slot, int n, const engine::Window & w) override;
    void ck_buffers(int slot, std::vector<std::pair<float *, size_t>> & out) override;

private:
    size_t hist_floats() const;
    struct Slot {
        float * hist = nullptr, * hist_snap = nullptr, * normed_rows = nullptr;
    };
    Linear & lin_;
    const PleWeights w_;
    PleTable & t_;
    std::vector<Slot> s_;
};

}  // namespace truss::blocks
