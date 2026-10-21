// DeepSeek-V4.1 attention mixer (dsref/model.py Attention, Compressor, Indexer): low-rank q (q_a -> rms -> q_b, 64
// heads of 512, rope on the last 64), one shared latent kv of 512 (rms, rope) that is both key and value, sparse
// attention over a 128-position sliding window plus, on compressing layers, the indexer's top-512 compressed entries,
// a per-head sink, inverse rope on the output, grouped output LoRA (8 groups x 1024) and the output projection.
//
// Layer roles (Config): a KV source compresses its own kv (ratio 1: a projection + rms per position; ratio 2: a
// softmax-gated pooling of 2 positions) into the compressed cache and its index keys; later layers with the same
// ratio read that cache. An index source scores the compressed positions with its own queries and publishes the
// top-512 for the layers after it; the candidate source (layer 20) also publishes its top 2048 blocks of 8, which
// later index sources restrict their scores to. Sources hand their buffers to consumers by pointer (set_sources);
// the layer split must keep a source and its consumers on one stage (DeepSeek-V4.1-Flash: split at 2, 8, 14, 20).
//
// Per slot: the window ring (R = 256 positions, >= window + any verify window), and on KV sources the compressed
// cache [seq_ctx / ratio][512] fp16, the index keys [seq_ctx / ratio][128] fp32 and, for ratio 2, the raw kv / score
// rings of the compressor. All are position-indexed and rewritten before they are read, so a verify window needs no
// undo. The top-k / candidate buffers are per chunk (max_rows rows), owned by the source block.
#pragma once
#include "blocks/linear.h"
#include "blocks/weights.h"
#include "engine/block.h"

#include <vector>

namespace truss::blocks {

struct Ds41AttnDims {
    int d_model, n_head, head_dim, rope_dim, q_rank, o_rank, o_groups, window;
    int index_heads, index_dim, index_top_k, candidate_top_blocks, candidate_block;
    int ratio;                 // this layer's compress ratio (0: window only)
    bool kv_source, index_source, candidate_source, uses_candidates;
    float eps;
    const float * inv_freq;    // device [rope_dim / 2]: this layer's rope (YaRN + compress theta when ratio > 0)
};

class AttnDs41 : public engine::Stateful {
public:
    // The window ring: the window (128 positions) plus the verify window plus the prompt path's row piece: a piece's
    // attention reads the positions before its start from the ring, so the ring must hold the piece's rows and the
    // window before them (the piece's rows are written to the ring before the piece's attention reads them).
    static constexpr int RING = 256 + 1024;

    AttnDs41(Linear & lin, const Ds41AttnWeights & w, const Ds41AttnDims & d) : lin_(lin), w_(w), d_(d) {}
    static void add_tensors(Linear & lin, const Ds41AttnWeights & w);
    // the prompt path's peak temporaries per row (the attention's output o is a piece, not a row) and the decode
    // path's (it also stacks the grouped LoRA's rows: the gemv path's ya)
    static size_t scratch_per_row(const Ds41AttnDims & d);
    static size_t scratch_decode_per_row(const Ds41AttnDims & d);
    static size_t scratch_piece(const Ds41AttnDims & d);   // the piece's output buffer (PREFILL_PIECE rows)
    static size_t scratch_fixed(const Ds41AttnDims & d, int n_ctx);
    // consumers: the layer whose compressed cache / top-k / candidates this layer reads (nullptr: itself / none)
    void set_sources(const AttnDs41 * kv, const AttnDs41 * index, const AttnDs41 * candidates)
    {
        kv_src_ = kv, idx_src_ = index, cand_src_ = candidates;
    }
    // in [T][d] (normed; fp32 and fp16 copies of the same rows) at positions pos0 .. -> out [T][d]
    // row_off: this layer's rows are the chunk's rows [row_off, row_off + T) (the tail prefill); clip: the window's
    // first position (the chunk's start: the positions before it were not written into the ring).
    void forward(engine::Gpu & g, int slot, const float * in, const half * in16, int pos0, int T, float * out,
                 int row_off = 0, int clip = 0);

    const int * topk() const { return topk_; }   // index sources: the chunk's top-k [rows][index_top_k] (device)
    void alloc_slots(engine::Stage & st, int n_slots, int seq_ctx, int spec_rows) override;
    // alloc_slots without a stage (tests/layer/ds41_attn_parity): n_ctx = n_slots * seq_ctx positions in all
    void alloc(engine::Gpu & g, int n_ctx, int max_rows, int n_slots, int seq_ctx, int spec_rows);
    void positional(std::vector<Positional> & out) const override;

private:
    struct Slot {
        half * ring = nullptr;                 // [RING][head_dim]
        half * comp = nullptr;                 // kv sources: [seq_ctx / ratio][head_dim]
        float * idx_k = nullptr;               // kv sources: [seq_ctx / ratio][index_dim]
        float * raw_kv = nullptr, * raw_sc = nullptr;   // ratio-2 kv sources: [RING][head_dim]
    };
    void compress(engine::Gpu & g, const Slot & S, const float * in, const half * in16, int pos0, int T);
    void index(engine::Gpu & g, int slot, const half * qr16, const half * in16, int pos0, int T);

    Linear & lin_;
    const Ds41AttnWeights w_;
    const Ds41AttnDims d_;
    const AttnDs41 * kv_src_ = nullptr, * idx_src_ = nullptr, * cand_src_ = nullptr;
    int n_ctx_ = 0, seq_ctx_ = 0;
    std::vector<Slot> sl_;
    int * topk_ = nullptr;                     // index sources: [max_rows][index_top_k] of the chunk in flight
    int * cand_ = nullptr;                     // candidate source: [max_rows][candidate_top_blocks]
};

}  // namespace truss::blocks
