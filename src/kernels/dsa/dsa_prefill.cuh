// DSA (qwen4exp QSA) prefill: block-sparse attention for a prompt chunk, one sequence.
//
// Keys live in blocks of RATIO cells. The indexer scores every complete block a query sees,
//   score(t, b) = sum_h relu(idx_q[t][h] . idx_k[b]),
// and the query attends to the TOP_BLOCKS best blocks (ties: the later block) plus its tail, the cells after its
// last complete block: the rule of ref::qsa_select (TRACKER #47). A query that sees <= TOP_BLOCKS blocks attends to
// all of them, i.e. plain causal attention.
//
// Two ops, so the selection can be reused and tested on its own:
//   select     indexer scores on tensor cores (fp16 inputs, fp32 accumulation) into a workspace, then a per-query
//              radix select; out: the chosen block ids, ascending
//   attention  per (query, KV head) the gathered cells through an online softmax on tensor cores: fp16 q/k/v/P,
//              fp32 scores, softmax state and P.V accumulation (llama-paw accumulates P.V in fp16, TRACKER #48);
//              the output gate is fused: out = attn * sigmoid(gate), fp16 for the out projection
// Caches are fp16, indexed by absolute position (block b = cells RATIO b .. RATIO b + RATIO - 1). The chunk's
// queries are positions pos0 .. pos0 + T - 1; the caches must already hold every cell and complete block up to
// pos0 + T - 1.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace truss::dsa {

struct FlashNext {                    // Qwen3.8 Flash-Next DSA layers
    static constexpr int H = 24, HKV = 2, D = 256;                   // attention heads, head dim
    static constexpr int IH = 4, ID = 128;                           // indexer heads, head dim
    static constexpr int RATIO = 4, TOP_BLOCKS = 2048 / RATIO;       // idx_top_k = 2048 cells
    static constexpr int ROPE_DIMS = 64;                             // NEOX rope on the first 64 dims of each head
};

// workspace for select() at context length n_ctx (queries are processed in groups that fit it)
template <class Shape> size_t select_workspace_bytes(int max_queries, int n_ctx);

// idx_q [T][IH][ID] (normed, roped), idx_k [n_blocks][ID] (pooled, normed, roped at the block start) ->
// blocks [T][TOP_BLOCKS] (the first n_blocks[t] entries used, ascending), n_blocks [T].
template <class Shape>
void select(const half * idx_q, const half * idx_k, int pos0, int T, int * blocks, int * n_blocks, void * ws,
            size_t ws_bytes, cudaStream_t stream);

// q [T][H][D] (normed, roped), gate [T][H][D] fp32, k, v [n_ctx][HKV][D] -> out [T][H * D] fp16 = softmax(q k / sqrt(D))
// v over the selected cells, times sigmoid(gate). q head h reads KV head h / (H / HKV).
template <class Shape>
void attention(const half * q, const float * gate, const half * k, const half * v, const int * blocks,
               const int * n_blocks, int pos0, int T, half * out, cudaStream_t stream);

}  // namespace truss::dsa
