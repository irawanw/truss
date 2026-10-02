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
#include <cstdint>

namespace truss::dsa {

// A layer's K/V cache [n_ctx][HKV][D]: fp16 (k16, v16), or int8 codes (kq, vq) with an fp16 scale per 64 values
// (ks, vs [n_ctx][HKV][D / 64]; value = code * scale, scale = max |x| / 127 of the group): Strata's int8 KV format,
// 1,056 B per cell per layer instead of 2,048 (TRACKER #87). int8 when kq is set.
struct KvCache {
    half * k16 = nullptr, * v16 = nullptr;
    int8_t * kq = nullptr, * vq = nullptr;
    half * ks = nullptr, * vs = nullptr;
    __host__ __device__ bool int8() const { return kq != nullptr; }
};
constexpr int KV_GROUP = 64;

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

// Chunks of up to SPLIT_ROWS queries split each query's cells over up to 32 CTAs (flash-decoding) and need a
// workspace of attention_workspace_bytes(T) (0 for longer chunks). Split and unsplit results differ by fp32
// reassociation only.
constexpr int SPLIT_ROWS = 32;
template <class Shape> size_t attention_workspace_bytes(int max_queries);

// q [T][H][D] (normed, roped), gate [T][H][D] fp32, k, v [n_ctx][HKV][D] -> out [T][H * D] fp16 = softmax(q k / sqrt(D))
// v over the selected cells, times sigmoid(gate). q head h reads KV head h / (H / HKV).
template <class Shape>
void attention(const half * q, const float * gate, const half * k, const half * v, const int * blocks,
               const int * n_blocks, int pos0, int T, half * out, void * ws, size_t ws_bytes, cudaStream_t stream);
// the same over either cache format (int8: the gather dequantizes into the fp16 tile; the math after it is unchanged)
template <class Shape>
void attention(const half * q, const float * gate, const KvCache & kv, const int * blocks, const int * n_blocks,
               int pos0, int T, half * out, void * ws, size_t ws_bytes, cudaStream_t stream);

}  // namespace truss::dsa
