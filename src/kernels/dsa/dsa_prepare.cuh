// DSA mixer inputs for prefill (the math of qwen4exp::reference::dsa before the selection; the projections are
// dense GEMMs run by the caller):
//   rope_table     cos/sin per (position, rotary pair) for the chunk, from fp64 angles: at 256K positions an fp32
//                  angle is off by ~0.02 rad
//   prepare_qkv    q | gate split, per-head RMSNorm and rope of q and k, fp16 q, fp32 gate, K/V cache rows
//   prepare_index  indexer queries normed and roped; raw indexer keys mean-pooled per complete block, normed, roped
//                  at the block start into the block cache (a block may span chunks: decode adds one cell at a time)
// Rope is NEOX on the first ROPE_DIMS dims: pair (p, p + ROPE_DIMS / 2), angle pos * base^(-2p / ROPE_DIMS).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace truss::dsa {

// cs [T][ROPE_DIMS / 2] for positions pos0 .. pos0 + T - 1
template <class Shape> void rope_table(int pos0, int T, float base, float2 * cs, cudaStream_t stream);

// qfull [T][H][2 D] (per head q | gate), k, v [T][HKV][D] raw -> q16 [T][H][D], gate [T][H][D],
// k_cache / v_cache [n_ctx][HKV][D] rows pos0 .. pos0 + T - 1. q_norm, k_norm [D].
template <class Shape>
void prepare_qkv(const float * qfull, const float * k, const float * v, const float * q_norm, const float * k_norm,
                 const float2 * cs, int pos0, int T, float eps, half * q16, float * gate, half * k_cache,
                 half * v_cache, cudaStream_t stream);

// idx_q [T][IH][ID], idx_k [T][ID] raw -> idx_q16 [T][IH][ID]; every block that completes in the chunk is mean-pooled,
// normed and roped at its start into idx_k_cache [n_ctx / RATIO][ID]. partial [RATIO - 1][ID] fp32 carries the raw
// keys of the block still open at the end of the previous chunk (row i = cell RATIO b + i) and is updated.
template <class Shape>
void prepare_index(const float * idx_q, const float * idx_k, const float * q_norm, const float * k_norm,
                   const float2 * cs, float rope_base, int pos0, int T, float eps, float * partial, half * idx_q16,
                   half * idx_k_cache, cudaStream_t stream);

}  // namespace truss::dsa
