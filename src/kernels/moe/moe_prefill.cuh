// MoE prefill: routed-expert SwiGLU FFN for a prompt chunk (thousands of tokens) on trellis expert weights.
//
// At prefill every expert gets ~T * TOPK / n_expert rows (160 at T = 8192), so the op is tensor-core bound, not
// read bound (TRACKER #40): each decoded weight tile is reused across up to 64 rows, accumulation is fp32.
// Same math and rounding points as moe_window (the verify-window op), so the two agree to fp32 summation order.
#pragma once
#include "moe_weights.cuh"

#include <cuda_runtime.h>
#include <cstddef>

namespace truss::moe {

// device workspace for up to max_tokens tokens per call
template <class Shape> size_t prefill_workspace_bytes(int max_tokens);

// out[t] = sum_s w[t][s] * FFN_{ids[t][s]}(x[t]); x, out fp32 [n_tokens][D_MODEL]; ids int32 / wts fp32
// [n_tokens][TOPK], each row's ids distinct; an id < 0 is skipped (adds 0). n_tokens <= the workspace's max_tokens. Device work only, on `stream`.
template <class Shape>
void prefill(const Weights & W, const float * x, const int * ids, const float * wts, int n_tokens, float * out,
             void * ws, int max_tokens, cudaStream_t stream);

}  // namespace truss::moe
