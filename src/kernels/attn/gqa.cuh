// Dense causal grouped-query attention with a positional fp16 K/V cache (LFM2's attention layers; any GQA model).
//
//   prepare   q [T][H][D], k / v [T][HKV][D] fp32 projections -> per-head RMSNorm of q and k (weights may be
//             nullptr: no norm) -> NEOX rope over the whole head (angles in fp64: an fp32 angle is off by ~0.02 rad
//             at 256K) -> q16 [T][H][D] fp16; k, v written to the cache at positions pos0 .. pos0 + T - 1
//   attention out16 [T][H][D] = softmax(q k^T / sqrt(D)) v over cache positions 0 .. pos0 + t (causal), query head h
//             reads KV head h / (H / HKV)
//
// v1 (2026-10-14): CUDA-core tiles - a CTA takes BQ queries x one KV head (all its G query heads), K / V tiles of BC
// cells through shared memory, online softmax. Chunks of <= 32 rows split the cells across CTAs (one tile each;
// partial m / l / o in the workspace, merged by a combine kernel). Decode (T x G <= 16 rows, 2026-10-15): its own
// kernel - CTA = (KV head, split of 128 cells), lane = cell (K row in registers), warp softmax by shuffles, up to 256
// splits (LFM K3.5 decode 117 -> 147 tok/s at 512 cells, 47 -> 112 at 32K). Tensor-core
// tiles are the next step (kernels/trellis/mma.cuh fragments); the contract stays.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace truss::attn {

struct GqaShape {   // runtime shape; D must be 64 (static_assert in the kernels' instantiation)
    int H, HKV, D;
    float rope_base, eps;
};

// k_cache / v_cache [n_ctx][HKV][D] fp16
void gqa_prepare(const GqaShape & s, const float * q, const float * k, const float * v, const float * q_norm,
                 const float * k_norm, int pos0, int T, half * q16, half * k_cache, half * v_cache, cudaStream_t stream);

size_t gqa_workspace_bytes(const GqaShape & s, int max_T, int n_ctx);   // the split-cells partials (0 if none)
void gqa_attention(const GqaShape & s, const half * q16, const half * k_cache, const half * v_cache, int pos0, int T,
                   half * out16, void * ws, size_t ws_bytes, cudaStream_t stream);

}  // namespace truss::attn
