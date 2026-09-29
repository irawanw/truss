// MoE window: routed-expert SwiGLU FFN for a small verify window (<= MAX_ROWS rows) on trellis expert weights.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace truss::moe {

constexpr int MAX_ROWS = 8;                // verify-window rows

// Routed-expert shape of a model. A new model is a new struct plus an explicit instantiation in moe_window.cu.
struct FlashNext {                         // Qwen3.8 Flash-Next: 512 experts, top-10
    static constexpr int D_MODEL = 2560, D_FF = 640, TOPK = 10;
};

struct ProjView {                          // one projection (gate, up or down) of all experts of a layer
    const uint16_t * trellis;              // concatenated tiles
    const int32_t * meta;                  // [n_expert][2] = (K, uint16 word offset)
    const half * suh;                      // [n_expert][in]
    const half * svh;                      // [n_expert][out]
};

struct Weights {
    ProjView proj[3];                      // gate, up, down
    int n_expert;
};

template <class Shape> size_t workspace_bytes();

// out[r] = sum_s w[r][s] * FFN_{ids[r][s]}(x[r]); x, out fp32 [n_rows][D_MODEL]; ids int32 / wts fp32
// [n_rows][TOPK]. n_rows <= MAX_ROWS. Device work only: capturable into a CUDA graph.
template <class Shape>
void window(const Weights & W, const float * x, const int * ids, const float * wts, int n_rows, float * out, void * ws,
            cudaStream_t stream);

}  // namespace truss::moe
