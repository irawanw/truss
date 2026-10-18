// Routed-expert weights and model shapes, shared by the MoE ops (moe_window: verify windows; moe_prefill: prompt
// chunks).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace truss::moe {

// Routed-expert shape of a model. A new model is a new struct plus explicit instantiations in the ops' .cu files.
// SWIGLU_LIMIT > 0: the gate is clamped from above and the up branch on both sides before silu(gate) * up.
struct FlashNext {                         // Qwen3.8 Flash-Next: 512 experts, top-10
    static constexpr int D_MODEL = 2560, D_FF = 640, TOPK = 10;
    static constexpr const char * NAME = "FlashNext";
    static constexpr float SWIGLU_LIMIT = 0.0f;
};
struct Lfm2 {                              // LFM2.5-24B-A2B: 64 experts, top-4
    static constexpr int D_MODEL = 2048, D_FF = 1536, TOPK = 4;
    static constexpr const char * NAME = "Lfm2";
    static constexpr float SWIGLU_LIMIT = 0.0f;
};
struct DsV41 {                             // DeepSeek-V4.1-Flash-1017: 384 experts (256 kept by the pack), top-6
    static constexpr int D_MODEL = 5120, D_FF = 2304, TOPK = 6;
    static constexpr const char * NAME = "DsV41";
    static constexpr float SWIGLU_LIMIT = 10.0f;   // config swiglu_limit (routed and shared experts)
};

#ifdef __CUDACC__
// silu(g) * u with the shape's SwiGLU clamp (none at SWIGLU_LIMIT 0; folded away at compile time), as both MoE ops
// compute their mid
template <class Shape> __device__ __forceinline__ float swiglu(float g, float u)
{
    constexpr float L = Shape::SWIGLU_LIMIT;
    if constexpr (L > 0.0f) {
        g = fminf(g, L);
        u = fminf(fmaxf(u, -L), L);
    }
    return g / (1.0f + __expf(-g)) * u;
}
#endif

struct ProjView {                          // one projection (gate, up or down) of all experts of a layer
    const uint16_t * trellis;              // base of the tiles
    const int32_t * meta;                  // [n_expert][2] = (K, offset): expert e at trellis + (offset << shift) words
    const half * suh;                      // [n_expert][in]
    const half * svh;                      // [n_expert][out]
    int shift = 0;                         // 0: GGUF word offsets; runtime::ExpertStore uses 4 (32-byte units, so an
                                           // int32 offset reaches +-64 GiB of the base)
};

struct Weights {
    ProjView proj[3];                      // gate, up, down
    int n_expert;
};

}  // namespace truss::moe
