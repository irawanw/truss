// Routed-expert weights and model shapes, shared by the MoE ops (moe_window: verify windows; moe_prefill: prompt
// chunks).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace truss::moe {

// Routed-expert shape of a model. A new model is a new struct plus explicit instantiations in the ops' .cu files.
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

}  // namespace truss::moe
