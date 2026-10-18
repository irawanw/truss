// deepseek41 weights as the blocks' views into a gguf::File (dense GGUF from tools/tk-pack/ds41_dense.py with the
// expert overlay of ds41_experts.py; mmnga/Strata deepseek41 tensor names). Routed experts are trellis tables;
// dense matrices Q8_0; norms, router, router bias, mHC, sinks, pooling compressor and Engram gate weights F32.
#pragma once
#include "blocks/weights.h"
#include "model/deepseek41/config.h"

#include <vector>

namespace truss::deepseek41 {

using T = blocks::T;

struct Layer {
    T attn_norm = nullptr, ffn_norm = nullptr;
    blocks::SkHcWeights hc_attn, hc_ffn;
    blocks::Ds41AttnWeights attn;
    blocks::MoeWeights moe;            // router, exp_bias, trellis experts, shared expert (no shared gate)
    blocks::EngramWeights engram;      // engram layers only
};

struct Weights {
    T token_embd = nullptr, final_norm = nullptr, output = nullptr;
    std::vector<Layer> layers;
};

Weights bind(const gguf::File & f, const Config & c);   // throws naming the tensor on any mismatch or unused tensor

}  // namespace truss::deepseek41
