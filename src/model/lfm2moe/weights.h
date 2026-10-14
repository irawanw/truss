// lfm2moe weights as the blocks' views into a gguf::File (llama.cpp tensor names: blk.N.attn_norm is the operator
// norm, token_embd_norm the final norm, the head is tied to token_embd). Routed experts are trellis tables
// (m3_trellis / m3_meta / m3_suh / m3_svh); dense matrices Q8_0; norms, conv, router and its bias F32.
#pragma once
#include "blocks/attn_gqa.h"
#include "blocks/ffn_dense.h"
#include "blocks/shortconv.h"
#include "blocks/weights.h"
#include "model/lfm2moe/config.h"

#include <vector>

namespace truss::lfm2moe {

using T = blocks::T;

struct Layer {
    T op_norm = nullptr, ffn_norm = nullptr;
    blocks::ShortConvWeights conv;          // short-conv layers
    blocks::GqaWeights attn;                // attention layers
    blocks::DenseFfnWeights dense;          // leading dense layers
    blocks::MoeWeights moe;                 // MoE layers (router, exp_bias, trellis experts)
};

struct Weights {
    T token_embd = nullptr, final_norm = nullptr;
    std::vector<Layer> layers;
};

Weights bind(const gguf::File & f, const Config & c);   // throws naming the tensor on any mismatch or unused tensor

}  // namespace truss::lfm2moe
