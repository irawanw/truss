// qwen4exp weights as typed references into a gguf::File (host side, zero-copy). bind() checks every shape
// against the Config and that every tensor in the file is used exactly once, so a converter change that adds,
// drops or reshapes a tensor fails at load instead of computing garbage. Device upload works from these references.
#pragma once
#include "formats/gguf.h"
#include "formats/trellis_table.h"
#include "blocks/weights.h"
#include "model/qwen4exp/config.h"

#include <vector>

namespace truss::qwen4exp {

using T = blocks::T;                       // nullptr = not present in this layer

// the blocks' weight views (src/blocks/), named as this family's binding names them
using HyperConnection = blocks::HcWeights;   // norm [hc_dim], down [hc_dim -> rank], up [rank -> hc_dim], inject
using Gdn = blocks::GdnWeights;
using Dsa = blocks::DsaWeights;
using Ple = blocks::PleWeights;
using Moe = blocks::MoeWeights;              // router fp32, routed experts trellis, gated shared expert

struct Layer {
    Mixer mixer;
    HyperConnection hc_attn, hc_ffn;
    Gdn gdn;                               // filled when mixer == GDN
    Dsa dsa;                               // filled when mixer == DSA
    Ple ple;                               // filled when Config::is_ple(layer)
    Moe moe;
};

struct Weights {
    T token_embd = nullptr, output = nullptr;
    HyperConnection hc_head;               // final hyper-connection mix before the head (no inject)
    T ple_table = nullptr, ple_scale = nullptr;   // int8 rows [ple_head_dim] with one fp16 scale per row
    std::vector<Layer> layers;
};

// Throws std::runtime_error naming the tensor on a missing tensor, a shape mismatch or an unused tensor.
Weights bind(const gguf::File & f, const Config & c);

// The MTP (multi-token prediction) draft block, blk.<n_layer> of its own file (flashnext_truss_mtp_pack.py): a DSA
// layer with routed experts, plus the join of the next token's embedding with the previous hidden state,
//     res = eh_proj [RMSNorm(embed(token)) * enorm | RMSNorm_per_stream(h) * hnorm]   (per hc stream)
// It uses the main model's token_embd, output and hc_head (llama-paw graph_mtp).
struct Mtp {
    Layer layer;
    T eh_proj = nullptr, enorm = nullptr, hnorm = nullptr;
};
Mtp bind_mtp(const gguf::File & f, const Config & c);

}  // namespace truss::qwen4exp
