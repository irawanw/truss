// qwen4exp weights as typed references into a gguf::File (host side, zero-copy). bind() checks every shape
// against the Config and that every tensor in the file is used exactly once, so a converter change that adds,
// drops or reshapes a tensor fails at load instead of computing garbage. Device upload works from these references.
#pragma once
#include "formats/gguf.h"
#include "formats/trellis_table.h"
#include "model/qwen4exp/config.h"

#include <vector>

namespace truss::qwen4exp {

using T = const gguf::Tensor *;            // nullptr = not present in this layer

struct HyperConnection {                   // norm [hc_dim], down [hc_dim -> rank], up [rank -> hc_dim], inject [hc_dim -> hc]
    T norm = nullptr, down = nullptr, up = nullptr, inject = nullptr;
};

struct Gdn {
    T qkv = nullptr, gate = nullptr, conv1d = nullptr, dt_bias = nullptr, a = nullptr, beta = nullptr,
      alpha = nullptr, norm = nullptr, out = nullptr;
};

struct Dsa {
    T q = nullptr, k = nullptr, v = nullptr, out = nullptr, q_norm = nullptr, k_norm = nullptr;
    T idx_q = nullptr, idx_k = nullptr, idx_q_norm = nullptr, idx_k_norm = nullptr;
};

struct Ple {
    T key = nullptr, value = nullptr, norm_key = nullptr, norm_query = nullptr, norm_conv = nullptr,
      conv1d = nullptr;
};

struct Moe {
    T router = nullptr;                    // [d_model -> n_expert]
    formats::ExpertTable gate, up, down;   // routed experts, trellis
    T shexp_gate_inp = nullptr, shexp_gate = nullptr, shexp_up = nullptr, shexp_down = nullptr;
};

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

}  // namespace truss::qwen4exp
