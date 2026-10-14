// The blocks' weight views: typed references into a gguf::File (host side, zero-copy), filled by a model family's
// binding (model/<family>/weights.cc) and read by the blocks. Host-only types, no CUDA.
#pragma once
#include "formats/gguf.h"
#include "formats/trellis_table.h"

namespace truss::blocks {

using T = const gguf::Tensor *;   // nullptr = not present

struct HcWeights {   // norm [hc_dim], down [hc_dim -> rank], up [rank -> hc_dim], inject [hc_dim -> hc]
    T norm = nullptr, down = nullptr, up = nullptr, inject = nullptr;
};

struct GdnWeights {
    T qkv = nullptr, gate = nullptr, conv1d = nullptr, dt_bias = nullptr, a = nullptr, beta = nullptr,
      alpha = nullptr, norm = nullptr, out = nullptr;
};

struct DsaWeights {
    T q = nullptr, k = nullptr, v = nullptr, out = nullptr, q_norm = nullptr, k_norm = nullptr;
    T idx_q = nullptr, idx_k = nullptr, idx_q_norm = nullptr, idx_k_norm = nullptr;
};

struct PleWeights {
    T key = nullptr, value = nullptr, norm_key = nullptr, norm_query = nullptr, norm_conv = nullptr, conv1d = nullptr;
};

struct MoeWeights {
    T router = nullptr;                    // [d_model -> n_expert], fp32
    formats::ExpertTable gate, up, down;   // routed experts, trellis
    T shexp_gate_inp = nullptr, shexp_gate = nullptr, shexp_up = nullptr, shexp_down = nullptr;   // optional
};

}  // namespace truss::blocks
