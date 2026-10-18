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
    T exp_bias = nullptr;                  // [n_expert] fp32 selection bias (sigmoid routers), optional
};

struct SkHcWeights {   // DeepSeek-V4 mHC around one sub-block: fn [(2 + hc) hc][hc d] fp32, base [(2 + hc) hc], scale [3]
    T fn = nullptr, base = nullptr, scale = nullptr;
};

struct Ds41AttnWeights {   // DeepSeek-V4.1 attention (blocks/attn_ds41): q / kv / grouped output LoRA, sinks; the
                           // compressor and indexer parts only on their source layers (nullptr elsewhere)
    T q_a = nullptr, q_a_norm = nullptr, q_b = nullptr, kv = nullptr, kv_norm = nullptr, out_a = nullptr,
      out_b = nullptr, sinks = nullptr;
    T cmp_kv = nullptr, cmp_gate = nullptr, cmp_norm = nullptr;          // compressor (kv sources; gate: ratio > 1)
    T idx_k = nullptr, idx_k_norm = nullptr;                             // index keys (kv sources)
    T idx_q_b = nullptr, idx_proj = nullptr;                             // index queries (index sources)
};

struct EngramWeights {   // DeepSeek Engram: wkv [buckets head_dim -> (hc + 1) d], q / k gate weights [hc][d]; the
                         // hash table (int4 rows, its own overlay file) is read on the host, never uploaded
    T wkv = nullptr, q = nullptr, k = nullptr, table = nullptr;
};

}  // namespace truss::blocks
