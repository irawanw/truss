// qwen4exp forward, block by block, on the reference ops (fp32). It defines the model math for TRUSS and is
// checked against llama-paw activations (tests/layer/qwen4exp_parity). Each block takes and returns device fp32
// activations; temporaries come from the caller's Scratch.
//
// Shapes: T tokens; the residual is hc parallel streams, res [T][hc][d_model]; block inputs/outputs [T][d_model].
#pragma once
#include "core/device_tensors.h"
#include "core/scratch.h"
#include "kernels/reference/ref.cuh"
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/weights.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace truss::qwen4exp::reference {

struct Ctx {
    const Config & c;
    const DeviceTensors & dev;
    Scratch & scratch;
    cudaStream_t stream;
    ref::Numerics num = ref::Numerics::FP32;   // LLAMA: reproduce llama-paw's rounding (parity tests)
};

// Hyper-connection mix (replaces the pre-norm): mixed [T][d]; inject [T][hc] (skipped when h.inject is null).
void hc_mix(const Ctx & x, const HyperConnection & h, const float * res, int T, float * mixed, float * inject);

// res[t][c] += block_out[t] * 2 * sigmoid(inject[t][c] / hc)
void hc_combine(const Ctx & x, float * res, const float * block_out, const float * inject, int T);

// --- GDN mixer (llama-paw build_layer_attn_linear), input = the hc_attn mix. One sequence starting at position 0
// (zero conv and recurrent state).

struct GdnTrace {                          // optional intermediates for parity tests (each may be null)
    float * qkv = nullptr;                 // [T][2 key_dim + value_dim], before the conv
    float * z = nullptr;                   // [T][value_dim], output gate input
    float * gate = nullptr;                // [T][v_heads], softplus(alpha + dt_bias) * a
    float * beta = nullptr;                // [T][v_heads], sigmoid
    float * conv = nullptr;                // [T][2 key_dim + value_dim], after conv + silu
    float * core = nullptr;                // [T][v_heads][head], delta rule output
    float * normed = nullptr;              // [T][value_dim], gated RMSNorm output
};

// qkv, z, alpha, beta projections; causal conv + silu; l2-normed q, k; gated delta rule (value head h reads key
// head h % key heads); RMSNorm(head) * sigmoid(z); out projection -> out [T][d_model]
void gdn(const Ctx & x, const Gdn & g, const float * in, int T, float * out, const GdnTrace * trace = nullptr);

// --- DSA mixer (llama-paw build_layer_attn + build_qsa_top_k), input = the hc_attn mix, one sequence from
// position 0. Attention is restricted to the indexer's selection (ref::qsa_select: top idx_top_k / ratio complete
// blocks plus the tail), which TRUSS defines as whole blocks; llama-paw's token-level top-k of idx_top_k + ratio - 1
// cells can add up to ratio - 1 cells of one more block, picked nondeterministically (CUB top-k), when the tail is
// short. Identical when every visible block fits the budget (prompts <= idx_top_k + ratio - 1 tokens).

struct DsaTrace {                          // optional intermediates for parity tests (each may be null)
    float * q_normed = nullptr;            // [T][n_head][head_dim], before rope
    float * k_normed = nullptr;            // [T][n_head_kv][head_dim], before rope
    float * q = nullptr, * k = nullptr, * v = nullptr;   // after rope (q, k)
    float * idx_q = nullptr;               // [T][idx_heads][idx_head_dim], normed + roped
    float * idx_k = nullptr;               // [T / ratio][idx_head_dim], pooled complete blocks, normed + roped
    float * gate = nullptr;                // [T][n_head][head_dim], the gate half of the q projection
    float * pregate = nullptr;             // [T][n_head][head_dim], attention output
    float * gated = nullptr;               // pregate * sigmoid(gate)
    uint8_t * sel = nullptr;               // [T][T], 1 = cell attended by the query (ref::qsa_select)
};

void dsa(const Ctx & x, const Dsa & a, int ratio, const float * in, int T, float * out, const DsaTrace * trace = nullptr);

// --- FFN (llama-paw build_layer_ffn): routed trellis experts + gated shared expert, input = the hc_ffn mix

struct Routing {                           // host, [T][n_expert_used]
    std::vector<int32_t> ids;
    std::vector<float> weights;            // softmax probabilities of the chosen experts, renormalized to sum 1
};

// logits = router(x) (written to `logits` [T][n_expert] when non-null); softmax; top-k; renormalize.
Routing route(const Ctx & x, const Moe & m, const float * in, int T, float * logits);

// sum over each row's choices, in order, of w * expert(in): expert = down(silu(gate(in)) * up(in)) with each
// projection y = svh * H128(W^T H128(suh * a)), W decoded exactly (ref::trellis_dequant)
void routed(const Ctx & x, const Moe & m, const float * in, const Routing & r, int T, float * out);

// sigmoid(gate_inp_shexp . in) * down(silu(gate(in)) * up(in))
void shared(const Ctx & x, const Moe & m, const float * in, int T, float * out);

// routed + shared
void ffn(const Ctx & x, const Moe & m, const float * in, int T, float * out);

}  // namespace truss::qwen4exp::reference
