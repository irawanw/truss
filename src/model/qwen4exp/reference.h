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
