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

namespace truss::qwen4exp::reference {

struct Ctx {
    const Config & c;
    const DeviceTensors & dev;
    Scratch & scratch;
    cudaStream_t stream;
    ref::ActQuant act = ref::ActQuant::NONE;   // Q8_1: reproduce llama-paw's activation rounding (parity mode)
};

// Hyper-connection mix (replaces the pre-norm): mixed [T][d]; inject [T][hc] (skipped when h.inject is null).
void hc_mix(const Ctx & x, const HyperConnection & h, const float * res, int T, float * mixed, float * inject);

// res[t][c] += block_out[t] * 2 * sigmoid(inject[t][c] / hc)
void hc_combine(const Ctx & x, float * res, const float * block_out, const float * inject, int T);

}  // namespace truss::qwen4exp::reference
