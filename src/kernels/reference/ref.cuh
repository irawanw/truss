// Reference ops: plain fp32 CUDA, one obvious thread mapping each, fixed reduction order. They define what the
// fast kernels (CP3) must compute and back the layer parity tests (CP2); speed is not a goal here.
// Activations are fp32 row-major [rows][n]; weights are DTensors in ggml layout (ne[0] = input dim, contiguous).
#pragma once
#include "core/device_tensors.h"

#include <cuda_runtime.h>

namespace truss::ref {

// How activations enter a quantized-weight dot product. NONE: fp32 (the reference). Q8_1: as llama.cpp/ggml CUDA
// does for Q8_0 weights (each 32-block of x quantized to int8 with d = amax/127 stored as fp16, integer dot per
// block); used to show the math matches llama-paw beyond its own activation rounding (~0.5% per matmul).
enum class ActQuant { NONE, Q8_1 };

// y[r][o] = sum_i W[o][i] * x[r][i]. W: F32, F16 or Q8_0 with ne = [in, out]. `aq` applies to Q8_0 weights only.
void linear(const DTensor & W, const float * x, float * y, int rows, cudaStream_t s, ActQuant aq = ActQuant::NONE);

// y[r] = x[r] / sqrt(mean(x[r]^2) + eps) * (gamma ? gamma[(r % gamma_rows)][:] : 1); gamma F32, n per row.
// gamma_rows lets one call normalize [rows] groups of n with a gamma of shape [gamma_rows][n] (hyper-connection
// streams: rows = tokens * hc, gamma_rows = hc).
void rms_norm(const float * x, const float * gamma, int gamma_rows, float * y, int rows, int n, float eps,
              cudaStream_t s);

}  // namespace truss::ref
