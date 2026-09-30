// Reference ops: plain fp32 CUDA, one obvious thread mapping each, fixed reduction order. They define what the
// fast kernels (CP3) must compute and back the layer parity tests (CP2); speed is not a goal here.
// Activations are fp32 row-major [rows][n]; weights are DTensors in ggml layout (ne[0] = input dim, contiguous).
#pragma once
#include "core/device_tensors.h"

#include <cuda_runtime.h>

namespace truss::ref {

// Rounding of activations inside the reference. FP32: none (the reference). LLAMA: where llama-paw's CUDA ops round,
// so parity tests can show the math matches beyond llama's own rounding (~0.5% per matmul, TRACKER #33):
//   Q8_0 weights: each 32-block of x quantized to int8 (d = amax/127, stored fp16), integer dot per block (Q8_1);
//   trellis weights: activations rounded to fp16 where the PAW X3 op feeds its GEMV.
enum class Numerics { FP32, LLAMA };

// y[r][o] = sum_i W[o][i] * x[r][i]. W: F32, F16 or Q8_0 with ne = [in, out]. `num` affects Q8_0 weights only.
void linear(const DTensor & W, const float * x, float * y, int rows, cudaStream_t s, Numerics num = Numerics::FP32);

// y[r] = x[r] / sqrt(mean(x[r]^2) + eps) * (gamma ? gamma[(r % gamma_rows)][:] : 1); gamma F32, n per row.
// gamma_rows lets one call normalize [rows] groups of n with a gamma of shape [gamma_rows][n] (hyper-connection
// streams: rows = tokens * hc, gamma_rows = hc).
void rms_norm(const float * x, const float * gamma, int gamma_rows, float * y, int rows, int n, float eps,
              cudaStream_t s);

// Walsh-Hadamard transform of each consecutive block of 128 values (Sylvester/natural order, scaled by
// 1/sqrt(128)), in place; n is a multiple of 128.
void hadamard128(float * x, int64_t n, cudaStream_t s);

// Decode one trellis projection (mul1 codebook, integer K = 1..8) to W[o][i] fp32, straight from the bitstream
// definition: 16x16 tiles stored k-tile-major (tile (kt, nt) at uint32 word (kt * out/16 + nt) * 8K); weight j of
// a tile has the 16-bit state ending at circular stream bit (j + 1) * K, bits taken MSB-first from the uint32 words;
// value = fp16 fma(bytesum(state * 0x83DCD12D) + 0x6400 as fp16, 1/147.7, -10.39); j = 8 * lane + i sits at
// (n, k) of the tile per the tensor-core fragment order (codec_mul1.cuh). No Hadamard, no suh/svh.
void trellis_dequant(const uint16_t * words, int K, int in, int out, float * W, cudaStream_t s);

// --- recurrent (gated delta net) ops, ref_ssm.cu

// Causal depthwise conv over tokens, then SiLU: y[t][c] = silu(sum_j w[c][j] * x[t - (kc - 1) + j][c]) with rows
// before t = 0 taken from `state` [kc - 1][C] (the previous tokens, oldest first; nullptr = zeros). w: [C][kc].
void causal_conv_silu(const float * x, const float * w, const float * state, float * y, int T, int C, int kc,
                      cudaStream_t s);

// y = x / sqrt(sum(x^2) + eps) per row of n (llama's GDN l2 norm: rms_norm(eps / n) / sqrt(n)); in place allowed.
void l2_norm(const float * x, float * y, int rows, int n, float eps, cudaStream_t s);

// Gated delta rule, one sequence, token by token. q, k: [T][Hk][S], v: [T][Hv][S], g, beta: [T][Hv]; value head h
// reads key head h % Hk. Per head, with state S [S_k][S_v]:
//   S = exp(g) S;  S += k (beta (v - S^T k))^T;  o = S^T q / sqrt(S)
// out: [T][Hv][S]. state: [Hv][S_k][S_v] in/out (nullptr: zero start, final state discarded).
void gated_delta_rule(const float * q, const float * k, const float * v, const float * g, const float * beta,
                      float * state, float * out, int T, int Hk, int Hv, int S, cudaStream_t s);

}  // namespace truss::ref
