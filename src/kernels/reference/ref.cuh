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

// Decode one trellis projection (mul1 codebook, rate code K: integer 1..8 or 15/25/35 = K1.5/2.5/3.5, formats/trellis_k.h)
// to W[o][i] fp32, straight from the bitstream definition: 16x16 tiles stored k-tile-major (tile (kt, nt) at uint16
// word (kt * out/16 + nt) * k_tile_u16(K)); weight j of a tile has the 16-bit state ending at circular stream bit
// k_window_end(K, j) ((j + 1) * K for integer K), bits taken MSB-first from the uint32 words;
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

// --- attention ops, ref_attn.cu

// RoPE, NEOX pairing, in place on the first n_rot dims of each head: x [rows][heads][hd], row r at position pos[r];
// pair (i, i + n_rot / 2) rotates by pos * base^(-2 i / n_rot). (qwen4exp's interleaved multi-section rope reduces to
// this for text, where all section positions are equal.)
void rope_neox(float * x, int rows, int heads, int hd, int n_rot, const int * pos, float base, cudaStream_t s);

// Block-sparse selection (qwen4exp QSA, one sequence from position 0): complete blocks of r tokens have one key each,
// idx_k [n_blocks][d]; query t scores the complete blocks it sees (r b + r - 1 <= t) with sum over heads of
// relu(q_h . k_b) (idx_q [T][heads][d]), keeps the top_blocks best (ties: the later block), plus its tail
// [r floor((t + 1) / r), t]. sel [T][T] = 1 for each attended cell, else 0.
void qsa_select(const float * idx_q, const float * idx_k, int T, int heads, int d, int r, int top_blocks,
                uint8_t * sel, cudaStream_t s);

// out[t][h] = sum_j softmax_j(scale q[t][h] . k[j][h / (Hq / Hkv)]) v[j][same kv head] over cells j with
// sel[t][j] (causal is up to sel). q [T][Hq][d], k, v [T][Hkv][d], out [T][Hq][d].
// FP32: fp32 scores, fp64 softmax accumulation. LLAMA: ggml-cuda's flash-attention mma kernel on NVIDIA (fp16 K/V
// cache): q * scale, k and v rounded to fp16, fp32 scores, online softmax over 64-cell tiles with the running max
// raised by FATTN_KQ_MAX_OFFSET, P rounded to fp16 and P.V accumulated in fp16 (one rounding per 16-cell mma step).
// Its fp16 accumulator puts it ~2e-3 from FP32 at ~2K cells (TRACKER #48); warp and stream-k splits are not modeled.
void masked_attention(const float * q, const float * k, const float * v, const uint8_t * sel, float * out, int T,
                      int Hq, int Hkv, int d, float scale, cudaStream_t s, Numerics num = Numerics::FP32);

}  // namespace truss::ref
