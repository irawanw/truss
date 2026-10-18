// DeepSeek-V4.1 Engram (dsref/model.py Engram) device ops:
//   dequant_q4   int4 table rows (tools/tk-pack/ds41_engram.py: 8 fp16 scales per 256 values, then 128 bytes of
//                nibbles, value = (nibble - 8) * scale) -> fp16 rows, the wkv projection's input
//   gate         per (row, copy c): rstd = rsqrt(mean(h^2) + eps) rsqrt(mean(key_c^2) + eps),
//                dot = sum(h * qk_c * key_c) rstd / sqrt(d), gate = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))),
//                h += gate * value   (kv = [key_0 .. key_{hc-1} | value], qk = q_weight * k_weight)
//   mul          y = a * b (the qk product, once at build)
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace truss::ds41 {

constexpr int ENGRAM_Q4_ROW = 144, ENGRAM_DIM = 256;

void engram_dequant_q4(const uint8_t * rows, int n, half * out, cudaStream_t stream);
// res [T][hc][d] in place; kv [T][(hc + 1) d]; qk [hc][d]
void engram_gate(float * res, const float * kv, const float * qk, int T, int hc, int d, float eps, cudaStream_t stream);
void mul(const float * a, const float * b, int n, float * y, cudaStream_t stream);

}  // namespace truss::ds41
