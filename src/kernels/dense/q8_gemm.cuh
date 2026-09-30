// Dense Q8 GEMMs for prefill: y = W x with Q8_0 weights. Two activation paths:
//   q8_gemm      Q8_1-quantized activations on int8 tensor cores (llama-paw's numerics)
//   q8_gemm_a16  fp16 activations, cuBLAS fp16 tensor cores (lower error)
// Q8_1 path:
// Numerics are llama-paw's (ref::Numerics::LLAMA, TRACKER #33): each 32-block of an activation row is quantized
// with d = amax / 127 (q = round(x / d), fp32 d), the block dot is exact in int32 and scaled by the fp16 weight
// scale times the fp16-rounded activation scale. One mma m16n8k32 spans exactly one Q8 block, so the scaling is
// one fp32 FMA per result per block. Why int8: 178-223 TOPS vs 57-66 TFLOPS fp16 on the dense prefill shapes
// (cuBLAS, GPU 2, TRACKER #40).
//
// Weights are repacked once from GGUF's 34-byte blocks into an aligned layout (16-byte async copies).
#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace truss::dense {

struct Q8Matrix {                 // q [out][in] int8, d [out][in / 32] fp16
    const int8_t * q;
    const half * d;
    int in, out;
};

// GGUF Q8_0 tensor (ne = [in, out]) -> Q8Matrix storage. in % 64 == 0.
void q8_repack(const void * blocks, int in, int out, int8_t * q, half * d, cudaStream_t stream);

// x fp32 [rows][in] -> xq int8 [rows][in], xd fp16 [rows][in / 32]
void q8_quantize_act(const float * x, int rows, int in, int8_t * xq, half * xd, cudaStream_t stream);
void q8_quantize_act(const half * x, int rows, int in, int8_t * xq, half * xd, cudaStream_t stream);

// y fp32 [rows][W.out] = W . x, from the quantized activations. W.in % 64 == 0; any rows, any W.out.
void q8_gemm(const Q8Matrix & W, const int8_t * xq, const half * xd, int rows, float * y, cudaStream_t stream);

// W8A16: y fp32 [rows][W.out] = W . x with fp16 activations x [rows][W.in], no activation quantization (lower error
// than Q8_1 by ~0.5% per matmul, TRACKER #33). The weights are dequantized to w16 (scratch, W.in * W.out halfs; one
// fp16 rounding of q * d) and multiplied by cuBLAS with fp32 accumulation. W.in % 64 == 0.
void q8_gemm_a16(const Q8Matrix & W, const half * x, int rows, float * y, half * w16, cublasHandle_t cublas,
                 cudaStream_t stream);

// q8_gemm for a few rows (decode, rows <= GEMV_ROWS): one warp per output, reads each weight once. Same numerics.
constexpr int GEMV_ROWS = 8;
void q8_gemv(const Q8Matrix & W, const int8_t * xq, const half * xd, int rows, float * y, cudaStream_t stream);

// out fp32 [n][W.in] = rows ids[0 .. n) of W (q * d), e.g. the token embedding
void q8_rows(const Q8Matrix & W, const int * ids, int n, float * out, cudaStream_t stream);

}  // namespace truss::dense
