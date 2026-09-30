// Dense Q8 GEMM for prefill: y = W x with Q8_0 weights and Q8_1-quantized activations on int8 tensor cores.
//
// Numerics are llama-paw's (ref::Numerics::LLAMA, TRACKER #33): each 32-block of an activation row is quantized
// with d = amax / 127 (q = round(x / d), fp32 d), the block dot is exact in int32 and scaled by the fp16 weight
// scale times the fp16-rounded activation scale. One mma m16n8k32 spans exactly one Q8 block, so the scaling is
// one fp32 FMA per result per block. Why int8: 178-223 TOPS vs 57-66 TFLOPS fp16 on the dense prefill shapes
// (cuBLAS, GPU 2, TRACKER #40).
//
// Weights are repacked once from GGUF's 34-byte blocks into an aligned layout (16-byte async copies).
#pragma once
#include <cuda_fp16.h>
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

// y fp32 [rows][W.out] = W . x, from the quantized activations. W.in % 64 == 0; any rows, any W.out.
void q8_gemm(const Q8Matrix & W, const int8_t * xq, const half * xd, int rows, float * y, cudaStream_t stream);

}  // namespace truss::dense
