// had128.cuh: the 128-point Hadamard helpers the trellis-dense reconstruct needs (half2 and float-pair forms).
// Ported from tqLLM kernels/formats/x3/had128.cuh, itself ported from llama-paw-k25 ggml/src/ggml-cuda/paw-x3.cu
// (MIT, exllamav3 lineage). The MIT notice of the original applies to this file.
//
// The runtime path (the window GEMV) transforms the ACTIVATIONS with the float 4-value butterfly in
// src/kernels/trellis/hadamard.cuh; the reconstruct transforms the WEIGHTS in half2 arithmetic (the matrix is
// dequantised once and then used by a plain GEMM), so it needs its own forms.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace truss::trellis_dense {

constexpr float TD_RSCALE = 0.08838834764831845f;   // 1/sqrt(128)

struct half4 { half2 x, y; };

__device__ inline void shuffle_had_f2x32(float & v, float & w, const int lane_id)
{
#pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        uint64_t vw = ((uint64_t) __float_as_uint(v)) | (((uint64_t) __float_as_uint(w)) << 32);
        uint64_t pvw = __shfl_xor_sync(0xffffffff, vw, i);
        float pv = __uint_as_float((uint32_t) (pvw & 0xffffffff));
        float pw = __uint_as_float((uint32_t) (pvw >> 32));
        uint32_t vi = __float_as_uint(v), wi = __float_as_uint(w);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        vi ^= (sfm & 0x80000000);
        wi ^= (sfm & 0x80000000);
        v = __uint_as_float(vi) + pv;
        w = __uint_as_float(wi) + pw;
    }
}

__device__ inline half2 shuffle_had_h2x32(half2 v, int lane_id)
{
    for (int i = 1; i < 32; i <<= 1) {
        half2 pv = __shfl_xor_sync(0xffffffff, v, i);
        uint32_t * vi = reinterpret_cast<uint32_t *>(&v);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        *vi ^= (sfm & 0x80008000);
        v = __hadd2(v, pv);
    }
    return v;
}

}  // namespace truss::trellis_dense
