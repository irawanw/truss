// 128-point Hadamard transform in one warp (4 values per lane, natural order, scaled by 1/sqrt(128)).
#pragma once
#include <cuda_runtime.h>

namespace truss {

__device__ inline void had4x32(float & h0, float & h1, float & h2, float & h3, const int lane)
{
    float s0 = h0 + h1, d0 = h0 - h1, s1 = h2 + h3, d1 = h2 - h3;
    h0 = s0 + s1; h1 = d0 + d1; h2 = s0 - s1; h3 = d0 - d1;
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        const float p0 = __shfl_xor_sync(0xffffffffu, h0, i);
        const float p1 = __shfl_xor_sync(0xffffffffu, h1, i);
        const float p2 = __shfl_xor_sync(0xffffffffu, h2, i);
        const float p3 = __shfl_xor_sync(0xffffffffu, h3, i);
        const bool hi = lane & i;
        h0 = hi ? p0 - h0 : h0 + p0;
        h1 = hi ? p1 - h1 : h1 + p1;
        h2 = hi ? p2 - h2 : h2 + p2;
        h3 = hi ? p3 - h3 : h3 + p3;
    }
    constexpr float r = 0.088388347648f;
    h0 *= r; h1 *= r; h2 *= r; h3 *= r;
}

}  // namespace truss
