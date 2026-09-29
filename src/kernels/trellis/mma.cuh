// Tensor-core fragments for trellis GEMV: decoded weights are the A operand (m = 16 output columns, k = 16),
// activations the B operand (n = 8 rows), so rows up to 8 cost one mma per 16x16 tile.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace truss {

// Decoded 16x16 tile, as a codec produces it: f0 = columns 0..7, f1 = columns 8..15, each in mma B-fragment layout
// (k pairs 2t, 2t+8 of column lane/4).
struct FragB { half2 v[2]; __device__ half2 & operator[](int i) { return v[i]; } };

// fp16 accumulator of one tile: [0] = (column lane/4, rows 2q, 2q+1), [1] = (column lane/4 + 8, same rows), q = lane%4
struct FragCh { half2 v[2]; __device__ half2 & operator[](int i) { return v[i]; } };

// c += W(tile)^T . act. The A fragment wants (m g, k 2t), (m g+8, k 2t), (m g, k 2t+8), (m g+8, k 2t+8):
// {f0[0], f1[0], f0[1], f1[1]}. act_lo/act_hi = row lane/4, k pairs 2q and 2q+8.
__device__ __forceinline__ void mma_w(const FragB & f0, const FragB & f1, half2 act_lo, half2 act_hi, FragCh & c)
{
    const uint32_t * p0 = reinterpret_cast<const uint32_t *>(&f0);
    const uint32_t * p1 = reinterpret_cast<const uint32_t *>(&f1);
    const uint32_t b0 = *reinterpret_cast<uint32_t *>(&act_lo);
    const uint32_t b1 = *reinterpret_cast<uint32_t *>(&act_hi);
    uint32_t * cc = reinterpret_cast<uint32_t *>(&c);
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        : "r"(p0[0]), "r"(p1[0]), "r"(p0[1]), "r"(p1[1]), "r"(b0), "r"(b1));
}

}  // namespace truss
