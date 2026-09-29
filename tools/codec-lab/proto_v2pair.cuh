// PROTOTYPE codec "v2pair" for decode-speed measurement only (no encoder, no bit-exact layout yet).
//
// V = 2: each 16-bit trellis state emits two weights (consecutive k of one column, i.e. one half2 of an mma
// fragment), and the state shifts 2K bits per step, so a lane's 8 weights are 4 states. Codebook:
//   x1 = s * A1 + B1, x2 = s * A2 + B2     (IMAD each)
//   h  = (x & 0x8fff8fff) ^ 0x3b603b60     (LOP3 each: sign, 2 exponent bits and the mantissa stay random)
//   w  = hadd2(h1, h2)                     (two fp16 halves per word: two weights)
// Its global scale (rms 1.245) folds into svh at pack time, so there is no hfma2.
// Quality: tools/codec-lab/viterbi_mse.cu row "V2 pair M8fff X3b60".
//
// Bench layout: a lane's 8K new bits follow the previous lane's (tail-biting across the tile); the state window
// is built from this lane's word(s) and the one below, the same loads as mul1.
#pragma once
#include "kernels/trellis/mma.cuh"

namespace truss::lab {

template <int K>
struct V2Pair {
    static_assert(K >= 2 && K <= 4, "prototype: integer rates 2..4");
    static constexpr const char * NAME = "v2pair(proto)";
    static constexpr int TILE_WORDS = 8 * K;
    static constexpr int TILES_PER_VEC = K == 2 ? 2 : 1;
    static constexpr int VEC_WORDS = TILE_WORDS * TILES_PER_VEC;

    int lane, src_hi, src_lo, off;

    __device__ explicit V2Pair(int lane_) : lane(lane_)
    {
        // this lane's new bits end at bit e = 8K(l+1) of the tile; the first state starts 16 - 2K bits before
        // its new bits, i.e. at e - 8K - 16 + 2K. Window = (word holding bit e-1) : (word below it).
        const int ebit = 8 * K * (lane + 1);
        const int hi = (ebit - 1) / 32;
        src_hi = hi;
        src_lo = (hi + TILE_WORDS - 1) % TILE_WORDS;
        off = (ebit - 8 * K - 16 + 2 * K) - 32 * (hi - 1);   // bit offset of state 0 in the 64-bit window
    }

    __device__ bool loads(int l) const { return l < VEC_WORDS; }

    __device__ __forceinline__ static half2 codebook(uint32_t s)
    {
        const uint32_t x1 = s * 0x83DCD12Du + 0x6A09E667u;
        const uint32_t x2 = s * 0xCBAC1FEDu + 0xBB67AE85u;
        const uint32_t h1 = (x1 & 0x8fff8fffu) ^ 0x3b603b60u;
        const uint32_t h2 = (x2 & 0x8fff8fffu) ^ 0x3b603b60u;
        return __hadd2(*reinterpret_cast<const half2 *>(&h1), *reinterpret_cast<const half2 *>(&h2));
    }

    __device__ __forceinline__ void tile(uint32_t w, int sub, FragB & f0, FragB & f1) const
    {
        const int base = K == 2 ? sub << 4 : 0;
        const uint32_t hi = __shfl_sync(0xffffffffu, w, base + src_hi);
        const uint32_t lo = __shfl_sync(0xffffffffu, w, base + src_lo);
        const uint64_t win = ((uint64_t) hi << 32) | lo;
        uint32_t s[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) s[j] = (uint32_t) (win >> (off + 2 * K * j)) & 0xffffu;
        f0[0] = codebook(s[0]);
        f0[1] = codebook(s[1]);
        f1[0] = codebook(s[2]);
        f1[1] = codebook(s[3]);
    }
};

}  // namespace truss::lab
