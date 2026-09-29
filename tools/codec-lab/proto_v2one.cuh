// PROTOTYPE codec "v2one" for decode-speed measurement only (no encoder yet; bit layout mirrors mul1 K2).
//
// V = 2: each 16-bit trellis state emits two weights (one half2 of an mma fragment) and the state shifts 2K = 4
// bits per step at K2, so a lane's 8 weights are 4 states taken from its 16 new bits and the 12 before them.
// Codebook (tools/codec-lab/viterbi_mse.cu row "V2 one M8fff X3b60", K2 MSE +10% vs mul1):
//   h = ((s * A + B) & 0x8fff8fff) ^ 0x3b603b60     (IMAD + LOP3: two fp16 weights per state)
// The global scale folds into svh at pack time.
// Why: mul1 spends ~6 INT-pipe ops per weight (TRACKER #17, #19); this spends ~1.5.
#pragma once
#include "kernels/trellis/mma.cuh"

namespace truss::lab {

template <int K>
struct V2One {
    static_assert(K == 2, "prototype: K2 only (aligned 16-bit lane windows)");
    static constexpr const char * NAME = "v2one(proto)";
    static constexpr int TILE_WORDS = 8 * K;
    static constexpr int TILES_PER_VEC = 2;
    static constexpr int VEC_WORDS = TILE_WORDS * TILES_PER_VEC;

    int lane, src_a, src_b;
    uint32_t xr;                               // the XOR constant in a register: one LOP3 does (x & M) ^ X

    __device__ explicit V2One(int lane_) : lane(lane_)
    {
        src_b = lane >> 1;
        src_a = (src_b + 15) & 15;
        asm volatile("mov.b32 %0, 0x3b603b60;" : "=r"(xr));   // opaque, so ptxas keeps it in a register
    }

    __device__ bool loads(int l) const { return l < VEC_WORDS; }

    __device__ __forceinline__ half2 codebook(uint32_t s) const
    {
        const uint32_t x = s * 0x83DCD12Du + 0x6A09E667u;
        uint32_t h;
        asm("lop3.b32 %0, %1, 0x8fff8fff, %2, 0x6a;" : "=r"(h) : "r"(x), "r"(xr));   // (x & M) ^ X
        return *reinterpret_cast<const half2 *>(&h);
    }

    __device__ __forceinline__ void tile(uint32_t w, int sub, FragB & f0, FragB & f1) const
    {
        const int base = sub << 4;
        uint32_t b = __shfl_sync(0xffffffffu, w, base + src_b);
        const uint32_t a = __shfl_sync(0xffffffffu, w, base + src_a);
        // this lane's 16 new bits to the low half, the 16 before them above (as mul1 K2)
        b = (uint32_t) ((((uint64_t) a << 32) | b) >> ((~(lane << 3) & 8) << 1));
        // states at bit offsets 12, 8, 4, 0: byte-aligned ones by one PRMT, nibble-aligned ones from b >> 4
        const uint32_t b4 = b >> 4;
        const uint32_t s0 = __byte_perm(b4, 0, 0x4421), s1 = __byte_perm(b, 0, 0x4421);
        const uint32_t s2 = __byte_perm(b4, 0, 0x4410), s3 = __byte_perm(b, 0, 0x4410);
        f0[0] = codebook(s0);
        f0[1] = codebook(s1);
        f1[0] = codebook(s2);
        f1[1] = codebook(s3);
    }
};

}  // namespace truss::lab
