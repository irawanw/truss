// PROTOTYPE codec "v2one" for decode-speed measurement only (no encoder yet).
//
// V = 2: each 16-bit trellis state emits two weights (one half2 of an mma fragment) and the state shifts S = 2K bits
// per step, so a lane's 8 weights are 4 states: its 4S new bits and the 16 - S before the first. Rates: S = 2..5
// (K1, K1.5, K2, K2.5); a tile is 128 S bits = 4 S words, stored like mul1 (lane l's bits follow lane l-1's,
// circular over the tile).
// Codebook (tools/codec-lab/viterbi_mse.cu row "V2 one M8fff X3b60"; real L24 experts: flashnext
// 20260930_truss_codec):
//   h = ((s * A + B) & 0x8fff8fff) ^ 0x3b603b60     (IMAD + one LOP3: two fp16 weights per state)
// The global scale folds into svh at pack time.
// Why: mul1 spends ~6 INT-pipe ops per weight (TRACKER #17, #19); this spends ~1.5-2 (#36).
#pragma once
#include "kernels/trellis/mma.cuh"

namespace truss::lab {

template <int S>
struct V2One {
    static_assert(S >= 2 && S <= 5, "prototype: 2K = 2..5 bits per state (a lane window must fit 32 bits)");
    static constexpr const char * NAME = "v2one(proto)";
    static constexpr int TILE_WORDS = 4 * S;
    static constexpr int TILES_PER_VEC = 32 / TILE_WORDS;
    static constexpr int VEC_WORDS = TILE_WORDS * TILES_PER_VEC;

    int lane, src_cur, src_prev, sh;
    uint32_t xr;                               // the XOR constant in a register: one LOP3 does (x & M) ^ X (#26)

    __device__ explicit V2One(int lane_) : lane(lane_)
    {
        // the lane's bits end at stream bit e; the 32 bits before e lie in word (e-1)/32 and the one before it
        const int e = 4 * S * (lane + 1);
        src_cur = (e - 1) / 32;
        src_prev = (src_cur + TILE_WORDS - 1) % TILE_WORDS;
        sh = 32 * src_cur + 32 - e;
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
        const int base = sub * TILE_WORDS;
        const uint32_t cur = __shfl_sync(0xffffffffu, w, base + src_cur);
        const uint32_t prev = __shfl_sync(0xffffffffu, w, base + src_prev);
        // stream bits [e - 32, e), the last one at bit 0 (words are MSB-first, the earlier word is the high half)
        const uint32_t b = __funnelshift_r(cur, prev, sh);
        uint32_t s0, s1, s2, s3;
        if constexpr (S == 4) {   // states at bit offsets 12, 8, 4, 0: byte-aligned ones by one PRMT each
            const uint32_t b4 = b >> 4;
            s0 = __byte_perm(b4, 0, 0x4421); s1 = __byte_perm(b, 0, 0x4421);
            s2 = __byte_perm(b4, 0, 0x4410); s3 = __byte_perm(b, 0, 0x4410);
        } else {
            s0 = (b >> (3 * S)) & 0xffffu; s1 = (b >> (2 * S)) & 0xffffu;
            s2 = (b >> S) & 0xffffu;       s3 = b & 0xffffu;
        }
        f0[0] = codebook(s0);
        f0[1] = codebook(s1);
        f1[0] = codebook(s2);
        f1[1] = codebook(s3);
    }
};

}  // namespace truss::lab
