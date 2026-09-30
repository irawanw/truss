// Trellis codec "v2one": two weights per 16-bit trellis state, rates K = 1, 1.5, 2, 2.5 bits per weight.
//
// Why: mul1 decodes one weight per state with ~6 INT-pipe ops, so at window sizes a mul1 expert is decode-bound
// (TRACKER #17, #30). v2one spends one IMAD + one LOP3 per two weights and decodes 1.5-1.7x faster; K2 and K2.5 run
// at memory bandwidth (#36, #37). Quality on real experts: +1.5 / +10 / +15% output error vs mul1 at K1.5 / 2 / 2.5
// (#37), so the pack solver picks the codec per expert (#38).
//
// Stream: a tile (16x16 weights, the mul1 tile order: weight j = 8 * lane + i, see mma.cuh) is 128 steps; step t
// emits weights 2t (low half) and 2t + 1 (high half) from its state, the 16 bits of the tile's bit stream ending at
// bit (t + 1) * S, circularly (tail-biting). S = 2K new bits per step; a tile is 128 S bits = 4 S uint32 words,
// MSB-first: stream bit p is bit 31 - p % 32 of word p / 32. Lane l's four states end at bit 4 S (l + 1).
// Codebook (tools/codec-lab/viterbi_mse.cu row "V2 one M8fff X3b60"):
//   h = ((state * A + B) & 0x8fff8fff) ^ 0x3b603b60,   weights = the two fp16 halves of h (raw rms RMS below)
// Rate parameter: the codec template takes S (bits per state), not K, since K is fractional.
// Codec interface: codec_mul1.cuh.
#pragma once
#include "mma.cuh"

#include <cstdint>

namespace truss {

namespace v2one_detail {
constexpr uint32_t A = 0x83DCD12Du, B = 0x6A09E667u, M = 0x8fff8fffu, X = 0x3b603b60u;
}  // namespace v2one_detail

// The codebook as bits: the two fp16 weights of a state (low half = the even weight). Host and device, so the
// encoder and the tests use the same definition as the kernel.
__host__ __device__ constexpr uint32_t v2one_codebook_bits(uint32_t state)
{
    using namespace v2one_detail;
    return ((state * A + B) & M) ^ X;
}

template <int S>
struct V2One {
    static_assert(S >= 2 && S <= 5, "v2one: S = 2K = 2..5 bits per state (a lane's window must fit 32 bits)");
    static constexpr const char * NAME = "v2one";
    static constexpr int TILE_WORDS = 4 * S;
    static constexpr int TILES_PER_VEC = 32 / TILE_WORDS;
    static constexpr int VEC_WORDS = TILE_WORDS * TILES_PER_VEC;
    static constexpr float RMS = 0.87996f;   // rms of all codebook values; the pack folds 1 / RMS into svh

    int lane, src_cur, src_prev, sh;
    uint32_t xr;   // the XOR constant in a register: one LOP3 does (x & M) ^ X (TRACKER #36, rule 26)

    __device__ explicit V2One(int lane_) : lane(lane_)
    {
        // the lane's bits end at stream bit e; the 32 bits before e lie in word (e - 1) / 32 and the one before it
        const int e = 4 * S * (lane + 1);
        src_cur = (e - 1) / 32;
        src_prev = (src_cur + TILE_WORDS - 1) % TILE_WORDS;
        sh = 32 * src_cur + 32 - e;
        asm volatile("mov.b32 %0, %1;" : "=r"(xr) : "n"(v2one_detail::X));   // opaque: ptxas keeps it in a register
    }

    __device__ bool loads(int l) const { return l < VEC_WORDS; }

    __device__ __forceinline__ half2 codebook(uint32_t s) const
    {
        const uint32_t x = s * v2one_detail::A + v2one_detail::B;
        uint32_t h;
        asm("lop3.b32 %0, %1, %2, %3, 0x6a;" : "=r"(h) : "r"(x), "n"(v2one_detail::M), "r"(xr));   // (x & M) ^ X
        return *reinterpret_cast<const half2 *>(&h);
    }

    __device__ __forceinline__ void tile(uint32_t w, int sub, FragB & f0, FragB & f1) const
    {
        const int base = sub * TILE_WORDS;
        const uint32_t cur = __shfl_sync(0xffffffffu, w, base + src_cur);
        const uint32_t prev = __shfl_sync(0xffffffffu, w, base + src_prev);
        // stream bits [e - 32, e), the last one at bit 0 (the earlier word is the high half)
        const uint32_t b = __funnelshift_r(cur, prev, sh);
        uint32_t s0, s1, s2, s3;
        if constexpr (S == 4) {   // states at bit offsets 12, 8, 4, 0: byte-aligned ones by one PRMT each
            const uint32_t b4 = b >> 4;
            s0 = __byte_perm(b4, 0, 0x4421);
            s1 = __byte_perm(b, 0, 0x4421);
            s2 = __byte_perm(b4, 0, 0x4410);
            s3 = __byte_perm(b, 0, 0x4410);
        } else {
            s0 = (b >> (3 * S)) & 0xffffu;
            s1 = (b >> (2 * S)) & 0xffffu;
            s2 = (b >> S) & 0xffffu;
            s3 = b & 0xffffu;
        }
        f0[0] = codebook(s0);
        f0[1] = codebook(s1);
        f1[0] = codebook(s2);
        f1[1] = codebook(s3);
    }
};

// Pack: the 128 states of one tile (a valid ring: state t = (state t-1 << S | new bits) & 0xffff, circularly) ->
// 4 S words. Only each state's S new bits are stored; the ring gives back the rest.
template <int S>
__host__ __device__ inline void v2one_pack_tile(const uint16_t * states, uint32_t * words)
{
    for (int i = 0; i < 4 * S; ++i) words[i] = 0;
    for (int t = 0; t < 128; ++t)
        for (int b = 0; b < S; ++b) {
            const int p = t * S + b;   // new bit b of step t, MSB first
            const uint32_t bit = (states[t] >> (S - 1 - b)) & 1u;
            words[p / 32] |= bit << (31 - p % 32);
        }
}

// The state of step t read back from packed words (the definition the decode must match).
template <int S>
__host__ __device__ inline uint32_t v2one_state(const uint32_t * words, int t)
{
    constexpr int BITS = 128 * S;
    uint32_t s = 0;
    for (int i = 0; i < 16; ++i) {
        const int p = ((t + 1) * S - 16 + i + BITS) % BITS;
        s = (s << 1) | ((words[p / 32] >> (31 - p % 32)) & 1u);
    }
    return s;
}

}  // namespace truss
