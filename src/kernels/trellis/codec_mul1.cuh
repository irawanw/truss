// Trellis codec "mul1" (PAW X3 / exllamav3 EXL3 tiles, integer rates K = 1, 2, 3, 4; half-integer rates
// K = KA + 0.5 in Mul1Frac, exllamav3 v1.5.1 "frac" tiles, formats/trellis_k.h).
//
// Tile: 16x16 weights, 256 16-bit trellis states shifted K bits per weight, stored as 8*K uint32. Weight value =
// mul1 codebook of its state: bytesum(state * 0x83DCD12D) mapped to fp16 by one hfma2.
// Decode primitives are from llama-paw paw-x3.cu, a port of exllamav3 (MIT).
//
// Codec interface (a new codec is a new header with the same shape; kernels take the codec as a template):
//   template <int K> struct Codec {
//       static constexpr const char * NAME;
//       static constexpr int TILE_WORDS;     // uint32 words per tile
//       static constexpr int VEC_WORDS;      // words one warp-wide load covers: lane l loads word l if l < VEC_WORDS
//       static constexpr int TILES_PER_VEC;  // tiles in one such load
//       __device__ explicit Codec(int lane); // per-lane constants, built once per kernel
//       __device__ bool loads(int lane) const;
//       // decode tile `sub` of the load whose word this lane holds in `w`; warp-synchronous (all 32 lanes)
//       __device__ void tile(uint32_t w, int sub, FragB & f0, FragB & f1) const;
//   };
#pragma once
#include "mma.cuh"

namespace truss {
namespace mul1_detail {

__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift)
{
    const uint64_t merged = ((uint64_t) a << 32) | (uint64_t) b;
    return (uint32_t) (merged >> shift);
}
#define TRUSS_FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define TRUSS_BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))

// two states -> two weights
__device__ __forceinline__ half2 codebook2(uint32_t x0, uint32_t x1)
{
    x0 *= 0x83DCD12Du;
    x1 *= 0x83DCD12Du;
    const uint32_t acc = 0x6400u;
    const uint32_t sum0 = __dp4a(x0, 0x01010101u, acc);
    const uint32_t sum1 = __dp4a(x1, 0x01010101u, acc);
    const half2 k_inv = __half2half2(__ushort_as_half(0x1eee));   //  1/147.7
    const half2 k_bias = __half2half2(__ushort_as_half(0xc931));  // -10.39
    const half2 h = __halves2half2(__ushort_as_half((uint16_t) sum0), __ushort_as_half((uint16_t) sum1));
    return __hfma2(h, k_inv, k_bias);
}

__device__ __forceinline__ void codebook8(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, uint32_t w4,
                                          uint32_t w5, uint32_t w6, uint32_t w7, FragB & f0, FragB & f1)
{
    f0[0] = codebook2(w0, w1);
    f0[1] = codebook2(w2, w3);
    f1[0] = codebook2(w4, w5);
    f1[1] = codebook2(w6, w7);
}

__device__ __forceinline__ void states8_k4(uint32_t a, uint32_t b, FragB & f0, FragB & f1)
{
    uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
    TRUSS_FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    TRUSS_BFE16_IMM(w6, b, 4);
    TRUSS_BFE16_IMM(w5, b, 8);
    TRUSS_BFE16_IMM(w4, b, 12);
    TRUSS_BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    TRUSS_BFE16_IMM(w1, s, 4);
    TRUSS_BFE16_IMM(w0, s, 8);
    codebook8(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

__device__ __forceinline__ void states8_k1(uint32_t a, uint32_t b, int t_offset, FragB & f0, FragB & f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    b = fshift(b, a, (~t_offset) & 24);
    w7 = b & 0xffff;
    TRUSS_BFE16_IMM(w6, b, 1);
    TRUSS_BFE16_IMM(w5, b, 2);
    TRUSS_BFE16_IMM(w4, b, 3);
    TRUSS_BFE16_IMM(w3, b, 4);
    TRUSS_BFE16_IMM(w2, b, 5);
    TRUSS_BFE16_IMM(w1, b, 6);
    TRUSS_BFE16_IMM(w0, b, 7);
    codebook8(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

__device__ __forceinline__ void states8_k2(uint32_t a, uint32_t b, int t_offset, FragB & f0, FragB & f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    TRUSS_BFE16_IMM(w6, b, 2);
    TRUSS_BFE16_IMM(w5, b, 4);
    TRUSS_BFE16_IMM(w4, b, 6);
    TRUSS_BFE16_IMM(w3, b, 8);
    TRUSS_BFE16_IMM(w2, b, 10);
    TRUSS_BFE16_IMM(w1, b, 12);
    TRUSS_BFE16_IMM(w0, b, 14);
    codebook8(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

__device__ __forceinline__ void states8_k3(uint32_t a, uint32_t b, int s2, FragB & f0, FragB & f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = fshift(b, a, s2);
    w6 = w7 >> 3;
    w5 = w6 >> 3;
    w4 = w5 >> 3;
    w3 = fshift(b, a, s2 + 12);
    w2 = w3 >> 3;
    w1 = w2 >> 3;
    w0 = w1 >> 3;
    codebook8(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff, w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff,
              f0, f1);
}

#undef TRUSS_FSHF_IMM
#undef TRUSS_BFE16_IMM

}  // namespace mul1_detail

template <int K>
struct Mul1 {
    static_assert(K >= 1 && K <= 4, "mul1: integer rates 1..4 (fractional rates not implemented)");
    static constexpr const char * NAME = "mul1";
    static constexpr int TILE_WORDS = 8 * K;
    static constexpr int TILES_PER_VEC = K == 1 ? 4 : K == 2 ? 2 : 1;
    static constexpr int VEC_WORDS = TILE_WORDS * TILES_PER_VEC;

    int lane, src_a = 0, src_b = 0, s2 = 0;

    __device__ explicit Mul1(int lane_) : lane(lane_)
    {
        if constexpr (K == 1) {
            src_b = lane >> 2;
            src_a = (src_b + 7) & 7;
        }
        if constexpr (K == 2) {
            const int i1 = lane >> 1;
            src_b = i1;
            src_a = (i1 + 15) & 15;
        }
        if constexpr (K == 3) {
            const int t_offset = lane << 3;
            const int b1 = (t_offset + 257) * 3;
            const int b2 = b1 + 21;
            const int i0 = (b1 - 16) / 32;
            const int i2 = (b2 - 1) / 32;
            s2 = (i2 + 1) * 32 - b2;
            src_a = i0 % 24;
            src_b = i2 % 24;
        }
    }

    __device__ bool loads(int l) const { return l < VEC_WORDS; }

    __device__ __forceinline__ void tile(uint32_t w, int sub, FragB & f0, FragB & f1) const
    {
        if constexpr (K == 4) {
            const uint32_t a = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
            mul1_detail::states8_k4(a, w, f0, f1);
        } else if constexpr (K == 1) {
            const int base = sub << 3;
            const uint32_t b = __shfl_sync(0xffffffffu, w, base + src_b);
            const uint32_t a = __shfl_sync(0xffffffffu, w, base + src_a);
            mul1_detail::states8_k1(a, b, lane << 3, f0, f1);
        } else if constexpr (K == 2) {
            const int base = sub << 4;
            const uint32_t b = __shfl_sync(0xffffffffu, w, base + src_b);
            const uint32_t a = __shfl_sync(0xffffffffu, w, base + src_a);
            mul1_detail::states8_k2(a, b, lane << 3, f0, f1);
        } else {
            const uint32_t a = __shfl_sync(0xffffffffu, w, src_a);
            const uint32_t b = __shfl_sync(0xffffffffu, w, src_b);
            mul1_detail::states8_k3(a, b, s2, f0, f1);
        }
    }
};

// Half-integer rate K = KA + 0.5 (exllamav3 v1.5.1 frac, MASK 0xAAAA; TRACKER #118): a tile is a ring of
// 128 * (2 KA + 1) bits = 8 KA + 4 uint32, weight j's window ends at S(j) = (2 KA + 1)(j >> 1) + KA + (j & 1)(KA + 1)
// (formats/trellis_k.h). A lane's 8 windows (j = 8 lane + m) form two groups of four; a group spans
// 3 KA + 18 <= 32 bits, so one funnel shift of two ring words holds it: 4 shuffles per tile.
template <int KA>
struct Mul1Frac {
    static_assert(KA >= 1 && KA <= 4, "mul1 frac: KA 1..4");
    static constexpr const char * NAME = "mul1-frac";
    static constexpr int TILE_WORDS = 8 * KA + 4;
    static constexpr int TILES_PER_VEC = 1;
    static constexpr int VEC_WORDS = TILE_WORDS;
    static constexpr int P = 2 * KA + 1;                  // bits per pair of weights
    // window m of a group ends this many bits after the group's first bit (16 = the first window's end)
    static constexpr int D1 = 16 + KA + 1, D2 = 16 + P, D3 = 16 + P + KA + 1;

    int lane, wa[2] = {}, wb[2] = {}, sh[2] = {};

    __device__ explicit Mul1Frac(int lane_) : lane(lane_)
    {
        constexpr int R = 32 * TILE_WORDS;
        for (int g = 0; g < 2; ++g) {
            const int end0 = P * (4 * lane + 2 * g) + KA;    // window 8 lane + 4 g
            const int lo = ((end0 - 16) % R + R) % R;         // ring bit of the group's first (MSB) bit
            wa[g] = lo >> 5;
            wb[g] = (wa[g] + 1) % TILE_WORDS;
            sh[g] = 32 - (lo & 31);                           // chunk = 32 ring bits from lo, MSB first
        }
    }

    __device__ bool loads(int l) const { return l < VEC_WORDS; }

    __device__ __forceinline__ void tile(uint32_t w, int, FragB & f0, FragB & f1) const
    {
        uint32_t s[8];
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const uint32_t a = __shfl_sync(0xffffffffu, w, wa[g]);
            const uint32_t b = __shfl_sync(0xffffffffu, w, wb[g]);
            const uint32_t c = mul1_detail::fshift(b, a, sh[g]);
            s[4 * g + 0] = c >> 16;
            s[4 * g + 1] = (c >> (32 - D1)) & 0xffff;
            s[4 * g + 2] = (c >> (32 - D2)) & 0xffff;
            s[4 * g + 3] = (c >> (32 - D3)) & 0xffff;
        }
        mul1_detail::codebook8(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], f0, f1);
    }
};

}  // namespace truss
