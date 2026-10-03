// CPU GEMV straight from the pack's trellis bytes (PAW X3 / EXL3 "mul1" codec), for the CPU expert tier.
//
// Why: Strata's CPU computes every missed expert from the same bytes the GPU would copy, so the tier costs no extra
// RAM. TRUSS's first CPU tier (expert_q4.h) used separate 4-bit copies: 2.76 MB per expert beside the pack's
// 1.77 MB pinned copy, 28 GB at share 0.15, which does not fit beside the renters (they take 58-77 GB of the
// 125 GB, TRACKER #73). Decoding the trellis on the CPU reads only the pinned pack bytes the ExpertStore already
// holds, and the CPU then computes the GPU's own weights (same numbers up to summation order).
//
// Projection math (as the GPU's moe::window and the reference qwen4exp::project):
//   a = fp16(H128(suh * x))            per 128-block fast Walsh-Hadamard, scaled 1/sqrt(128)
//   c[o] = sum_i W[o][i] a[i]          W decoded from 16x16 tiles
//   y = svh * H128(c)
// Tile (ref_trellis.cu, codec_mul1.cuh): 8*K uint32 words = a cyclic 256*K-bit stream, MSB first in each word.
// Weight j of the tile (j = lane*8 + m) is the 16-bit state ending at bit (j+1)*K, value
// fp16(bytesum(state * 0x83DCD12D) + 1024) * 1/147.7 - 10.39 (one fp16 fma). Lane l covers outputs n = l/4 and
// n + 8, inputs k = 2*(l%4) + {0, 1} and + 8. Tiles are k-slice major: tile (kt, nt) at index kt * (out/16) + nt.
//
// AVX2 inner loop, per 8 weights (one lane of one tile): a 64-bit window of the stream (scalar), two variable
// 64-bit shifts and one shuffle give the 8 states, then mullo, byte sum (maddubs + madd), fma to the codebook
// value, fp16 rounding (cvtps_ph / cvtph_ps), and one fma per activation row into that lane-group's accumulator.
// The activation side is pre-permuted to the same 8-lane order (trellis_prep), so no shuffles in the hot loop.
#pragma once
#include <cstdint>

namespace truss::cpu {

struct TrellisMat {
    const uint32_t * tiles;   // (in/16) * (out/16) tiles of 8*K words, k-slice major
    int K;                    // rate code (formats/trellis_k.h): 1 .. 6, or 25 / 35 = K2.5 / K3.5
    int in, out;              // multiples of 128
    const uint16_t * suh;     // [in] input signs/scales, fp16 bits (the pack's own tensor rows)
    const uint16_t * svh;     // [out]
};

struct TrellisExpert {
    TrellisMat gate, up, down;
};

// floats of a prepared activation block for R rows of a projection with `in` inputs (the most any K needs): K <= 4
// lays out each 16-input slice as [8 weight indices][8 lanes] (4 * in per row), K 5-6 as [4 lane quads][8] (2 * in
// used of the same 4 * in row stride, so a row prepared alone at r * trellis_prep_floats(in, 1) is row r)
constexpr int trellis_prep_floats(int in, int R) { return 4 * R * in; }

// P [R][in/16][4][8] = a = fp16(H128(suh * x[r])), permuted to the kernel's lane order. x rows have stride ldx.
void trellis_prep(const TrellisMat & W, const float * x, int ldx, int R, float * P);

// y[r][c] (stride ld_y) = (svh * H128(W a_r))[c] for c in [c0, c1), both multiples of 128.
void trellis_gemv(const TrellisMat & W, const float * P, int R, int c0, int c1, float * y, int ld_y);

// The same product in pieces, for the pool's input-split items (TRACKER #82): an item reads one contiguous run of
// k-slices (tiles are k-slice major) instead of a strided column range, and prepares only its own inputs (the input
// Hadamard is per 128-block). Partial products over input blocks add up to the whole one before the output
// Hadamard; the caller sums them in a fixed order, then applies trellis_out.
// W restricted to inputs [i0, i1) (multiples of 128), all outputs: prepare it with trellis_prep on x + i0
TrellisMat trellis_inputs(const TrellisMat & W, int i0, int i1);
// c[r][j] (stride ldc) = (W a_r)[c0 + j], j < c1 - c0: the raw product, before the output Hadamard and svh
void trellis_gemv_raw(const TrellisMat & W, const float * P, int R, int c0, int c1, float * c, int ldc);
// y[r][j] (stride ld_y) = svh[c0 + j] * H128(c_r)[j] for the columns [c0, c1) of W (multiples of 128) held at c
// (stride ldc); c is overwritten by the Hadamard
void trellis_out(const TrellisMat & W, float * c, int R, int ldc, int c0, int c1, float * y, int ld_y);

// one weight, decoded bit by bit (the reference for tests). round16: the GPU's fp16 value (trellis_gemv uses it only
// with TRUSS_TRELLIS_ROUND=1; by default the fp32 codebook value, at most half an fp16 ulp away)
float trellis_weight_ref(const TrellisMat & W, int o, int i, bool round16 = false);

}  // namespace truss::cpu
