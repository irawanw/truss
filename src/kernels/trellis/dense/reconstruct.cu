// reconstruct.cu: dequantise a trellis-dense matrix to fp16 [out][in] (the engine's GEMM layout), templated on a
// codec. Ported from tqLLM kernels/formats/x3/reconstruct.cu, itself ported from llama-paw-k25
// ggml/src/ggml-cuda/paw-x3.cu (MIT, exllamav3 lineage). The MIT notice of the original applies to this file.
//
// reconstruct_had_kernel: fused dequant + both-side Hadamard, emits ORIGINAL-basis weights
//   W = diag(suh) . H128 . W_hat . H128 . diag(svh) per 128x128 tile.
// The prompt path uses it: the trellis weights are the resident copy, the prompt rows need the matrix once per chunk.
#include "kernels/trellis/dense/had128.cuh"
#include "kernels/trellis/dense/plan.hpp"
#include "kernels/trellis/dense/trellis_dense.h"

#include "kernels/trellis/codec_mul1.cuh"
#include "kernels/trellis/mma.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace truss::trellis_dense {

namespace {

#define TD_RH_THREADS 256

// T is the (rows/16, cols/16, TILE_WORDS) trellis (uint32 words); suh[cols], svh[rows]; rows/cols multiples of 128.
// OutMajor: the tile is written as [out][in] with the row stride ld_out (the engine's GEMM layout).
template <class Codec>
__global__ __launch_bounds__(TD_RH_THREADS)
void reconstruct_had_kernel(half * __restrict__ g_unpacked, const uint16_t * __restrict__ g_packed,
                            const half * __restrict__ suh, const half * __restrict__ svh, int row_tiles, int row0,
                            int ld_out)
{
    constexpr int words = Codec::TILE_WORDS;        // uint32 words per 16x16 tile
    constexpr int j_int4 = words * 4 / 16;          // int4 copies per tile
    constexpr float r_scale = TD_RSCALE;

    const int t = threadIdx.x;
    const int lane_id = t % 32;
    const int warp_id = t / 32;
    const int kb = blockIdx.y;      // the column block (128 columns)
    const int nb = blockIdx.x;      // the row block (128 rows)
    const int n = row0 / 16 + nb * 8;   // the first row tile of the block (the matrix's, not the chunk's)

    __shared__ __align__(16) uint32_t s_packed[8][8][32];   // the tile's words (28 for K3.5, 32 for K4)
    __shared__ half2 stile[128 * 64];

    auto tix = [&](int R, int q, int p) { return R * 64 + (q ^ ((R >> 2) & 31)) * 2 + p; };

    for (int u = t; u < 8 * 8 * j_int4; u += TD_RH_THREADS) {
        const int j = u / (8 * j_int4);
        const int r = u % (8 * j_int4);
        const int wn = r / j_int4, r4 = r % j_int4;   // the word group inside the tile, the int4 inside it
        // the row tiles are minor in the file: the block's 8 sub-tiles are consecutive, so tile wn sits wn tiles on
        const size_t tix_lin = (size_t) tile_offset(kb * 8 + j, n, row_tiles, 2 * words) + (size_t) wn * 2 * words;
        const uint16_t * gp = g_packed + tix_lin;
        ((int4 *) &s_packed[j][wn][0])[r4] = ((const int4 *) gp)[r4];
    }
    __syncthreads();

    for (int jj = 0; jj < 8 * 8 / (TD_RH_THREADS / 32); ++jj) {
        const int j = (warp_id / 8) * (8 / (TD_RH_THREADS / 256)) + jj;
        const int wn = warp_id % 8;
        const uint32_t w = s_packed[j][wn][lane_id];   // the lane's word of the tile
        FragB f0, f1;
        const Codec codec(lane_id);
        codec.tile(w, 0, f0, f1);
        half2 frag[4] = { f0[0], f0[1], f1[0], f1[1] };

        const half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0], 4, 32);
        const half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[1], 4, 32);
        const half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[2], 4, 32);
        const half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[3], 4, 32);

        if (!(lane_id & 4)) {
            const half2 m0 = __halves2half2(__low2half(frag[0]), __low2half(n0));
            const half2 m1 = __halves2half2(__high2half(frag[0]), __high2half(n0));
            const half2 m2 = __halves2half2(__low2half(frag[1]), __low2half(n1));
            const half2 m3 = __halves2half2(__high2half(frag[1]), __high2half(n1));
            const half2 m4 = __halves2half2(__low2half(frag[2]), __low2half(n2));
            const half2 m5 = __halves2half2(__high2half(frag[2]), __high2half(n2));
            const half2 m6 = __halves2half2(__low2half(frag[3]), __low2half(n3));
            const half2 m7 = __halves2half2(__high2half(frag[3]), __high2half(n3));
            const int r0 = j * 16 + (lane_id % 4) * 2;
            const int r1 = r0 + 1;
            const int r2 = r0 + 8;
            const int r3 = r0 + 9;
            const int c0 = lane_id / 8;
            const int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
            const int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
            stile[tix(r0, q0, p0)] = m0;
            stile[tix(r1, q0, p0)] = m1;
            stile[tix(r2, q0, p0)] = m2;
            stile[tix(r3, q0, p0)] = m3;
            stile[tix(r0, q1, p1)] = m4;
            stile[tix(r1, q1, p1)] = m5;
            stile[tix(r2, q1, p1)] = m6;
            stile[tix(r3, q1, p1)] = m7;
        }
    }
    __syncthreads();

    const half2 rs2 = __float2half2_rn(r_scale);
    constexpr int CHUNKS_PW = 32 / (TD_RH_THREADS / 32);
#pragma unroll
    for (int qq = 0; qq < CHUNKS_PW; ++qq) {
        const int q = warp_id * CHUNKS_PW + qq;
        const int qs = q ^ lane_id;
        half2 a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const half4 v = *((const half4 *) (stile + (lane_id * 4 + i) * 64 + qs * 2));
            a[i] = v.x;
            b[i] = v.y;
        }
#pragma unroll
        for (int x = 0; x < 2; ++x) {
            half2 * v = x == 0 ? a : b;
            const half2 s0 = __hadd2(v[0], v[1]), d0 = __hsub2(v[0], v[1]);
            const half2 s1 = __hadd2(v[2], v[3]), d1 = __hsub2(v[2], v[3]);
            v[0] = __hmul2(__hadd2(s0, s1), rs2);
            v[1] = __hmul2(__hadd2(d0, d1), rs2);
            v[2] = __hmul2(__hsub2(s0, s1), rs2);
            v[3] = __hmul2(__hsub2(d0, d1), rs2);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = shuffle_had_h2x32(v[i], lane_id);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            half4 v;
            v.x = a[i];
            v.y = b[i];
            *((half4 *) (stile + (lane_id * 4 + i) * 64 + qs * 2)) = v;
        }
    }
    __syncthreads();

    constexpr int ROWS_PW = 128 / (TD_RH_THREADS / 32);
    const half4 sv4 = ((const half4 *) (svh + row0))[nb * 32 + lane_id];   // the chunk's rows are row0 rows on
#pragma unroll
    for (int rr = 0; rr < ROWS_PW; ++rr) {
        const int R = warp_id * ROWS_PW + rr;
        const int base = R * 64 + (lane_id ^ ((R >> 2) & 31)) * 2;
        const half2 v01 = stile[base];
        const half2 v23 = stile[base + 1];
        const float v0 = __low2float(v01), v1 = __high2float(v01);
        const float v2 = __low2float(v23), v3 = __high2float(v23);
        const float s0 = v0 + v1, d0 = v0 - v1;
        const float s1 = v2 + v3, d1 = v2 - v3;
        half2 h01 = __hmul2(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
        half2 h23 = __hmul2(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
        h01 = shuffle_had_h2x32(h01, lane_id);
        h23 = shuffle_had_h2x32(h23, lane_id);
        const half2 su2 = __half2half2(suh[kb * 128 + R]);
        half4 v;
        v.x = __hmul2(__hmul2(h01, su2), sv4.x);
        v.y = __hmul2(__hmul2(h23, su2), sv4.y);
        stile[base] = v.x;
        stile[base + 1] = v.y;
    }
    // stile row R (in index) holds out columns o at half2 R*64 + ((o>>2) ^ ((R>>2)&31))*2 + ((o&3)>>1).
    // Each warp writes 16 out rows; a lane gathers in indices R = lane*4 .. +3 (so (R>>2)&31 == lane) and
    // stores them as one half4: 256 contiguous bytes per warp. The rows are the chunk's (row0 selects the tiles).
    __syncthreads();
    const half * sh = reinterpret_cast<const half *>(stile);
#pragma unroll 4
    for (int oo = 0; oo < 128 / (TD_RH_THREADS / 32); ++oo) {
        const int o = warp_id * (128 / (TD_RH_THREADS / 32)) + oo;
        const int col = (((o >> 2) ^ lane_id) * 2 + ((o & 3) >> 1)) * 2 + (o & 1);   // half index in a row
        half4 v;
        v.x = __halves2half2(sh[(lane_id * 4 + 0) * 128 + col], sh[(lane_id * 4 + 1) * 128 + col]);
        v.y = __halves2half2(sh[(lane_id * 4 + 2) * 128 + col], sh[(lane_id * 4 + 3) * 128 + col]);
        *((half4 *) (g_unpacked + (size_t) (nb * 128 + o) * ld_out + kb * 128 + lane_id * 4)) = v;
    }
}

// The rows [row0, row0 + rows) of the matrix: row_tiles is the whole matrix's (the tiles are addressed in the
// matrix's row tiles, a chunk is a slice of them).
template <class Codec>
void reconstruct_rows_codec(half * W, int ld, const uint16_t * T, const half * suh, const half * svh, int rows,
                            int cols, int row0, int row_tiles, cudaStream_t stream)
{
    dim3 grid((rows + 127) / 128, (cols + 127) / 128);
    reconstruct_had_kernel<Codec><<<grid, TD_RH_THREADS, 0, stream>>>(W, T, suh, svh, row_tiles, row0, ld);
}

}  // namespace

// W[out * ld + in] = the matrix in the original basis (out rows, in columns); ld a multiple of 4.
void reconstruct_rows(half * W, int ld, const uint16_t * T, const half * suh, const half * svh, int k, int rows,
                      int cols, int row0, int row_tiles, cudaStream_t stream)
{
    switch (k) {
        case 1: reconstruct_rows_codec<Mul1<1>>(W, ld, T, suh, svh, rows, cols, row0, row_tiles, stream); break;
        case 2: reconstruct_rows_codec<Mul1<2>>(W, ld, T, suh, svh, rows, cols, row0, row_tiles, stream); break;
        case 3: reconstruct_rows_codec<Mul1<3>>(W, ld, T, suh, svh, rows, cols, row0, row_tiles, stream); break;
        case 4: reconstruct_rows_codec<Mul1<4>>(W, ld, T, suh, svh, rows, cols, row0, row_tiles, stream); break;
        case 15: reconstruct_rows_codec<Mul1Frac<1>>(W, ld, T, suh, svh, rows, cols, row0, row_tiles, stream); break;
        case 25: reconstruct_rows_codec<Mul1Frac<2>>(W, ld, T, suh, svh, rows, cols, row0, row_tiles, stream); break;
        case 35: reconstruct_rows_codec<Mul1Frac<3>>(W, ld, T, suh, svh, rows, cols, row0, row_tiles, stream); break;
        default: break;   // unsupported rate: the caller checks before calling
    }
}

// The rows [row0, row0 + rows) of the matrix in the original basis: fp16 [rows][in], ld = in.
void reconstruct(const Matrix & M, half * w, int row0, int rows, cudaStream_t s)
{
    reconstruct_rows(w, M.in, M.B, M.suh, M.svh, M.k, rows, M.in, row0, M.out / 16, s);
}

}  // namespace truss::trellis_dense
