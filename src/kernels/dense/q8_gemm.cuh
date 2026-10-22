// Dense Q8 GEMMs for prefill: y = W x with Q8_0 weights. Two activation paths:
//   q8_gemm      Q8_1-quantized activations on int8 tensor cores (llama-paw's numerics)
//   q8_gemm_a16  fp16 activations, cuBLAS fp16 tensor cores (lower error)
// Q8_1 path:
// Numerics are llama-paw's (ref::Numerics::LLAMA, TRACKER #33): each 32-block of an activation row is quantized
// with d = amax / 127 (q = round(x / d), fp32 d), the block dot is exact in int32 and scaled by the fp16 weight
// scale times the fp16-rounded activation scale. One mma m16n8k32 spans exactly one Q8 block, so the scaling is
// one fp32 FMA per result per block. Why int8: 178-223 TOPS vs 57-66 TFLOPS fp16 on the dense prefill shapes
// (cuBLAS, GPU 2, TRACKER #40).
//
// Weights are repacked once from GGUF's 34-byte blocks into an aligned layout (16-byte async copies).
#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace truss::dense {

struct Q8Matrix {                 // q [out][in] int8, d [out][in / 32] fp16
    const int8_t * q;
    const half * d;
    int in, out;
};

// GGUF Q8_0 tensor (ne = [in, out]) -> Q8Matrix storage. in % 64 == 0.
void q8_repack(const void * blocks, int in, int out, int8_t * q, half * d, cudaStream_t stream);

// x fp32 [rows][in] -> xq int8 [rows][in], xd fp16 [rows][in / 32]
void q8_quantize_act(const float * x, int rows, int in, int8_t * xq, half * xd, cudaStream_t stream);
void q8_quantize_act(const half * x, int rows, int in, int8_t * xq, half * xd, cudaStream_t stream);

// y fp32 [rows][W.out] = W . x, from the quantized activations. W.in % 64 == 0; any rows, any W.out.
void q8_gemm(const Q8Matrix & W, const int8_t * xq, const half * xd, int rows, float * y, cudaStream_t stream);

// W8A16: y fp32 [rows][W.out] = W . x with fp16 activations x [rows][W.in], no activation quantization (lower error
// than Q8_1 by ~0.5% per matmul, TRACKER #33). The weights are dequantized to w16 (scratch, W.in * W.out halfs; one
// fp16 rounding of q * d) and multiplied by cuBLAS with fp32 accumulation. W.in % 64 == 0.
void q8_gemm_a16(const Q8Matrix & W, const half * x, int rows, float * y, half * w16, cublasHandle_t cublas,
                 cudaStream_t stream);
// Block-diagonal (grouped) q8_gemm_a16: W [G * R][K] holds G blocks of R outputs; row t's group g input is
// x[t][g * K .. (g + 1) * K) (x [rows][G * K]) and its outputs go to y[t][g * R .. (g + 1) * R) (y [rows][G * R]).
// One dequant, one strided-batched cuBLAS GEMM (batch = group): G times fewer FLOPs than the dense product of the
// rows x G stacked inputs with all of W (DeepSeek-V4.1's grouped output LoRA wo_a, TRACKER #194). W.out % G == 0.
void q8_gemm_a16_grouped(const Q8Matrix & W, int G, const half * x, int rows, float * y, half * w16,
                         cublasHandle_t cublas, cudaStream_t stream);

// Grouped Q8_1: W [G * R][K] holds G blocks of R outputs; row t's group g input is x[t][g * K .. (g + 1) * K) (the
// quantized activations xq [rows][G * K], the scales xd) and its outputs go to y[t][g * R .. (g + 1) * R)
// (y [rows][G * R]). G times fewer FLOPs than the dense product of the rows x G stacked inputs with all of W, and
// G times less output memory (no stacked ya). Bit-identical to that dense product: the int32 block dots and their
// fp32 folding depend on the output element alone, not on the tile it shares.
void q8_gemm_grouped(const Q8Matrix & W, int G, const int8_t * xq, const half * xd, int rows, float * y,
                     cudaStream_t stream);

// q8_gemm for a few rows (decode, rows <= GEMV_ROWS): one warp per output, reads each weight once. Same numerics.
constexpr int GEMV_ROWS = 8;
void q8_gemv(const Q8Matrix & W, const int8_t * xq, const half * xd, int rows, float * y, cudaStream_t stream);

// y fp32 [rows][out] = W x for fp32 W [out][in] (ggml [in, out]), x [rows][in]; in % 128 == 0. One warp per output,
// fixed reduction order: every row's result is independent of `rows` (the router must not change with the batch:
// cuBLAS picks its SGEMM by row count and 1e-6 flips near-tied experts, TRACKER #60). For decode-sized rows.
void f32_gemv(const float * W, int in, int out, const float * x, int rows, float * y, cudaStream_t stream);

// Several matrices that read the same activations in one launch (decode): y[i] fp32 [rows][W[i]->out] = W[i] . x.
// Same per-output arithmetic as q8_gemv / f32_gemv up to the fp32 summation order, which depends only on the shapes
// (lanes per output for short rows, warps per output for long ones: from `in` only), never on `rows` or on the other
// matrices of the launch: a matrix gives bit-identical results fused or alone. All W[i] share `in`.
constexpr int MULTI_MAX = 8;
void q8_gemv_multi(const Q8Matrix * const * W, float * const * y, int n, const int8_t * xq, const half * xd, int rows,
                   cudaStream_t stream);
// W8A16 for decode rows (rows <= GEMV_ROWS): the same mapping as q8_gemv(_multi), fp16 activations, int8 weights
// converted in registers (no dequantized copy, so the same weight traffic as the Q8_1 path), fp32 accumulation.
void q8_gemv_multi_a16(const Q8Matrix * const * W, float * const * y, int n, const half * x, int rows, cudaStream_t stream);
void q8_gemv_a16(const Q8Matrix & W, const half * x, int rows, float * y, cudaStream_t stream);
// q8_gemm_a16_grouped for decode rows (rows <= GEMV_ROWS) on the tensor-core gemv: the G blocks are G matrices of one
// launch, block g reading row slice x[t][g K .. (g + 1) K) (x [rows][G K]) into y[t][g R ..) (y [rows][G R]). Reads
// the int8 weights once: no dequantized fp16 copy (the grouped GEMM wrote and re-read 2 bytes per weight: 217 us vs
// ~40 us per DS41 layer for wo_a at 4 rows, 10-22). R % 16 == 0 (a tile stays in one group), G <= MULTI_MAX.
void q8_gemv_grouped_a16(const Q8Matrix & W, int G, const half * x, int rows, float * y, cudaStream_t stream);
// Its k split: S warps per 16-output tile in a block, KB blocks per tile (KB > 1: the last block of a tile adds the
// block partials in order). Chosen from the shape (in, total outputs of the launch) only, never from rows: a row's
// result does not depend on the row count, but a matrix fused with others may differ from it alone in the last bits.
struct MmaSplit { int S, KB; };
MmaSplit mma_split(int in, int outs);
// The tensor-core gemv is on unless TRUSS_GEMV_MMA=0; the fused epilogues below exist only in it.
bool mma_on();
// the same with a given split (tuning: tools/tk-bench/gemv_split_bench)
void q8_gemv_multi_a16_cfg(const Q8Matrix * const * W, float * const * y, int n, const half * x, int rows, MmaSplit c,
                           cudaStream_t stream);

// Two fused launches for the hyper-connection (the engine puts them behind TRUSS_HC_SILU / TRUSS_HC_COLLAPSE).
// Neither changes a single arithmetic operation of the gemv it replaces: only what happens to the result does.
//   q8_gemv_multi_a16_silu: a matrix whose y16[i] is non-null writes silu(row * y16s[i]) as fp16 [rows][out] in the
//     gemv's epilogue and leaves y[i] alone. The hc down projection's fp32 output feeds nothing but the silu the up
//     projection then reads, so that separate kernel disappears (same expression, same value, same rounding).
//   q8_gemv_a16_up_collapse: the hc up projection (W.out == hc * d) plus the collapse of the streams in one launch:
//     gate never reaches memory and mixed/mixed16 come straight out of the CTA that holds all hc gate values of its
//     own 16 columns. Needs the shape's own k split to stay inside a block (up_collapse_ok).
void q8_gemv_multi_a16_silu(const Q8Matrix * const * W, float * const * y, half * const * y16,
                            const float * y16s, int n, const half * x, int rows, cudaStream_t stream);
bool up_collapse_ok(int in, int out, int hc, int d);
void q8_gemv_a16_up_collapse(const Q8Matrix & W, const half * x, int rows, const float * res, const float * rstd,
                             const float * gamma, float * mixed, half * mixed16, int hc, int d, cudaStream_t stream);

void f32_gemv_multi(const float * const * W, const int * out, float * const * y, int n, int in, const float * x,
                    int rows, cudaStream_t stream);

// out = rows ids[0 .. n) of W as a new matrix (q [n][in], d [n][in / 32]), e.g. the MTP draft head's vocabulary subset
void q8_gather(const Q8Matrix & W, const int * ids, int n, int8_t * q, half * d, cudaStream_t stream);

// out fp32 [n][W.in] = rows ids[0 .. n) of W (q * d), e.g. the token embedding
void q8_rows(const Q8Matrix & W, const int * ids, int n, float * out, cudaStream_t stream);

}  // namespace truss::dense
