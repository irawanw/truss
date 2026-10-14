// LFM2 short convolution (HF Lfm2MoeShortConv, llama.cpp build_shortconv_block), between its two projections:
//   [B | C | x] = in_proj(h)                      bcx [T][3d] fp32
//   bx[t] = B[t] * x[t]
//   conv[t][c] = sum_k w[c][k] * bx[t - (L - 1) + k][c]   causal depthwise, rows before the chunk from `state`
//   y[t] = C[t] * conv[t]                         y16 [T][d] fp16: the out projection's input
// state [L - 1][d] holds the last L - 1 bx rows of the sequence; it is advanced to the chunk's last L - 1 rows.
// bx_out (may be nullptr): the chunk's bx rows, which a verify window keeps to rebuild the state of accepted rows.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace truss::conv {

// w [d][L] (GGUF shortconv.conv: ne0 = L); L <= 8
void short_conv(const float * bcx, const float * w, float * state, int T, int d, int L, half * y16, float * bx_out,
                cudaStream_t stream);

}  // namespace truss::conv
