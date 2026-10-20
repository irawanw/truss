// The trellis-dense format: a dense matrix whose weights are PAW X3.1 trellis tiles (mul1 codec) instead of a
// quantised block format. Two paths, picked by the number of rows:
//   decode (rows <= TD_MAX_N): the window GEMV (gemv_window.cu) reads the trellis words in place - the weights are
//     never converted, so the matrix costs 3.5 bits per weight instead of 8 (Q8_0).
//   prompt (more rows): the matrix is reconstructed to fp16 (reconstruct.cu) and multiplied by a plain GEMM.
// The parts of the matrix (the words, the two scale vectors) live in the model file as "<name>.m3_trellis" /
// ".m3_suh" / ".m3_svh" (formats/gguf.h builds the group; the binding rule is in formats/binder.h).
#pragma once
#include "kernels/trellis/dense/plan.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace truss::trellis_dense {

// One trellis-coded dense matrix on the device.
struct Matrix {
    const uint16_t * B = nullptr;   // the trellis words (tile-major: k tile major, row tile minor)
    const half * suh = nullptr;     // [in]  the input-side scales
    const half * svh = nullptr;     // [out] the output-side scales
    int k = 0;                      // the rate code (formats/trellis_k.h): 1..4, 15/25/35
    int in = 0, out = 0;
};

// The window GEMV over a list of parts of one rate (one input, one output each): y[t][c] = W x[t], x fp16 [n][cols].
void matvec(const List & L, int k, const half * x, cudaStream_t s);

// The rows [row0, row0 + rows) of the matrix in the original basis (the scales and both Hadamards applied):
// fp16 [rows][in], ld = in.
void reconstruct(const Matrix & M, half * w, int row0, int rows, cudaStream_t s);

}  // namespace truss::trellis_dense
