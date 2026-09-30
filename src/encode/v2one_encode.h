// Pack-time encoder for the v2one trellis codec (src/kernels/trellis/codec_v2one.cuh): tail-biting Viterbi over
// 256-weight tiles, minimizing the tile's squared error. The caller (the X3 recipe: Hadamard, regularization,
// LDLQ with the expert's Hessian) supplies the tiles already in codebook units and in the tile order.
//
// Tail-biting as exllamav3's quantize_tiles_kernel.cuh: pass 1 runs the ring rotated by half with a free start and
// reads the state overlap at the ring's seam; pass 2 runs in order with the seam fixed at both ends. Near-optimal
// (the exact tail-biting search is 2^(16-S) passes).
#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace truss::encode {

// device workspace per tile for v2one_encode at rate S (backpointers, and the step costs when S = 2)
size_t v2one_workspace_bytes_per_tile(int S);

// tiles: [n, 256] fp32 (device), in codebook units and tile order. Outputs (device): q [n, 256] the decoded values
// (bit-exact with the kernel decode, as float), states [n, 128] the tile's ring of states (codec_v2one.cuh
// v2one_pack_tile packs them). Runs on `stream` in chunks that fit `ws_bytes`; throws if one tile does not fit.
void v2one_encode(int S, const float * tiles, int n, float * q, uint16_t * states, void * ws, size_t ws_bytes,
                  cudaStream_t stream);

}  // namespace truss::encode
