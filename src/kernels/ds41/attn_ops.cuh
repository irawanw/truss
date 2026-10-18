// DeepSeek-V4.1 attention, elementwise pieces (math: dsref/model.py Attention / Compressor; blocks/attn_ds41 calls
// them in order):
//   rope_tail     rotate the last RD dims of each head, adjacent pairs as complex numbers (GPT-J order), position
//                 pos_base + row * pos_mul (compressed latents: group j sits at j * ratio), inverse = conjugate
//   ring_write    the chunk's rows into a ring of R rows (row of position p at p % R); only the last R rows are kept
//   pool          ratio-r compressor: every group of r positions completed by this chunk -> sum over the group of
//                 kv * softmax(score) per dim; the group's earlier positions come from the raw ring (written by
//                 earlier chunks) or this chunk
//   pick_groups   y [T * G][G * C] (all groups' outputs for every group's input: the block-diagonal wo_a as one
//                 dense projection of T * G rows) -> out [T][G * C], block g of row (t, g)
// Positions are absolute token positions of the sequence. Device work on the caller's stream; deterministic.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

namespace truss::ds41 {

// Rotary inverse frequencies [rd / 2] as dsref precompute_freqs_cis builds them in fp32: with orig_ctx > 0 YaRN
// (dims whose wavelength fits the training context keep their frequency, those beyond are / factor, a linear ramp
// between beta_fast and beta_slow); orig_ctx = 0: plain.
std::vector<float> rope_inv_freq(int rd, float base, int orig_ctx, float factor, float beta_fast, float beta_slow);

// x: rows of `heads` heads of D floats (row stride heads * D); the last rd of each head rotated in place.
void rope_tail(float * x, int rows, int heads, int D, int rd, int pos_base, int pos_mul, const float * inv_freq,
               bool inverse, cudaStream_t stream);

// ring [R][W] fp16 <- src [T][W] fp32 at positions pos0 .. pos0 + T - 1 (the last min(T, R) rows)
void ring_write(half * ring, int R, const float * src, int W, int pos0, int T, cudaStream_t stream);
// same for an fp32 ring (the ratio-2 compressor's raw kv | score rows)
void ring_write_f32(float * ring, int R, const float * src, int W, int pos0, int T, cudaStream_t stream);

// Groups completed by a chunk: group j ends at position (j + 1) * ratio - 1, so the chunk [pos0, pos0 + T) completes
// j0 = pos0 / ratio .. (pos0 + T) / ratio - 1
inline void groups_done(int pos0, int T, int ratio, int & j0, int & n)
{
    j0 = pos0 / ratio;
    n = (pos0 + T) / ratio - j0;
}

// kv, sc: this chunk's compressor rows [T][D] (fp32); raw_kv, raw_sc [R][D] rings hold earlier positions.
// out [n][D] for groups j0 .. j0 + n - 1 (groups_done)
void pool(const float * kv, const float * sc, const float * raw_kv, const float * raw_sc, int R, int D, int ratio,
          int pos0, int j0, int n, float * out, cudaStream_t stream);

// y [T * G][G * C] -> out [T][G * C]
void pick_groups(const float * y, int T, int G, int C, float * out, cudaStream_t stream);

// fp32 -> fp16 rows (the window kv the attention reads for this chunk)
void to_half(const float * x, size_t n, half * y, cudaStream_t stream);

}  // namespace truss::ds41
