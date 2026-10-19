// DeepSeek-V4.1 sparse attention (dsref kernel sparse_attn): MQA with one latent of D = 512 that is both key and
// value; a query at position p attends to
//   the sliding window  positions max(0, p - window + 1) .. p  (raw kv: this chunk's rows, earlier ones from the ring)
//   compressed entries  idx[t][0 .. k)  (>= 0; -1 = none) of the compressed cache, when idx != nullptr
// score = q . kv * scale; softmax over the scores plus a per-head sink logit (den += exp(sink - max)); out = sum p kv / den.
// v1 (sparse_attn.cu): one CTA per (row, head), CUDA cores, every head reads every key (the reference kernel).
// v2 (sparse_attn_tc.cu): tensor cores, 16 heads per CTA share each key tile, online softmax, key splits with a
// combine pass for few rows (decode); D = 512 and H % 16 == 0 only (TRACKER #189).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace truss::ds41 {

struct SparseAttnArgs {
    const float * q;          // [T][H][D] (rope applied)
    const half * chunk_kv;    // [T][D] this chunk's window kv (positions pos0 ..)
    const half * ring;        // [R][D] window kv of earlier positions (position p at p % R)
    int R;
    const half * comp;        // [n][D] compressed kv cache (nullptr: none)
    const int * idx;          // [T][k] chosen compressed entries (nullptr: none)
    int k;
    const float * sinks;      // [H]
    int H, D, window, pos0, T;
    float scale;
    float * out;              // [T][H][D]
};

void sparse_attn(const SparseAttnArgs & a, cudaStream_t stream);

// v2: whether it takes these shapes; the fp32 workspace its key-split launch needs (0: none); the launch
bool sparse_attn_tc_ok(const SparseAttnArgs & a);
size_t sparse_attn_tc_ws_floats(const SparseAttnArgs & a);
void sparse_attn_tc(const SparseAttnArgs & a, float * ws, cudaStream_t stream);

}  // namespace truss::ds41
