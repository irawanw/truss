// DeepSeek-V4.1 indexer (dsref/model.py Indexer + select_candidate_blocks): which compressed positions a query
// attends to.
//   scores      s[t][n] = sum_h relu(q[t][h] . k[n]) * w[t][h] for the compressed positions the query at position
//               pos0 + t can see (n < (pos0 + t + 1) / ratio), -inf for the rest (n < n_max = (pos0 + T) / ratio)
//   select      per row the top min(k, count of entries > -inf) entries of a row, ties by the lower index, emitted in
//               ascending index order, -1 padded to k (dsref keeps the set; attention does not depend on order, but
//               a fixed order keeps its sums deterministic)
//   block_max   per row max over blocks of `block` positions, the block holding the row's newest visible position
//               pinned to +inf (select_candidate_blocks)
//   mask_blocks scores outside a row's chosen blocks -> -inf
// Index keys are fp32 [n][ID] (selection is sensitive to rounding; TRACKER #177).
#pragma once
#include <cuda_runtime.h>

#include <cstddef>

namespace truss::ds41 {

// q [T][NH][ID], w [T][NH] (scaled by w_scale = ID^-0.5 NH^-0.5 here), keys [n_max][ID] -> s [T][n_max]
void idx_scores(const float * q, const float * w, float w_scale, const float * keys, int NH, int ID, int pos0, int T,
                int ratio, float * s, cudaStream_t stream);

// x [T][n] -> idx [T][k]; one CTA per row (radix select over the order-preserving uint32 image of the floats)
void select_topk(const float * x, int T, int n, int k, int * idx, cudaStream_t stream);

// s [T][n] -> bm [T][ceil(n / block)]; vis[t] = visible positions of row t = (pos0 + t + 1) / ratio
void block_max(const float * s, int T, int n, int block, int pos0, int ratio, float * bm, cudaStream_t stream);

// blocks [T][kb] chosen block ids (-1 = none) -> s [T][n] = -inf outside them
void mask_blocks(float * s, int T, int n, int block, const int * blocks, int kb, cudaStream_t stream);

}  // namespace truss::ds41
