// Argmax of a row of logits on the device (greedy decoding without copying the logits to the host).
#pragma once
#include <cuda_runtime.h>

namespace truss::sampling {

// out[0] = the index of the largest of x[0 .. n) (ties: the lowest index)
void argmax(const float * x, int n, int * out, cudaStream_t stream);

// the same over a subset: out[0] = map ? map[argmax] : argmax, prob[0] = the maximum's softmax probability over x
// (the MTP draft head: the subset's token ids, and the confidence that decides whether to keep drafting)
void argmax_prob(const float * x, int n, const int * map, int * out, float * prob, cudaStream_t stream);

}  // namespace truss::sampling
