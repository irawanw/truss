// Argmax of a row of logits on the device (greedy decoding without copying the logits to the host).
#pragma once
#include <cuda_runtime.h>

namespace truss::sampling {

// out[0] = the index of the largest of x[0 .. n) (ties: the lowest index)
void argmax(const float * x, int n, int * out, cudaStream_t stream);

}  // namespace truss::sampling
