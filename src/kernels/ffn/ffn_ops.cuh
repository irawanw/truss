// FFN elementwise ops for prefill (the math of qwen4exp::reference::route / shared / ffn; the router logits and the
// shared expert's projections are GEMMs run by the caller):
//   route       logits -> softmax -> top k (ties: the lower expert id), weights renormalized over the k (sum clamped
//               to >= 2^-14), ids in descending probability as the reference's partial_sort
//   swiglu      silu(g) * u -> fp16 (the down projection's input)
//   shared_add  out = routed + y * sigmoid(gate)
//   count       counts[ids[i]] += 1 (routing profile for the expert hot set)
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace truss::ffn {

// logits [T][E] -> ids, wts [T][k]. E % 32 == 0, E <= 1024, k <= 32.
void route(const float * logits, int T, int E, int k, int * ids, float * wts, cudaStream_t stream);

// g, u [n] -> mid16 [n]
void swiglu(const float * g, const float * u, int n, half * mid16, cudaStream_t stream);

// out [T][d] = routed + y * sigmoid(gate [T])
void shared_add(const float * routed, const float * y, const float * gate, int T, int d, float * out,
                cudaStream_t stream);

// y [n] += x [n]
void add(float * y, const float * x, int n, cudaStream_t stream);

// counts [E] += how often each id appears in ids [n] (float: a profile of many tokens stays exact up to 2^24 per id)
void count(const int * ids, int n, float * counts, cudaStream_t stream);

}  // namespace truss::ffn
