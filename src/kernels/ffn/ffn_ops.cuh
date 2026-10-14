// FFN elementwise ops for prefill (the math of qwen4exp::reference::route / shared / ffn; the router logits and the
// shared expert's projections are GEMMs run by the caller):
//   route       logits -> softmax -> top k (ties: the lower expert id), weights renormalized over the k (sum clamped
//               to >= 2^-14), ids in descending probability as the reference's partial_sort
//   route_publish  the same top k plus the doorbell publish of a decode FFN, in one launch (Phase 2)
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

// sigmoid router with a selection bias (route_sigmoid.cu): top k of sigmoid(logit) + bias (nullptr: none), weights the
// chosen sigmoids, / (sum + 1e-6) when renorm, x scale. LFM2-MoE, DeepSeek-V3-style routing.
void route_sigmoid(const float * logits, const float * bias, int T, int E, int k, bool renorm, float scale, int * ids,
                   float * wts, cudaStream_t stream);

// The publish of a doorbell decode FFN folded into its routing (runtime/doorbell.cuh's publish, three launches
// today): one kernel computes this layer's top-k into ids/wts (device) and the next layer's predicted top-k
// (pred_logits, null without the pre-gating hint), writes ids/weights/prediction to the mapped host buffers
// dst_ids/dst_wts/dst_pred, streams x [n_x] into dst_x, and rings the doorbell once every CTA has fenced. n_x = 0
// publishes no rows (no CPU tier reads them). arrive is a device word, zero before the first launch; the kernel
// resets it, so it counts one launch at a time. T <= 8 (cpu::MAX_ROWS).
// doorbell = arrive = nullptr: write and fence only, no ring - for a caller that ships the rows with the copy
// engine and rings behind that copy, so the compute stream never waits for the rows to cross the link.
void route_publish(const float * logits, const float * pred_logits, int T, int E, int k, int * ids, float * wts,
                   const float * x, int n_x, int * dst_ids, int * dst_pred, float * dst_wts, float * dst_x,
                   volatile int * doorbell, int seq, unsigned * arrive, int copy_ctas, cudaStream_t stream);

// The engine's choice of copy CTAs for n_x floats of FFN input (0: nothing to publish). One, measured: the link,
// not the CTA count, is the ceiling, and extra CTAs each pay a system fence.
int publish_ctas(int n_x);

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
