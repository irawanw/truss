// GPU <-> host signalling for decode layers without a synchronizing driver call (Strata's "doorbell", TRACKER #61).
//
// A decode FFN needs the host between the router and the experts (which cold experts to copy, which run on the CPU).
// Instead of cudaStreamSynchronize + relaunch per layer, the host enqueues the whole pass ahead and:
//   publish       the GPU copies the layer's routing (ids, the next layer's predicted ids, weights) and FFN input rows
//                 into mapped pinned host memory, then raises the layer's doorbell (a mapped int = the pass's seq);
//   spin_until    a one-thread kernel holds the stream until a flag reaches seq: the "go" word the copy stream writes
//                 after the layer's expert copies, or the CPU tier's done flag;
//   add_mapped    routed += the CPU tier's rows, read from mapped host memory.
// A host driver thread (qwen4exp::Forward) spins on each doorbell, splits the experts, issues the copies and the CPU
// work, and raises the flags. Flags only increase (seq per pass), so no reset is needed between passes.
#pragma once
#include <cuda_runtime.h>

namespace truss::runtime {

// dst_* are mapped host pointers (device view). n_ids = T * k, n_x = T * d.
void publish(const int * ids, const int * pred, const float * wts, int n_ids, const float * x, int n_x, int * dst_ids,
             int * dst_pred, float * dst_wts, float * dst_x, volatile int * doorbell, int seq, cudaStream_t stream);

// waits until *flag >= seq (flag: device memory or a mapped host word)
void spin_until(const volatile int * flag, int seq, cudaStream_t stream);

// y [n] += x [n], x in mapped host memory
void add_mapped(float * y, const float * x, int n, cudaStream_t stream);

}  // namespace truss::runtime
