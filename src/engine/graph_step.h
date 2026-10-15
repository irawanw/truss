// engine::GraphStep: one all-resident T=1 decode step (run + head + greedy argmax + next-token copy-back) as a
// reusable CUDA graph (B1, TRACKER #167-168).
//
// Per step the caller passes the position, the token and the device buffers; the step refreshes pos0 in the
// model's pos-taking nodes, rewrites the pinned token, launches, syncs, and returns the next token. Re-capture
// happens when the model's epoch(pos) changes (perf-only: any splits value covers all cells, TRACKER #168).
//
// Requires: no layer split, all experts resident with no CPU tier, T=1 steps on the selected slot, and the SAME
// logits/next_d addresses for the graph's lifetime. The capture skips the per-chunk host work it cannot replay:
// check_driver/begin_ring are no-ops in this mode; kv_reserve past 4K boundaries has nothing to order (no ring
// copies ever target reclaimed rows when every expert is resident); adapt_every/admit_idle/aux must stay off
// (their host-side copies would race the replayed kernels).
#pragma once
#include "core/cuda_check.h"
#include "engine/model.h"
#include "engine/pipeline.h"
#include "engine/util.h"

#include <cstdint>
#include <vector>

namespace truss::engine {

class GraphStep {
public:
    // throws if graph mode is unsupported (a layer split, non-resident experts, or no pos-taking nodes named)
    GraphStep(Pipeline & p, int vocab);
    ~GraphStep();
    GraphStep(const GraphStep &) = delete;
    GraphStep & operator=(const GraphStep &) = delete;
    // one decode step at pos with token tok; logits/next_d are device buffers, same addresses every call
    int32_t step(int pos, int32_t tok, float * logits, int * next_d);
    int captures() const { return captures_; }   // capture count, re-captures included (diagnostics)

private:
    struct PosNode {
        cudaGraphNode_t node = nullptr;
        int arg = -1, nargs = 0;
        cudaKernelNodeParams kp{};
        void * args[12] = {};
        int pos_store = 0;
    };
    void capture(int pos, float * logits, int * next_d);
    Pipeline & p_;
    const int vocab_;
    const StageModel::GraphInfo gi_;
    cudaStream_t s_ = nullptr;
    int32_t * tok_ = nullptr;   // pinned: the embed H2D node re-reads it at every launch
    int * next_h_ = nullptr;    // pinned: the graph's token copy-back target
    cudaGraph_t g_ = nullptr;
    cudaGraphExec_t e_ = nullptr;
    std::vector<PosNode> nodes_;
    int64_t epoch_ = -1;
    int captures_ = 0;
};

}  // namespace truss::engine
