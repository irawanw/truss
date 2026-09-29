// MoE window: routed-expert SwiGLU FFN for a small verify window (<= MAX_ROWS rows) on trellis expert weights,
// as one persistent kernel launch.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace truss::moe {

constexpr int MAX_ROWS = 8;                // verify-window rows

// Routed-expert shape of a model. A new model is a new struct plus an explicit instantiation in moe_window.cu.
struct FlashNext {                         // Qwen3.8 Flash-Next: 512 experts, top-10
    static constexpr int D_MODEL = 2560, D_FF = 640, TOPK = 10;
};

struct ProjView {                          // one projection (gate, up or down) of all experts of a layer
    const uint16_t * trellis;              // concatenated tiles
    const int32_t * meta;                  // [n_expert][2] = (K, uint16 word offset)
    const half * suh;                      // [n_expert][in]
    const half * svh;                      // [n_expert][out]
};

struct Weights {
    ProjView proj[3];                      // gate, up, down
    int n_expert;
};

// Optional per-item timeline (tools/tk-bench/moe_trace). One record per work item, in queue order; kind 0 = H,
// 1 = gate/up, 2 = down. Times are %globaltimer ns: the block's kernel entry, item start, after the item's
// dependency wait, item end.
struct TraceEvent {
    int block, kind, slot, group;
    unsigned long long t_enter, t_start, t_ready, t_end;
};
constexpr int MAX_TRACE_EVENTS = 4096;

template <class Shape> size_t workspace_bytes();

// Zero the workspace's queue counters once after allocation; each window() launch leaves them zero again.
template <class Shape> void workspace_init(void * ws, cudaStream_t stream);

// out[r] = sum_s w[r][s] * FFN_{ids[r][s]}(x[r]); x, out fp32 [n_rows][D_MODEL]; ids int32 / wts fp32
// [n_rows][TOPK]. n_rows <= MAX_ROWS. Device work only: capturable into a CUDA graph.
// trace: nullptr (normal use) or a device array of MAX_TRACE_EVENTS.
template <class Shape>
void window(const Weights & W, const float * x, const int * ids, const float * wts, int n_rows, float * out, void * ws,
            cudaStream_t stream, TraceEvent * trace = nullptr);

}  // namespace truss::moe
