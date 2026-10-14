// MoeExec's private state types: the CPU tier and the per-layer doorbell. Included only by runtime/moe_*.cu.
#pragma once
#include "runtime/moe_exec.h"

namespace truss::runtime {

// Options::cpu_dir / cpu_trellis: which experts of each store layer run on the host, their copies, the pool, and
// pinned staging for one layer's rows (x, routing weights in; y out) and the GPU kernel's masked routing
struct MoeExec::CpuTier {
    std::vector<std::vector<uint8_t>> is_cpu;         // [store layer][expert]
    std::vector<std::vector<uint8_t>> data;           // [store layer] the CPU experts' q4s bytes
    std::vector<std::vector<float>> fscales;          // [store layer] preconverted fp32 weight scales (one per q4s
                                                      // block; the per-block _cvtsh_ss in the hot loop cost ~0.9 ms)
    std::vector<std::vector<cpu::Q4Expert>> view;     // [store layer][expert]
    bool trellis = false;                             // Options::cpu_trellis: tview, not view
    std::vector<std::vector<cpu::TrellisExpert>> tview;   // [store layer][expert] views of the pinned pack bytes
    cpu::Slot slot(int row, int l, int e, float w)
    {
        return trellis ? cpu::Slot{ row, nullptr, w, &tview[l][e] } : cpu::Slot{ row, &view[l][e], w };
    }
    std::shared_ptr<cpu::ExpertPool> pool;
    std::shared_ptr<std::mutex> pool_mu;              // a split's stages share the pool: one call at a time
    float * x = nullptr, * w = nullptr, * y = nullptr;   // pinned [rows][d], [rows][k], [rows][d]
    int * ids = nullptr;                                 // pinned [rows][k] routing with CPU slots as -1
    size_t bytes = 0;
};

// Doorbell decode (TRACKER #61): FFNs of windows up to cpu::MAX_ROWS rows run without a host sync. Per store layer:
// mapped host buffers the GPU publishes the routing and FFN input into, the doorbell, the CPU tier's result and done
// flag (mapped), a device "go" word the copy stream writes after the layer's copies, and the masked ids (CPU slots
// = -1) the copy stream uploads. The driver thread services the layers in enqueue order.
struct MoeExec::Bell {
    int * ids = nullptr, * pred = nullptr, * mids = nullptr;     // mapped [8][k]; mids: pinned source
    float * w = nullptr, * x = nullptr, * y = nullptr;           // mapped [8][k], [8][d], [8][d]
    int * doorbell = nullptr, * cpu_done = nullptr;             // mapped words
    int * plan = nullptr, * need_copy = nullptr;                // mapped words the driver writes (wait_plan)
    int * go = nullptr, * d_mids = nullptr;                     // device
    unsigned * arrive = nullptr;   // device: the fused route+publish's CTA counter (zeroed once, resets itself)
    int seq = 0;                   // host: the last seq enqueued
};

}  // namespace truss::runtime
