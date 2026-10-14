// Every engine option, model-agnostic, with the TRACKER row behind each default. apply_env() reads the TRUSS_* knobs
// on top (the C API and the benches both call it, so a config benched is the config served).
#pragma once
#include "blocks/linear.h"
#include "runtime/moe_exec.h"

#include <cstddef>
#include <functional>
#include <vector>

namespace truss::engine {

struct Options {
    blocks::Activations act = blocks::Activations::Q8_1;
    // routed counts [layer][expert] (tk-profile; n_layer + 1 layers with a drafter): the hot set; empty: index order
    std::vector<float> expert_usage;
    runtime::MoeOptions moe;          // residency, CPU tier, doorbell, hints, admission (runtime/moe_exec.h)
    size_t ring_bytes_override = 0;   // TRUSS_RING_GB
    int spec_rows = 0;                // > 0: verify windows of up to spec_rows (<= 8) rows
    // rows the prompt path's buffers are sized for (0: max_chunk). They are carved from the expert ring's spare
    // region, so rows past the prompt's real length evict cached experts for the whole run (TRACKER #70)
    int prefill_rows = 0;
    bool kv_int8 = false;             // attention K/V as int8 + fp16 scale per 64 (Strata's KV, TRACKER #87)
    bool direct_tables = true;        // per-token tables (PLE) read with O_DIRECT, not page faults (TRACKER #88)
    // E6 (TRACKER #114): the K/V caches' unused tail lent to the expert ring (one slot only)
    bool kv_lend = false;
    // Layer split over two devices: layers [0, layer_split) on layer_split_device (-1: the current device too, a
    // test), the rest, the head and the drafter on the current device (TRACKER #154)
    int layer_split = 0, layer_split_device = -1;
    float layer_split_frac = 0.5f;
    int slots = 1;                    // independent sequences (n_ctx / slots positions each)
    // called once the weights are on the device: the owner drops the model file's pages (TRACKER #72)
    std::function<void()> after_upload;
};

void apply_env(Options & o);

}  // namespace truss::engine
