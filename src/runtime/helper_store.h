// Helper experts (PLAN Phase 1/2): part of ANOTHER stage's hot set, resident on this GPU. Uploaded once at load and
// never touched again: no ring, no host copy, no stream mode, no eviction. The owner of these layers keeps its own
// pinned cold copy and streams it as today; Phase 2 lets the owner hand those rows to this GPU's helper stream and
// read the partial result back instead.
//
// Device layout: one arena, the experts in fill order (each expert's three projections contiguous, every projection
// a multiple of 32 bytes). One meta table per (layer, projection) holds (K, offset in 32-byte units from the arena
// start), so weights(l) is an ordinary moe::Weights and moe::window needs nothing new. suh / svh are the whole
// tensors, indexed by expert id like the kernel expects.
#pragma once
#include "runtime/expert_store.h"
#include "kernels/moe/moe_weights.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <utility>
#include <vector>

namespace truss::runtime {

class HelperStore {
public:
    using Item = std::pair<int, int>;                 // (store layer, expert), in fill order: best score first

    // The scales and the meta tables: the VRAM the arena is not. Reserve it beside the arena.
    static size_t fixed_bytes(const std::vector<ExpertLayer> & layers);

    HelperStore(std::vector<ExpertLayer> layers, const std::vector<Item> & items);
    ~HelperStore();
    HelperStore(const HelperStore &) = delete;
    HelperStore & operator=(const HelperStore &) = delete;

    bool has(int layer, int expert) const { return set_[(size_t) layer][expert] != 0; }
    // The kernel's view of one layer: only the stored experts are readable (an unstored expert's meta is zeroed, so
    // the caller must check has() — serve() masks everything else).
    moe::Weights weights(int layer) const { return w_[(size_t) layer]; }
    size_t trellis_bytes() const { return trellis_; }
    size_t fixed_bytes_used() const { return fixed_; }
    int count() const { return count_; }
    int stored(int layer) const { return stored_[(size_t) layer]; }
    int layers() const { return (int) set_.size(); }

private:
    std::vector<ExpertLayer> layers_;
    std::vector<std::vector<uint8_t>> set_;           // [layer][expert]
    std::vector<int> stored_;
    std::vector<moe::Weights> w_;
    std::vector<void *> device_;
    uint8_t * arena_ = nullptr;
    size_t trellis_ = 0, fixed_ = 0;
    int count_ = 0;
};

}  // namespace truss::runtime
