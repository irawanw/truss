// Routed-expert weights of every MoE layer on one GPU: a hot set stays resident, the rest (cold) lives in pinned
// host memory and is copied per layer into one of two device slots while the previous layer computes.
//
// The MoE kernels address an expert as ProjView.trellis + meta word offset (int32, so +-4 GiB of the base). Each
// projection (gate, up, down) gets one arena laid out as
//     [hot experts of the first half of the layers][slot 0][slot 1][hot experts of the second half]
// with the base at slot 0: every hot expert is at a fixed (possibly negative) offset, and layer l's cold experts
// always land at the same place in slot l % 2, so the meta tables are built once and never rewritten. The load
// throws if an offset does not fit int32.
//
// Per use of layer l (all on the caller's compute stream, except the copy):
//     acquire(l)   compute waits for layer l's copy
//     ...          moe ops on weights(l)
//     release(l)   the slot may be overwritten once compute gets here
//     prefetch(l + 2), which reuses the slot after release(l)
// A decode step, whose routing is known only at its layer, calls fetch(l, ids) (the few cold experts it uses) right
// before acquire(l) instead of prefetching whole layers.
#pragma once
#include "formats/trellis_table.h"
#include "kernels/moe/moe_weights.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <vector>

namespace truss::runtime {

using ExpertLayer = std::array<const formats::ExpertTable *, 3>;   // gate, up, down of one layer

class ExpertStore {
public:
    using HotSet = std::vector<std::vector<uint8_t>>;                 // [layer][expert] = 1: resident

    // The same number of hot experts in every layer (experts in index order), the most whose device bytes (hot,
    // two slots, scales, meta) fit `budget`.
    static HotSet plan(const std::vector<ExpertLayer> & layers, size_t budget);

    ExpertStore(const std::vector<ExpertLayer> & layers, const HotSet & hot);
    ~ExpertStore();
    ExpertStore(const ExpertStore &) = delete;
    ExpertStore & operator=(const ExpertStore &) = delete;

    moe::Weights weights(int layer) const;
    void prefetch(int layer);                                      // all cold experts, on the store's copy stream
    // decode: only the cold experts among ids[0 .. n) (host), same slot places; used instead of prefetch
    void fetch(int layer, const int * ids, int n);
    void acquire(int layer, cudaStream_t compute);
    void release(int layer, cudaStream_t compute);

    size_t device_bytes() const { return device_bytes_; }
    size_t cold_bytes() const { return cold_total_; }                // pinned host bytes, streamed once per chunk
    int layers() const { return (int) layers_.size(); }

private:
    struct Layer {
        size_t cold[3] = {};                                        // cold bytes per projection
        const void * host[3] = {};                                  // pinned source of the cold experts
        std::vector<int64_t> cold_off[3];                           // [expert] byte offset in host / slot, -1: hot
        std::vector<size_t> bytes[3];                               // [expert]
        int32_t * meta[3] = {};                                     // device (K, word offset from the slot-0 base)
        const half * suh[3] = {}, * svh[3] = {};
        int n_expert = 0;
    };
    std::vector<Layer> layers_;
    uint16_t * base_[3] = {};                                      // slot 0 of each projection's arena
    size_t slot_words_[3] = {};
    std::vector<void *> device_, pinned_;
    cudaStream_t copy_ = nullptr;
    cudaEvent_t copied_[2] = {}, released_[2] = {};
    bool released_recorded_[2] = {};
    size_t device_bytes_ = 0, cold_total_ = 0;
};

}  // namespace truss::runtime
