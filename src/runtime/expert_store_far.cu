// ExpertStore's far tier (expert_far.h, docs/PLAN-20261024-ds41-nvme-tier.md): far experts have no host copy; they are
// read from the mapped pack into FarCache slots and copied to the ring from there. Setup, the driver's far_ready(), the
// copy-side hold and release, and the report.
#include "runtime/expert_store.h"
#include "runtime/expert_store_util.h"

#include "core/cuda_check.h"
#include "formats/gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace truss::runtime {

using namespace store_detail;

// the far experts' spans in the mapped pack (never read at load) and the pinned slots; nothing without a far set
void ExpertStore::init_far(const std::vector<ExpertLayer> & layers, const Sizes & z)
{
    const int L = (int) layers_.size();
    size_t slot_b = 0;
    long n_far = 0;
    double far_gb = 0;
    int E_max = 0;
    for (const Layer & Y : layers_) E_max = std::max(E_max, Y.n_expert);
    std::vector<std::array<FarCache::Span, 3>> spans((size_t) L * E_max);
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < layers_[l].n_expert; ++e)
            if (is_far(l, e)) {
                for (int p = 0; p < 3; ++p) {
                    const formats::ExpertTable & t = *layers[l][p];
                    spans[(size_t) l * E_max + e][p] = { reinterpret_cast<const uint8_t *>(t.trellis->data) + t.offset[e] * 2,
                                                         proj_bytes(t, e) };
                }
                slot_b = std::max(slot_b, layers_[l].bytes[e]);
                ++n_far, far_gb += layers_[l].bytes[e] / 1e9;
            }
    if (!n_far || !z.far_cache_bytes) return;   // 0: the far set is routed around (MoeOptions::far_drop)
    slot_b = align_up(slot_b, gguf::DIRECT_ALIGN);
    FarCache::Config fc;
    fc.n_layer = L, fc.n_expert = E_max, fc.slot_bytes = slot_b, fc.threads = z.far_threads;
    fc.piece_bytes = z.far_piece_bytes;
    fc.n_slots = (int) std::max<size_t>(16, z.far_cache_bytes / slot_b);
    uint8_t * region = nullptr;
    TRUSS_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&region), (size_t) fc.n_slots * slot_b, cudaHostAllocDefault));
    pinned_.push_back(region);
    far_ = std::make_unique<FarCache>(fc, std::move(spans), region);
    far_ev_pool_.resize(FAR_EVENTS);   // all made here: cudaEventCreate later can deadlock (expert_store.h, FarHold)
    for (cudaEvent_t & ev : far_ev_pool_) TRUSS_CUDA(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    std::fprintf(stderr, "expert store: far tier %ld experts (%.2f GB on the NVMe), %d slots x %.2f MB = %.2f GB pinned, "
                         "%d readers\n", n_far, far_gb, fc.n_slots, slot_b / 1e6, fc.n_slots * slot_b / 1e9, fc.threads);
}

bool ExpertStore::far_ready(int l, int e, std::chrono::steady_clock::time_point deadline)
{
    Layer & Y = layers_[l];
    far_poll_release();
    if (Y.far_ptr[e]) return true;
    if ((Y.far_ptr[e] = far_->acquire(l, e))) {
        ++far_st_.ready;
        return true;
    }
    far_->want(l, e, true);
    const auto t0 = std::chrono::steady_clock::now();
    if (t0 >= deadline) return false;
    Y.far_ptr[e] = far_->wait_until(l, e, deadline);
    far_st_.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    far_st_.waited_ok += Y.far_ptr[e] != nullptr;
    return Y.far_ptr[e] != nullptr;
}

const uint8_t * ExpertStore::far_src(int l, int e)
{
    Layer & Y = layers_[l];
    const uint8_t * p = Y.far_ptr[e];
    if (!far_) throw std::logic_error("ExpertStore: far expert " + std::to_string(e) + " of layer " + std::to_string(l) +
                                      " routed with the far set dropped (its router bias is not masked on this path)");
    if (!p) {   // a path that did not ask far_ready() first (fetch path, verify window): wait for the read
        // in 1 ms steps, releasing landed copies' slots meanwhile: with every slot held the reader needs one of them
        const auto t0 = std::chrono::steady_clock::now();
        for (bool first = true; !(p = far_->wait(l, e, 1, first)); first = false) {
            if (far_->failed(l, e))
                throw std::runtime_error("ExpertStore: far expert " + std::to_string(e) + " of layer " +
                                         std::to_string(l) + ": NVMe read failed");
            far_poll_release();
        }
        ++far_st_.fetch_waits;
        far_st_.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    Y.far_ptr[e] = nullptr;   // the copy now owns the hold; far_copied() ends it
    return p;
}

void ExpertStore::far_copied(int l, int e)
{
    if (far_ev_pool_.empty()) far_poll_release();
    if (far_ev_pool_.empty()) {   // every event in flight: the newest hold takes this expert, its event moves here
        far_held_.back().le.push_back({ l, e });
        TRUSS_CUDA(cudaEventRecord(far_held_.back().ev, copy_));
        return;
    }
    const cudaEvent_t ev = far_ev_pool_.back();
    far_ev_pool_.pop_back();
    TRUSS_CUDA(cudaEventRecord(ev, copy_));
    far_held_.push_back({ ev, { { l, e } } });
}

void ExpertStore::far_poll_release()
{
    while (!far_held_.empty() && cudaEventQuery(far_held_.front().ev) == cudaSuccess) {
        for (const auto & [l, e] : far_held_.front().le) far_->release(l, e);
        far_ev_pool_.push_back(far_held_.front().ev);
        far_held_.pop_front();
    }
}

std::string ExpertStore::far_report() const
{
    if (!far_) return "";
    const FarStats & f = far_st_;
    char b[256];
    std::snprintf(b, sizeof b, "far experts: routed %ld, in ring %ld, slot ready %ld, waited in time %ld, skipped %ld, "
                               "fetch waits %ld, wait %.1f ms; ", f.routed, f.ring_hits, f.ready, f.waited_ok, f.skipped,
                  f.fetch_waits, f.wait_ms);
    return b + far_->report();
}

}  // namespace truss::runtime
