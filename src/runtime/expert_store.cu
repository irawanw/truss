#include "runtime/expert_store.h"

#include "core/cuda_check.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <string>

namespace truss::runtime {
namespace {

size_t expert_bytes(const formats::ExpertTable & t, int e) { return (size_t) t.words(e) * 2; }

size_t scale_bytes(const formats::ExpertTable & t) { return (size_t) t.n_expert * (t.in + t.out) * 2; }

size_t meta_bytes(const formats::ExpertTable & t) { return (size_t) t.n_expert * 2 * sizeof(int32_t); }

// device bytes of "the first n_hot experts of every layer resident"
size_t plan_bytes(const std::vector<ExpertLayer> & layers, int n_hot)
{
    size_t total = 0, slot[3] = {};
    for (const ExpertLayer & L : layers)
        for (int p = 0; p < 3; ++p) {
            const formats::ExpertTable & t = *L[p];
            size_t cold = 0;
            for (int e = 0; e < t.n_expert; ++e) (e < n_hot ? total : cold) += expert_bytes(t, e);
            slot[p] = std::max(slot[p], cold);
            total += scale_bytes(t) + meta_bytes(t);
        }
    return total + 2 * (slot[0] + slot[1] + slot[2]);
}

void * device_alloc(size_t bytes, std::vector<void *> & owned)
{
    void * p = nullptr;
    if (bytes) {
        TRUSS_CUDA(cudaMalloc(&p, bytes));
        owned.push_back(p);
    }
    return p;
}

}  // namespace

ExpertStore::HotSet ExpertStore::plan(const std::vector<ExpertLayer> & layers, size_t budget)
{
    const int n_expert = layers.empty() ? 0 : layers[0][0]->n_expert;
    int n_hot = n_expert;
    while (n_hot > 0 && plan_bytes(layers, n_hot) > budget) --n_hot;
    if (plan_bytes(layers, n_hot) > budget)
        throw std::runtime_error("ExpertStore::plan: " + std::to_string(budget >> 20) + " MiB cannot hold even the slots (" +
                                 std::to_string(plan_bytes(layers, 0) >> 20) + " MiB)");
    HotSet hot(layers.size(), std::vector<uint8_t>(n_expert, 0));
    for (auto & h : hot) std::fill(h.begin(), h.begin() + n_hot, 1);
    return hot;
}

ExpertStore::ExpertStore(const std::vector<ExpertLayer> & layers, const HotSet & hot)
{
    const int L = (int) layers.size(), half_l = (L + 1) / 2;
    if ((int) hot.size() != L) throw std::invalid_argument("ExpertStore: hot set has the wrong layer count");
    layers_.resize(L);
    TRUSS_CUDA(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
    for (int s = 0; s < 2; ++s) {
        TRUSS_CUDA(cudaEventCreateWithFlags(&copied_[s], cudaEventDisableTiming));
        TRUSS_CUDA(cudaEventCreateWithFlags(&released_[s], cudaEventDisableTiming));
    }

    for (int p = 0; p < 3; ++p) {
        // arena: [hot, layers < half_l][slot 0][slot 1][hot, layers >= half_l]
        size_t hot_lo = 0, hot_hi = 0, slot = 0;
        for (int l = 0; l < L; ++l) {
            const formats::ExpertTable & t = *layers[l][p];
            size_t cold = 0;
            for (int e = 0; e < t.n_expert; ++e) (hot[l][e] ? (l < half_l ? hot_lo : hot_hi) : cold) += expert_bytes(t, e);
            slot = std::max(slot, cold);
            layers_[l].cold[p] = cold;
        }
        auto * arena = static_cast<uint8_t *>(device_alloc(hot_lo + 2 * slot + hot_hi, device_));
        base_[p] = reinterpret_cast<uint16_t *>(arena + hot_lo);
        slot_words_[p] = slot / 2;
        device_bytes_ += hot_lo + 2 * slot + hot_hi;

        uint8_t * next_hot[2] = { arena, arena + hot_lo + 2 * slot };
        for (int l = 0; l < L; ++l) {
            const formats::ExpertTable & t = *layers[l][p];
            Layer & Y = layers_[l];
            Y.n_expert = t.n_expert;
            const auto * src = reinterpret_cast<const uint8_t *>(t.trellis->data);
            std::vector<int32_t> meta(2 * t.n_expert);
            uint8_t * cold_dst = reinterpret_cast<uint8_t *>(base_[p]) + (l % 2) * slot;
            uint8_t * pinned = nullptr;
            if (Y.cold[p]) {
                TRUSS_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&pinned), Y.cold[p], cudaHostAllocDefault));
                pinned_.push_back(pinned);
                Y.host[p] = pinned;
                cold_total_ += Y.cold[p];
            }
            size_t cold_pos = 0;
            Y.cold_off[p].assign(t.n_expert, -1);
            Y.bytes[p].resize(t.n_expert);
            for (int e = 0; e < t.n_expert; ++e) {
                const size_t bytes = expert_bytes(t, e);
                Y.bytes[p][e] = bytes;
                if (!hot[l][e]) Y.cold_off[p][e] = (int64_t) cold_pos;
                const uint8_t * from = src + t.offset[e] * 2;
                uint8_t * at;
                if (hot[l][e]) {
                    uint8_t *& h = next_hot[l < half_l ? 0 : 1];
                    at = h;
                    h += bytes;
                    TRUSS_CUDA(cudaMemcpy(at, from, bytes, cudaMemcpyHostToDevice));
                } else {
                    at = cold_dst + cold_pos;
                    std::memcpy(pinned + cold_pos, from, bytes);
                    cold_pos += bytes;
                }
                const ptrdiff_t words = (at - reinterpret_cast<uint8_t *>(base_[p])) / 2;
                if (words < INT_MIN || words > INT_MAX)
                    throw std::runtime_error("ExpertStore: layer " + std::to_string(l) + " expert offset exceeds int32 words");
                meta[2 * e] = t.k[e];
                meta[2 * e + 1] = (int32_t) words;
            }
            Y.meta[p] = static_cast<int32_t *>(device_alloc(meta.size() * 4, device_));
            TRUSS_CUDA(cudaMemcpy(Y.meta[p], meta.data(), meta.size() * 4, cudaMemcpyHostToDevice));
            auto * suh = static_cast<half *>(device_alloc(t.suh->bytes, device_));
            auto * svh = static_cast<half *>(device_alloc(t.svh->bytes, device_));
            TRUSS_CUDA(cudaMemcpy(suh, t.suh->data, t.suh->bytes, cudaMemcpyHostToDevice));
            TRUSS_CUDA(cudaMemcpy(svh, t.svh->data, t.svh->bytes, cudaMemcpyHostToDevice));
            Y.suh[p] = suh, Y.svh[p] = svh;
            device_bytes_ += meta.size() * 4 + t.suh->bytes + t.svh->bytes;
        }
    }
}

ExpertStore::~ExpertStore()
{
    if (copy_) cudaStreamSynchronize(copy_);
    for (void * p : device_) cudaFree(p);
    for (void * p : pinned_) cudaFreeHost(p);
    for (int s = 0; s < 2; ++s) {
        if (copied_[s]) cudaEventDestroy(copied_[s]);
        if (released_[s]) cudaEventDestroy(released_[s]);
    }
    if (copy_) cudaStreamDestroy(copy_);
}

moe::Weights ExpertStore::weights(int l) const
{
    const Layer & Y = layers_.at(l);
    moe::Weights w{};
    for (int p = 0; p < 3; ++p) w.proj[p] = { base_[p], Y.meta[p], Y.suh[p], Y.svh[p] };
    w.n_expert = Y.n_expert;
    return w;
}

void ExpertStore::prefetch(int l)
{
    const Layer & Y = layers_.at(l);
    const int s = l % 2;
    if (released_recorded_[s]) TRUSS_CUDA(cudaStreamWaitEvent(copy_, released_[s], 0));
    for (int p = 0; p < 3; ++p)
        if (Y.cold[p])
            TRUSS_CUDA(cudaMemcpyAsync(base_[p] + s * slot_words_[p], Y.host[p], Y.cold[p], cudaMemcpyHostToDevice, copy_));
    TRUSS_CUDA(cudaEventRecord(copied_[s], copy_));
}

void ExpertStore::fetch(int l, const int * ids, int n)
{
    const Layer & Y = layers_.at(l);
    const int s = l % 2;
    if (released_recorded_[s]) TRUSS_CUDA(cudaStreamWaitEvent(copy_, released_[s], 0));
    std::vector<uint8_t> seen(Y.n_expert, 0);
    for (int i = 0; i < n; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= Y.n_expert) throw std::out_of_range("ExpertStore::fetch: expert id " + std::to_string(e));
        if (seen[e] || Y.cold_off[0][e] < 0) continue;
        seen[e] = 1;
        for (int p = 0; p < 3; ++p) {
            const int64_t off = Y.cold_off[p][e];
            TRUSS_CUDA(cudaMemcpyAsync(reinterpret_cast<uint8_t *>(base_[p] + s * slot_words_[p]) + off,
                                       static_cast<const uint8_t *>(Y.host[p]) + off, Y.bytes[p][e],
                                       cudaMemcpyHostToDevice, copy_));
        }
    }
    TRUSS_CUDA(cudaEventRecord(copied_[s], copy_));
}

void ExpertStore::acquire(int l, cudaStream_t compute)
{
    TRUSS_CUDA(cudaStreamWaitEvent(compute, copied_[l % 2], 0));
}

void ExpertStore::release(int l, cudaStream_t compute)
{
    TRUSS_CUDA(cudaEventRecord(released_[l % 2], compute));
    released_recorded_[l % 2] = true;
}

}  // namespace truss::runtime
