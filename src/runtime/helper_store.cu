#include "runtime/helper_store.h"

#include "core/cuda_check.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace truss::runtime {
namespace {

constexpr int SHIFT = 4;                 // meta offsets in 16-word (32-byte) units, as ExpertStore's
constexpr size_t UNIT = 2u << SHIFT;

size_t proj_bytes(const formats::ExpertTable & t, int e) { return (size_t) t.words(e) * 2; }

size_t expert_bytes(const ExpertLayer & L, int e)
{
    return proj_bytes(*L[0], e) + proj_bytes(*L[1], e) + proj_bytes(*L[2], e);
}

void * dev(size_t bytes, std::vector<void *> & owned)
{
    void * p = nullptr;
    if (bytes) {
        TRUSS_CUDA(cudaMalloc(&p, bytes));
        owned.push_back(p);
    }
    return p;
}

}  // namespace

size_t HelperStore::fixed_bytes(const std::vector<ExpertLayer> & layers)
{
    size_t b = 0;
    for (const auto & L : layers)
        for (int p = 0; p < 3; ++p) {
            const formats::ExpertTable & t = *L[p];
            b += (size_t) t.n_expert * (t.in + t.out) * 2 + (size_t) t.n_expert * 2 * sizeof(int32_t);
        }
    return b;
}

HelperStore::HelperStore(std::vector<ExpertLayer> layers, const std::vector<Item> & items)
    : layers_(std::move(layers))
{
    const int L = (int) layers_.size();
    const int E = L ? layers_[0][0]->n_expert : 0;
    if (L == 0) throw std::invalid_argument("HelperStore: no layers");
    set_.assign(L, std::vector<uint8_t>(E, 0));
    stored_.assign(L, 0);
    w_.assign((size_t) L, moe::Weights{});
    for (auto & W : w_) W.n_expert = E;

    // The arena holds each stored expert's three projections, contiguous in fill order. A projection is a whole
    // number of 16-word tiles, so every expert starts on a 32-byte boundary and its offset fits the meta unit.
    for (const auto & it : items)
        if (it.first >= 0 && it.first < L && it.second >= 0 && it.second < E && !set_[it.first][it.second])
            trellis_ += expert_bytes(layers_[it.first], it.second);
    if (trellis_) TRUSS_CUDA(cudaMalloc(reinterpret_cast<void **>(&arena_), trellis_));
    device_.push_back(arena_);

    std::vector<int32_t> meta((size_t) L * 3 * 2 * E, 0);      // [layer][proj][expert][K, offset]
    uint8_t * at = arena_;
    {
        // One pinned staging buffer, filled from the shard with pread (the model's file pages were released after
        // upload; read_mapped re-reads them without faulting the mapping).
        constexpr size_t STAGE = 64u << 20;
        uint8_t * stage = nullptr;
        TRUSS_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&stage), STAGE, cudaHostAllocDefault));
        cudaStream_t st = nullptr;
        TRUSS_CUDA(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        uint8_t * dst_run = arena_;
        size_t fill = 0;
        auto flush = [&] {
            if (!fill) return;
            TRUSS_CUDA(cudaMemcpyAsync(dst_run, stage, fill, cudaMemcpyHostToDevice, st));
            TRUSS_CUDA(cudaStreamSynchronize(st));
            fill = 0;
        };
        for (const auto & it : items) {
            const int l = it.first, e = it.second;
            if (l < 0 || l >= L || e < 0 || e >= E || set_[l][e]) continue;
            const size_t b = expert_bytes(layers_[l], e);
            if (fill && fill + b > STAGE) { flush(); dst_run = at; }
            size_t part = 0;
            for (int p = 0; p < 3; ++p) {
                const formats::ExpertTable & t = *layers_[l][p];
                const uint8_t * from = reinterpret_cast<const uint8_t *>(t.trellis->data) + t.offset[e] * 2;
                uint8_t * to = stage + fill + part;
                if (!gguf::read_mapped(from, proj_bytes(t, e), to)) std::memcpy(to, from, proj_bytes(t, e));
                const ptrdiff_t d = (at + part) - arena_;
                if (d % (ptrdiff_t) UNIT) throw std::logic_error("HelperStore: expert not aligned to its offset unit");
                meta[(((size_t) l * 3 + p) * 2 * E) + 2 * e] = t.k[e];
                meta[(((size_t) l * 3 + p) * 2 * E) + 2 * e + 1] = (int32_t) (d / (ptrdiff_t) UNIT);
                part += proj_bytes(t, e);
            }
            set_[l][e] = 1;
            ++stored_[l];
            ++count_;
            at += b;
            fill += b;
        }
        flush();
        cudaStreamDestroy(st);
        cudaFreeHost(stage);
    }
    if ((size_t) (at - arena_) != trellis_) throw std::logic_error("HelperStore: the arena and its items disagree");

    for (int l = 0; l < L; ++l)
        for (int p = 0; p < 3; ++p) {
            const formats::ExpertTable & t = *layers_[l][p];
            auto * dm = static_cast<int32_t *>(dev(sizeof(int32_t) * 2 * E, device_));
            TRUSS_CUDA(cudaMemcpy(dm, meta.data() + ((size_t) l * 3 + p) * 2 * E, sizeof(int32_t) * 2 * E,
                                  cudaMemcpyHostToDevice));
            auto * suh = static_cast<half *>(dev(t.suh->bytes, device_));
            auto * svh = static_cast<half *>(dev(t.svh->bytes, device_));
            TRUSS_CUDA(cudaMemcpy(suh, t.suh->data, t.suh->bytes, cudaMemcpyHostToDevice));
            TRUSS_CUDA(cudaMemcpy(svh, t.svh->data, t.svh->bytes, cudaMemcpyHostToDevice));
            fixed_ += 2 * sizeof(int32_t) * E + t.suh->bytes + t.svh->bytes;
            w_[l].proj[p] = { reinterpret_cast<const uint16_t *>(arena_), dm, suh, svh, SHIFT };
        }
}

HelperStore::~HelperStore()
{
    for (void * p : device_)
        if (p) cudaFree(p);
}

}  // namespace truss::runtime
