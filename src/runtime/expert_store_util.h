// ExpertStore internals shared by its implementation files (expert_store*.cu): sizes, offsets, page release.
#pragma once
#include "core/cuda_check.h"
#include "formats/gguf.h"
#include "runtime/expert_store.h"

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace truss::runtime::store_detail {


// MADV_DONTNEED over the whole pages of a mapped range (file pages only: read-only and shared, a later read re-faults)
inline void release_range(const void * p, size_t bytes)
{
    const long ps = sysconf(_SC_PAGESIZE);
    if (!p || !bytes || ps <= 0) return;
    const uintptr_t a = (uintptr_t) p, m = (uintptr_t) ps - 1;
    const uintptr_t lo = (a + m) & ~m, hi = (a + (uintptr_t) bytes) & ~m;   // pages wholly inside the range
    if (hi > lo) madvise((void *) lo, (size_t) (hi - lo), MADV_DONTNEED);
}

inline void release_pages(const gguf::Tensor * t)
{
    if (t) release_range(t->data, t->bytes);
}

constexpr int SHIFT = 4;                 // meta offsets in 16-word (32-byte) units
constexpr size_t UNIT = 2u << SHIFT;     // bytes per offset unit
constexpr size_t ALIGN = 256;            // ring allocations and the slots

inline size_t align_up(size_t x, size_t a = ALIGN) { return (x + a - 1) / a * a; }

// trellis projections are 16*K words per 16x16 tile, so a whole projection is a multiple of 32 bytes
inline size_t proj_bytes(const formats::ExpertTable & t, int e) { return (size_t) t.words(e) * 2; }

inline size_t expert_bytes(const ExpertLayer & L, int e)
{
    return proj_bytes(*L[0], e) + proj_bytes(*L[1], e) + proj_bytes(*L[2], e);
}

inline size_t fixed_bytes(const ExpertLayer & L)   // scales + stream and ring meta tables
{
    size_t b = 0;
    for (int p = 0; p < 3; ++p) {
        const formats::ExpertTable & t = *L[p];
        b += (size_t) t.n_expert * (t.in + t.out) * 2 + (size_t) t.n_expert * 2 * sizeof(int32_t) * 2;
    }
    return b;
}

inline void * device_alloc(size_t bytes, std::vector<void *> & owned)
{
    void * p = nullptr;
    if (bytes) {
        TRUSS_CUDA(cudaMalloc(&p, bytes));
        owned.push_back(p);
    }
    return p;
}

inline int32_t unit_offset(const uint8_t * at, const uint8_t * base, int layer)
{
    const ptrdiff_t d = at - base;
    if (d % (ptrdiff_t) UNIT) throw std::logic_error("ExpertStore: expert not aligned to its offset unit");
    const ptrdiff_t u = d / (ptrdiff_t) UNIT;
    if (u < INT_MIN || u > INT_MAX)
        throw std::runtime_error("ExpertStore: layer " + std::to_string(layer) + " expert offset exceeds int32 units");
    return (int32_t) u;
}

// The kept cold copy's ids (cold_keep.h): slot = where every layer's projections come from (file identity + offset,
// expert count), key = slot + the cold expert set + the copy's size. False if a projection is in no open mapping.
inline bool cold_keep_ids(const std::vector<ExpertLayer> & layers, const ExpertStore::HotSet & hot,
                          const ExpertStore::HotSet & far, size_t bytes, uint64_t & slot, uint64_t & key)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) h = (h ^ ((v >> (8 * i)) & 0xff)) * 1099511628211ull;
    };
    for (size_t l = 0; l < layers.size(); ++l)
        for (int p = 0; p < 3; ++p) {
            uint64_t file = 0, off = 0;
            if (!gguf::span_ident(layers[l][p]->trellis->data, file, off)) return false;
            mix(file), mix(off), mix((uint64_t) layers[l][p]->n_expert);
        }
    slot = h;
    mix(bytes);
    for (size_t l = 0; l < hot.size(); ++l)
        for (size_t e = 0; e < hot[l].size(); ++e)
            if (!hot[l][e] && (far.empty() || !far[l][e])) mix(l << 32 | e);   // the host copy's experts
    key = h;
    return true;
}

}  // namespace truss::runtime::store_detail
