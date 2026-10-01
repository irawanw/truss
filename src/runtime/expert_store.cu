#include "runtime/expert_store.h"

#include "core/cuda_check.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <stdexcept>
#include <string>

namespace truss::runtime {
namespace {

// MADV_DONTNEED over the whole pages of a mapped tensor (file pages only: the tensor is read-only and shared)
void release_pages(const gguf::Tensor * t)
{
    const long ps = sysconf(_SC_PAGESIZE);
    if (!t || !t->data || !t->bytes || ps <= 0) return;
    const uintptr_t a = (uintptr_t) t->data, m = (uintptr_t) ps - 1;
    const uintptr_t lo = (a + m) & ~m, hi = (a + (uintptr_t) t->bytes) & ~m;   // pages wholly inside the tensor
    if (hi > lo) madvise((void *) lo, (size_t) (hi - lo), MADV_DONTNEED);
}

constexpr int SHIFT = 4;                 // meta offsets in 16-word (32-byte) units
constexpr size_t UNIT = 2u << SHIFT;     // bytes per offset unit
constexpr size_t ALIGN = 256;            // ring allocations and the slots

size_t align_up(size_t x, size_t a = ALIGN) { return (x + a - 1) / a * a; }

// trellis projections are 16*K words per 16x16 tile, so a whole projection is a multiple of 32 bytes
size_t proj_bytes(const formats::ExpertTable & t, int e) { return (size_t) t.words(e) * 2; }

size_t expert_bytes(const ExpertLayer & L, int e)
{
    return proj_bytes(*L[0], e) + proj_bytes(*L[1], e) + proj_bytes(*L[2], e);
}

size_t fixed_bytes(const ExpertLayer & L)   // scales + stream and ring meta tables
{
    size_t b = 0;
    for (int p = 0; p < 3; ++p) {
        const formats::ExpertTable & t = *L[p];
        b += (size_t) t.n_expert * (t.in + t.out) * 2 + (size_t) t.n_expert * 2 * sizeof(int32_t) * 2;
    }
    return b;
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

int32_t unit_offset(const uint8_t * at, const uint8_t * base, int layer)
{
    const ptrdiff_t d = at - base;
    if (d % (ptrdiff_t) UNIT) throw std::logic_error("ExpertStore: expert not aligned to its offset unit");
    const ptrdiff_t u = d / (ptrdiff_t) UNIT;
    if (u < INT_MIN || u > INT_MAX)
        throw std::runtime_error("ExpertStore: layer " + std::to_string(layer) + " expert offset exceeds int32 units");
    return (int32_t) u;
}

}  // namespace

size_t ExpertStore::ring_size(const std::vector<ExpertLayer> & layers, const HotSet & hot, const Sizes & z)
{
    size_t slot = 0;
    for (size_t l = 0; l < layers.size(); ++l) {
        size_t cold = 0;
        for (int e = 0; e < layers[l][0]->n_expert; ++e)
            if (!hot[l][e]) cold += expert_bytes(layers[l], e);
        slot = std::max(slot, align_up(cold));
    }
    return align_up(std::max(z.ring_bytes, 2 * slot + z.stream_extra));
}

size_t ExpertStore::device_bytes(const std::vector<ExpertLayer> & layers, const HotSet & hot, const Sizes & z)
{
    size_t total = ring_size(layers, hot, z);
    for (size_t l = 0; l < layers.size(); ++l) {
        total += fixed_bytes(layers[l]);
        for (int e = 0; e < layers[l][0]->n_expert; ++e)
            if (hot[l][e]) total += expert_bytes(layers[l], e);
    }
    return total;
}

ExpertStore::HotSet ExpertStore::plan(const std::vector<ExpertLayer> & layers, size_t budget, const Sizes & z,
                                      const std::vector<float> & usage)
{
    const int L = (int) layers.size(), n_expert = layers.empty() ? 0 : layers[0][0]->n_expert;
    HotSet hot(L, std::vector<uint8_t>(n_expert, 0));
    if (device_bytes(layers, hot, z) > budget)
        throw std::runtime_error("ExpertStore::plan: " + std::to_string(budget >> 20) + " MiB cannot hold even the ring (" +
                                 std::to_string(device_bytes(layers, hot, z) >> 20) + " MiB)");
    if (usage.empty()) {   // the same count in every layer, index order
        for (int n = 0; n < n_expert; ++n) {
            for (auto & h : hot) h[n] = 1;
            if (device_bytes(layers, hot, z) > budget) {
                for (auto & h : hot) h[n] = 0;
                break;
            }
        }
        return hot;
    }
    if (usage.size() != (size_t) L * n_expert)
        throw std::invalid_argument("ExpertStore::plan: usage has " + std::to_string(usage.size()) + " values, expected " +
                                    std::to_string((size_t) L * n_expert));
    size_t fixed = 0;
    std::vector<size_t> size(L * (size_t) n_expert), cold_all(L);
    for (int l = 0; l < L; ++l) {
        fixed += fixed_bytes(layers[l]);
        for (int e = 0; e < n_expert; ++e) {
            size[(size_t) l * n_expert + e] = expert_bytes(layers[l], e);
            cold_all[l] += size[(size_t) l * n_expert + e];
        }
    }
    auto better = [&](int a, int b) { return (double) usage[a] / size[a] > (double) usage[b] / size[b]; };
    std::vector<int> order(size.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int) i;
    std::stable_sort(order.begin(), order.end(), better);
    std::vector<std::vector<int>> by_layer(L);
    for (int i : order) by_layer[i / n_expert].push_back(i);

    auto fill = [&](int floor, HotSet & h) -> double {   // the hot usage, or -1 if the floor does not fit
        std::vector<size_t> cold = cold_all;
        for (auto & x : h) std::fill(x.begin(), x.end(), 0);
        auto total = [&](size_t hot_bytes) {
            size_t slot = 0;
            for (int l = 0; l < L; ++l) slot = std::max(slot, align_up(cold[l]));
            return fixed + hot_bytes + align_up(std::max(z.ring_bytes, 2 * slot + z.stream_extra));
        };
        size_t hot_bytes = 0;
        double used = 0;
        auto take = [&](int i) {
            cold[i / n_expert] -= size[i];
            hot_bytes += size[i];
            h[i / n_expert][i % n_expert] = 1;
            used += usage[i];
        };
        for (int l = 0; l < L; ++l)
            for (int j = 0; j < floor; ++j) take(by_layer[l][j]);
        if (total(hot_bytes) > budget) return -1;
        for (int i : order) {
            const int l = i / n_expert;
            if (h[l][i % n_expert]) continue;
            cold[l] -= size[i];
            const bool fits = total(hot_bytes + size[i]) <= budget;
            cold[l] += size[i];
            if (fits) take(i);
        }
        return used;
    };
    double best = -1;
    HotSet trial = hot;
    for (int floor = 0; floor <= n_expert; floor += 8) {
        const double used = fill(floor, trial);
        if (used < 0) break;
        if (used > best) best = used, hot = trial;
    }
    return hot;
}

std::vector<float> ExpertStore::load_usage(const std::string & path, int n_layer, int n_expert)
{
    std::vector<float> u((size_t) n_layer * n_expert);
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("ExpertStore::load_usage: cannot open " + path);
    const size_t got = std::fread(u.data(), sizeof(float), u.size(), f);
    const bool extra = std::fgetc(f) != EOF;
    std::fclose(f);
    if (got != u.size() || extra)
        throw std::runtime_error("ExpertStore::load_usage: " + path + " is not " + std::to_string(n_layer) + " x " +
                                 std::to_string(n_expert) + " float32");
    return u;
}

ExpertStore::ExpertStore(const std::vector<ExpertLayer> & layers, const HotSet & hot, const Sizes & z)
{
    const int L = (int) layers.size(), half_l = (L + 1) / 2;
    if ((int) hot.size() != L) throw std::invalid_argument("ExpertStore: hot set has the wrong layer count");
    layers_.resize(L);
    TRUSS_CUDA(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
    for (int s = 0; s < 2; ++s) {
        TRUSS_CUDA(cudaEventCreateWithFlags(&copied_[s], cudaEventDisableTiming));
        TRUSS_CUDA(cudaEventCreateWithFlags(&released_[s], cudaEventDisableTiming));
    }
    TRUSS_CUDA(cudaEventCreateWithFlags(&compute_mark_, cudaEventDisableTiming));
    TRUSS_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&signal_src_), sizeof(int) * SIGNAL_RING, cudaHostAllocDefault));
    pinned_.push_back(signal_src_);

    size_t hot_lo = 0, hot_hi = 0;
    for (int l = 0; l < L; ++l) {
        Layer & Y = layers_[l];
        const int E = layers[l][0]->n_expert;
        Y.n_expert = E;
        Y.bytes.resize(E);
        for (int p = 0; p < 3; ++p) Y.part[p].resize(E);
        for (int e = 0; e < E; ++e) {
            Y.part[0][e] = 0;
            Y.part[1][e] = proj_bytes(*layers[l][0], e);
            Y.part[2][e] = Y.part[1][e] + proj_bytes(*layers[l][1], e);
            Y.bytes[e] = expert_bytes(layers[l], e);
            (hot[l][e] ? (l < half_l ? hot_lo : hot_hi) : Y.cold) += Y.bytes[e];
        }
    }
    ring_ = ring_size(layers, hot, z);
    for (const Layer & Y : layers_) slot_ = std::max(slot_, align_up(Y.cold));
    auto * arena = static_cast<uint8_t *>(device_alloc(hot_lo + ring_ + hot_hi, device_));
    base_ = arena + hot_lo;
    spare_ = z.stream_extra ? base_ + 2 * slot_ : nullptr;
    device_bytes_ = hot_lo + ring_ + hot_hi;

    uint8_t * next_hot[2] = { arena, arena + hot_lo + ring_ };
    for (int l = 0; l < L; ++l) {
        Layer & Y = layers_[l];
        const int E = Y.n_expert;
        if (Y.cold) {
            uint8_t * pinned;
            TRUSS_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&pinned), Y.cold, cudaHostAllocDefault));
            pinned_.push_back(pinned);
            Y.host = pinned;
            cold_total_ += Y.cold;
        }
        TRUSS_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&Y.ring_meta_host), sizeof(int32_t) * 3 * 2 * E,
                                 cudaHostAllocDefault));
        pinned_.push_back(Y.ring_meta_host);
        std::vector<int32_t> smeta(3 * 2 * E);
        Y.cold_off.assign(E, -1);
        Y.ring_at.assign(E, -1);
        uint8_t * slot_dst = base_ + (l % 2) * slot_;
        size_t cold_pos = 0;
        for (int e = 0; e < E; ++e) {
            uint8_t * at;
            if (hot[l][e]) {
                uint8_t *& h = next_hot[l < half_l ? 0 : 1];
                at = h;
                h += Y.bytes[e];
            } else {
                Y.cold_off[e] = (int64_t) cold_pos;
                at = slot_dst + cold_pos;
                cold_pos += Y.bytes[e];
            }
            for (int p = 0; p < 3; ++p) {
                const formats::ExpertTable & t = *layers[l][p];
                const auto * from = reinterpret_cast<const uint8_t *>(t.trellis->data) + t.offset[e] * 2;
                const size_t pb = proj_bytes(t, e);
                if (hot[l][e]) TRUSS_CUDA(cudaMemcpy(at + Y.part[p][e], from, pb, cudaMemcpyHostToDevice));
                else std::memcpy(const_cast<uint8_t *>(Y.host) + Y.cold_off[e] + Y.part[p][e], from, pb);
                int32_t * sm = smeta.data() + (size_t) p * 2 * E, * rm = Y.ring_meta_host + (size_t) p * 2 * E;
                sm[2 * e] = rm[2 * e] = t.k[e];
                sm[2 * e + 1] = unit_offset(at + Y.part[p][e], base_, l);
                rm[2 * e + 1] = hot[l][e] ? sm[2 * e + 1] : 0;   // cold: set when fetched
            }
        }
        // The layer's tiles now live in the pinned copy or on the device: drop the file pages they were read from.
        // The mapping stays valid (read-only, shared; a later read re-faults), but without this the load's resident
        // peak counts the pack twice (pinned + page cache, ~69 GB measured) beside renters holding 58-77 GB.
        for (int p = 0; p < 3; ++p) release_pages(layers[l][p]->trellis);
        for (int p = 0; p < 3; ++p) {
            const formats::ExpertTable & t = *layers[l][p];
            Y.stream_meta[p] = static_cast<int32_t *>(device_alloc(sizeof(int32_t) * 2 * E, device_));
            Y.ring_meta[p] = static_cast<int32_t *>(device_alloc(sizeof(int32_t) * 2 * E, device_));
            TRUSS_CUDA(cudaMemcpy(Y.stream_meta[p], smeta.data() + (size_t) p * 2 * E, sizeof(int32_t) * 2 * E,
                                  cudaMemcpyHostToDevice));
            TRUSS_CUDA(cudaMemcpy(Y.ring_meta[p], Y.ring_meta_host + (size_t) p * 2 * E, sizeof(int32_t) * 2 * E,
                                  cudaMemcpyHostToDevice));
            auto * suh = static_cast<half *>(device_alloc(t.suh->bytes, device_));
            auto * svh = static_cast<half *>(device_alloc(t.svh->bytes, device_));
            TRUSS_CUDA(cudaMemcpy(suh, t.suh->data, t.suh->bytes, cudaMemcpyHostToDevice));
            TRUSS_CUDA(cudaMemcpy(svh, t.svh->data, t.svh->bytes, cudaMemcpyHostToDevice));
            Y.suh[p] = suh, Y.svh[p] = svh;
            device_bytes_ += 2 * sizeof(int32_t) * 2 * E + t.suh->bytes + t.svh->bytes;
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
    if (compute_mark_) cudaEventDestroy(compute_mark_);
    if (copy_) cudaStreamDestroy(copy_);
}

moe::Weights ExpertStore::weights(int l) const
{
    const Layer & Y = layers_.at(l);
    moe::Weights w{};
    for (int p = 0; p < 3; ++p) {
        w.proj[p] = { reinterpret_cast<const uint16_t *>(base_), mode_ == Mode::STREAM ? Y.stream_meta[p] : Y.ring_meta[p],
                      Y.suh[p], Y.svh[p] };
        w.proj[p].shift = SHIFT;
    }
    w.n_expert = Y.n_expert;
    return w;
}

void ExpertStore::wait_compute(cudaStream_t compute)
{
    TRUSS_CUDA(cudaEventRecord(compute_mark_, compute));
    TRUSS_CUDA(cudaStreamWaitEvent(copy_, compute_mark_, 0));
}

void ExpertStore::begin_stream(cudaStream_t compute)
{
    wait_compute(compute);   // decode kernels may still read the ring the slots overwrite
    for (const RingEntry & r : fifo_) layers_[r.layer].ring_at[r.expert] = -1;
    fifo_.clear();
    head_ = 0;
    released_recorded_[0] = released_recorded_[1] = false;
    mode_ = Mode::STREAM;
}

void ExpertStore::prefetch(int l)
{
    if (mode_ != Mode::STREAM) throw std::logic_error("ExpertStore::prefetch outside a begin_stream() chunk");
    const Layer & Y = layers_.at(l);
    const int s = l % 2;
    if (released_recorded_[s]) TRUSS_CUDA(cudaStreamWaitEvent(copy_, released_[s], 0));
    if (Y.cold) TRUSS_CUDA(cudaMemcpyAsync(base_ + s * slot_, Y.host, Y.cold, cudaMemcpyHostToDevice, copy_));
    TRUSS_CUDA(cudaEventRecord(copied_[s], copy_));
}

// FIFO allocation: evict the oldest entries overlapping [head, head + bytes) (wrapping to 0 when the tail is too
// short), copy the expert, point the ring meta at it.
bool ExpertStore::ring_put(int l, int e, bool hint)
{
    Layer & Y = layers_[l];
    const int64_t need = (int64_t) align_up(Y.bytes[e]);
    if (hint) {   // would this allocation evict an expert the current layer's kernel reads?
        const int64_t start = head_ + need > (int64_t) ring_ ? 0 : head_;
        for (const RingEntry & r : fifo_) {
            const bool gone = (start == 0 && r.off >= head_) || (r.off >= start && r.off < start + need);
            if (!gone) {
                if (r.off >= start + need) break;   // FIFO order: later entries lie further on
                continue;
            }
            if (r.layer == protect_layer_ && std::find(protect_.begin(), protect_.end(), r.expert) != protect_.end())
                return false;
        }
    }
    if (head_ + need > (int64_t) ring_) {   // wrap: the tail [head, end) goes with its entries
        while (!fifo_.empty() && fifo_.front().off >= head_) {
            layers_[fifo_.front().layer].ring_at[fifo_.front().expert] = -1;
            fifo_.pop_front();
        }
        head_ = 0;
    }
    while (!fifo_.empty() && fifo_.front().off >= head_ && fifo_.front().off < head_ + need) {
        layers_[fifo_.front().layer].ring_at[fifo_.front().expert] = -1;
        fifo_.pop_front();
    }
    const int64_t off = head_;
    head_ += need;
    fifo_.push_back({ l, e, off });
    Y.ring_at[e] = off;
    uint8_t * at = base_ + off;
    TRUSS_CUDA(cudaMemcpyAsync(at, Y.host + Y.cold_off[e], Y.bytes[e], cudaMemcpyHostToDevice, copy_));
    for (int p = 0; p < 3; ++p) Y.ring_meta_host[(size_t) p * 2 * Y.n_expert + 2 * e + 1] = unit_offset(at + Y.part[p][e], base_, l);
    if (hint) stats_.hint_bytes += Y.bytes[e], ++stats_.hinted;
    else stats_.bytes += Y.bytes[e], ++stats_.misses;
    return true;
}

void ExpertStore::begin_ring()
{
    if (mode_ != Mode::RING) {   // first decode step after a prompt: the ring starts empty over the slots and spare
        mode_ = Mode::RING;
        head_ = 0;
    }
}

void ExpertStore::fetch(int l, const int * ids, int n, cudaStream_t compute)
{
    if (mode_ != Mode::RING) throw std::logic_error("ExpertStore::fetch outside ring mode (begin_ring)");
    if (compute) wait_compute(compute);   // earlier kernels may read what the ring overwrites
    Layer & Y = layers_.at(l);
    ++stats_.fetch_calls;
    std::vector<int> need;
    need.reserve(n);
    for (int i = 0; i < n; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= Y.n_expert) throw std::out_of_range("ExpertStore::fetch: expert id " + std::to_string(e));
        if (Y.cold_off[e] < 0 || std::find(need.begin(), need.end(), e) != need.end()) continue;
        need.push_back(e);
    }
    stats_.experts_asked += (long) need.size();
    protect_layer_ = l;
    protect_.clear();
    for (int i = 0; i < n; ++i) protect_.push_back(ids[i]);
    // A copy may evict another expert this call needs (one lying just ahead of the head): repeat until all are
    // resident. A re-copied expert lands at the head and is not evicted again by this call, so each round settles at
    // least one needed expert and the loop ends within need.size() + 1 rounds unless the ring is smaller than them.
    bool changed = false;
    for (int round = 0;; ++round) {
        bool missing = false;
        for (int e : need)
            if (Y.ring_at[e] < 0) {
                if (round > (int) need.size()) throw std::runtime_error("ExpertStore::fetch: ring too small for one layer's experts");
                ring_put(l, e);
                missing = changed = true;
            }
        if (!missing) break;
    }
    if (changed)
        for (int p = 0; p < 3; ++p)
            TRUSS_CUDA(cudaMemcpyAsync(Y.ring_meta[p], Y.ring_meta_host + (size_t) p * 2 * Y.n_expert,
                                       sizeof(int32_t) * 2 * Y.n_expert, cudaMemcpyHostToDevice, copy_));
    TRUSS_CUDA(cudaEventRecord(copied_[l % 2], copy_));
}

void ExpertStore::prefetch_hint(int l, const int * ids, int n)
{
    const int next = l + 1;
    if (mode_ != Mode::RING || next >= (int) layers_.size()) return;
    Layer & Y = layers_[next];
    bool changed = false;
    std::vector<int> seen;
    for (int i = 0; i < n; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= Y.n_expert || Y.cold_off[e] < 0 || Y.ring_at[e] >= 0) continue;
        if (std::find(seen.begin(), seen.end(), e) != seen.end()) continue;
        seen.push_back(e);
        if (!ring_put(next, e, true)) break;   // the next slot holds an expert this layer still reads
        changed = true;
    }
    if (changed)
        for (int p = 0; p < 3; ++p)
            TRUSS_CUDA(cudaMemcpyAsync(Y.ring_meta[p], Y.ring_meta_host + (size_t) p * 2 * Y.n_expert,
                                       sizeof(int32_t) * 2 * Y.n_expert, cudaMemcpyHostToDevice, copy_));
}

void ExpertStore::signal(int l, int * flag, int seq)
{
    (void) l;
    int * src = signal_src_ + signal_next_;
    signal_next_ = (signal_next_ + 1) % SIGNAL_RING;
    *src = seq;
    TRUSS_CUDA(cudaMemcpyAsync(flag, src, sizeof(int), cudaMemcpyHostToDevice, copy_));
}

void ExpertStore::upload(void * dst, const void * src, size_t bytes)
{
    TRUSS_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, copy_));
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
