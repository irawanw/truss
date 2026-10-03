#include "runtime/expert_store.h"

#include "core/cuda_check.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
    std::vector<double> score(size.size());
    for (size_t i = 0; i < size.size(); ++i) score[i] = usage[i] / std::pow((double) size[i], z.plan_alpha);
    auto better = [&](int a, int b) { return score[a] > score[b]; };
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
    // Every event the driver thread records is created here, on the constructing thread: cuEventCreate takes the
    // context's write lock, which a host thread blocked in cudaLaunchKernel (its queue full behind a doorbell spin
    // that waits for the driver) holds as a reader: the driver then never writes the plan (TRACKER #115).
    for (cudaEvent_t * e : { &demand_ev_, &piece_ev_, &fence_ev_, &adm_ev_ })
        TRUSS_CUDA(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
    ev_pool_.resize(LANDING_EVENTS);
    for (cudaEvent_t & e : ev_pool_) TRUSS_CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
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
    vend_ = (int64_t) ring_;
    for (const Layer & Y : layers_) slot_ = std::max(slot_, align_up(Y.cold));
    auto * arena = static_cast<uint8_t *>(device_alloc(hot_lo + ring_ + hot_hi, device_));
    base_ = arena + hot_lo;
    spare_ = z.stream_extra ? base_ + 2 * slot_ : nullptr;
    stream_end_ = (int64_t) align_up(2 * slot_ + z.stream_extra);
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
        Y.pend.assign(E, 0);
        Y.ref.assign(E, 0);
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
    if (adm_ev_) cudaEventDestroy(adm_ev_);
    for (auto & x : landing_) cudaEventDestroy(x.first);
    for (cudaEvent_t e : ev_pool_) cudaEventDestroy(e);
    if (piece_ev_) cudaEventDestroy(piece_ev_);
    if (demand_ev_) cudaEventDestroy(demand_ev_);
    if (fence_ev_) cudaEventDestroy(fence_ev_);
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
    std::lock_guard<std::mutex> g(adm_mu_);
    wait_compute(compute);   // decode kernels may still read the ring the slots overwrite
    // The slots and the chunk's buffers take [0, stream_end_): only the entries there go. The rest of the ring and the
    // lent KV segments keep theirs for the next decode (TRACKER #115: an agent turn's prompt used to empty the ring).
    // The head moves past the stream area, so address order from the head is still FIFO order.
    for (auto it = fifo_.begin(); it != fifo_.end();)
        if (it->off < stream_end_) drop(*it), it = fifo_.erase(it);
        else ++it;
    if (head_ < stream_end_) head_ = stream_end_;
    hinted_.clear();
    adm_.clear();
    claims_.clear(), adm_backlog_ = 0;   // landing_ events stay: their claims are gone from the ring (checked by offset)
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
uint8_t * ExpertStore::real(int64_t v) const
{
    if (v < (int64_t) ring_) return base_ + v;
    for (const Segment & g : segs_)
        if (v >= g.vbase && v < g.vbase + g.size) return g.ptr + (v - g.vbase);
    throw std::logic_error("ExpertStore: ring offset outside every region");
}

bool ExpertStore::region(int64_t v, int64_t * b, int64_t * e) const
{
    if (v >= 0 && v < (int64_t) ring_) return *b = 0, *e = (int64_t) ring_, true;
    for (const Segment & g : segs_)
        if (v >= g.vbase + g.lo && v < g.vbase + g.size) return *b = g.vbase + g.lo, *e = g.vbase + g.size, true;
    return false;
}

int64_t ExpertStore::next_region(int64_t v, int64_t need) const
{
    for (const Segment & g : segs_)
        if (g.vbase + g.lo > v && g.size - g.lo >= need) return g.vbase + g.lo;
    return 0;   // back to the ring's own region
}

void ExpertStore::drop(const RingEntry & r)
{
    layers_[r.layer].ring_at[r.expert] = -1;
    layers_[r.layer].pend[r.expert] = 0;
}

int ExpertStore::add_segment(uint8_t * ptr, size_t bytes)
{
    const uintptr_t u = reinterpret_cast<uintptr_t>(ptr);
    const int64_t a = (int64_t) ((u + 255) / 256 * 256 - u);
    const int64_t size = ((int64_t) bytes - a) / 256 * 256;
    if (size <= 0) return -1;
    unit_offset(ptr + a, base_, 0);   // throws if it cannot be addressed from the ring base
    segs_.push_back({ ptr + a, vend_, size, size });   // lent by reclaim()
    vend_ += size;
    return (int) segs_.size() - 1;
}

void ExpertStore::reclaim(int j, size_t lo_bytes)
{
    std::lock_guard<std::mutex> g(adm_mu_);
    Segment & S = segs_.at(j);
    const int64_t lo = std::min<int64_t>(S.size, (int64_t) align_up(lo_bytes));
    if (lo > S.lo) {   // evict what lies in [vbase, vbase + lo)
        const int64_t a = S.vbase, b = S.vbase + lo;
        for (auto it = fifo_.begin(); it != fifo_.end();)
            if (it->off >= a && it->off < b) drop(*it), it = fifo_.erase(it);
            else ++it;
        // a head inside moves to the boundary; the next ring_put() advances from there in address order
        if (head_ >= a && head_ < b) head_ = b;
    }
    S.lo = lo;
}

void ExpertStore::fence(cudaStream_t compute)
{
    TRUSS_CUDA(cudaEventRecord(fence_ev_, copy_));
    TRUSS_CUDA(cudaStreamWaitEvent(compute, fence_ev_, 0));
}

size_t ExpertStore::lent_bytes() const
{
    size_t n = 0;
    for (const Segment & g : segs_) n += (size_t) (g.size - g.lo);
    return n;
}

bool ExpertStore::ring_put(int l, int e, bool hint, bool copy)
{
    Layer & Y = layers_[l];
    const int64_t need = (int64_t) align_up(Y.bytes[e]);
    // where this allocation lands: at the head if its region has room, else at the next region with room (0: the
    // ring's own region again); everything from the head to there goes
    auto target = [&] {
        int64_t b, e2;
        return region(head_, &b, &e2) && head_ + need <= e2 ? head_ : next_region(head_, need);
    };
    if (hint) {   // would this allocation evict an expert the current layer's kernel reads?
        const int64_t start = target();
        for (const RingEntry & r : fifo_) {
            const bool skipped = start > head_ ? r.off >= head_ && r.off < start : start < head_ && r.off >= head_;
            const bool gone = skipped || (r.off >= start && r.off < start + need);
            if (!gone) {   // FIFO order is address order from the head: later entries lie further on
                if (start >= head_ ? r.off >= start + need : r.off < head_ && r.off >= start + need) break;
                continue;
            }
            if (r.layer == protect_layer_ && std::find(protect_.begin(), protect_.end(), r.expert) != protect_.end())
                return false;
        }
    }
    auto advance = [&] {   // move the head to target(), evicting what lies between (the tail of a region)
        const int64_t to = target();
        if (to == head_) return false;
        while (!fifo_.empty() && fifo_.front().off >= head_ && (to < head_ || fifo_.front().off < to))
            drop(fifo_.front()), fifo_.pop_front();
        head_ = to;
        return true;
    };
    advance();
    // CLOCK in place (TRACKER #91): the oldest entry in the way that was used since it came in stays where it is,
    // loses its bit and goes to the back of the order, and the head skips past it (no copy). Replay at 256K: FIFO
    // 240 -> ~207 misses/pass, LRU 208. At most `clock_max` skips per copy (TRUSS_RING_CLOCK; 0 = plain FIFO).
    static const int clock_max = std::getenv("TRUSS_RING_CLOCK") ? std::atoi(std::getenv("TRUSS_RING_CLOCK")) : 8;
    for (int skips = 0;;) {
        advance();   // again: a skip may have pushed the head to a region's end
        if (fifo_.empty() || fifo_.front().off < head_ || fifo_.front().off >= head_ + need) break;
        const RingEntry f = fifo_.front();
        Layer & F = layers_[f.layer];
        fifo_.pop_front();
        // not for hints: their protection check above covers only the unskipped region, and the current layer's
        // kernel may still read the ring (demand fetches and admissions wait for compute first)
        if (!hint && F.ref[f.expert] && skips < clock_max && !(f.layer == l && f.expert == e)) {
            F.ref[f.expert] = 0;
            fifo_.push_back(f);
            head_ = f.off + (int64_t) align_up(F.bytes[f.expert]);
            ++skips;
            continue;
        }
        F.ring_at[f.expert] = -1;
        F.pend[f.expert] = 0;
    }
    const int64_t off = head_;
    head_ += need;
    fifo_.push_back({ l, e, off });
    Y.ring_at[e] = off;
    Y.ref[e] = 0;
    uint8_t * at = real(off);
    if (copy) TRUSS_CUDA(cudaMemcpyAsync(at, Y.host + Y.cold_off[e], Y.bytes[e], cudaMemcpyHostToDevice, copy_));
    for (int p = 0; p < 3; ++p) Y.ring_meta_host[(size_t) p * 2 * Y.n_expert + 2 * e + 1] = unit_offset(at + Y.part[p][e], base_, l);
    if (!copy) return true;
    if (hint) stats_.hint_bytes += Y.bytes[e], ++stats_.hinted;
    else stats_.bytes += Y.bytes[e], ++stats_.misses;
    return true;
}

void ExpertStore::poll_admitted()
{
    if (adm_.empty() || cudaEventQuery(adm_ev_) != cudaSuccess) return;
    for (const auto & [l, e] : adm_) layers_[l].pend[e] = 0;
    adm_.clear();
}

void ExpertStore::admit(const std::vector<std::pair<int, int>> & le, cudaStream_t compute)
{
    if (mode_ != Mode::RING) return;
    poll_admitted();
    if (!adm_.empty() || le.empty()) return;
    wait_compute(compute);   // the copies overwrite ring slots that queued kernels may still read
    std::vector<uint8_t> touched(layers_.size(), 0);
    for (const auto & [l, e] : le) {
        Layer & Y = layers_.at(l);
        if (Y.cold_off[e] < 0 || Y.ring_at[e] >= 0) continue;
        ring_put(l, e);
        Y.pend[e] = 1;
        adm_.push_back({ l, e });
        touched[l] = 1;
    }
    for (size_t l = 0; l < layers_.size(); ++l)
        if (touched[l])
            for (int p = 0; p < 3; ++p)
                TRUSS_CUDA(cudaMemcpyAsync(layers_[l].ring_meta[p], layers_[l].ring_meta_host + (size_t) p * 2 * layers_[l].n_expert,
                                           sizeof(int32_t) * 2 * layers_[l].n_expert, cudaMemcpyHostToDevice, copy_));
    // a ring_put of a later admission may have evicted an earlier one of this batch: keep only the resident
    adm_.erase(std::remove_if(adm_.begin(), adm_.end(), [&](const auto & x) { return layers_[x.first].ring_at[x.second] < 0; }),
               adm_.end());
    TRUSS_CUDA(cudaEventRecord(adm_ev_, copy_));
    adm_total_ += (long) adm_.size();
}

void ExpertStore::begin_ring()
{
    std::lock_guard<std::mutex> g(adm_mu_);
    if (mode_ != Mode::RING) mode_ = Mode::RING;   // decode after a prompt: the stream area is empty, the rest kept
}

bool ExpertStore::fetch(int l, const int * ids, int n, cudaStream_t compute)
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
        if (Y.ring_at[e] >= 0) Y.ref[e] = 1;   // a ring hit: CLOCK's second chance
        need.push_back(e);
    }
    stats_.experts_asked += (long) need.size();
    bool hinted_needed = false;   // a hinted expert's copy may not have landed yet
    if ((size_t) l < hinted_.size()) {
        for (int e : need) hinted_needed |= std::find(hinted_[l].begin(), hinted_[l].end(), e) != hinted_[l].end();
        hinted_[l].clear();
    }
    poll_admitted();
    for (int e : need)
        if (Y.pend[e]) {   // admitted or claimed, copy maybe in flight: wait for go (after the rest of a claim)
            claim_flush(l, e);
            hinted_needed = true;
        }
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
    if (changed) {   // admit_step() holds back while these demand copies run
        TRUSS_CUDA(cudaEventRecord(demand_ev_, copy_));
        demand_out_ = true;
    }
    return changed || hinted_needed;
}

bool ExpertStore::claim(int l, int e)
{
    if (mode_ != Mode::RING) return false;
    Layer & Y = layers_.at(l);
    if (Y.cold_off[e] < 0 || Y.ring_at[e] >= 0) return false;
    if (!ring_put(l, e, true, false)) return false;
    Y.pend[e] = 1;
    claims_.push_back({ l, e, Y.ring_at[e], 0 });
    adm_backlog_ += Y.bytes[e];
    return true;
}

void ExpertStore::claim_issue(Claim & c, size_t upto)
{
    Layer & Y = layers_[c.layer];
    const size_t n = upto - c.done;
    TRUSS_CUDA(cudaMemcpyAsync(real(c.off) + c.done, Y.host + Y.cold_off[c.expert] + c.done, n, cudaMemcpyHostToDevice,
                               copy_));
    c.done = upto;
    adm_backlog_ -= n;
    stats_.admit_bytes += n;
    if (c.done < Y.bytes[c.expert]) return;
    // the last piece: the meta table behind it, then an event that clears pend once it has landed
    for (int p = 0; p < 3; ++p)
        TRUSS_CUDA(cudaMemcpyAsync(Y.ring_meta[p], Y.ring_meta_host + (size_t) p * 2 * Y.n_expert,
                                   sizeof(int32_t) * 2 * Y.n_expert, cudaMemcpyHostToDevice, copy_));
    if (ev_pool_.empty()) {   // every landing event in flight (never in practice): the newest batch takes this claim
        landing_.back().second.push_back({ c.layer * 65536 + c.expert, c.off });
        TRUSS_CUDA(cudaEventRecord(landing_.back().first, copy_));
    } else {
        const cudaEvent_t ev = ev_pool_.back();
        ev_pool_.pop_back();
        TRUSS_CUDA(cudaEventRecord(ev, copy_));
        landing_.push_back({ ev, { { c.layer * 65536 + c.expert, c.off } } });
    }
    ++stats_.admitted_claims;
}

void ExpertStore::claim_flush(int l, int e)
{
    for (auto it = claims_.begin(); it != claims_.end(); ++it)
        if (it->layer == l && it->expert == e) {
            if (layers_[l].ring_at[e] == it->off) claim_issue(*it, layers_[l].bytes[e]);
            else adm_backlog_ -= layers_[l].bytes[e] - it->done;
            claims_.erase(it);
            return;
        }
}

bool ExpertStore::admit_step()
{
    std::unique_lock<std::mutex> g(adm_mu_, std::try_to_lock);
    if (!g.owns_lock()) return false;
    // landed claims: on the device from now on (unless the ring moved on since)
    for (size_t i = 0; i < landing_.size();) {
        if (cudaEventQuery(landing_[i].first) != cudaSuccess) { ++i; continue; }
        for (const auto & [le, off] : landing_[i].second) {
            Layer & Y = layers_[le / 65536];
            if (Y.ring_at[le % 65536] == off) Y.pend[le % 65536] = 0, ++adm_landed_;
        }
        ev_pool_.push_back(landing_[i].first);
        landing_.erase(landing_.begin() + (ptrdiff_t) i);
    }
    if (mode_ != Mode::RING) return false;
    while (!claims_.empty()) {   // drop claims the ring has moved past
        Claim & c = claims_.front();
        if (layers_[c.layer].ring_at[c.expert] == c.off) break;
        adm_backlog_ -= layers_[c.layer].bytes[c.expert] - c.done;
        claims_.pop_front();
    }
    if (claims_.empty()) return false;
    if (piece_out_) {
        if (cudaEventQuery(piece_ev_) != cudaSuccess) return false;
        piece_out_ = false;
    }
    if (demand_out_) {
        if (cudaEventQuery(demand_ev_) != cudaSuccess) return false;
        demand_out_ = false;
    }
    Claim & c = claims_.front();
    claim_issue(c, std::min(layers_[c.layer].bytes[c.expert], c.done + PIECE));
    if (c.done == layers_[c.layer].bytes[c.expert]) claims_.pop_front();
    TRUSS_CUDA(cudaEventRecord(piece_ev_, copy_));
    piece_out_ = true;
    return true;
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
        if (hinted_.size() < layers_.size()) hinted_.resize(layers_.size());
        hinted_[next].push_back(e);
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
