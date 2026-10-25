// ExpertStore decode side: the ring (virtual offsets over segments and eviction), fetch of a layer's routed cold
// experts, hints, the adaptive and idle-link admission, E6 segment lending.
#include "runtime/expert_store.h"
#include "runtime/expert_store_util.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace truss::runtime {

using namespace store_detail;

uint8_t * ExpertStore::real(int64_t v) const
{
    if (v < (int64_t) ring_) return base_ + v;
    for (const Segment & g : segs_)
        if (v >= g.vbase && v < g.vbase + g.size) return g.ptr + (v - g.vbase);
    throw std::logic_error("ExpertStore: ring offset outside every region");
}

bool ExpertStore::region(int64_t v, int64_t * b, int64_t * e) const
{
    // E7: the FIFO's own region stops at ring_hi_; everything above it is the pinned refill's (never allocated here)
    if (v >= 0 && v < ring_hi_) return *b = 0, *e = ring_hi_, true;
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
    Layer & Y = layers_[r.layer];
    if (adapt_reuse_on_ && !Y.adapt_set.empty() && Y.adapt_set[r.expert]) {   // the admission's reuse count
        ++adapt_reuse_[Y.adapt_hits[r.expert]];
        Y.adapt_set[r.expert] = 0;
    }
    Y.ring_at[r.expert] = -1;
    Y.pend[r.expert] = 0;
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

bool ExpertStore::ring_put(int l, int e, bool hint, bool copy, bool adapt)
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
    if (copy) {
        const bool far = is_far(l, e);
        TRUSS_CUDA(cudaMemcpyAsync(at, far ? far_src(l, e) : Y.host + Y.cold_off[e], Y.bytes[e], cudaMemcpyHostToDevice, copy_));
        if (far) far_copied(l, e);
    }
    for (int p = 0; p < 3; ++p) Y.ring_meta_host[(size_t) p * 2 * Y.n_expert + 2 * e + 1] = unit_offset(at + Y.part[p][e], base_, l);
    Y.meta_dirty = true;   // uploaded once at the next fetch/hint/acquire of this layer (1.A)
    if (!copy) return true;
    if (hint) stats_.hint_bytes += Y.bytes[e], ++stats_.hinted;
    else if (adapt) stats_.adapt_bytes += Y.bytes[e], ++stats_.adapt_copies;
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
        if (Y.cold_off[e] < 0 || Y.ring_at[e] >= 0 || is_far(l, e)) continue;   // far: no host copy to admit from
        ring_put(l, e, false, true, true);   // the adaptive tier's copy: its own counter
        if (adapt_reuse_on_ && Y.adapt_set.empty()) {
            Y.adapt_set.assign(Y.n_expert, 0);
            Y.adapt_hits.assign(Y.n_expert, 0);
        }
        if (adapt_reuse_on_) Y.adapt_set[e] = 1, Y.adapt_hits[e] = 0;   // sized only with TRUSS_ADAPT_REUSE
        Y.pend[e] = 1;
        adm_.push_back({ l, e });
        touched[l] = 1;
    }
    for (size_t l = 0; l < layers_.size(); ++l)
        if (touched[l] && layers_[l].meta_dirty)
            for (int p = 0; p < 3; ++p)
            {
                TRUSS_CUDA(cudaMemcpyAsync(layers_[l].ring_meta[p], layers_[l].ring_meta_host + (size_t) p * 2 * layers_[l].n_expert,
                                           sizeof(int32_t) * 2 * layers_[l].n_expert, cudaMemcpyHostToDevice, copy_));
                layers_[l].meta_dirty = false;
            }
    // a ring_put of a later admission may have evicted an earlier one of this batch: keep only the resident
    adm_.erase(std::remove_if(adm_.begin(), adm_.end(), [&](const auto & x) { return layers_[x.first].ring_at[x.second] < 0; }),
               adm_.end());
    TRUSS_CUDA(cudaEventRecord(adm_ev_, copy_));
    adm_total_ += (long) adm_.size();
}

void ExpertStore::begin_ring()
{
    prof_harvest();
    std::lock_guard<std::mutex> g(adm_mu_);
    if (mode_ == Mode::RING) return;   // decode after a prompt: the stream area is empty, the rest kept
    mode_ = Mode::RING;
    // E7: arm the refill. pin_step() (driver thread) places the experts in the ring's free space, admit_step()
    // copies them; the FIFO keeps the last Sizes::pin_fill bytes.
    if (pin_budget_ && !fill_.empty()) {
        fill_pos_ = 0, pin_bytes_ = 0, pin_queued_ = false;
        pins_.clear();
        ring_hi_ = (int64_t) ring_;
        pin_t0_ = std::chrono::steady_clock::now();
        pin_active_.store(true, std::memory_order_release);
    }
}

// E7 (E6 option c): reserve the top of the ring for `fill`'s next cold experts and queue one claim each. Pinned
// experts lie in [ring_hi_, ring_) so ring_put()'s FIFO, which only allocates in [0, ring_hi_), never evicts them.
// Driver thread (it shares the ring with fetch()); the copies run later in admit_step().
void ExpertStore::pin_step()
{
    if (!pin_active_.load(std::memory_order_acquire) || pin_queued_) return;
    std::lock_guard<std::mutex> g(adm_mu_);
    if (mode_ != Mode::RING) { pin_queued_ = true; return; }
    while (fill_pos_ < fill_.size()) {
        const auto [l, e] = fill_[fill_pos_];
        Layer & Y = layers_[l];
        if (Y.cold_off[e] < 0 || Y.ring_at[e] >= 0 || is_far(l, e)) { ++fill_pos_; continue; }   // hot, in the ring, far
        const int64_t need = (int64_t) align_up(Y.bytes[e]);
        if (pin_bytes_ + (size_t) need > pin_budget_ || ring_hi_ - need < 0) break;   // the FIFO's share is reached
        const int64_t off = ring_hi_ - need;
        // the FIFO's region shrinks to [0, off): everything now above it goes, and the cursor moves back if it is there
        for (auto it = fifo_.begin(); it != fifo_.end();)
            if (it->off >= off) drop(*it), it = fifo_.erase(it);
            else ++it;
        if (head_ >= off) head_ = 0;
        ring_hi_ = off;
        Y.ring_at[e] = off;
        Y.pend[e] = 1;
        Y.ref[e] = 0;
        uint8_t * at = real(off);
        for (int p = 0; p < 3; ++p)
            Y.ring_meta_host[(size_t) p * 2 * Y.n_expert + 2 * e + 1] = unit_offset(at + Y.part[p][e], base_, l);
        Y.meta_dirty = true;
        claims_.push_back({ l, e, off, 0 });   // admit_step() copies it piecewise, like any claim
        adm_backlog_ += Y.bytes[e];
        pin_bytes_ += (size_t) need;
        pins_.push_back({ l, e, off });
        ++fill_pos_;
    }
    pin_queued_ = true;
}

// E7: how much of the refill landed and how long it ran. `tag` says whether it finished or a prompt superseded it.
void ExpertStore::pin_report(const char * tag)
{
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pin_t0_).count();
    std::fprintf(stderr, "E7 pin fill: %zu experts, %.2f GiB, %.1f ms%s\n", pins_.size(), pin_bytes_ / 1073741824.0, ms,
                 tag);
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
        if (is_far(l, e)) ++far_st_.routed, far_st_.ring_hits += Y.ring_at[e] >= 0;
        need.push_back(e);
    }
    stats_.experts_asked += (long) need.size();
    bool hinted_needed = false;   // a hinted expert's copy may not have landed yet
    if ((size_t) l < hinted_.size()) {
        for (int e : need) hinted_needed |= std::find(hinted_[l].begin(), hinted_[l].end(), e) != hinted_[l].end();
        hinted_[l].clear();
    }
    poll_admitted();
    if (adapt_reuse_on_)
        for (int e : need)
            if (!Y.adapt_set.empty() && Y.adapt_set[e]) ++Y.adapt_hits[e];   // the admission paid off: routed again
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
    if (Y.meta_dirty)   // the table goes up once here, behind every copy queued so far (1.A)
    {
        for (int p = 0; p < 3; ++p)
            TRUSS_CUDA(cudaMemcpyAsync(Y.ring_meta[p], Y.ring_meta_host + (size_t) p * 2 * Y.n_expert,
                                       sizeof(int32_t) * 2 * Y.n_expert, cudaMemcpyHostToDevice, copy_));
        Y.meta_dirty = false;
    }
    if (!Y.far_ptr.empty())   // a slot far_ready() held for an expert the caller then dropped (a skip rule)
        for (int e = 0; e < Y.n_expert; ++e)
            if (Y.far_ptr[e]) Y.far_ptr[e] = nullptr, far_->release(l, e);
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
    if (Y.cold_off[e] < 0 || Y.ring_at[e] >= 0 || is_far(l, e)) return false;   // a claim copies from the host copy
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
    // the last piece: the meta table behind it if it is not up already (1.A), then an event that clears pend once it has landed
    if (Y.meta_dirty)
    {
        for (int p = 0; p < 3; ++p)
            TRUSS_CUDA(cudaMemcpyAsync(Y.ring_meta[p], Y.ring_meta_host + (size_t) p * 2 * Y.n_expert,
                                       sizeof(int32_t) * 2 * Y.n_expert, cudaMemcpyHostToDevice, copy_));
        Y.meta_dirty = false;
    }
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
    // E7: the refill is complete once every claim it queued has been issued and its copies have landed
    if (pin_active_.load(std::memory_order_acquire) && pin_queued_ && claims_.empty() && landing_.empty()) {
        pin_active_.store(false, std::memory_order_release);
        pin_report("");
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

void ExpertStore::prefetch_hint(int l, const int * ids, int n, bool far_only)
{
    const int next = l + 1;
    if (mode_ != Mode::RING || next >= (int) layers_.size()) return;
    Layer & Y = layers_[next];
    std::vector<int> seen;
    for (int i = 0; i < n; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= Y.n_expert || Y.cold_off[e] < 0 || Y.ring_at[e] >= 0) continue;
        if (far_only && !is_far(next, e)) continue;
        if (std::find(seen.begin(), seen.end(), e) != seen.end()) continue;
        seen.push_back(e);
        if (is_far(next, e)) {   // far: to the GPU only if its slot is ready; else start the NVMe read for later
            if (!far_) continue;   // far_drop: never routed, never read
            if (!Y.far_ptr[e] && !(Y.far_ptr[e] = far_->acquire(next, e))) {
                far_->want(next, e, false);
                continue;
            }
            if (!ring_put(next, e, true)) {   // refused before its copy: the slot is still held here
                far_->release(next, e), Y.far_ptr[e] = nullptr;
                break;
            }
        } else if (!ring_put(next, e, true)) {
            break;   // the next slot holds an expert this layer still reads
        }
        if (hinted_.size() < layers_.size()) hinted_.resize(layers_.size());
        hinted_[next].push_back(e);
    }
    if (Y.meta_dirty)   // one upload for the hinted layer, behind its copies (1.A)
    {
        for (int p = 0; p < 3; ++p)
            TRUSS_CUDA(cudaMemcpyAsync(Y.ring_meta[p], Y.ring_meta_host + (size_t) p * 2 * Y.n_expert,
                                       sizeof(int32_t) * 2 * Y.n_expert, cudaMemcpyHostToDevice, copy_));
        Y.meta_dirty = false;
    }
}

std::string ExpertStore::adapt_reuse_report() const
{
    std::string s = "adaptive tier admissions: reuse histogram";
    for (const auto & [hits, n] : adapt_reuse_) s += " " + std::to_string(hits) + "x:" + std::to_string(n);
    long tot = 0, sum = 0;
    for (const auto & [hits, n] : adapt_reuse_) tot += n, sum += hits * n;
    s += " (experts " + std::to_string(tot) + ", mean reuses " + std::to_string(tot ? double(sum) / tot : 0.0) + ")";
    return s;
}

}  // namespace truss::runtime
