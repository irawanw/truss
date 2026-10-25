// The far tier's slot cache and reader threads: see expert_far.h.
#include "runtime/expert_far.h"

#include "formats/gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace truss::runtime {

namespace {
bool default_read(const void * src, size_t n, void * dst, void * bounce, size_t bounce_bytes)
{
    // O_DIRECT first (no page cache: the pinned cold copy already fills the RAM), buffered pread if the file system
    // refuses it, the mapping itself as the last resort (a test file that was never registered)
    if (bounce && gguf::read_direct(src, n, dst, bounce, bounce_bytes)) return true;
    if (gguf::read_mapped(src, n, dst)) return true;
    std::memcpy(dst, src, n);
    return true;
}
}  // namespace

FarCache::FarCache(const Config & c, std::vector<std::array<Span, 3>> far, uint8_t * region, ReadFn read)
    : cfg_(c), far_(std::move(far)), region_(region), read_(read ? std::move(read) : ReadFn(default_read))
{
    const size_t K = (size_t) c.n_layer * c.n_expert;
    if (far_.size() != K) throw std::invalid_argument("FarCache: far table size != n_layer * n_expert");
    if (c.n_slots <= 0 || !region) throw std::invalid_argument("FarCache: no slots");
    for (size_t k = 0; k < K; ++k) {
        size_t b = far_[k][0].bytes + far_[k][1].bytes + far_[k][2].bytes;
        if (b > c.slot_bytes) throw std::invalid_argument("FarCache: a far expert is larger than slot_bytes");
    }
    state_.assign(K, EMPTY);
    slot_of_.assign(K, -1);
    failed_.assign(K, 0);
    slots_.resize(c.n_slots);
    for (int s = c.n_slots - 1; s >= 0; --s) free_.push_back(s);
    for (int t = 0; t < std::max(1, c.threads); ++t) threads_.emplace_back([this] { worker(); });
}

FarCache::~FarCache()
{
    {
        std::lock_guard<std::mutex> g(mu_);
        stop_ = true;
    }
    work_cv_.notify_all();
    for (std::thread & t : threads_) t.join();
}

size_t FarCache::bytes_of(int l, int e) const
{
    const auto & f = far_[key(l, e)];
    return f[0].bytes + f[1].bytes + f[2].bytes;
}

void FarCache::lru_remove(int s)
{
    if (slots_[s].in_lru) lru_.erase(slots_[s].lru), slots_[s].in_lru = false;
}

void FarCache::lru_push(int s)
{
    lru_remove(s);
    slots_[s].lru = lru_.insert(lru_.end(), s);
    slots_[s].in_lru = true;
}

int FarCache::victim(bool hint)
{
    if (!free_.empty()) {
        const int s = free_.back();
        free_.pop_back();
        return s;
    }
    int s = -1;   // every slot held or being read: -1
    for (int x : lru_)
        if (!hint || !slots_[x].fresh) {
            s = x;
            break;
        }
    if (s < 0) return -1;
    lru_remove(s);
    slots_[s].fresh = false;
    const int k = slots_[s].key;
    state_[k] = EMPTY, slot_of_[k] = -1;
    ++st_.evictions;
    return s;
}

bool FarCache::want(int l, int e, bool demand)
{
    const int k = key(l, e);
    if (!is_far(l, e)) return false;
    std::lock_guard<std::mutex> g(mu_);
    switch (state_[k]) {
    case READY: return true;
    case READING: return false;
    case QUEUED:
        if (demand) {   // promote a queued hint (out of hint_q_: an overflow there must not drop it)
            for (auto it = hint_q_.begin(); it != hint_q_.end(); ++it)
                if (*it == k) {
                    hint_q_.erase(it);
                    demand_q_.push_back(k);
                    work_cv_.notify_one();
                    break;
                }
        }
        return false;
    case EMPTY: break;
    }
    failed_[k] = 0;
    state_[k] = QUEUED;
    if (demand) {
        demand_q_.push_back(k);
    } else {
        hint_q_.push_back(k);
        while ((int) hint_q_.size() > cfg_.max_hints) {   // oldest guess out
            const int old = hint_q_.front();
            hint_q_.pop_front();
            if (state_[old] == QUEUED) state_[old] = EMPTY, ++st_.hints_dropped;
        }
    }
    work_cv_.notify_one();
    return false;
}

void FarCache::drop_hints()
{
    std::lock_guard<std::mutex> g(mu_);
    for (int k : hint_q_)
        if (state_[k] == QUEUED) state_[k] = EMPTY, ++st_.hints_dropped;
    hint_q_.clear();
}

const uint8_t * FarCache::acquire(int l, int e)
{
    const int k = key(l, e);
    std::lock_guard<std::mutex> g(mu_);
    if (state_[k] != READY) {
        ++st_.misses;
        return nullptr;
    }
    ++st_.hits;
    const int s = slot_of_[k];
    slots_[s].fresh = false;
    if (slots_[s].held++ == 0) lru_remove(s);
    return region_ + (size_t) s * cfg_.slot_bytes;
}

void FarCache::release(int l, int e)
{
    const int k = key(l, e);
    std::lock_guard<std::mutex> g(mu_);
    const int s = slot_of_[k];
    if (s < 0 || slots_[s].held <= 0) return;
    if (--slots_[s].held == 0) lru_push(s);   // the most recently used
}

const uint8_t * FarCache::wait(int l, int e, int timeout_ms, bool count)
{
    return wait_until(l, e, timeout_ms < 0 ? std::chrono::steady_clock::time_point::max()
                                           : std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms),
                      count);
}

bool FarCache::failed(int l, int e)
{
    std::lock_guard<std::mutex> g(mu_);
    return failed_[key(l, e)] != 0;
}

const uint8_t * FarCache::wait_until(int l, int e, std::chrono::steady_clock::time_point deadline, bool count)
{
    const int k = key(l, e);
    if (!is_far(l, e)) return nullptr;
    want(l, e, true);
    std::unique_lock<std::mutex> g(mu_);
    if (count) {
        if (state_[k] == READY) ++st_.hits;
        else ++st_.misses;
    }
    auto done = [&] { return state_[k] == READY || failed_[k] || stop_; };
    if (deadline == std::chrono::steady_clock::time_point::max()) done_cv_.wait(g, done);
    else if (!done_cv_.wait_until(g, deadline, done)) return nullptr;
    if (state_[k] != READY) return nullptr;
    const int s = slot_of_[k];
    slots_[s].fresh = false;
    if (slots_[s].held++ == 0) lru_remove(s);
    return region_ + (size_t) s * cfg_.slot_bytes;
}

FarCache::Job * FarCache::open_job(bool demand_only)
{
    for (Job & j : jobs_)
        if (j.next < j.pieces.size() && (!demand_only || !j.hint)) return &j;
    return nullptr;
}

FarCache::Job * FarCache::start_job(bool hint)
{
    std::deque<int> & q = hint ? hint_q_ : demand_q_;
    if (q.empty()) return nullptr;
    const int k = q.front();
    const int s = victim(hint);
    if (s < 0) {   // no slot it may take (rare: slots >> in flight)
        if (hint) {   // a guess is not worth waiting for: dropped
            q.pop_front();
            state_[k] = EMPTY, ++st_.hints_dropped;
        }
        return nullptr;   // a demand read stays queued until a release
    }
    q.pop_front();
    state_[k] = READING, slot_of_[k] = s;
    slots_[s].key = k, slots_[s].held = 0;
    Job j{ k, s, hint, {} };
    const size_t piece = cfg_.piece_bytes ? cfg_.piece_bytes : (size_t) -1;
    for (const Span & p : far_[k])
        for (size_t o = 0; o < p.bytes; o += piece) {
            const size_t n = std::min(piece, p.bytes - o);
            j.pieces.push_back({ static_cast<const uint8_t *>(p.src) + o, n, j.bytes });
            j.bytes += n;
        }
    jobs_.push_back(std::move(j));
    return &jobs_.back();
}

void FarCache::finish_job(std::list<Job>::iterator it)
{
    const Job & j = *it;
    const int k = j.key, s = j.slot;
    if (j.ok) {
        state_[k] = READY;
        slots_[s].fresh = !j.hint;
        lru_push(s);
        ++st_.reads, st_.hint_reads += j.hint, st_.read_bytes += j.bytes;
    } else {
        state_[k] = EMPTY, slot_of_[k] = -1, failed_[k] = 1;
        slots_[s].key = -1;
        free_.push_back(s);
        ++st_.read_fail;
    }
    jobs_.erase(it);
    done_cv_.notify_all();
}

void FarCache::worker()
{
    void * bounce = nullptr;
    if (posix_memalign(&bounce, gguf::DIRECT_ALIGN, cfg_.bounce_bytes) != 0) bounce = nullptr;
    std::unique_lock<std::mutex> g(mu_);
    for (;;) {
        bool demand_stuck = false;   // a demand read waits for a slot release
        work_cv_.wait(g, [&] {
            if (stop_) return true;
            // a queued key whose state moved on (dropped hint, promoted duplicate) is skipped
            while (!demand_q_.empty() && state_[demand_q_.front()] != QUEUED) demand_q_.pop_front();
            while (!hint_q_.empty() && state_[hint_q_.front()] != QUEUED) hint_q_.pop_front();
            return open_job(false) || !demand_q_.empty() || (!hint_q_.empty() && (!free_.empty() || !lru_.empty()));
        });
        if (stop_) break;
        // demand pieces first, then a new demand read, then hint pieces, then a new hint
        Job * j = open_job(true);
        if (!j && !demand_q_.empty() && !(j = start_job(false))) demand_stuck = true;
        if (!j) j = open_job(false);
        if (!j && !demand_stuck && !hint_q_.empty()) j = start_job(true);   // no slot for a demand: none for a hint
        if (!j) {
            if (demand_stuck) done_cv_.wait_for(g, std::chrono::milliseconds(1));
            continue;
        }
        const Piece p = j->pieces[j->next++];
        if (j->next < j->pieces.size()) work_cv_.notify_one();   // more pieces: wake another reader
        uint8_t * dst = region_ + (size_t) j->slot * cfg_.slot_bytes + p.at;
        g.unlock();
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = read_(p.src, p.n, dst, bounce, cfg_.bounce_bytes);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        g.lock();
        st_.read_ms += ms;
        j->ok = j->ok && ok;
        if (++j->done == j->pieces.size())
            for (auto it = jobs_.begin(); it != jobs_.end(); ++it)
                if (&*it == j) {
                    finish_job(it);
                    break;
                }
    }
    g.unlock();
    std::free(bounce);
}

FarCache::Stats FarCache::stats() const
{
    std::lock_guard<std::mutex> g(mu_);
    return st_;
}

std::string FarCache::report() const
{
    const Stats s = stats();
    char b[256];
    std::snprintf(b, sizeof b,
                  "far tier: hits %ld misses %ld reads %ld (hint %ld, failed %ld) %.2f GB at %.2f GB/s per thread, "
                  "evictions %ld, hints dropped %ld",
                  s.hits, s.misses, s.reads, s.hint_reads, s.read_fail, s.read_bytes / 1e9,
                  s.read_ms > 0 ? s.read_bytes / 1e6 / s.read_ms : 0.0, s.evictions, s.hints_dropped);
    return b;
}

}  // namespace truss::runtime
