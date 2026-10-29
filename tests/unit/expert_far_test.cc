// runtime::FarCache (expert_far.h) on host memory, no GPU, no file: the "file" is a byte array and the read function
// copies from it (optionally slowly, or failing on chosen experts). Checks: the bytes land as gate | up | down; LRU
// eviction order; a held slot is never overwritten; hint overflow drops the oldest, never a promoted one; a failed
// read ends wait() with nullptr; and a multi-threaded stress where every acquired slot must hold its own expert's bytes.
// usage: expert_far_test        (prints "expert_far_test: ok" or the failed checks; exit 1 on failure)
#include "runtime/expert_far.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using truss::runtime::FarCache;

namespace {
int fails = 0;
#define CHECK(c)                                                                 \
    do {                                                                         \
        if (!(c)) std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c), ++fails; \
    } while (0)

struct Fake {
    int L, E;
    std::vector<uint8_t> file;
    std::vector<std::array<FarCache::Span, 3>> far;
    // expert (l, e) projection p: 100 + 37 * p + e bytes, each byte = (l * 131 + e * 7 + p * 3 + i) & 255
    Fake(int L, int E, int every = 2) : L(L), E(E), far((size_t) L * E)
    {
        size_t at = 0;
        std::vector<std::array<std::pair<size_t, size_t>, 3>> pos((size_t) L * E);
        for (int l = 0; l < L; ++l)
            for (int e = 0; e < E; ++e) {
                if (e % every) continue;   // only every `every`-th expert is far
                for (int p = 0; p < 3; ++p) {
                    const size_t n = 100 + 37 * p + e;
                    pos[l * E + e][p] = { at, n };
                    for (size_t i = 0; i < n; ++i) file.push_back((uint8_t) (l * 131 + e * 7 + p * 3 + i));
                    at += n;
                }
            }
        for (size_t k = 0; k < pos.size(); ++k)
            for (int p = 0; p < 3; ++p)
                if (pos[k][p].second) far[k][p] = { file.data() + pos[k][p].first, pos[k][p].second };
    }
    bool ok(int l, int e, const uint8_t * got) const
    {
        size_t n = 0;
        for (int p = 0; p < 3; ++p)
            for (size_t i = 0; i < 100 + 37 * (size_t) p + e; ++i, ++n)
                if (got[n] != (uint8_t) (l * 131 + e * 7 + p * 3 + i)) return false;
        return true;
    }
};

FarCache::Config cfg(const Fake & f, int slots, int threads = 2)
{
    FarCache::Config c;
    c.n_layer = f.L, c.n_expert = f.E, c.n_slots = slots, c.threads = threads, c.max_hints = 4;
    c.slot_bytes = 3 * 100 + 37 * 3 + f.E * 3;
    return c;
}

FarCache::ReadFn copy_fn(int delay_us = 0, int fail_e = -1, const Fake * f = nullptr)
{
    return [=](const void * src, size_t n, void * dst, void *, size_t) {
        if (delay_us) std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
        if (f && fail_e >= 0 && src == f->far[fail_e][0].src) return false;
        std::memcpy(dst, src, n);
        return true;
    };
}
}  // namespace

int main()
{
    {   // bytes land, hit / miss counts, non-far ignored
        Fake f(2, 8);
        std::vector<uint8_t> region(4 * cfg(f, 4).slot_bytes);
        FarCache c(cfg(f, 4), f.far, region.data(), copy_fn());
        CHECK(c.is_far(1, 2) && !c.is_far(1, 3));
        CHECK(!c.want(1, 3, true));
        CHECK(c.acquire(1, 2) == nullptr);
        const uint8_t * p = c.wait(1, 2, 1000);
        CHECK(p && f.ok(1, 2, p));
        c.release(1, 2);
        CHECK(c.want(1, 2, false));   // READY now
        const uint8_t * q = c.acquire(1, 2);
        CHECK(q == p);
        c.release(1, 2);
        CHECK(c.stats().reads == 1 && c.stats().hits == 1);
    }
    {   // LRU: 2 slots; read A, B; touch A; read C -> B evicted, A stays
        Fake f(1, 8);
        std::vector<uint8_t> region(2 * cfg(f, 2).slot_bytes);
        FarCache c(cfg(f, 2, 1), f.far, region.data(), copy_fn());
        c.wait(0, 0), c.release(0, 0);
        c.wait(0, 2), c.release(0, 2);
        c.acquire(0, 0), c.release(0, 0);
        const uint8_t * p = c.wait(0, 4);
        CHECK(p && f.ok(0, 4, p));
        c.release(0, 4);
        CHECK(c.acquire(0, 2) == nullptr);
        const uint8_t * a = c.acquire(0, 0);
        CHECK(a && f.ok(0, 0, a));
        c.release(0, 0);
        CHECK(c.stats().evictions == 1);
    }
    {   // a held slot is never overwritten: 1 slot held, a demand for another times out, then lands after release
        Fake f(1, 8);
        std::vector<uint8_t> region(cfg(f, 1).slot_bytes);
        FarCache c(cfg(f, 1, 1), f.far, region.data(), copy_fn());
        const uint8_t * p = c.wait(0, 0);
        CHECK(c.wait(0, 2, 50) == nullptr);
        CHECK(f.ok(0, 0, p));
        c.release(0, 0);
        const uint8_t * q = c.wait(0, 2, 1000);
        CHECK(q && f.ok(0, 2, q));
        c.release(0, 2);
    }
    {   // hints: max 4 queued, oldest dropped; a promoted hint survives later overflow
        Fake f(1, 32);
        std::vector<uint8_t> region(cfg(f, 1).slot_bytes);
        FarCache c(cfg(f, 1, 1), f.far, region.data(), copy_fn(20000));   // slow reads: the queue backs up
        const uint8_t * held = c.wait(0, 0);   // the only slot is held: no hint can be read
        for (int e = 2; e <= 12; e += 2) c.want(0, e, false);   // 6 hints, 4 kept
        c.want(0, 10, true);                                    // promote one
        for (int e = 14; e <= 20; e += 2) c.want(0, e, false);  // more overflow
        CHECK(c.stats().hints_dropped >= 2);
        (void) held;
        c.release(0, 0);
        const uint8_t * p = c.wait(0, 10, 2000);   // the promoted one must still come
        CHECK(p && f.ok(0, 10, p));
        if (p) c.release(0, 10);
    }
    {   // a failed read: wait() returns nullptr, the slot is freed, read_fail counted
        Fake f(1, 8);
        std::vector<uint8_t> region(2 * cfg(f, 2).slot_bytes);
        FarCache c(cfg(f, 2, 1), f.far, region.data(), copy_fn(0, 4, &f));
        CHECK(c.wait(0, 4, 1000) == nullptr);
        CHECK(c.stats().read_fail == 1);
        const uint8_t * p = c.wait(0, 6, 1000);
        CHECK(p && f.ok(0, 6, p));
        if (p) c.release(0, 6);
    }
    for (size_t piece : { (size_t) 0, (size_t) 16 }) {   // stress: 6 threads, 40 layers x 64 experts (half far), 24
        // slots, 8 readers, random demand / hint / acquire; whole reads, then 16-byte pieces (every read in many)
        Fake f(40, 64);
        std::vector<uint8_t> region(24 * cfg(f, 24).slot_bytes);
        FarCache::Config fc = cfg(f, 24, 8);
        fc.piece_bytes = piece;
        FarCache c(fc, f.far, region.data(), copy_fn(50));
        std::atomic<int> bad{ 0 }, got{ 0 };
        std::vector<std::thread> ts;
        for (int t = 0; t < 6; ++t)
            ts.emplace_back([&, t] {
                std::mt19937 r(t);
                for (int i = 0; i < 3000; ++i) {
                    const int l = r() % 40, e = 2 * (r() % 32);
                    if (r() % 3 == 0) {
                        c.want(l, (e + 2) % 64, false);
                        continue;
                    }
                    const uint8_t * p = r() % 2 ? c.acquire(l, e) : c.wait(l, e, 1000);
                    if (!p) continue;
                    ++got;
                    std::this_thread::sleep_for(std::chrono::microseconds(r() % 30));   // a "PCIe copy" reads it
                    if (!f.ok(l, e, p)) ++bad;
                    c.release(l, e);
                }
            });
        for (auto & t : ts) t.join();
        CHECK(bad == 0);
        CHECK(got > 1000);
        std::printf("%s\n", c.report().c_str());
    }
    if (fails) {
        std::fprintf(stderr, "expert_far_test: %d checks failed\n", fails);
        return 1;
    }
    std::printf("expert_far_test: ok\n");
    return 0;
}
