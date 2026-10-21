// gguf::read_direct (O_DIRECT through a bounce buffer) against the mapping's own bytes: bit-identical for random
// spans of every size class (a few bytes, straddling 4 KiB pages, larger than the bounce buffer, ending at the last
// byte of the file) and odd destination alignments. Also the read rate of whole tensors, direct vs buffered pread.
// usage: read_direct_test <any gguf>
#include "formats/gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace truss;

int main(int argc, char ** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: read_direct_test <gguf>\n");
        return 2;
    }
    const auto file = gguf::File::open(argv[1]);
    const auto & ts = file->tensors();
    constexpr size_t BOUNCE = 1u << 20;   // small, so spans above it take several reads
    void * bounce = nullptr;
    if (posix_memalign(&bounce, gguf::DIRECT_ALIGN, BOUNCE) != 0) return 2;
    std::mt19937_64 rng(1021);
    std::vector<uint8_t> got;
    int fails = 0, n = 0;
    auto check = [&](const std::byte * p, size_t len, size_t dst_pad) {
        got.assign(len + dst_pad, 0xA5);
        if (!gguf::read_direct(p, len, got.data() + dst_pad, bounce, BOUNCE)) {
            std::fprintf(stderr, "read_direct refused a span of %zu bytes\n", len);
            ++fails;
        } else if (std::memcmp(got.data() + dst_pad, p, len) != 0) {
            std::fprintf(stderr, "MISMATCH: span of %zu bytes at %p\n", len, (const void *) p);
            ++fails;
        }
        ++n;
    };
    const size_t sizes[] = { 1, 7, 4095, 4096, 4097, 65537, BOUNCE - 1, BOUNCE, BOUNCE + 3, 3 * BOUNCE + 4099 };
    for (int i = 0; i < 400; ++i) {
        const gguf::Tensor & t = ts[rng() % ts.size()];
        const size_t len = std::min<size_t>(sizes[i % 10], t.bytes);
        if (!len) continue;
        const size_t at = (size_t) (rng() % (t.bytes - len + 1));
        check(t.data + at, len, (size_t) (rng() % 64));
    }
    // the file's last bytes: the aligned window runs past the end of the file
    const gguf::Tensor & last = *std::max_element(ts.begin(), ts.end(), [](const auto & a, const auto & b) {
        return a.shard != b.shard ? a.shard < b.shard : a.file_offset < b.file_offset;
    });
    for (size_t len : { (size_t) 1, (size_t) 5000, std::min<size_t>(last.bytes, 2 * BOUNCE + 1) })
        if (len <= last.bytes) check(last.data + last.bytes - len, len, 3);
    std::printf("read_direct: %d of %d spans identical\n", n - fails, n);

    // rate: the largest tensors, up to 4 GB, read once direct and once buffered (the buffered pass reads them after
    // the direct pass, so it may hit the page cache the direct pass did not fill: an upper bound for buffered)
    std::vector<const gguf::Tensor *> big;
    for (const auto & t : ts) big.push_back(&t);
    std::sort(big.begin(), big.end(), [](auto a, auto b) { return a->bytes > b->bytes; });
    std::vector<uint8_t> buf;
    size_t total = 0;
    std::vector<const gguf::Tensor *> pick;
    for (auto * t : big) {
        if (total + t->bytes > (4ull << 30)) continue;
        pick.push_back(t), total += t->bytes;
    }
    constexpr size_t RB = 16u << 20;
    void * rbounce = nullptr;
    if (posix_memalign(&rbounce, gguf::DIRECT_ALIGN, RB) != 0) return 2;
    for (int mode = 0; mode < 2; ++mode) {
        const auto t0 = std::chrono::steady_clock::now();
        for (auto * t : pick) {
            buf.resize(t->bytes);
            const bool ok = mode == 0 ? gguf::read_direct(t->data, t->bytes, buf.data(), rbounce, RB)
                                      : gguf::read_mapped(t->data, t->bytes, buf.data());
            if (!ok) ++fails;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%s: %.2f GB in %.2f s, %.2f GB/s (one thread)\n", mode == 0 ? "direct" : "buffered", total / 1e9,
                    s, total / 1e9 / s);
    }
    std::free(bounce), std::free(rbounce);
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
