// The registry of open GGUF mappings (address range -> descriptor) and the bulk reads through it: read_mapped()
// (buffered pread) and read_direct() (O_DIRECT through a bounce buffer). gguf.cc registers each shard it maps.
#include "formats/gguf.h"
#include "formats/gguf_io.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace truss::gguf {

namespace {
// dfd: an O_DIRECT descriptor of the same file for read_direct(), opened on first use (-1 before, -2 if the file
// system refused O_DIRECT)
struct Registered { void * addr; size_t size; int fd; std::string path; int dfd = -1; uint64_t ident = 0; };
std::mutex & registry_mu() { static std::mutex m; return m; }
std::vector<Registered> & registry() { static std::vector<Registered> r; return r; }
}  // namespace

void register_mapping(void * addr, size_t size, int fd, const std::string & path)
{
    // the file's identity for span_ident(): path, size and modification time (a rewritten pack gets a new one)
    struct stat st {};
    fstat(fd, &st);
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void * b, size_t n) {
        for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t *>(b)[i]) * 1099511628211ull;
    };
    mix(path.data(), path.size());
    mix(&st.st_size, sizeof st.st_size), mix(&st.st_mtim, sizeof st.st_mtim);
    std::lock_guard<std::mutex> g(registry_mu());
    registry().push_back({ addr, size, fd, path, -1, h });
}
void unregister_mapping(void * addr)
{
    std::lock_guard<std::mutex> g(registry_mu());
    auto & r = registry();
    for (const Registered & x : r)
        if (x.addr == addr && x.dfd >= 0) ::close(x.dfd);
    r.erase(std::remove_if(r.begin(), r.end(), [&](const Registered & x) { return x.addr == addr; }), r.end());
}

bool read_mapped(const void * p, size_t n, void * dst)
{
    int fd = -1;
    off_t off = 0;
    {
        std::lock_guard<std::mutex> g(registry_mu());
        for (const auto & m : registry())
            if ((const char *) p >= (const char *) m.addr && (const char *) p + n <= (const char *) m.addr + m.size) {
                fd = m.fd;
                off = (off_t) ((const char *) p - (const char *) m.addr);
                break;
            }
    }
    if (fd < 0) return false;
    auto * d = static_cast<char *>(dst);
    while (n) {
        const ssize_t r = ::pread(fd, d, n, off);
        if (r <= 0) {
            if (r < 0 && errno == EINTR) continue;
            return false;
        }
        d += r, off += r, n -= (size_t) r;
    }
    return true;
}

bool read_direct(const void * p, size_t n, void * dst, void * bounce, size_t bounce_bytes)
{
    constexpr size_t A = DIRECT_ALIGN;
    if (((uintptr_t) bounce % A) || bounce_bytes < 2 * A) return false;
    int dfd = -1;
    off_t off = 0;
    {
        std::lock_guard<std::mutex> g(registry_mu());
        for (auto & m : registry())
            if ((const char *) p >= (const char *) m.addr && (const char *) p + n <= (const char *) m.addr + m.size) {
                if (m.dfd == -1) {
                    m.dfd = ::open(m.path.c_str(), O_RDONLY | O_DIRECT);
                    if (m.dfd < 0) m.dfd = -2;
                }
                dfd = m.dfd;
                off = (off_t) ((const char *) p - (const char *) m.addr);
                break;
            }
    }
    if (dfd < 0) return false;
    auto * d = static_cast<char *>(dst);
    const size_t window = bounce_bytes / A * A;
    while (n) {
        // the aligned window around [off, off + n): O_DIRECT needs the offset, the length and the buffer aligned
        const off_t a0 = off / (off_t) A * (off_t) A;
        const size_t lead = (size_t) (off - a0);
        const size_t want = std::min(window, (lead + n + A - 1) / A * A);
        size_t got = 0;
        while (got < want) {
            const ssize_t r = ::pread(dfd, static_cast<char *>(bounce) + got, want - got, a0 + (off_t) got);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) break;   // end of file: a short tail is fine as long as it covers the bytes we need
            got += (size_t) r;
        }
        if (got <= lead) return false;
        const size_t take = std::min(n, got - lead);
        std::memcpy(d, static_cast<char *>(bounce) + lead, take);
        d += take, off += (off_t) take, n -= take;
    }
    return true;
}

bool span_ident(const void * p, uint64_t & file, uint64_t & offset)
{
    std::lock_guard<std::mutex> g(registry_mu());
    for (const auto & m : registry())
        if ((const char *) p >= (const char *) m.addr && (const char *) p < (const char *) m.addr + m.size) {
            file = m.ident, offset = (uint64_t) ((const char *) p - (const char *) m.addr);
            return true;
        }
    return false;
}

}  // namespace truss::gguf
