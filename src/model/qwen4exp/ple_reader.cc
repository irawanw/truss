#include "model/qwen4exp/ple_reader.h"

#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <thread>
#include <string>

namespace truss::qwen4exp {
namespace {

constexpr uint64_t PAGE = 4096;

// buffered by default: pages already in the page cache cost microseconds, and only the misses wait for the SSD
// (O_DIRECT, Strata's choice, re-reads them all; TRUSS_PLE_ODIRECT=1 for the A/B)
int open_direct(const std::string & path)
{
    static const bool odirect = std::getenv("TRUSS_PLE_ODIRECT") && std::atoi(std::getenv("TRUSS_PLE_ODIRECT"));
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (odirect ? O_DIRECT : 0));
    if (fd < 0) throw std::runtime_error("PleReader: cannot open " + path + ": " + std::strerror(errno));
    return fd;
}

uint64_t key(int fd, uint64_t page) { return (uint64_t) fd << 48 | page; }

// AVX2 + F16C: 16 int8 -> 16 fp16 per iteration; bit-identical to the scalar loop (int8 -> float is exact,
// vmulps is IEEE RNE, vcvtps2ph rounds to nearest even like __float2half; denormals are kept). Zen 2 has both.
__attribute__((target("avx2,f16c")))
void put_row(const int8_t * b, float sf, half * o, int n)
{
    const __m256 vsf = _mm256_set1_ps(sf);
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        const __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + j));
        const __m256i w16 = _mm256_cvtepi8_epi16(in);
        const __m256 flo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(w16))), vsf);
        const __m256 fhi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(w16, 1))), vsf);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o + j),
                         _mm256_cvtps_ph(flo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o + j + 8),
                         _mm256_cvtps_ph(fhi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
    for (; j < n; ++j) o[j] = __float2half(b[j] * sf);
}

}  // namespace

PleReader::PleReader(const gguf::File & file, const gguf::Tensor & table, const gguf::Tensor & scale, int row_bytes,
                     int threads, size_t cache_rows)
    : row_bytes_(row_bytes), table_off_(table.file_offset), scale_off_(scale.file_offset), cap_(cache_rows)
{
    fd_table_ = open_direct(file.shard_paths().at(table.shard));
    fd_scale_ = scale.shard == table.shard ? fd_table_ : open_direct(file.shard_paths().at(scale.shard));
    cache_row_.assign(cap_, -1);
    cache_bytes_.resize(cap_ * row_bytes_);
    cache_scale_.resize(cap_);
    ref_.assign(cap_, 0);
    if (const char * e = std::getenv("TRUSS_PLE_THREADS")) threads = std::max(1, std::atoi(e));
    for (int i = 0; i < threads; ++i) pool_.emplace_back([this] { worker(); });
}

PleReader::~PleReader()
{
    {
        std::lock_guard<std::mutex> g(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto & t : pool_) t.join();
    for (uint8_t * b : bufs_) std::free(b);
    if (fd_scale_ >= 0 && fd_scale_ != fd_table_) ::close(fd_scale_);
    if (fd_table_ >= 0) ::close(fd_table_);
}

void PleReader::worker()
{
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
            if (stop_) return;
            seen = gen_;
        }
        size_t done = 0;
        int err = 0;
        for (size_t i; (i = next_.fetch_add(1)) < n_pages_;) {
            Page p;
            { std::lock_guard<std::mutex> g(mu_); p = pages_[i]; }   // append() may grow pages_ concurrently (1.B)
            const ssize_t got = ::pread(p.fd, p.buf, PAGE, (off_t) p.off);
            if (got < 0) err = errno;   // a short read is the file's end: the rows never reach it
            ++done;
        }
        std::lock_guard<std::mutex> g(mu_);
        if (err) io_errno_ = err;
        finished_ += done;
        if (finished_ == n_pages_) done_cv_.notify_all();
    }
}

int PleReader::begin(bool cache_misses)
{
    {   // one ticket in flight: the previous one's reads write into buffers this one reuses
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [&] { return finished_ == n_pages_; });
        rows_.clear();
        slot_.clear();
        pages_.clear();
        page_at_.clear();
        n_pages_ = 0;
        finished_ = 0;
        next_ = 0;
        buf_used_ = 0;
        io_errno_ = 0;
        cache_misses_ = cache_misses;
    }
    return ++ticket_;
}

void PleReader::append(int ticket, const int32_t * rows, size_t n)
{
    if (ticket != ticket_) throw std::logic_error("PleReader::append: not the ticket in flight");
    std::unique_lock<std::mutex> lk(mu_);   // pages_ grows here; workers copy pages_[i] under the same lock
    const size_t old_pages = pages_.size();
    rows_.insert(rows_.end(), rows, rows + n);
    slot_.resize(rows_.size(), -1);
    auto need = [&](int fd, uint64_t off) {
        const uint64_t pg = off / PAGE;
        auto [it, fresh] = page_at_.try_emplace(key(fd, pg), nullptr);
        if (!fresh) return;
        if (buf_used_ == bufs_.size()) {
            void * b = nullptr;
            if (posix_memalign(&b, PAGE, PAGE)) throw std::bad_alloc();
            bufs_.push_back(static_cast<uint8_t *>(b));
        }
        it->second = bufs_[buf_used_++];
        pages_.push_back({ fd, pg * PAGE, it->second });
    };
    for (size_t i = 0; i < n; ++i) {
        const int32_t r = rows[i];
        if (cap_) {
            auto it = where_.find(r);
            if (it != where_.end()) {
                slot_[rows_.size() - n + i] = it->second;
                ref_[it->second] = 1;
                ++stats_.cache_hits;
                continue;
            }
        }
        const uint64_t a = table_off_ + (uint64_t) r * row_bytes_;
        need(fd_table_, a);
        need(fd_table_, a + row_bytes_ - 1);   // a row may straddle two pages
        need(fd_scale_, scale_off_ + (uint64_t) r * sizeof(half));
    }
    stats_.rows += n;
    stats_.pages += pages_.size() - old_pages;
    const bool more = pages_.size() > n_pages_;
    n_pages_ = pages_.size();
    ++gen_;
    lk.unlock();
    if (more) cv_.notify_all();
}

int PleReader::issue(const int32_t * rows, size_t n, bool cache_misses)
{
    const int t = begin(cache_misses);
    append(t, rows, n);
    return t;
}

const uint8_t * PleReader::page_of(int fd, uint64_t off) const
{
    return page_at_.at(key(fd, off / PAGE)) + off % PAGE;
}

int PleReader::insert(int32_t row, const int8_t * bytes, half scale)
{
    while (ref_[hand_]) ref_[hand_] = 0, hand_ = (hand_ + 1) % cap_;   // clock: a second chance for used slots
    const int s = (int) hand_;
    hand_ = (hand_ + 1) % cap_;
    if (cache_row_[s] >= 0) where_.erase(cache_row_[s]);
    cache_row_[s] = row;
    where_[row] = s;
    std::memcpy(&cache_bytes_[(size_t) s * row_bytes_], bytes, row_bytes_);
    cache_scale_[s] = scale;
    return s;
}

void PleReader::copy_put_rows(size_t lo, size_t hi, half * emb, double & ms) const
{
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int8_t> row(row_bytes_);
    for (size_t i = lo; i < hi; ++i) {
        if (slot_[i] >= 0) {
            put_row(&cache_bytes_[(size_t) slot_[i] * row_bytes_], __half2float(cache_scale_[slot_[i]]),
                    emb + i * row_bytes_, row_bytes_);
            continue;
        }
        const uint64_t a = table_off_ + (uint64_t) rows_[i] * row_bytes_;
        const uint64_t split = std::min<uint64_t>(row_bytes_, PAGE - a % PAGE);   // bytes in the first page
        std::memcpy(row.data(), page_of(fd_table_, a), split);
        if (split < (uint64_t) row_bytes_) std::memcpy(row.data() + split, page_of(fd_table_, a + split), row_bytes_ - split);
        half s;
        std::memcpy(&s, page_of(fd_scale_, scale_off_ + (uint64_t) rows_[i] * sizeof(half)), sizeof(half));
        put_row(row.data(), __half2float(s), emb + i * row_bytes_, row_bytes_);
    }
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void PleReader::collect(int ticket, half * emb)
{
    if (ticket != ticket_) throw std::logic_error("PleReader::collect: not the ticket in flight");
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [&] { return finished_ == n_pages_; });
        stats_.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (io_errno_) throw std::runtime_error(std::string("PleReader: read failed: ") + std::strerror(io_errno_));
    }
    const size_t n = rows_.size();
    // A prefill ticket inserts nothing, so the row cache and this ticket's pages are read-only here: copy+put
    // fans out over ranges (hits and misses fused per row). Only decode tickets need the serial hits-first order.
    if (!cache_misses_ && n >= 4096) {
        constexpr unsigned P = 16;
        const size_t span = (n + P - 1) / P;
        std::vector<std::thread> th;
        std::vector<double> ms(P, 0.0);
        std::vector<std::exception_ptr> errs(P);
        const auto tc0 = std::chrono::steady_clock::now();
        for (unsigned p = 1; p < P; ++p)
            th.emplace_back([this, p, span, n, emb, &ms, &errs] {
                try { copy_put_rows((size_t) p * span, std::min((size_t) p * span + span, n), emb, ms[p]); }
                catch (...) { errs[p] = std::current_exception(); }
            });
        try { copy_put_rows(0, std::min(span, n), emb, ms[0]); } catch (...) { errs[0] = std::current_exception(); }
        for (auto & t : th) t.join();
        for (auto & e : errs)
            if (e) std::rethrow_exception(e);
        stats_.phase_ms[1] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0).count();
        return;
    }
    auto put = [&](size_t i, const int8_t * b, half s) { put_row(b, __half2float(s), emb + i * row_bytes_, row_bytes_); };
    // cache hits first: inserting the misses may evict a slot a hit of this ticket still points at
    const auto th0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n; ++i)
        if (slot_[i] >= 0) put(i, &cache_bytes_[(size_t) slot_[i] * row_bytes_], cache_scale_[slot_[i]]);
    std::vector<int8_t> row(row_bytes_);
    stats_.phase_ms[0] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - th0).count();
    double copy_ms = 0, ins_ms = 0;   // phases 1-2, folded into stats_ once (two clock reads per miss row)
    auto tc = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n; ++i) {
        if (slot_[i] >= 0) continue;
        const uint64_t a = table_off_ + (uint64_t) rows_[i] * row_bytes_;
        const uint64_t split = std::min<uint64_t>(row_bytes_, PAGE - a % PAGE);   // bytes in the first page
        std::memcpy(row.data(), page_of(fd_table_, a), split);
        if (split < (uint64_t) row_bytes_) std::memcpy(row.data() + split, page_of(fd_table_, a + split), row_bytes_ - split);
        half s;
        std::memcpy(&s, page_of(fd_scale_, scale_off_ + (uint64_t) rows_[i] * sizeof(half)), sizeof(half));
        put(i, row.data(), s);
        const auto ti = std::chrono::steady_clock::now();
        copy_ms += std::chrono::duration<double, std::milli>(ti - tc).count();
        if (cache_misses_ && cap_ && !where_.count(rows_[i])) insert(rows_[i], row.data(), s);
        tc = std::chrono::steady_clock::now();
        ins_ms += std::chrono::duration<double, std::milli>(tc - ti).count();
    }
    stats_.phase_ms[1] += copy_ms, stats_.phase_ms[2] += ins_ms;
}

double PleReader::take_wait_ms() const
{
    std::lock_guard<std::mutex> lk(mu_);
    const double w = stats_.wait_ms;
    stats_.wait_ms = 0;
    return w;
}

void PleReader::take_phase_ms(double out[3]) const
{
    std::lock_guard<std::mutex> lk(mu_);
    for (int i = 0; i < 3; ++i) out[i] = stats_.phase_ms[i], stats_.phase_ms[i] = 0;
}

}  // namespace truss::qwen4exp
