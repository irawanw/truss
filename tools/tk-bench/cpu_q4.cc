// CPU expert throughput probe for a CPU miss tier: one routed expert (gate, up 2560 -> 640, down 640 -> 2560) with
// 4-bit weights (blocks of 32: 16 bytes of nibbles + one fp32 scale, 0.625 B/weight incl. scale) against int8
// activations (per-32 scale), AVX2 maddubs, for a window of R rows. Many distinct experts (more than the L3 holds)
// are cycled so every expert streams from DRAM, as misses would. Threads take whole experts.
// usage: tk-bench-cpu-q4 [threads=8] [rows=4] [experts in the pool=256] [seconds=3]
#include <immintrin.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace {

constexpr int D = 2560, F = 640, B = 32;

struct Mat {                      // rows x cols, blocks of 32 along cols
    int rows, cols;
    std::vector<uint8_t> q;        // rows * cols / 2
    std::vector<float> s;          // rows * cols / 32
};

Mat make(int rows, int cols, std::mt19937 & rng)
{
    Mat m{ rows, cols, std::vector<uint8_t>((size_t) rows * cols / 2), std::vector<float>((size_t) rows * cols / B) };
    for (auto & b : m.q) b = (uint8_t) rng();
    for (auto & x : m.s) x = 0.01f;
    return m;
}

inline float hsum(__m256 v)
{
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_hadd_ps(a, a);
    a = _mm_hadd_ps(a, a);
    return _mm_cvtss_f32(a);
}

// y [R][rows] = W x, x as int8 [R][cols] with scales xs [R][cols / 32]
void gemv(const Mat & W, const int8_t * x, const float * xs, int R, float * y)
{
    const __m256i lo = _mm256_set1_epi8(0x0f), eight = _mm256_set1_epi8(8), ones = _mm256_set1_epi16(1);
    const int nb = W.cols / B;
    for (int r = 0; r < W.rows; ++r) {
        __m256 acc[8];
        for (int i = 0; i < R; ++i) acc[i] = _mm256_setzero_ps();
        const uint8_t * q = W.q.data() + (size_t) r * W.cols / 2;
        const float * s = W.s.data() + (size_t) r * nb;
        for (int b = 0; b < nb; ++b) {
            const __m128i raw = _mm_loadu_si128((const __m128i *) (q + 16 * b));
            __m256i w = _mm256_set_m128i(_mm_srli_epi16(raw, 4), raw);
            w = _mm256_sub_epi8(_mm256_and_si256(w, lo), eight);   // [-8, 7]
            const __m256i aw = _mm256_sign_epi8(w, w);
            for (int i = 0; i < R; ++i) {
                const __m256i xv = _mm256_loadu_si256((const __m256i *) (x + (size_t) i * W.cols + B * b));
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(xv, w)), ones);
                acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(s[b] * xs[(size_t) i * nb + b]), acc[i]);
            }
        }
        for (int i = 0; i < R; ++i) y[(size_t) i * W.rows + r] = hsum(acc[i]);
    }
}

void quant(const float * x, int n, int8_t * q, float * s)
{
    for (int b = 0; b < n / B; ++b) {
        float m = 0;
        for (int i = 0; i < B; ++i) m = std::max(m, std::abs(x[B * b + i]));
        const float d = m / 127.f, inv = d ? 1.f / d : 0.f;
        s[b] = d;
        for (int i = 0; i < B; ++i) q[B * b + i] = (int8_t) std::lrint(x[B * b + i] * inv);
    }
}

void expert(const Mat & g, const Mat & u, const Mat & dn, const int8_t * x, const float * xs, int R, float * y)
{
    std::vector<float> a((size_t) R * F), b((size_t) R * F);
    gemv(g, x, xs, R, a.data());
    gemv(u, x, xs, R, b.data());
    for (size_t i = 0; i < a.size(); ++i) a[i] = a[i] / (1.f + std::exp(-a[i])) * b[i];
    std::vector<int8_t> hq((size_t) R * F);
    std::vector<float> hs((size_t) R * F / B);
    for (int i = 0; i < R; ++i) quant(a.data() + (size_t) i * F, F, hq.data() + (size_t) i * F, hs.data() + (size_t) i * F / B);
    gemv(dn, hq.data(), hs.data(), R, y);
}

}  // namespace

int main(int argc, char ** argv)
{
    const int threads = argc > 1 ? std::atoi(argv[1]) : 8, R = argc > 2 ? std::atoi(argv[2]) : 4;
    const int pool = argc > 3 ? std::atoi(argv[3]) : 256;
    const double secs = argc > 4 ? std::atof(argv[4]) : 3.0;
    std::mt19937 rng(1);
    struct E { Mat g, u, d; };
    std::vector<E> ex;
    for (int i = 0; i < pool; ++i) ex.push_back({ make(F, D, rng), make(F, D, rng), make(D, F, rng) });
    const double mb = (double) (3 * (size_t) D * F) * 0.625 / 1e6;
    std::vector<float> xf((size_t) R * D);
    for (auto & v : xf) v = std::uniform_real_distribution<float>(-1, 1)(rng);
    std::vector<int8_t> xq(xf.size());
    std::vector<float> xs(xf.size() / B);
    for (int i = 0; i < R; ++i) quant(xf.data() + (size_t) i * D, D, xq.data() + (size_t) i * D, xs.data() + (size_t) i * D / B);
    std::atomic<long> next{ 0 }, done{ 0 };
    std::atomic<bool> stop{ false };
    std::vector<std::thread> th;
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < threads; ++t)
        th.emplace_back([&] {
            std::vector<float> y((size_t) R * D);
            while (!stop.load(std::memory_order_relaxed)) {
                const long i = next.fetch_add(1);
                const E & e = ex[(size_t) (i * 7919) % pool];
                expert(e.g, e.u, e.d, xq.data(), xs.data(), R, y.data());
                done.fetch_add(1);
            }
        });
    std::this_thread::sleep_for(std::chrono::duration<double>(secs));
    stop = true;
    for (auto & t : th) t.join();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("threads %d, rows %d: %.1f experts/ms (%.2f MB each) = %.1f GB/s of weights, %.1f us per expert per thread\n",
                threads, R, done / s / 1e3, mb, done * mb / 1e3 / s, s * 1e6 * threads / done);
    return 0;
}
