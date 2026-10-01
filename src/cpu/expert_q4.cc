#include "cpu/expert_q4.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace truss::cpu {
namespace {

// Work-item shape. gate/up and down rows per item (gate/up 640 / gu rows, down 2560 / dn rows). Fewer, larger items
// stream each expert's weights in one pass, which is what the DRAM needs: with 20 items per expert the measured
// in-situ tier reached 23.7 GB/s against the 50.9 GB/s of a probe that gives each thread a whole expert (TRACKER
// #66). Overridable for the sweep; the defaults are what was measured best.
constexpr int GU_CHUNK_DEF = 64, DN_CHUNK_DEF = 256;
constexpr int SPIN_DEF = 4000;   // pause iterations before a worker blocks
int gu_chunk()
{
    static const int v = getenv("TRUSS_CPU_GU_CHUNK") ? atoi(getenv("TRUSS_CPU_GU_CHUNK")) : GU_CHUNK_DEF;
    return v;
}
int dn_chunk()
{
    static const int v = getenv("TRUSS_CPU_DN_CHUNK") ? atoi(getenv("TRUSS_CPU_DN_CHUNK")) : DN_CHUNK_DEF;
    return v;
}

inline float fp16(uint16_t h) { return _cvtsh_ss(h); }

inline float hsum(__m256 v)
{
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_hadd_ps(a, a);
    a = _mm_hadd_ps(a, a);
    return _mm_cvtss_f32(a);
}

// per 32-block int8 quantization, d = amax / 127 (the GPU's Q8_1 activations). AVX2: bit-identical to the
// scalar form (max reduction order-free, cvtps uses MXCSR round-to-nearest-even like lrintf).
void quantize(const float * x, int n, int8_t * q, float * d)
{
    const __m128 sign = _mm_set1_ps(-0.f);
    for (int b = 0; b < n / BLOCK; ++b) {
        const float * xb = x + BLOCK * b;
        __m128 a = _mm_andnot_ps(sign, _mm_loadu_ps(xb));
        a = _mm_max_ps(a, _mm_andnot_ps(sign, _mm_loadu_ps(xb + 4)));
        a = _mm_max_ps(a, _mm_andnot_ps(sign, _mm_loadu_ps(xb + 8)));
        a = _mm_max_ps(a, _mm_andnot_ps(sign, _mm_loadu_ps(xb + 12)));
        a = _mm_max_ps(a, _mm_andnot_ps(sign, _mm_loadu_ps(xb + 16)));
        a = _mm_max_ps(a, _mm_andnot_ps(sign, _mm_loadu_ps(xb + 20)));
        a = _mm_max_ps(a, _mm_andnot_ps(sign, _mm_loadu_ps(xb + 24)));
        a = _mm_max_ps(a, _mm_andnot_ps(sign, _mm_loadu_ps(xb + 28)));
        a = _mm_max_ps(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 0, 3, 2)));
        a = _mm_max_ps(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)));
        const float m = _mm_cvtss_f32(a);
        const float s = m / 127.f, inv = s > 0.f ? 1.f / s : 0.f;
        d[b] = s;
        const __m128 v = _mm_set1_ps(inv);
        for (int k = 0; k < 2; ++k) {
            const __m128i p0 =
                _mm_packs_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(xb + 16 * k), v)),
                                _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(xb + 16 * k + 4), v)));
            const __m128i p1 =
                _mm_packs_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(xb + 16 * k + 8), v)),
                                _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(xb + 16 * k + 12), v)));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(q + BLOCK * b + 16 * k), _mm_packs_epi16(p0, p1));
        }
    }
}

// y[i][r] = W[r] . x[i] for rows r in [r0, r1) and R activation rows (int8 [R][cols], scales [R][cols / 32])
void gemv(const Q4Matrix & W, int r0, int r1, const int8_t * const * xq, const float * const * xd, int R, float * y,
          int ld_y)
{
    const __m256i lo = _mm256_set1_epi8(0x0f), eight = _mm256_set1_epi8(8), ones = _mm256_set1_epi16(1);
    const int nb = W.cols / BLOCK;
    const bool pre = W.df != nullptr;
    for (int r = r0; r < r1; ++r) {
        __m256 acc[MAX_ROWS];
        for (int i = 0; i < R; ++i) acc[i] = _mm256_setzero_ps();
        const uint8_t * q = W.q + (size_t) r * W.cols / 2;
        const uint16_t * s = W.d + (size_t) r * nb;
        const float * sf = pre ? W.df + (size_t) r * nb : nullptr;
        for (int b = 0; b < nb; ++b) {
            const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q + 16 * b));
            __m256i w = _mm256_set_m128i(_mm_srli_epi16(raw, 4), raw);    // low nibbles = elements 0..15
            w = _mm256_sub_epi8(_mm256_and_si256(w, lo), eight);         // [-8, 7]
            const __m256i aw = _mm256_sign_epi8(w, w);
            const float ws = pre ? sf[b] : fp16(s[b]);
            for (int i = 0; i < R; ++i) {
                const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq[i] + BLOCK * b));
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(xv, w)), ones);
                acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(ws * xd[i][b]), acc[i]);
            }
        }
        for (int i = 0; i < R; ++i) y[(size_t) i * ld_y + r] = hsum(acc[i]);
    }
}

Q4Matrix view(const uint8_t *& p, int rows, int cols)
{
    Q4Matrix m{ p, reinterpret_cast<const uint16_t *>(p + (size_t) rows * cols / 2), nullptr, rows, cols };
    p += (size_t) rows * cols / 2 + (size_t) rows * cols / BLOCK * 2;
    return m;
}

}  // namespace

Q4Expert expert_view(const uint8_t * p)
{
    Q4Expert e;
    e.gate = view(p, D_FF, D_MODEL);
    e.up = view(p, D_FF, D_MODEL);
    e.down = view(p, D_MODEL, D_FF);
    return e;
}

void preconvert_scales(const Q4Expert & e, float * out)
{
    for (const Q4Matrix * M : { &e.gate, &e.up, &e.down }) {
        const size_t n = (size_t) M->rows * M->cols / BLOCK;
        for (size_t k = 0; k < n; ++k) out[k] = fp16(M->d[k]);
        const_cast<Q4Matrix *>(M)->df = out;
        out += n;
    }
}

ExpertPool::ExpertPool(int threads)
{
    // NOTE: workers are deliberately *not* pinned: this box always hosts renter load (TRACKER #27) and pinning
    // traps threads on busy cores (measured: pinning cost ~40% at loadavg 28). The scheduler dodges better.
    spin_ = getenv("TRUSS_CPU_SPIN") ? atoi(getenv("TRUSS_CPU_SPIN")) : SPIN_DEF;
    for (int i = 0; i < threads; ++i) workers_.emplace_back([this] { worker(); });
}

ExpertPool::~ExpertPool()
{
    stop_ = true;
    cv_.notify_all();
    for (auto & t : workers_) t.join();
}

void ExpertPool::start(const float * x, int T, const std::vector<Slot> & slots, float * y)
{
    if (T < 1 || T > MAX_ROWS) throw std::invalid_argument("cpu::ExpertPool: 1 .. 8 rows");
    std::unique_lock<std::mutex> g(mu_);
    if (busy_) throw std::logic_error("cpu::ExpertPool::start while busy");
    x_ = x, y_ = y, T_ = T, slots_ = slots;
    groups_.clear();
    for (const Slot & s : slots_) {
        if (s.row < 0 || s.row >= T) throw std::out_of_range("cpu::ExpertPool: slot row");
        auto it = std::find_if(groups_.begin(), groups_.end(), [&](const Group & gr) { return gr.e == s.e; });
        if (it == groups_.end()) groups_.push_back({ s.e, {} }), it = groups_.end() - 1;
        if (std::find(it->rows.begin(), it->rows.end(), s.row) == it->rows.end()) it->rows.push_back(s.row);
    }
    xq_.resize((size_t) T * D_MODEL);
    xd_.resize((size_t) T * D_MODEL / BLOCK);
    for (int t = 0; t < T; ++t) quantize(x + (size_t) t * D_MODEL, D_MODEL, &xq_[(size_t) t * D_MODEL], &xd_[(size_t) t * D_MODEL / BLOCK]);
    const size_t G = groups_.size();
    // grow-only scratch: every cell read below is written first (h/out rows only for rows in each group's list),
    // so the zeroing of assign() is pure overhead, ~100 KB per layer per pass.
    if (h_.size() < G * MAX_ROWS * D_FF) h_.resize(G * MAX_ROWS * D_FF);
    if (hq_.size() < G * MAX_ROWS * D_FF) hq_.resize(G * MAX_ROWS * D_FF);
    if (hd_.size() < G * MAX_ROWS * D_FF / BLOCK) hd_.resize(G * MAX_ROWS * D_FF / BLOCK);
    if (out_.size() < G * MAX_ROWS * D_MODEL) out_.resize(G * MAX_ROWS * D_MODEL);
    phase_items_[0] = (int) G * (D_FF / gu_chunk());
    phase_items_[1] = (int) G * (D_MODEL / dn_chunk());
    pending_[0] = phase_items_[0], pending_[1] = phase_items_[1];
    phase_ = 0, next_item_ = 0;
    ++calls_;
    groups_sum_ += (long long) G;
    slots_sum_ += (long long) slots_.size();
    for (const Group & gr : groups_) rows_sum_ += (long long) gr.rows.size();
    busy_ = G > 0;   // release: the item fields above are visible to any worker that sees busy_
    g.unlock();
    if (G) cv_.notify_all();
}

// phase 0: gate and up rows of one chunk, then h = silu(gate) * up; phase 1: h quantized, down rows of one chunk
void ExpertPool::item(int i)
{
    const auto t0 = std::chrono::steady_clock::now();
    if (phase_ == 0) {
        const int gc = gu_chunk(), per = D_FF / gc, gi = i / per, r0 = (i % per) * gc, r1 = r0 + gc;
        const Group & gr = groups_[gi];
        const int R = (int) gr.rows.size();
        const int8_t * xq[MAX_ROWS];
        const float * xd[MAX_ROWS];
        for (int k = 0; k < R; ++k)
            xq[k] = &xq_[(size_t) gr.rows[k] * D_MODEL], xd[k] = &xd_[(size_t) gr.rows[k] * D_MODEL / BLOCK];
        float g[MAX_ROWS * D_FF], u[MAX_ROWS * D_FF];
        const auto t1 = std::chrono::steady_clock::now();
        gemv(gr.e->gate, r0, r1, xq, xd, R, g, D_FF);
        gemv(gr.e->up, r0, r1, xq, xd, R, u, D_FF);
        const auto t2 = std::chrono::steady_clock::now();
        float * h = &h_[(size_t) gi * MAX_ROWS * D_FF];
        for (int k = 0; k < R; ++k)
            for (int r = r0; r < r1; ++r) {
                const float a = g[k * D_FF + r];
                h[k * D_FF + r] = a / (1.f + std::exp(-a)) * u[k * D_FF + r];
            }
        const auto t3 = std::chrono::steady_clock::now();
        gate_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        silu_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    } else {
        const int dc = dn_chunk(), per = D_MODEL / dc, gi = i / per, r0 = (i % per) * dc, r1 = r0 + dc;
        const Group & gr = groups_[gi];
        const int R = (int) gr.rows.size();
        const int8_t * xq[MAX_ROWS];
        const float * xd[MAX_ROWS];
        for (int k = 0; k < R; ++k) {
            xq[k] = &hq_[((size_t) gi * MAX_ROWS + k) * D_FF];
            xd[k] = &hd_[((size_t) gi * MAX_ROWS + k) * D_FF / BLOCK];
        }
        gemv(gr.e->down, r0, r1, xq, xd, R, &out_[(size_t) gi * MAX_ROWS * D_MODEL], D_MODEL);
        down_us_ += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    }
    item_us_ += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
}

// every group's h rows quantized once, at the phase 0 -> 1 flip (under lock: no phase-1 item has started).
// Before, each of the 10 down-chunk items re-quantized its group's full h rows.
void ExpertPool::quantize_h()
{
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t gi = 0; gi < groups_.size(); ++gi) {
        const Group & gr = groups_[gi];
        for (size_t k = 0; k < gr.rows.size(); ++k)
            quantize(&h_[((size_t) gi * MAX_ROWS + k) * D_FF], D_FF, &hq_[((size_t) gi * MAX_ROWS + k) * D_FF],
                     &hd_[((size_t) gi * MAX_ROWS + k) * D_FF / BLOCK]);
    }
    hq_us_ += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
}

bool ExpertPool::grab(int & i, int & phase)
{
    if (!busy_.load(std::memory_order_acquire)) return false;
    const int ph = phase_.load(std::memory_order_relaxed);
    if (next_item_.load(std::memory_order_relaxed) >= phase_items_[ph]) return false;
    i = next_item_.fetch_add(1, std::memory_order_relaxed);
    phase = ph;
    return i < phase_items_[ph];
}

// the last item of a phase: flip to phase 1 (quantizing h first, so no phase-1 item sees stale hq_) or end the call
void ExpertPool::retire(int ph)
{
    if (pending_[ph].fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    if (ph == 0) {
        quantize_h();
        next_item_.store(0, std::memory_order_relaxed);
        phase_.store(1, std::memory_order_relaxed);
        cv_.notify_all();   // a worker blocked on the flip must see it
    } else {
        busy_.store(false, std::memory_order_release);
    }
    done_cv_.notify_all();   // the caller helps with phase 1 or collects the result
}

void ExpertPool::run_items()
{
    for (int i, ph; grab(i, ph);) {
        item(i);
        retire(ph);
    }
}

void ExpertPool::worker()
{
    for (;;) {
        for (int s = spin_; s > 0; --s) {   // a layer's call is a ~1 ms burst: blocking costs more than spinning
            if (stop_.load(std::memory_order_relaxed) ||
                (busy_.load(std::memory_order_acquire) &&
                 next_item_.load(std::memory_order_relaxed) < phase_items_[phase_.load(std::memory_order_relaxed)]))
                break;
            _mm_pause();
        }
        if (!busy_.load(std::memory_order_acquire) ||
            next_item_.load(std::memory_order_relaxed) >= phase_items_[phase_.load(std::memory_order_relaxed)]) {
            std::unique_lock<std::mutex> g(mu_);
            cv_.wait(g, [&] {
                return stop_.load(std::memory_order_relaxed) ||
                       (busy_.load(std::memory_order_acquire) &&
                        next_item_.load(std::memory_order_relaxed) < phase_items_[phase_.load(std::memory_order_relaxed)]);
            });
            if (stop_.load(std::memory_order_relaxed)) return;
        }
        run_items();
    }
}

void ExpertPool::wait()
{
    const auto t0 = std::chrono::steady_clock::now();
    {
        int i, ph;
        while (grab(i, ph)) {   // the caller is a worker too
            item(i);
            retire(ph);
        }
        std::unique_lock<std::mutex> g(mu_);
        while (busy_.load(std::memory_order_acquire)) done_cv_.wait(g);
    }
    wait_us_ += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    ++waits_;
    finish();
}

void ExpertPool::reset_stats()
{
    wait_us_ = 0;
    waits_ = 0;
    item_us_ = 0;
    gate_us_ = 0;
    silu_us_ = 0;
    hq_us_ = 0;
    down_us_ = 0;
    calls_ = 0;
    groups_sum_ = 0;
    slots_sum_ = 0;
    rows_sum_ = 0;
}

// y[t] = sum over the slots of row t, in slot order, of w * the expert's output row
void ExpertPool::finish()
{
    std::fill(y_, y_ + (size_t) T_ * D_MODEL, 0.f);
    for (const Slot & s : slots_) {
        const int gi = (int) (std::find_if(groups_.begin(), groups_.end(), [&](const Group & gr) { return gr.e == s.e; }) -
                              groups_.begin());
        const std::vector<int> & rows = groups_[gi].rows;
        const int k = (int) (std::find(rows.begin(), rows.end(), s.row) - rows.begin());
        const float * o = &out_[((size_t) gi * MAX_ROWS + k) * D_MODEL];
        float * y = y_ + (size_t) s.row * D_MODEL;
        for (int j = 0; j < D_MODEL; ++j) y[j] += s.w * o[j];
    }
}

}  // namespace truss::cpu
