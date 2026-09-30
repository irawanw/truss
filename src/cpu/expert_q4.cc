#include "cpu/expert_q4.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace truss::cpu {
namespace {

constexpr int GU_CHUNK = 64;    // gate/up rows per work item (640 / 64 = 10 items per expert)
constexpr int DN_CHUNK = 256;   // down rows per work item (2560 / 256 = 10 items per expert)

inline float fp16(uint16_t h) { return _cvtsh_ss(h); }

inline float hsum(__m256 v)
{
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_hadd_ps(a, a);
    a = _mm_hadd_ps(a, a);
    return _mm_cvtss_f32(a);
}

// per 32-block int8 quantization, d = amax / 127 (the GPU's Q8_1 activations)
void quantize(const float * x, int n, int8_t * q, float * d)
{
    for (int b = 0; b < n / BLOCK; ++b) {
        float m = 0.f;
        for (int i = 0; i < BLOCK; ++i) m = std::max(m, std::fabs(x[BLOCK * b + i]));
        const float s = m / 127.f, inv = s > 0.f ? 1.f / s : 0.f;
        d[b] = s;
        for (int i = 0; i < BLOCK; ++i) q[BLOCK * b + i] = (int8_t) std::lrintf(x[BLOCK * b + i] * inv);
    }
}

// y[i][r] = W[r] . x[i] for rows r in [r0, r1) and R activation rows (int8 [R][cols], scales [R][cols / 32])
void gemv(const Q4Matrix & W, int r0, int r1, const int8_t * const * xq, const float * const * xd, int R, float * y,
          int ld_y)
{
    const __m256i lo = _mm256_set1_epi8(0x0f), eight = _mm256_set1_epi8(8), ones = _mm256_set1_epi16(1);
    const int nb = W.cols / BLOCK;
    for (int r = r0; r < r1; ++r) {
        __m256 acc[MAX_ROWS];
        for (int i = 0; i < R; ++i) acc[i] = _mm256_setzero_ps();
        const uint8_t * q = W.q + (size_t) r * W.cols / 2;
        const uint16_t * s = W.d + (size_t) r * nb;
        for (int b = 0; b < nb; ++b) {
            const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q + 16 * b));
            __m256i w = _mm256_set_m128i(_mm_srli_epi16(raw, 4), raw);    // low nibbles = elements 0..15
            w = _mm256_sub_epi8(_mm256_and_si256(w, lo), eight);         // [-8, 7]
            const __m256i aw = _mm256_sign_epi8(w, w);
            const float ws = fp16(s[b]);
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
    Q4Matrix m{ p, reinterpret_cast<const uint16_t *>(p + (size_t) rows * cols / 2), rows, cols };
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

ExpertPool::ExpertPool(int threads)
{
    pending_.assign(2, 0);
    for (int i = 0; i < threads; ++i) workers_.emplace_back([this] { worker(); });
}

ExpertPool::~ExpertPool()
{
    {
        std::lock_guard<std::mutex> g(mu_);
        stop_ = true;
    }
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
    h_.assign(G * MAX_ROWS * D_FF, 0.f);
    out_.assign(G * MAX_ROWS * D_MODEL, 0.f);
    phase_items_[0] = (int) G * (D_FF / GU_CHUNK);
    phase_items_[1] = (int) G * (D_MODEL / DN_CHUNK);
    pending_[0] = phase_items_[0], pending_[1] = phase_items_[1];
    phase_ = 0, next_item_ = 0;
    busy_ = G > 0;
    ++generation_;
    g.unlock();
    if (G) cv_.notify_all();
}

// phase 0: gate and up rows of one chunk, then h = silu(gate) * up; phase 1: h quantized, down rows of one chunk
void ExpertPool::item(int i)
{
    if (phase_ == 0) {
        const int per = D_FF / GU_CHUNK, gi = i / per, r0 = (i % per) * GU_CHUNK, r1 = r0 + GU_CHUNK;
        const Group & gr = groups_[gi];
        const int R = (int) gr.rows.size();
        const int8_t * xq[MAX_ROWS];
        const float * xd[MAX_ROWS];
        for (int k = 0; k < R; ++k)
            xq[k] = &xq_[(size_t) gr.rows[k] * D_MODEL], xd[k] = &xd_[(size_t) gr.rows[k] * D_MODEL / BLOCK];
        float g[MAX_ROWS * D_FF], u[MAX_ROWS * D_FF];
        gemv(gr.e->gate, r0, r1, xq, xd, R, g, D_FF);
        gemv(gr.e->up, r0, r1, xq, xd, R, u, D_FF);
        float * h = &h_[(size_t) gi * MAX_ROWS * D_FF];
        for (int k = 0; k < R; ++k)
            for (int r = r0; r < r1; ++r) {
                const float a = g[k * D_FF + r];
                h[k * D_FF + r] = a / (1.f + std::exp(-a)) * u[k * D_FF + r];
            }
    } else {
        const int per = D_MODEL / DN_CHUNK, gi = i / per, r0 = (i % per) * DN_CHUNK, r1 = r0 + DN_CHUNK;
        const Group & gr = groups_[gi];
        const int R = (int) gr.rows.size();
        int8_t hq[MAX_ROWS * D_FF];
        float hd[MAX_ROWS * D_FF / BLOCK];
        const int8_t * xq[MAX_ROWS];
        const float * xd[MAX_ROWS];
        const float * h = &h_[(size_t) gi * MAX_ROWS * D_FF];
        for (int k = 0; k < R; ++k) {
            quantize(h + k * D_FF, D_FF, hq + k * D_FF, hd + k * D_FF / BLOCK);
            xq[k] = hq + k * D_FF, xd[k] = hd + k * D_FF / BLOCK;
        }
        gemv(gr.e->down, r0, r1, xq, xd, R, &out_[(size_t) gi * MAX_ROWS * D_MODEL], D_MODEL);
    }
}

void ExpertPool::worker()
{
    std::unique_lock<std::mutex> g(mu_);
    for (;;) {
        cv_.wait(g, [&] { return stop_ || (busy_ && next_item_ < phase_items_[phase_]); });
        if (stop_) return;
        const int i = next_item_++;
        const int ph = phase_;
        ++running_;
        g.unlock();
        item(i);
        g.lock();
        --running_;
        if (--pending_[ph] == 0) {
            if (ph == 0) {
                phase_ = 1, next_item_ = 0;
                cv_.notify_all();
            } else {
                busy_ = false;
            }
            done_cv_.notify_all();   // the caller helps with phase 1 or collects the result
        }
    }
}

void ExpertPool::wait()
{
    {
        std::unique_lock<std::mutex> g(mu_);
        // the caller helps with the items instead of sleeping
        while (busy_) {
            if (next_item_ < phase_items_[phase_]) {
                const int i = next_item_++;
                const int ph = phase_;
                ++running_;
                g.unlock();
                item(i);
                g.lock();
                --running_;
                if (--pending_[ph] == 0) {
                    if (ph == 0) {
                        phase_ = 1, next_item_ = 0;
                        cv_.notify_all();
                    } else {
                        busy_ = false;
                    }
                }
            } else {
                done_cv_.wait(g, [&] { return !busy_ || next_item_ < phase_items_[phase_]; });
            }
        }
    }
    finish();
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
