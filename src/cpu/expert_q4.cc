#include "cpu/expert_q4.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <cstdio>
#include <pthread.h>
#include <sched.h>

namespace truss::cpu {
namespace {

// Work-item shape. gate/up and down rows per item (gate/up 640 / gu rows, down 2560 / dn rows). Fewer, larger items
// stream each expert's weights in one pass, which is what the DRAM needs: with 20 items per expert the measured
// in-situ tier reached 23.7 GB/s against the 50.9 GB/s of a probe that gives each thread a whole expert (TRACKER
// #66). Overridable for the sweep; the defaults are what was measured best.
constexpr int GU_CHUNK_DEF = 64, DN_CHUNK_DEF = 256;
// microseconds an idle worker spins before it blocks: Strata's kSpinBeforeSleep (20 ms). Decode calls the pool every
// ~0.5-1 ms; the old 40,000 pauses (~0.65 ms on Zen 2) let the workers fall asleep between layers, and start() then
// spent 0.27 ms per layer on the mutex and the futex wakes (13.9 ms per pass, TRACKER #77)
constexpr int SPIN_US_DEF = 20000;
// trellis items are input-split (TRACKER #82): an item reads one contiguous run of k-slices of one matrix and
// prepares only its own inputs (the input Hadamard is per 128-block), so there is no serial preparation step and no
// strided column walk (D0: column items of 128 cost 1.25x the thread time of contiguous ones, and the two serial
// preparation steps 40 us of a 211 us two-expert call). Gate/up: TR_GU_IN inputs x all 640 columns per item. Down:
// one 128-input block (its h slice is rebuilt by the item from the gate/up partials: their output Hadamard is per
// 128-block too) x TR_DN_COLS columns. Overridable (TRUSS_CPU_TR_GU_IN, TRUSS_CPU_TR_DN_COLS) for the sweep.
// Phase 2 (the end of the call): item = 128 output columns, for every slot: down's partials summed in block order,
// output transform, weighted add into y in slot order (was serial in the caller: 13 us per expert, 20% of a call).
constexpr int TR_GU_IN_DEF = 256, TR_DN_COLS_DEF = 640, TR_DN_IN = 128, TR_OUT_COLS = 128;
int tr_gu_in()
{
    static const int v = getenv("TRUSS_CPU_TR_GU_IN") ? atoi(getenv("TRUSS_CPU_TR_GU_IN")) : TR_GU_IN_DEF;
    return v;
}
int tr_dn_cols()
{
    static const int v = getenv("TRUSS_CPU_TR_DN_COLS") ? atoi(getenv("TRUSS_CPU_TR_DN_COLS")) : TR_DN_COLS_DEF;
    return v;
}
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

// one logical CPU per physical core this process may run on (the first allowed SMT sibling), from sysfs, as Strata's
// pool (kernels/cpu/pool.cpp physical_cores): two workers on SMT siblings share one core's pipes. The siblings follow.
static std::vector<int> physical_cores()
{
    std::vector<int> out;
    std::vector<std::pair<long, long>> seen;
    std::vector<std::pair<std::pair<long, long>, int>> siblings;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return out;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &set)) continue;
        auto topo = [c](const char * what) {
            char path[96];
            std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", c, what);
            long v = -1;
            if (FILE * f = std::fopen(path, "r")) {
                if (std::fscanf(f, "%ld", &v) != 1) v = -1;
                std::fclose(f);
            }
            return v;
        };
        const std::pair<long, long> key{ topo("physical_package_id"), topo("core_id") };
        if (std::find(seen.begin(), seen.end(), key) != seen.end()) {
            siblings.push_back({ key, c });
            continue;
        }
        seen.push_back(key);
        out.push_back(c);
    }
    // then the SMT siblings, in core order, without core 0's (the driver thread's core): pool sizes above the
    // physical core count pin their extra workers there (TRACKER #93)
    const auto core0 = seen.empty() ? std::pair<long, long>{ -1, -1 } : seen[0];
    for (const auto & [key, c] : siblings)
        if (key != core0) out.push_back(c);
    return out;
}

ExpertPool::ExpertPool(int threads)
{
    // Pinning (TRUSS_CPU_PIN=1): worker i on its own physical core, skipping core 0 (the driver thread's), as Strata
    // does. Off by default: the renters' threads float over every core, and pinning measured ~40% slower at loadavg
    // 28 with the q4s kernel (TRACKER #27); TRACKER #76 re-measures it with the int16 trellis kernel.
    spin_ = getenv("TRUSS_CPU_SPIN_US") ? atoi(getenv("TRUSS_CPU_SPIN_US")) : SPIN_US_DEF;
    const bool pin = getenv("TRUSS_CPU_PIN") && atoi(getenv("TRUSS_CPU_PIN"));
    const std::vector<int> cores = pin ? physical_cores() : std::vector<int>{};
    for (int i = 0; i < threads; ++i) {
        const int core = i + 1 < (int) cores.size() ? cores[i + 1] : -1;
        workers_.emplace_back([this, core] {
            if (core >= 0) {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(core, &set);
                pthread_setaffinity_np(pthread_self(), sizeof set, &set);
            }
            worker();
        });
    }
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
    if (T < 1) throw std::invalid_argument("cpu::ExpertPool: no rows");
    // no lock: no worker reads the call's fields while busy_ is false (a late grab() takes an index of this call only
    // after ticket_ is reset below, i.e. after the fields are written); wake() handles sleepers
    if (busy_) throw std::logic_error("cpu::ExpertPool::start while busy");
    x_ = x, y_ = y, T_ = T, slots_ = slots;
    groups_.clear();
    trellis_ = !slots_.empty() && slots_[0].t != nullptr;
    // groups in first-slot order; an expert with more than MAX_ROWS rows (prompt chunks, TRACKER #117) takes one group
    // per MAX_ROWS of them. Each slot's group and its row index in it.
    slot_g_.resize(slots_.size()), slot_k_.resize(slots_.size());
    std::unordered_map<const void *, std::vector<int>> of;   // expert -> its groups
    for (size_t i = 0; i < slots_.size(); ++i) {
        const Slot & s = slots_[i];
        if (s.row < 0 || s.row >= T) throw std::out_of_range("cpu::ExpertPool: slot row");
        if ((s.t != nullptr) != trellis_) throw std::invalid_argument("cpu::ExpertPool: q4s and trellis slots mixed");
        std::vector<int> & gs = of[trellis_ ? (const void *) s.t : (const void *) s.e];
        int g = -1, k = -1;
        for (int gi : gs) {   // a row routed twice to the same expert shares its row
            const std::vector<int> & rows = groups_[gi].rows;
            const auto it = std::find(rows.begin(), rows.end(), s.row);
            if (it != rows.end()) g = gi, k = (int) (it - rows.begin());
        }
        if (g < 0) {
            if (gs.empty() || groups_[gs.back()].rows.size() == MAX_ROWS)
                gs.push_back((int) groups_.size()), groups_.push_back({ s.e, s.t, {} });
            g = gs.back(), k = (int) groups_[g].rows.size();
            groups_[g].rows.push_back(s.row);
        }
        slot_g_[i] = g, slot_k_[i] = k;
    }
    if (!trellis_) {
        xq_.resize((size_t) T * D_MODEL);
        xd_.resize((size_t) T * D_MODEL / BLOCK);
        for (int t = 0; t < T; ++t)
            quantize(x + (size_t) t * D_MODEL, D_MODEL, &xq_[(size_t) t * D_MODEL], &xd_[(size_t) t * D_MODEL / BLOCK]);
    }
    const size_t G = groups_.size();
    // grow-only scratch: every cell read below is written first (h/out rows only for rows in each group's list),
    // so the zeroing of assign() is pure overhead, ~100 KB per layer per pass.
    if (h_.size() < G * MAX_ROWS * D_FF) h_.resize(G * MAX_ROWS * D_FF);
    if (hq_.size() < G * MAX_ROWS * D_FF) hq_.resize(G * MAX_ROWS * D_FF);
    if (hd_.size() < G * MAX_ROWS * D_FF / BLOCK) hd_.resize(G * MAX_ROWS * D_FF / BLOCK);
    if (out_.size() < G * MAX_ROWS * D_MODEL) out_.resize(G * MAX_ROWS * D_MODEL);
    if (trellis_) {   // 0: gate/up partials per input block, 1: down partials per (h block, column range)
        if (D_MODEL % tr_gu_in() || tr_gu_in() % 128 || D_MODEL % tr_dn_cols() || tr_dn_cols() % 128)
            throw std::invalid_argument("cpu::ExpertPool: TRUSS_CPU_TR_GU_IN / TR_DN_COLS");
        const size_t nbg = D_MODEL / tr_gu_in(), nbd = D_FF / TR_DN_IN;
        if (pg_.size() < G * 2 * nbg * MAX_ROWS * D_FF) pg_.resize(G * 2 * nbg * MAX_ROWS * D_FF);
        if (pd_.size() < G * nbd * MAX_ROWS * D_MODEL) pd_.resize(G * nbd * MAX_ROWS * D_MODEL);
        n_phases_ = 3;
        phase_items_[0] = (int) (G * 2 * nbg);
        phase_items_[1] = (int) (G * nbd * (D_MODEL / tr_dn_cols()));
        phase_items_[2] = D_MODEL / TR_OUT_COLS;
    } else {
        n_phases_ = 2;
        phase_items_[0] = (int) G * (D_FF / gu_chunk());
        phase_items_[1] = (int) G * (D_MODEL / dn_chunk());
        phase_items_[2] = 0;
    }
    for (int p = 0; p < 3; ++p) pending_[p] = phase_items_[p];
    ticket_ = 0;
    ++calls_;
    groups_sum_ += (long long) G;
    slots_sum_ += (long long) slots_.size();
    for (const Group & gr : groups_) rows_sum_ += (long long) gr.rows.size();
    t_start_ = std::chrono::steady_clock::now();
    last_ms_ = 0;
    busy_ = G > 0;   // seq_cst: the item fields above are visible to any worker that sees busy_
    if (G) wake();
}

void ExpertPool::wake()
{
    if (sleepers_.load() == 0) return;
    { std::lock_guard<std::mutex> g(mu_); }   // a sleeper between its predicate check and its wait holds mu_
    cv_.notify_all();
}

// trellis items (input-split, see TR_GU_IN_DEF). Fixed summation orders throughout: the result does not depend on
// which thread ran which item.
// Phase 0, item (group, gate|up, input block b): prepare the group's rows over inputs [b * TR_GU_IN, +TR_GU_IN) and
// write the raw partial product, all 640 columns, into pg_.
// Phase 1, item (group, h block b of 128, column range): h[b] = silu(gate) * up on that block, where gate =
// svh * H128(sum of gate partials in block order) (the output Hadamard is per 128-block, so the block is
// self-contained); prepare it and write down's raw partial over that input block into pd_.
// Phase 2, item (128 output columns): every slot's down output on those columns = svh * H128(sum of down's partials
// in block order), added into y with its weight in slot order (rows without slots: 0).
void ExpertPool::item_trellis(int i, int phase)
{
    const auto t0 = std::chrono::steady_clock::now();
    thread_local std::vector<float> P;   // the item's prepared rows
    const int nbg = D_MODEL / tr_gu_in(), nbd = D_FF / TR_DN_IN;
    if (phase == 2) {
        const int c0 = i * TR_OUT_COLS;
        for (int t = 0; t < T_; ++t) std::fill(y_ + (size_t) t * D_MODEL + c0, y_ + (size_t) t * D_MODEL + c0 + TR_OUT_COLS, 0.f);
        alignas(32) float c[TR_OUT_COLS], o[TR_OUT_COLS];
        for (size_t si = 0; si < slots_.size(); ++si) {
            const int gi = slot_g_[si], k = slot_k_[si];
            std::fill(c, c + TR_OUT_COLS, 0.f);
            for (int b = 0; b < nbd; ++b) {   // block order
                const float * p = &pd_[(((size_t) gi * nbd + b) * MAX_ROWS + k) * D_MODEL + c0];
                for (int j = 0; j < TR_OUT_COLS; j += 8)
                    _mm256_store_ps(c + j, _mm256_add_ps(_mm256_load_ps(c + j), _mm256_loadu_ps(p + j)));
            }
            trellis_out(groups_[gi].t->down, c, 1, TR_OUT_COLS, c0, c0 + TR_OUT_COLS, o, TR_OUT_COLS);
            float * y = y_ + (size_t) slots_[si].row * D_MODEL + c0;
            const __m256 w = _mm256_set1_ps(slots_[si].w);
            for (int j = 0; j < TR_OUT_COLS; j += 8) _mm256_storeu_ps(y + j, _mm256_fmadd_ps(w, _mm256_load_ps(o + j), _mm256_loadu_ps(y + j)));
        }
        hq_us_ += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    } else if (phase == 0) {
        const int per = 2 * nbg, gi = i / per, r = i % per, up = r >= nbg, b = r % nbg, IN = tr_gu_in();
        const Group & gr = groups_[gi];
        const int R = (int) gr.rows.size();
        const TrellisMat W = trellis_inputs(up ? gr.t->up : gr.t->gate, b * IN, (b + 1) * IN);
        const size_t step = (size_t) trellis_prep_floats(IN, 1);
        if (P.size() < step * MAX_ROWS) P.resize(step * MAX_ROWS);
        for (int k = 0; k < R; ++k) trellis_prep(W, x_ + (size_t) gr.rows[k] * D_MODEL + b * IN, D_MODEL, 1, &P[k * step]);
        trellis_gemv_raw(W, P.data(), R, 0, D_FF, &pg_[(((size_t) gi * 2 + up) * nbg + b) * MAX_ROWS * D_FF], D_FF);
        gate_us_ += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    } else {
        const int ncr = D_MODEL / tr_dn_cols(), per = nbd * ncr, gi = i / per, b = (i % per) / ncr, cr = (i % per) % ncr;
        const Group & gr = groups_[gi];
        const int R = (int) gr.rows.size(), j0 = b * TR_DN_IN;
        alignas(32) float g[MAX_ROWS][TR_DN_IN], u[MAX_ROWS][TR_DN_IN], h[MAX_ROWS][TR_DN_IN];
        for (int k = 0; k < R; ++k) {
            for (int up = 0; up < 2; ++up) {
                float * a = up ? u[k] : g[k];
                std::fill(a, a + TR_DN_IN, 0.f);
                for (int bb = 0; bb < nbg; ++bb) {   // block order
                    const float * p = &pg_[((((size_t) gi * 2 + up) * nbg + bb) * MAX_ROWS + k) * D_FF + j0];
                    for (int j = 0; j < TR_DN_IN; j += 8)
                        _mm256_store_ps(a + j, _mm256_add_ps(_mm256_load_ps(a + j), _mm256_loadu_ps(p + j)));
                }
                trellis_out(up ? gr.t->up : gr.t->gate, a, 1, TR_DN_IN, j0, j0 + TR_DN_IN, a, TR_DN_IN);
            }
            for (int j = 0; j < TR_DN_IN; ++j) h[k][j] = g[k][j] / (1.f + std::exp(-g[k][j])) * u[k][j];
        }
        const TrellisMat W = trellis_inputs(gr.t->down, j0, j0 + TR_DN_IN);
        const size_t step = (size_t) trellis_prep_floats(TR_DN_IN, 1);
        if (P.size() < step * MAX_ROWS) P.resize(step * MAX_ROWS);
        for (int k = 0; k < R; ++k) trellis_prep(W, h[k], TR_DN_IN, 1, &P[k * step]);
        const int c0 = cr * tr_dn_cols();
        trellis_gemv_raw(W, P.data(), R, c0, c0 + tr_dn_cols(), &pd_[((size_t) gi * nbd + b) * MAX_ROWS * D_MODEL + c0], D_MODEL);
        down_us_ += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    }
    item_us_ += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
}

// phase 0: gate and up rows of one chunk, then h = silu(gate) * up; phase 1: h quantized, down rows of one chunk
void ExpertPool::item(int i, int ph)
{
    if (trellis_) return item_trellis(i, ph);
    const auto t0 = std::chrono::steady_clock::now();
    if (ph == 0) {
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

bool ExpertPool::open() const
{
    if (!busy_.load(std::memory_order_acquire)) return false;
    const int t = ticket_.load(std::memory_order_acquire);
    return (t & ((1 << TICKET_SHIFT) - 1)) < phase_items_[t >> TICKET_SHIFT];
}

bool ExpertPool::grab(int & i, int & phase)
{
    if (!busy_.load(std::memory_order_acquire)) return false;
    if (!open()) return false;
    const int t = ticket_.fetch_add(1, std::memory_order_acq_rel);   // phase and index from the same value
    phase = t >> TICKET_SHIFT, i = t & ((1 << TICKET_SHIFT) - 1);
    return i < phase_items_[phase];
}

// the last item of a phase: flip to the next (first preparing what it reads: q4s quantizes h before down, trellis
// prepares h for down) or end the call
void ExpertPool::retire(int ph)
{
    if (pending_[ph].fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    if (ph + 1 < n_phases_) {
        if (!trellis_) quantize_h();   // trellis: nothing to prepare (each item prepares its own inputs)
        // seq_cst store, then wake(): a sleeper registered in sleepers_ before checking open() under mu_, so either
        // it sees the flip or wake() sees it (a lost wakeup hung calls before, TRACKER #73)
        ticket_.store((ph + 1) << TICKET_SHIFT);   // after the flip's preparation
        wake();   // workers blocked on the flip
    } else {
        last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start_).count();
        busy_.store(false);   // the caller spins on it (wait())
    }
}

void ExpertPool::run_items()
{
    for (int i, ph; grab(i, ph);) {
        item(i, ph);
        retire(ph);
    }
}

void ExpertPool::worker()
{
    for (;;) {
        // spin up to spin_ us (checked every 256 pauses): decode calls come every ~0.5-1 ms, a wake costs more.
        // Sleep only when the spin timed out. Before (TRACKER #77), a worker that left the spin because items opened
        // re-checked open() and slept when the other workers had already taken them all: every phase put most of
        // the pool to sleep (the first sleep came after ~0.2 ms, not the spin limit), and each start() paid futex
        // wakes, 0.2 ms per call in the engine.
        const auto t0 = std::chrono::steady_clock::now();
        bool timed_out = false;
        for (int n = 0; !stop_.load(std::memory_order_relaxed) && !open(); ++n) {
            _mm_pause();
            if ((n & 255) == 255 && std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_)) {
                timed_out = true;
                break;
            }
        }
        if (timed_out) {
            std::unique_lock<std::mutex> g(mu_);
            sleepers_.fetch_add(1);   // before the predicate check (see sleepers_)
            cv_.wait(g, [&] { return stop_.load() || open(); });
            sleepers_.fetch_sub(1);
        }
        if (stop_.load()) return;
        run_items();
    }
}

void ExpertPool::wait()
{
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {   // the caller is a worker too, in every phase, and spins (as Strata's host) until the call is done
        int i, ph;
        while (grab(i, ph)) {
            item(i, ph);
            retire(ph);
        }
        if (!busy_.load()) break;
        _mm_pause();
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
    if (trellis_) return;   // phase 2 items wrote y
    std::fill(y_, y_ + (size_t) T_ * D_MODEL, 0.f);
    for (size_t i = 0; i < slots_.size(); ++i) {
        const Slot & s = slots_[i];
        const float * o = &out_[((size_t) slot_g_[i] * MAX_ROWS + slot_k_[i]) * D_MODEL];
        float * y = y_ + (size_t) s.row * D_MODEL;
        for (int j = 0; j < D_MODEL; ++j) y[j] += s.w * o[j];
    }
}

}  // namespace truss::cpu
