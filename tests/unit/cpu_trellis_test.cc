// cpu::trellis_gemv against a dense reference built from cpu::trellis_weight_ref (bit-by-bit decode, the formula of
// ref_trellis.cu), for K = 1..4 and 1..8 rows; then speed: one expert (gate + up 2560 -> 640, down 640 -> 2560) per
// call, single thread from cache, and N threads over a set of experts larger than the L3 (DRAM, as in decode).
// usage: cpu_trellis_test [threads=12] [K for the speed run=2]
// Pass: within 1e-5 of the fp16-activation reference (gemv_tiles4), or, for the int16 kernel (gemv_i16, int16
// activations), no further from the exact (unrounded-activation) product than the GPU's fp16 activations are.
#include "cpu/expert_q4.h"
#include "cpu/expert_trellis.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using namespace truss::cpu;

namespace {

void h128(double * v)
{
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const double a = v[j], b = v[j + h];
                v[j] = a + b, v[j + h] = a - b;
            }
    for (int j = 0; j < 128; ++j) v[j] /= std::sqrt(128.0);
}

float r16(float x) { return _cvtsh_ss(_cvtss_sh(x, 0)); }

struct Mat {
    std::vector<uint32_t> tiles;
    std::vector<uint16_t> suh, svh;   // fp16 bits, as in the pack
    TrellisMat m;
};

Mat make(int K, int in, int out, std::mt19937 & g)
{
    Mat M;
    M.tiles.resize((size_t) in / 16 * (out / 16) * 8 * K);
    for (auto & w : M.tiles) w = g();
    std::uniform_real_distribution<float> s(0.5f, 1.5f);
    M.suh.resize(in), M.svh.resize(out);
    for (auto & v : M.suh) v = _cvtss_sh((g() & 1 ? 1.f : -1.f) * s(g), 0);
    for (auto & v : M.svh) v = _cvtss_sh((g() & 1 ? 1.f : -1.f) * s(g) * 0.01f, 0);
    M.m = { M.tiles.data(), K, in, out, M.suh.data(), M.svh.data() };
    return M;
}

// relative error of trellis_gemv against the reference with fp16-rounded activations (the GPU's) and, in *exact,
// against exact (unrounded) activations; *gpu = the GPU-equivalent fp16-activation reference's own error vs exact
double check(int K, int in, int out, int R, std::mt19937 & g, double * exact = nullptr, double * gpu = nullptr)
{
    const bool round = getenv("TRUSS_TRELLIS_ROUND") && atoi(getenv("TRUSS_TRELLIS_ROUND"));
    Mat M = make(K, in, out, g);
    std::normal_distribution<float> nd;
    std::vector<float> x((size_t) R * in);
    for (auto & v : x) v = nd(g);
    std::vector<float> W((size_t) out * in);
    for (int o = 0; o < out; ++o)
        for (int i = 0; i < in; ++i) W[(size_t) o * in + i] = trellis_weight_ref(M.m, o, i, round);
    std::vector<float> P(trellis_prep_floats(in, R)), y((size_t) R * out);
    trellis_prep(M.m, x.data(), in, R, P.data());
    trellis_gemv(M.m, P.data(), R, 0, out, y.data(), out);
    double err = 0, ref = 0, errx = 0, refx = 0, errg = 0;
    for (int r = 0; r < R; ++r) {
        std::vector<double> a(in), ax(in), c(out), cx(out);
        for (int i = 0; i < in; ++i) a[i] = (double) x[(size_t) r * in + i] * _cvtsh_ss(M.suh[i]);
        for (int b = 0; b < in; b += 128) h128(&a[b]);
        ax = a;
        for (int i = 0; i < in; ++i) a[i] = r16((float) a[i]);
        for (int o = 0; o < out; ++o) {
            double s = 0, sx = 0;
            for (int i = 0; i < in; ++i) s += (double) W[(size_t) o * in + i] * a[i], sx += (double) W[(size_t) o * in + i] * ax[i];
            c[o] = s, cx[o] = sx;
        }
        for (int b = 0; b < out; b += 128) h128(&c[b]), h128(&cx[b]);
        for (int o = 0; o < out; ++o) {
            const double yr = c[o] * _cvtsh_ss(M.svh[o]), d = y[(size_t) r * out + o] - yr;
            const double yx = cx[o] * _cvtsh_ss(M.svh[o]), dx = y[(size_t) r * out + o] - yx;
            err += d * d, ref += yr * yr, errx += dx * dx, refx += yx * yx, errg += (yr - yx) * (yr - yx);
        }
    }
    if (exact) *exact = std::sqrt(errx / refx);
    if (gpu) *gpu = std::sqrt(errg / refx);
    return std::sqrt(err / ref);
}

// one projection in fp64 from bit-decoded weights (dense cache per matrix): y = svh * H128(W a), a = H128(suh * x),
// fp16-rounded first when act16 (the GPU's activations)
void proj_exact(const TrellisMat & m, const double * x, bool act16, double * y)
{
    static std::vector<std::pair<const uint32_t *, std::vector<float>>> cache;
    auto it = std::find_if(cache.begin(), cache.end(), [&](const auto & e) { return e.first == m.tiles; });
    if (it == cache.end()) {
        std::vector<float> W((size_t) m.out * m.in);
        for (int o = 0; o < m.out; ++o)
            for (int i = 0; i < m.in; ++i) W[(size_t) o * m.in + i] = trellis_weight_ref(m, o, i);
        cache.emplace_back(m.tiles, std::move(W));
        it = cache.end() - 1;
    }
    std::vector<double> a(m.in), c(m.out);
    for (int i = 0; i < m.in; ++i) a[i] = x[i] * _cvtsh_ss(m.suh[i]);
    for (int b = 0; b < m.in; b += 128) h128(&a[b]);
    if (act16)
        for (auto & v : a) v = r16((float) v);
    for (int o = 0; o < m.out; ++o) {
        double s = 0;
        for (int i = 0; i < m.in; ++i) s += (double) it->second[(size_t) o * m.in + i] * a[i];
        c[o] = s;
    }
    for (int b = 0; b < m.out; b += 128) h128(&c[b]);
    for (int o = 0; o < m.out; ++o) y[o] = c[o] * _cvtsh_ss(m.svh[o]);
}

// a whole expert in fp64 (silu * up between the projections, rounded to fp16 like the GPU's h when act16)
void expert_exact(const TrellisExpert & e, const float * xf, bool act16, double * y)
{
    std::vector<double> x(xf, xf + 2560), g(640), u(640);
    proj_exact(e.gate, x.data(), act16, g.data());
    proj_exact(e.up, x.data(), act16, u.data());
    for (int j = 0; j < 640; ++j) g[j] = g[j] / (1.0 + std::exp(-g[j])) * u[j];
    proj_exact(e.down, g.data(), act16, y);
}

double ms_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// one expert for R rows: prep + gate + up, silu * up, prep + down
void expert(const TrellisExpert & e, const float * x, int R, float * P, float * g, float * u, float * y)
{
    trellis_prep(e.gate, x, 2560, R, P);
    trellis_gemv(e.gate, P, R, 0, 640, g, 640);
    trellis_prep(e.up, x, 2560, R, P);
    trellis_gemv(e.up, P, R, 0, 640, u, 640);
    for (int i = 0; i < R * 640; ++i) g[i] = g[i] / (1.f + std::exp(-g[i])) * u[i];
    trellis_prep(e.down, g, 640, R, P);
    trellis_gemv(e.down, P, R, 0, 2560, y, 2560);
}

}  // namespace

int main(int argc, char ** argv)
{
    const int threads = argc > 1 ? std::atoi(argv[1]) : 12, Ks = argc > 2 ? std::atoi(argv[2]) : 2;
    std::mt19937 g(7);
    bool ok = true;
    for (int K = 1; K <= 4; ++K)
        for (int R : { 1, 3, 8 }) {
            // pass: within 1e-5 of the fp16-activation reference (gemv_tiles4), or (int16 activations, gemv_i16) no
            // further from the exact product than the GPU's own fp16 activations are
            double x1, g1, x2, g2;
            const double e1 = check(K, 256, 128, R, g, &x1, &g1), e2 = check(K, 128, 384, R, g, &x2, &g2);
            const bool pass = (e1 < 1e-5 && e2 < 1e-5) || (x1 <= std::max(1e-5, g1) && x2 <= std::max(1e-5, g2));
            ok &= pass;
            std::printf("K %d rows %d: rel err %.2e / %.2e (vs exact %.2e / %.2e, GPU fp16 act %.2e / %.2e)  %s\n", K, R,
                        e1, e2, x1, x2, g1, g2, pass ? "ok" : "FAIL");
        }

    // speed: N experts of random K-bit tiles (the bytes are what matters; values are random)
    const int n_exp = 256;   // 256 x 1.6-6.6 MB > 128 MB L3
    const size_t gu = (size_t) 2560 / 16 * (640 / 16) * 8 * Ks, dn = gu;
    std::vector<uint32_t> buf((gu * 2 + dn) * n_exp);
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = (uint32_t) (i * 2654435761u) ^ (uint32_t) (i >> 7);
    std::vector<uint16_t> s2560(2560, _cvtss_sh(1.f, 0)), s640(640, _cvtss_sh(0.01f, 0));
    std::vector<TrellisExpert> ex(n_exp);
    for (int e = 0; e < n_exp; ++e) {
        const uint32_t * b = buf.data() + (gu * 2 + dn) * e;
        ex[e].gate = { b, Ks, 2560, 640, s2560.data(), s640.data() };
        ex[e].up = { b + gu, Ks, 2560, 640, s2560.data(), s640.data() };
        ex[e].down = { b + 2 * gu, Ks, 640, 2560, s640.data(), s2560.data() };
    }
    const double mb = (gu * 2 + dn) * 4 / 1e6;
    std::printf("speed, K %d (%.2f MB per expert):\n", Ks, mb);
    std::vector<float> x(8 * 2560, 0.1f);
    for (int R : { 1, 2, 4 }) {
        std::vector<float> P(4 * 8 * 2560), gg(8 * 640), uu(8 * 640), y(8 * 2560);
        expert(ex[0], x.data(), R, P.data(), gg.data(), uu.data(), y.data());
        double one = 1e9;   // best of 5 batches: the renters' load makes single runs noisy
        for (int b = 0; b < 5; ++b) {
            const auto t1 = std::chrono::steady_clock::now();
            const int reps = 10;
            for (int i = 0; i < reps; ++i) expert(ex[0], x.data(), R, P.data(), gg.data(), uu.data(), y.data());
            one = std::min(one, ms_since(t1) / reps);
        }
        auto t0 = std::chrono::steady_clock::now();
        std::atomic<int> next{ 0 };
        const int total = n_exp * 4;
        t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back([&] {
                std::vector<float> P2(4 * 8 * 2560), g2(8 * 640), u2(8 * 640), y2(8 * 2560);
                for (int i; (i = next++) < total;)
                    expert(ex[(i * 37) % n_exp], x.data(), R, P2.data(), g2.data(), u2.data(), y2.data());
            });
        for (auto & t : ts) t.join();
        const double ms = ms_since(t0);
        std::printf("  rows %d: 1 thread %.3f ms/expert (cache) | %d threads %.1f experts/ms = %.1f GB/s\n", R, one,
                    threads, total / ms, total * mb / ms);
    }
    // ExpertPool on trellis slots: 4 rows x 6 experts (shared between rows), vs expert() per slot summed in slot
    // order; and each row equal to its own 1-row call (rows are independent)
    {
        ExpertPool pool(threads);
        std::normal_distribution<float> nd;
        std::vector<float> xs(4 * 2560);
        for (auto & v : xs) v = nd(g);
        std::vector<Slot> slots;
        for (int t = 0; t < 4; ++t)
            for (int j = 0; j < 3; ++j) slots.push_back({ t, nullptr, 0.1f * (j + 1), &ex[(t + 2 * j) % 6] });
        std::vector<float> y(4 * 2560), ref(4 * 2560, 0.f);
        pool.run(xs.data(), 4, slots, y.data());
        std::vector<float> P(4 * 8 * 2560), gg(8 * 640), uu(8 * 640), yo(8 * 2560);
        for (const Slot & sl : slots) {
            expert(*sl.t, xs.data() + sl.row * 2560, 1, P.data(), gg.data(), uu.data(), yo.data());
            for (int c = 0; c < 2560; ++c) ref[sl.row * 2560 + c] += sl.w * yo[c];
        }
        // vs an fp64 reference from bit-decoded weights: the pool (input-split items, one int16 scale per input block)
        // and expert() per slot (one scale per row) against exact activations, and the GPU's fp16 activations
        std::vector<double> ex64(4 * 2560, 0.0), g64(4 * 2560, 0.0), o64(2560);
        for (const Slot & sl : slots) {
            expert_exact(*sl.t, xs.data() + sl.row * 2560, false, o64.data());
            for (int c = 0; c < 2560; ++c) ex64[sl.row * 2560 + c] += sl.w * o64[c];
            expert_exact(*sl.t, xs.data() + sl.row * 2560, true, o64.data());
            for (int c = 0; c < 2560; ++c) g64[sl.row * 2560 + c] += sl.w * o64[c];
        }
        double e = 0, r = 0, ep = 0, eg = 0, rx = 0;
        for (int i = 0; i < 4 * 2560; ++i) {
            e += (y[i] - ref[i]) * (double) (y[i] - ref[i]), r += ref[i] * (double) ref[i];
            ep += (y[i] - ex64[i]) * (y[i] - ex64[i]), eg += (g64[i] - ex64[i]) * (g64[i] - ex64[i]);
            rx += ex64[i] * ex64[i];
        }
        double es = 0;
        for (int i = 0; i < 4 * 2560; ++i) es += (ref[i] - ex64[i]) * (ref[i] - ex64[i]);
        bool same = true;
        for (int t = 0; t < 4; ++t) {
            std::vector<Slot> one;
            for (const Slot & sl : slots)
                if (sl.row == t) one.push_back({ 0, nullptr, sl.w, sl.t });
            std::vector<float> y1(2560);
            pool.run(xs.data() + t * 2560, 1, one, y1.data());
            same &= std::memcmp(y1.data(), y.data() + t * 2560, 2560 * 4) == 0;
        }
        // pass: no further from the exact product than the GPU's fp16 activations are (as the kernel checks above)
        const bool pass = std::sqrt(ep / rx) <= std::sqrt(eg / rx) && same;
        ok &= pass;
        std::printf("pool, 4 rows x 3 trellis slots: vs exact %.2e (per-slot expert() %.2e, GPU fp16 act %.2e), vs "
                    "per-slot %.1e, rows equal to 1-row calls: %s  %s\n",
                    std::sqrt(ep / rx), std::sqrt(es / rx), std::sqrt(eg / rx), std::sqrt(e / r), same ? "yes" : "no",
                    pass ? "ok" : "FAIL");
        // stress: many small calls (1-3 rows, 1-4 experts): phase flips race with workers grabbing items; a lost
        // item would hang here (the pre-ticket pool did, TRACKER #73)
        {
            std::mt19937 gs(3);
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < 3000; ++it) {
                const int T = 1 + (int) (gs() % 3), n = 1 + (int) (gs() % 4);
                std::vector<Slot> sl;
                for (int j = 0; j < n; ++j) sl.push_back({ (int) (gs() % T), nullptr, 0.5f, &ex[gs() % 8] });
                std::vector<float> ys(T * 2560);
                pool.run(xs.data(), T, sl, ys.data());
            }
            std::printf("stress: 3000 small calls in %.0f ms, no hang\n", ms_since(t0));
        }
        // small-call latency: what a decode layer usually hands the CPU (~0.5-2 experts)
        for (int ne : { 1, 2, 4 }) {
            std::vector<Slot> sl;
            for (int j = 0; j < ne; ++j) sl.push_back({ 0, nullptr, 0.5f, &ex[(11 * j + 5) % n_exp] });
            std::vector<float> ys(2560);
            double best = 1e9;
            for (int k = 0; k < 30; ++k) {
                const auto t0 = std::chrono::steady_clock::now();
                pool.run(xs.data(), 1, sl, ys.data());
                best = std::min(best, ms_since(t0));
            }
            std::printf("pool latency: 1 row x %d expert(s): best %.3f ms\n", ne, best);
        }
        // pool speed: 4 rows x 10 slots over 30 distinct experts (a verify window's CPU share, from DRAM)
        std::vector<Slot> big;
        for (int t = 0; t < 4; ++t)
            for (int j = 0; j < 10; ++j) big.push_back({ t, nullptr, 0.1f, &ex[(7 * t + 3 * j) % n_exp] });
        std::vector<float> yb(4 * 2560);
        pool.run(xs.data(), 4, big, yb.data());
        pool.reset_stats();   // the shape below covers these calls only
        double best = 1e9;
        for (int k = 0; k < 20; ++k) {
            const auto t0 = std::chrono::steady_clock::now();
            pool.run(xs.data(), 4, big, yb.data());
            best = std::min(best, ms_since(t0));
        }
        long long sh[4];
        pool.shape(sh);
        std::printf("pool speed: 4 rows x 10 slots, %lld distinct experts per call: best %.3f ms (%.1f experts/ms)\n",
                    sh[1] / sh[0], best, (sh[1] / sh[0]) / best);
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
