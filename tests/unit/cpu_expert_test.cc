// cpu::ExpertPool vs a double-precision reference on random q4s experts: y = sum_slots w * down(silu(gate x) * up x)
// with the same activation quantization (x and h per 32-block int8). Cases: 1, 4 and 8 rows, shared and distinct
// experts, rows with no slot; row independence (a row's result equals its 1-row call); timing of a 4-row window
// with ~8 experts (a decode layer's CPU share).
#include "cpu/expert_q4.h"

#include <immintrin.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace truss::cpu;

namespace {

float fp16(uint16_t h) { return _cvtsh_ss(h); }

void quant_ref(const double * x, int n, std::vector<double> & out)   // x -> dequantized int8 per block
{
    out.resize(n);
    for (int b = 0; b < n / BLOCK; ++b) {
        float m = 0.f;
        for (int i = 0; i < BLOCK; ++i) m = std::max(m, (float) std::fabs(x[BLOCK * b + i]));
        const float s = m / 127.f, inv = s > 0.f ? 1.f / s : 0.f;
        for (int i = 0; i < BLOCK; ++i) out[BLOCK * b + i] = std::lrintf((float) x[BLOCK * b + i] * inv) * (double) s;
    }
}

double w_at(const Q4Matrix & M, int r, int c)
{
    const uint8_t byte = M.q[(size_t) r * M.cols / 2 + (c / BLOCK) * 16 + (c % BLOCK) % 16];
    const int nib = (c % BLOCK) < 16 ? byte & 15 : byte >> 4;
    return (nib - 8) * (double) fp16(M.d[(size_t) r * (M.cols / BLOCK) + c / BLOCK]);
}

std::vector<double> matvec(const Q4Matrix & M, const std::vector<double> & x)
{
    std::vector<double> y(M.rows);
    for (int r = 0; r < M.rows; ++r) {
        double a = 0;
        for (int c = 0; c < M.cols; ++c) a += w_at(M, r, c) * x[c];
        y[r] = a;
    }
    return y;
}

std::vector<double> expert_ref(const Q4Expert & e, const float * x)
{
    std::vector<double> xd(x, x + D_MODEL), xq, h(D_FF), hq;
    quant_ref(xd.data(), D_MODEL, xq);
    const std::vector<double> g = matvec(e.gate, xq), u = matvec(e.up, xq);
    for (int i = 0; i < D_FF; ++i) h[i] = (float) ((float) g[i] / (1.f + std::exp(-(float) g[i])) * (float) u[i]);
    quant_ref(h.data(), D_FF, hq);
    return matvec(e.down, hq);
}

}  // namespace

int main()
{
    std::mt19937 rng(5);
    const int NE = 12;
    std::vector<std::vector<uint8_t>> blobs(NE, std::vector<uint8_t>(EXPERT_BYTES));
    std::vector<Q4Expert> ex;
    for (auto & b : blobs) {
        for (auto & v : b) v = (uint8_t) rng();
        Q4Expert e = expert_view(b.data());
        // sane scales (fp16 ~ 0.01 .. 0.03) in place of random bits
        for (const Q4Matrix * M : { &e.gate, &e.up, &e.down }) {
            uint16_t * d = const_cast<uint16_t *>(M->d);
            for (size_t i = 0; i < (size_t) M->rows * M->cols / BLOCK; ++i)
                d[i] = _cvtss_sh(0.01f + 0.02f * (rng() % 1000) / 1000.f, 0);
        }
        ex.push_back(e);
    }
    ExpertPool pool(8);
    int fails = 0;
    auto check = [&](int T, const std::vector<Slot> & slots, const char * name) {
        std::vector<float> x((size_t) T * D_MODEL), y((size_t) T * D_MODEL);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & v : x) v = nd(rng);
        pool.run(x.data(), T, slots, y.data());
        double num = 0, den = 0;
        std::vector<double> want((size_t) T * D_MODEL, 0.0);
        for (const Slot & s : slots) {
            const std::vector<double> o = expert_ref(*s.e, x.data() + (size_t) s.row * D_MODEL);
            for (int j = 0; j < D_MODEL; ++j) want[(size_t) s.row * D_MODEL + j] += s.w * o[j];
        }
        for (size_t i = 0; i < y.size(); ++i) num += (y[i] - want[i]) * (y[i] - want[i]), den += want[i] * want[i];
        // row independence: each row alone
        long diff = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<Slot> one;
            for (const Slot & s : slots)
                if (s.row == t) one.push_back({ 0, s.e, s.w });
            std::vector<float> y1(D_MODEL);
            pool.run(x.data() + (size_t) t * D_MODEL, 1, one, y1.data());
            diff += std::memcmp(y1.data(), y.data() + (size_t) t * D_MODEL, D_MODEL * 4) != 0;
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        const bool ok = rel < 2e-3 && diff == 0;
        fails += !ok;
        std::printf("%-34s rows %d, %2zu slots: rel %.1e vs reference, rows equal to 1-row calls: %s  %s\n", name, T,
                    slots.size(), rel, diff ? "NO" : "yes", ok ? "PASS" : "FAIL");
    };
    check(1, { { 0, &ex[0], 0.7f } }, "one expert");
    check(1, { { 0, &ex[1], 0.4f }, { 0, &ex[2], 0.35f }, { 0, &ex[3], 0.25f } }, "three experts");
    check(4, { { 0, &ex[0], 0.5f }, { 1, &ex[0], 0.3f }, { 1, &ex[4], 0.2f }, { 3, &ex[5], 0.9f }, { 3, &ex[0], 0.1f } },
          "shared expert, an empty row");
    std::vector<Slot> big;
    for (int t = 0; t < 8; ++t)
        for (int k = 0; k < 3; ++k) big.push_back({ t, &ex[(t + 5 * k) % NE], 0.3f });
    check(8, big, "8 rows x 3 experts");

    // timing: a 4-row window, 8 distinct experts, streamed from DRAM (pool of distinct experts)
    const int POOL = 256;
    std::vector<std::vector<uint8_t>> many(POOL, std::vector<uint8_t>(EXPERT_BYTES, 0x88));
    std::vector<Q4Expert> mex;
    for (auto & b : many) mex.push_back(expert_view(b.data()));
    std::vector<float> x(4 * D_MODEL, 0.5f), y(4 * D_MODEL);
    const int iters = 200;
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        std::vector<Slot> s;
        for (int k = 0; k < 8; ++k) s.push_back({ k % 4, &mex[(it * 8 + k) % POOL], 0.1f });
        pool.run(x.data(), 4, s, y.data());
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    std::printf("timing: 4 rows, 8 experts per call, %d threads + caller: %.0f us per call (%.1f experts/ms)\n",
                pool.threads(), us, 8e3 / us);
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
