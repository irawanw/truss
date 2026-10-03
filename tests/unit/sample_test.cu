// sampling::sample against the exact distribution: for small vocabularies (and one of the real size), the empirical
// frequencies over many rows (one launch, each row its own random stream) match softmax(x / T) restricted to the
// top-k / top-p / min-p set (total variation distance within sampling noise), no token outside the set is ever
// drawn, and the same (seed, counter) draws the same token. sampling::penalize against a host reference: bit-exact
// logits per row (each row its own history, -1 padding and out-of-range ids ignored), and the count scratch back to
// zero afterwards.
#include "core/cuda_check.h"
#include "kernels/sampling/penalty.cuh"
#include "kernels/sampling/sample.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using namespace truss;

namespace {

// exact kept distribution: sort by logit, apply top-k, min-p, top-p (smallest prefix with mass >= top_p)
std::vector<double> exact(const std::vector<float> & x, const sampling::SampleParams & p)
{
    const int n = (int) x.size();
    std::vector<int> o(n);
    std::iota(o.begin(), o.end(), 0);
    std::sort(o.begin(), o.end(), [&](int a, int b) { return x[a] > x[b]; });
    std::vector<double> y(n);
    const double m = x[o[0]] / p.temperature;
    double z = 0;
    for (int i = 0; i < n; ++i) z += y[i] = std::exp(x[i] / p.temperature - m);
    int keep = n;
    if (p.top_k > 0) keep = std::min(keep, p.top_k);
    if (p.min_p > 0)
        for (int i = 0; i < keep; ++i)
            if (y[o[i]] < p.min_p) { keep = i; break; }
    if (p.top_p > 0 && p.top_p < 1) {
        double zk = 0;
        for (int i = 0; i < keep; ++i) zk += y[o[i]];
        double acc = 0;
        for (int i = 0; i < keep; ++i) {
            acc += y[o[i]];
            if (acc >= p.top_p * zk) { keep = i + 1; break; }
        }
    }
    std::vector<double> q(n, 0.0);
    double zk = 0;
    for (int i = 0; i < keep; ++i) zk += y[o[i]];
    for (int i = 0; i < keep; ++i) q[o[i]] = y[o[i]] / zk;
    return q;
}

int run(int n, int rows, const sampling::SampleParams & p, std::mt19937 & rng, const char * what)
{
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<float> x(n);
    for (float & v : x) v = nd(rng);
    std::vector<float> xs((size_t) rows * n);
    for (int r = 0; r < rows; ++r) std::copy(x.begin(), x.end(), xs.begin() + (size_t) r * n);
    float * dx;
    int * dout;
    TRUSS_CUDA(cudaMalloc(&dx, xs.size() * 4));
    TRUSS_CUDA(cudaMalloc(&dout, rows * 4));
    TRUSS_CUDA(cudaMemcpy(dx, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice));
    sampling::sample(dx, n, rows, p, 1000, dout, nullptr);
    std::vector<int> got(rows), again(rows);
    TRUSS_CUDA(cudaMemcpy(got.data(), dout, rows * 4, cudaMemcpyDeviceToHost));
    sampling::sample(dx, n, rows, p, 1000, dout, nullptr);
    TRUSS_CUDA(cudaMemcpy(again.data(), dout, rows * 4, cudaMemcpyDeviceToHost));
    const std::vector<double> q = exact(x, p);
    std::vector<double> f(n, 0.0);
    long outside = 0;
    for (int t : got) {
        f[t] += 1.0 / rows;
        outside += q[t] == 0.0;
    }
    double tv = 0;
    int support = 0;
    for (int i = 0; i < n; ++i) tv += std::fabs(f[i] - q[i]) / 2, support += q[i] > 0;
    // noise: E[TV] ~ sum sqrt(q (1 - q) / rows) / 2 * sqrt(2 / pi); allow 3x
    double noise = 0;
    for (double v : q) noise += std::sqrt(v * (1 - v) / rows);
    noise *= 0.5 * std::sqrt(2 / M_PI);
    const bool ok = outside == 0 && got == again && tv <= 3 * noise + 1e-3;
    std::printf("%-28s n %6d, support %5d, rows %6d: TV %.4f (noise %.4f), outside %ld, deterministic %s  %s\n", what, n,
                support, rows, tv, noise, outside, got == again ? "yes" : "NO", ok ? "PASS" : "FAIL");
    cudaFree(dx), cudaFree(dout);
    return ok ? 0 : 1;
}

int run_penalty(int n, int rows, int h, const sampling::PenaltyParams & pp, std::mt19937 & rng, const char * what)
{
    std::normal_distribution<float> nd(0.f, 3.f);
    std::uniform_int_distribution<int> tok(0, 40), pad(0, h / 2);
    std::vector<float> x((size_t) rows * n);
    for (float & v : x) v = nd(rng);
    std::vector<int> hist((size_t) rows * h);
    for (int r = 0; r < rows; ++r) {
        const int np = pad(rng);   // a short history: -1 in front, ids clustered so tokens repeat
        for (int i = 0; i < h; ++i) hist[(size_t) r * h + i] = i < np ? -1 : (i % 97 == 0 ? n + 5 : tok(rng) * (r + 1) % n);
    }
    std::vector<float> want = x;
    for (int r = 0; r < rows; ++r) {
        std::vector<int> c(n, 0);
        for (int i = 0; i < h; ++i) {
            const int t = hist[(size_t) r * h + i];
            if (t >= 0 && t < n) ++c[t];
        }
        for (int t = 0; t < n; ++t) {
            if (!c[t]) continue;
            float v = want[(size_t) r * n + t];
            if (pp.repeat != 1.f) v = v > 0.f ? v / pp.repeat : v * pp.repeat;
            volatile float prod = (float) c[t] * pp.frequency;   // rounded on its own, as the kernel does
            want[(size_t) r * n + t] = v - prod - pp.presence;
        }
    }
    float * dx;
    int * dh, * dc;
    TRUSS_CUDA(cudaMalloc(&dx, x.size() * 4));
    TRUSS_CUDA(cudaMalloc(&dh, hist.size() * 4));
    TRUSS_CUDA(cudaMalloc(&dc, (size_t) rows * n * 4));
    TRUSS_CUDA(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(dh, hist.data(), hist.size() * 4, cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemset(dc, 0, (size_t) rows * n * 4));
    sampling::penalize(dx, n, rows, dh, h, pp, dc, nullptr);
    std::vector<float> got(x.size());
    std::vector<int> cnt((size_t) rows * n);
    TRUSS_CUDA(cudaMemcpy(got.data(), dx, got.size() * 4, cudaMemcpyDeviceToHost));
    TRUSS_CUDA(cudaMemcpy(cnt.data(), dc, cnt.size() * 4, cudaMemcpyDeviceToHost));
    long diff = 0, changed = 0, dirty = 0;
    for (size_t i = 0; i < x.size(); ++i) diff += got[i] != want[i], changed += want[i] != x[i];
    for (int v : cnt) dirty += v != 0;
    const bool ok = diff == 0 && dirty == 0 && changed > 0;
    std::printf("%-28s n %6d, rows %d, h %4d: %ld logits penalized, %ld differ from the host, %ld counts left  %s\n",
                what, n, rows, h, changed, diff, dirty, ok ? "PASS" : "FAIL");
    cudaFree(dx), cudaFree(dh), cudaFree(dc);
    return ok ? 0 : 1;
}

}  // namespace

int main()
{
    std::mt19937 rng(3);
    int fails = 0;
    sampling::SampleParams p;
    p.seed = 42;
    p.temperature = 1.f;
    fails += run(64, 200000, p, rng, "T 1");
    p.temperature = 0.6f;
    fails += run(64, 200000, p, rng, "T 0.6");
    p.top_k = 20;
    fails += run(256, 200000, p, rng, "T 0.6 top_k 20");
    p.top_k = 0, p.top_p = 0.95f;
    fails += run(256, 200000, p, rng, "T 0.6 top_p 0.95");
    p.top_p = 0.8f, p.top_k = 40, p.min_p = 0.05f;
    fails += run(512, 200000, p, rng, "top_p 0.8 top_k 40 min_p 0.05");
    p = {};
    p.seed = 9, p.temperature = 1.f, p.top_p = 0.95f;
    fails += run(248077, 4000, p, rng, "real vocab, top_p 0.95");
    sampling::PenaltyParams pp;
    pp.presence = 1.5f;
    fails += run_penalty(512, 4, 64, pp, rng, "presence 1.5");
    pp = {}, pp.frequency = 0.3f;
    fails += run_penalty(512, 4, 256, pp, rng, "frequency 0.3");
    pp = {}, pp.repeat = 1.1f, pp.frequency = 0.1f, pp.presence = 0.5f;
    fails += run_penalty(248077, 4, 4096, pp, rng, "real vocab, all three");
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
