// The small ops DeepSeek-V4.1 adds, each vs an fp64 CPU transcription of dsref/model.py on N(0,1)-scaled data:
//   ffn::route_biased SqrtSoftplus (256 experts, top 6, bias picks, raw scores weigh, renorm eps 1e-20, x1.5):
//                     ids exact, weights rel 1e-5
//   hc::sk_mix        (hc 4, d 5120: 24 dot products over 20480, rms, sigmoid / 2 sigmoid / softmax + 20 Sinkhorn
//                     rounds): coef rel 1e-4
//   hc::sk_collapse, sk_post, sk_expand: rel 1e-6
// Rows T = 1, 5, 37. Outputs NaN-filled before each call. usage: ds41_ops_test
#include "kernels/ffn/ffn_ops.cuh"
#include "kernels/hc/hc_sinkhorn.cuh"

#include "core/cuda_check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {

std::mt19937 rng(41);
std::vector<float> randn(size_t n, float s = 1.f)
{
    std::normal_distribution<float> d(0.f, s);
    std::vector<float> v(n);
    for (float & x : v) x = d(rng);
    return v;
}
template <class X> X * up(const std::vector<X> & v)
{
    X * p;
    TRUSS_CUDA(cudaMalloc(&p, v.size() * sizeof(X)));
    TRUSS_CUDA(cudaMemcpy(p, v.data(), v.size() * sizeof(X), cudaMemcpyHostToDevice));
    return p;
}
template <class X> X * nan_buf(size_t n)
{
    X * p;
    TRUSS_CUDA(cudaMalloc(&p, n * sizeof(X)));
    TRUSS_CUDA(cudaMemset(p, 0xff, n * sizeof(X)));
    return p;
}
template <class X> std::vector<X> down(const X * p, size_t n)
{
    std::vector<X> v(n);
    TRUSS_CUDA(cudaDeviceSynchronize());
    TRUSS_CUDA(cudaMemcpy(v.data(), p, n * sizeof(X), cudaMemcpyDeviceToHost));
    return v;
}
double rel(const std::vector<float> & a, const std::vector<double> & b)
{
    double d = 0, r = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        if (!std::isfinite(a[i])) return INFINITY;
        d += (a[i] - b[i]) * (a[i] - b[i]), r += b[i] * b[i];
    }
    return std::sqrt(d / std::max(r, 1e-300));
}
int report(const char * name, int T, double err, double gate)
{
    const bool ok = err <= gate;
    printf("%-14s T=%-3d rel %.3g (gate %.0e) %s\n", name, T, err, gate, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int test_route(int T)
{
    const int E = 256, K = 6;
    auto lg = randn((size_t) T * E, 2.f), bias = randn(E, 0.3f);
    int * d_ids = nan_buf<int>((size_t) T * K);
    float * d_w = nan_buf<float>((size_t) T * K);
    truss::ffn::route_biased(up(lg), up(bias), T, E, K, truss::ffn::Score::SqrtSoftplus, true, 1e-20f, 1.5f, d_ids, d_w,
                             0);
    auto ids = down(d_ids, (size_t) T * K);
    auto w = down(d_w, (size_t) T * K);
    int bad_ids = 0;
    std::vector<float> got;
    std::vector<double> want;
    for (int t = 0; t < T; ++t) {
        std::vector<double> s(E), key(E);
        for (int e = 0; e < E; ++e) {
            const double x = lg[(size_t) t * E + e];
            s[e] = std::sqrt(x > 20 ? x : std::log1p(std::exp(x)));
            key[e] = s[e] + bias[e];
        }
        std::vector<int> o(E);
        std::iota(o.begin(), o.end(), 0);
        std::stable_sort(o.begin(), o.end(), [&] (int a, int b) { return key[a] > key[b]; });
        double sum = 0;
        for (int j = 0; j < K; ++j) sum += s[o[j]];
        for (int j = 0; j < K; ++j) {
            bad_ids += ids[(size_t) t * K + j] != o[j];
            got.push_back(w[(size_t) t * K + j]);
            want.push_back(s[o[j]] / (sum + 1e-20) * 1.5);
        }
    }
    if (bad_ids) printf("route: %d ids differ\n", bad_ids);
    return report("route_biased", T, bad_ids ? INFINITY : rel(got, want), 1e-5);
}

int test_hc(int T)
{
    const truss::hc::SkDims k { 4, 5120, 1e-20f, 1e-6f, 20 };
    const int HC = 4, D = k.d, HCD = HC * D, NM = k.n_mix();
    auto res = randn((size_t) T * HCD), fn = randn((size_t) NM * HCD, 0.01f), scale = randn(3, 0.5f),
         base = randn(NM, 0.5f), x = randn((size_t) T * D);
    float * d_res = up(res), * d_coef = nan_buf<float>((size_t) T * NM);
    void * ws = nan_buf<char>(truss::hc::sk_mix_ws_bytes(T, k));
    truss::hc::sk_mix(d_res, up(fn), up(scale), up(base), T, k, ws, d_coef, 0);
    auto coef = down(d_coef, (size_t) T * NM);

    std::vector<double> want_coef;
    for (int t = 0; t < T; ++t) {
        const float * r = res.data() + (size_t) t * HCD;
        double ss = 0;
        for (int i = 0; i < HCD; ++i) ss += (double) r[i] * r[i];
        const double rs = 1.0 / std::sqrt(ss / HCD + k.norm_eps);
        std::vector<double> m(NM);
        for (int j = 0; j < NM; ++j) {
            double a = 0;
            for (int i = 0; i < HCD; ++i) a += (double) r[i] * fn[(size_t) j * HCD + i];
            m[j] = a * rs;
        }
        auto sg = [] (double v) { return 1.0 / (1.0 + std::exp(-v)); };
        double c[4][4];
        for (int i = 0; i < HC; ++i) want_coef.push_back(sg(m[i] * scale[0] + base[i]) + k.hc_eps);
        for (int i = 0; i < HC; ++i) want_coef.push_back(2 * sg(m[HC + i] * scale[1] + base[HC + i]));
        for (int i = 0; i < HC; ++i) {
            double mx = -INFINITY, s = 0;
            for (int j = 0; j < HC; ++j) mx = std::max(mx, c[i][j] = m[8 + i * 4 + j] * scale[2] + base[8 + i * 4 + j]);
            for (int j = 0; j < HC; ++j) s += (c[i][j] = std::exp(c[i][j] - mx));
            for (int j = 0; j < HC; ++j) c[i][j] = c[i][j] / s + k.hc_eps;
        }
        auto coln = [&] { for (int j = 0; j < HC; ++j) { double s = 0; for (int i = 0; i < HC; ++i) s += c[i][j];
                                                         for (int i = 0; i < HC; ++i) c[i][j] /= s + k.hc_eps; } };
        coln();
        for (int it = 1; it < k.iters; ++it) {
            for (int i = 0; i < HC; ++i) { double s = 0; for (int j = 0; j < HC; ++j) s += c[i][j];
                                           for (int j = 0; j < HC; ++j) c[i][j] /= s + k.hc_eps; }
            coln();
        }
        for (int i = 0; i < HC; ++i)
            for (int j = 0; j < HC; ++j) want_coef.push_back(c[i][j]);
    }
    int fails = report("hc sk_mix", T, rel(coef, want_coef), 1e-4);

    // collapse / post / expand on the GPU's coef (their own math only)
    float * d_y = nan_buf<float>((size_t) T * D);
    half * d_y16 = nan_buf<half>((size_t) T * D);
    truss::hc::sk_collapse(d_res, d_coef, T, k, d_y, d_y16, 0);
    auto y = down(d_y, (size_t) T * D);
    std::vector<double> want_y((size_t) T * D);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < D; ++i) {
            double a = 0;
            for (int c = 0; c < HC; ++c) a += (double) coef[(size_t) t * NM + c] * res[((size_t) t * HC + c) * D + i];
            want_y[(size_t) t * D + i] = a;
        }
    fails += report("hc sk_collapse", T, rel(y, want_y), 1e-6);

    float * d_out = nan_buf<float>((size_t) T * HCD);
    truss::hc::sk_post(d_res, up(x), d_coef, T, k, d_out, 0);
    auto out = down(d_out, (size_t) T * HCD);
    std::vector<double> want_out((size_t) T * HCD);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < HC; ++j)
            for (int i = 0; i < D; ++i) {
                const float * p = coef.data() + (size_t) t * NM;
                double a = (double) p[HC + j] * x[(size_t) t * D + i];
                for (int c = 0; c < HC; ++c) a += (double) p[8 + c * HC + j] * res[((size_t) t * HC + c) * D + i];
                want_out[((size_t) t * HC + j) * D + i] = a;
            }
    fails += report("hc sk_post", T, rel(out, want_out), 1e-6);
    truss::hc::sk_post(d_res, up(x), d_coef, T, k, d_res, 0);          // in place
    fails += report("hc sk_post(in)", T, rel(down(d_res, (size_t) T * HCD), want_out), 1e-6);

    float * d_pre = nan_buf<float>((size_t) T * NM);
    truss::hc::sk_expand(up(x), T, k, d_out, d_pre, 0);
    auto ex = down(d_out, (size_t) T * HCD);
    auto pre = down(d_pre, (size_t) T * NM);
    std::vector<double> want_ex((size_t) T * HCD), want_pre((size_t) T * NM, 0.0);
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < HC; ++c)
            for (int i = 0; i < D; ++i) want_ex[((size_t) t * HC + c) * D + i] = x[(size_t) t * D + i];
        want_pre[(size_t) t * NM] = 1.0;
    }
    fails += report("hc sk_expand", T, std::max(rel(ex, want_ex), rel(pre, want_pre)), 0);

    // the same ops on residual rows of hc d + 24 floats with the coef in each row's tail (deepseek41's layout):
    // results must equal the contiguous ones exactly
    truss::hc::SkDims ks = k;
    ks.res_ld = HCD + NM, ks.coef_ld = HCD + NM;
    std::vector<float> rs((size_t) T * ks.res_ld, 0.f);
    for (int t = 0; t < T; ++t) std::copy(res.begin() + (size_t) t * HCD, res.begin() + (size_t) (t + 1) * HCD,
                                          rs.begin() + (size_t) t * ks.res_ld);
    float * d_rs = up(rs);
    truss::hc::sk_mix(d_rs, up(fn), up(scale), up(base), T, ks, ws, d_rs + HCD, 0);
    auto rs_out = down(d_rs, rs.size());
    double worst = 0;
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < NM; ++j) worst = std::max(worst, (double) std::fabs(rs_out[(size_t) t * ks.res_ld + HCD + j] -
                                                                            coef[(size_t) t * NM + j]));
    float * d_y2 = nan_buf<float>((size_t) T * D);
    truss::hc::sk_collapse(d_rs, d_rs + HCD, T, ks, d_y2, nullptr, 0);
    auto y2 = down(d_y2, (size_t) T * D);
    for (size_t i = 0; i < y2.size(); ++i) worst = std::max(worst, (double) std::fabs(y2[i] - y[i]));
    fails += report("hc strided", T, worst, 0);
    return fails;
}

}  // namespace

int main()
{
    int fails = 0;
    for (int T : { 1, 5, 37 }) fails += test_route(T);
    for (int T : { 1, 5, 37 }) fails += test_hc(T);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
