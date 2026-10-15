// The small ops LFM2 added, each vs an fp64 CPU reference on N(0,1) data:
//   norm::rms        (2048 wide, fp32 + fp16 out)                         gate rel 1e-6 (fp32) / 1e-3 (fp16)
//   norm::rms_heads  (64-wide heads in place)                             gate rel 1e-6
//   conv::short_conv (d 2048, L 3; a chunk of 7 then 1 + 4 rows: the carried state; bx rows out)   gate rel 1e-3
//   ffn::route_sigmoid (64 experts, top 4, bias; ids exact, weights rel 1e-6)
// Outputs NaN-filled before each call. usage: lfm_small_ops_test
#include "kernels/conv/short_conv.cuh"
#include "kernels/dense/q8_gemm.cuh"
#include "kernels/ffn/ffn_ops.cuh"
#include "kernels/norm/rms_norm.cuh"

#include "core/cuda_check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {

std::mt19937 rng(5);
std::vector<float> randn(size_t n)
{
    std::normal_distribution<float> d(0.f, 1.f);
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
template <class X> std::vector<X> down(const X * p, size_t n)
{
    std::vector<X> v(n);
    TRUSS_CUDA(cudaMemcpy(v.data(), p, n * sizeof(X), cudaMemcpyDeviceToHost));
    return v;
}
template <class X> X * nanbuf(size_t n)
{
    X * p;
    TRUSS_CUDA(cudaMalloc(&p, n * sizeof(X)));
    TRUSS_CUDA(cudaMemset(p, 0xff, n * sizeof(X)));
    return p;
}
double rel(const std::vector<double> & ref, const std::vector<float> & got)
{
    double n = 0, d = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::isnan(got[i])) return INFINITY;
        n += (got[i] - ref[i]) * (got[i] - ref[i]), d += ref[i] * ref[i];
    }
    return std::sqrt(n / std::max(d, 1e-30));
}
int fails = 0;
void report(const char * what, double r, double gate)
{
    const bool ok = r <= gate;
    fails += !ok;
    std::printf("%-44s rel %.2e (gate %.0e)  %s\n", what, r, gate, ok ? "PASS" : "FAIL");
}

void test_rms()
{
    const int T = 5, d = 2048;
    const std::vector<float> x = randn((size_t) T * d), w = randn(d);
    float * y32 = nanbuf<float>((size_t) T * d);
    half * y16 = nanbuf<half>((size_t) T * d);
    truss::norm::rms(up(x), up(w), T, d, 1e-5f, y32, y16, nullptr);
    std::vector<double> ref((size_t) T * d);
    for (int t = 0; t < T; ++t) {
        double ss = 0;
        for (int i = 0; i < d; ++i) ss += (double) x[(size_t) t * d + i] * x[(size_t) t * d + i];
        const double r = 1.0 / std::sqrt(ss / d + 1e-5);
        for (int i = 0; i < d; ++i) ref[(size_t) t * d + i] = x[(size_t) t * d + i] * r * w[i];
    }
    report("norm::rms fp32", rel(ref, down(y32, ref.size())), 1e-6);
    const std::vector<half> h = down(y16, ref.size());
    std::vector<float> hf(h.size());
    for (size_t i = 0; i < h.size(); ++i) hf[i] = __half2float(h[i]);
    report("norm::rms fp16", rel(ref, hf), 1e-3);
    // heads in place
    const int H = 32, D = 64;
    std::vector<float> xh = randn((size_t) T * H * D), wh = randn(D);
    float * dxh = up(xh);
    truss::norm::rms_heads(dxh, up(wh), T, H, D, 1e-5f, nullptr);
    std::vector<double> rh(xh.size());
    for (int r = 0; r < T * H; ++r) {
        double ss = 0;
        for (int i = 0; i < D; ++i) ss += (double) xh[(size_t) r * D + i] * xh[(size_t) r * D + i];
        const double s = 1.0 / std::sqrt(ss / D + 1e-5);
        for (int i = 0; i < D; ++i) rh[(size_t) r * D + i] = xh[(size_t) r * D + i] * s * wh[i];
    }
    report("norm::rms_heads", rel(rh, down(dxh, rh.size())), 1e-6);
}

void test_conv()
{
    const int d = 2048, L = 3;
    const std::vector<float> w = randn((size_t) d * L);
    float * dw = up(w), * state;
    TRUSS_CUDA(cudaMalloc(&state, sizeof(float) * (L - 1) * d));
    TRUSS_CUDA(cudaMemset(state, 0, sizeof(float) * (L - 1) * d));
    std::vector<double> hist((size_t) (L - 1) * d, 0.0);   // the reference's carried bx rows
    for (int T : { 7, 1, 4 }) {
        const std::vector<float> bcx = randn((size_t) T * 3 * d);
        half * y16 = nanbuf<half>((size_t) T * d);
        float * bx = nanbuf<float>((size_t) T * d);
        truss::conv::short_conv(up(bcx), dw, state, T, d, L, y16, bx, nullptr);
        std::vector<double> seq(hist);   // [L-1 + T][d] bx rows
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < d; ++c) seq.push_back((double) bcx[(size_t) t * 3 * d + c] * bcx[(size_t) t * 3 * d + 2 * d + c]);
        std::vector<double> ref((size_t) T * d), rbx((size_t) T * d);
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < d; ++c) {
                double a = 0;
                for (int k = 0; k < L; ++k) a += w[(size_t) c * L + k] * seq[(size_t) (t + k) * d + c];
                ref[(size_t) t * d + c] = bcx[(size_t) t * 3 * d + d + c] * a;
                rbx[(size_t) t * d + c] = seq[(size_t) (L - 1 + t) * d + c];
            }
        hist.assign(seq.end() - (size_t) (L - 1) * d, seq.end());
        const std::vector<half> h = down(y16, ref.size());
        std::vector<float> hf(h.size());
        for (size_t i = 0; i < h.size(); ++i) hf[i] = __half2float(h[i]);
        char name[64];
        std::snprintf(name, sizeof name, "conv::short_conv T %d (state carried)", T);
        report(name, rel(ref, hf), 1e-3);
        std::snprintf(name, sizeof name, "conv::short_conv T %d bx rows", T);
        report(name, rel(rbx, down(bx, rbx.size())), 1e-6);
        report("conv::short_conv state", rel(hist, down(state, hist.size())), 1e-6);
    }
}

void test_route()
{
    const int T = 9, E = 64, k = 4;
    const std::vector<float> lg = randn((size_t) T * E), bias = randn(E);
    int * ids = nanbuf<int>((size_t) T * k);
    float * wts = nanbuf<float>((size_t) T * k);
    truss::ffn::route_sigmoid(up(lg), up(bias), T, E, k, true, 1.f, ids, wts, nullptr);
    const std::vector<int> gi = down(ids, (size_t) T * k);
    const std::vector<float> gw = down(wts, (size_t) T * k);
    int bad = 0;
    std::vector<double> rw;
    std::vector<float> gws;
    for (int t = 0; t < T; ++t) {
        std::vector<int> o(E);
        std::iota(o.begin(), o.end(), 0);
        std::vector<double> s(E);
        for (int e = 0; e < E; ++e) s[e] = 1.0 / (1.0 + std::exp(-(double) lg[(size_t) t * E + e]));
        std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return s[a] + bias[a] > s[b] + bias[b]; });
        double sum = 0;
        for (int j = 0; j < k; ++j) sum += s[o[j]];
        for (int j = 0; j < k; ++j) {
            bad += gi[(size_t) t * k + j] != o[j];
            rw.push_back(s[o[j]] / (sum + 1e-6)), gws.push_back(gw[(size_t) t * k + j]);
        }
    }
    std::printf("%-44s %d of %d ids differ  %s\n", "ffn::route_sigmoid ids", bad, T * k, bad ? "FAIL" : "PASS");
    fails += bad != 0;
    report("ffn::route_sigmoid weights", rel(rw, gws), 1e-6);
}

// B2: route_sigmoid_gemv (fused lone-router projection + route, T == 1) must be bit-identical to f32_gemv +
// route_sigmoid: the same WPO=2 arithmetic, the same tail. Words compared exactly (routing flips on last bits).
void test_route_fused()
{
    const int cases[][2] = { { 64, 2048 }, { 256, 2048 } };
    for (const auto & c : cases) {
        const int E = c[0], in = c[1], k = 4;
        const float *dW = up(randn((size_t) E * in)), *dbias = up(randn(E)), *dx = up(randn(in));
        float *lg_a = nanbuf<float>(E), *lg_b = nanbuf<float>(E);
        int *ida = nanbuf<int>(k), *idb = nanbuf<int>(k);
        float *wta = nanbuf<float>(k), *wtb = nanbuf<float>(k);
        unsigned * arrive;
        TRUSS_CUDA(cudaMalloc(&arrive, sizeof(unsigned)));
        TRUSS_CUDA(cudaMemset(arrive, 0, sizeof(unsigned)));
        truss::dense::f32_gemv(dW, in, E, dx, 1, lg_a, nullptr);
        truss::ffn::route_sigmoid(lg_a, dbias, 1, E, k, true, 1.f, ida, wta, nullptr);
        truss::ffn::route_sigmoid_gemv(dW, in, dbias, dx, E, k, lg_b, idb, wtb, arrive, nullptr);
        TRUSS_CUDA(cudaDeviceSynchronize());
        const std::vector<float> ga = down(lg_a, E), gb = down(lg_b, E);
        const std::vector<int> ia = down(ida, k), ib = down(idb, k);
        const std::vector<float> wa = down(wta, k), wb = down(wtb, k);
        size_t nd = 0;
        for (int i = 0; i < E; ++i) nd += std::memcmp(&ga[i], &gb[i], 4) != 0;
        for (int i = 0; i < k; ++i)
            nd += std::memcmp(&ia[i], &ib[i], 4) != 0 || std::memcmp(&wa[i], &wb[i], 4) != 0;
        char name[64];
        std::snprintf(name, sizeof name, "ffn::route_sigmoid_gemv E %d words differ", E);
        std::printf("%-44s %zu  %s\n", name, nd, nd ? "FAIL" : "PASS");
        fails += nd != 0;
        cudaFree((void *) dW), cudaFree((void *) dbias), cudaFree((void *) dx);
        cudaFree(lg_a), cudaFree(lg_b), cudaFree(ida), cudaFree(idb), cudaFree(wta), cudaFree(wtb), cudaFree(arrive);
    }
}

}  // namespace

int main()
{
    test_rms();
    test_conv();
    test_route();
    test_route_fused();
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
