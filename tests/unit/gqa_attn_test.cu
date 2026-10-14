// attn::gqa_prepare + gqa_attention vs an fp64 CPU reference (per-head RMSNorm, NEOX rope, causal softmax over the
// cache), LFM2 shape (H 32, HKV 8, D 64, rope base 1e7): a prompt in two chunks (the second sees the first's cache),
// then decode steps and a 5-row window at a long position (the split-cells path). Inputs N(0,1); outputs NaN-filled
// first. Gate: rel <= 3e-3 (fp16 q / K / V / output, as the DSA kernel's 2.9e-4 per element plus the long sums).
// usage: gqa_attn_test
#include "kernels/attn/gqa.cuh"

#include "core/cuda_check.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using truss::attn::GqaShape;

std::vector<float> randn(size_t n, std::mt19937 & g)
{
    std::normal_distribution<float> d(0.f, 1.f);
    std::vector<float> v(n);
    for (float & x : v) x = d(g);
    return v;
}

template <class X> X * up(const std::vector<X> & v)
{
    X * p;
    TRUSS_CUDA(cudaMalloc(&p, v.size() * sizeof(X)));
    TRUSS_CUDA(cudaMemcpy(p, v.data(), v.size() * sizeof(X), cudaMemcpyHostToDevice));
    return p;
}

// norm + rope of one head [D] at position pos
void prep(const GqaShape & s, const float * x, const float * w, int pos, bool rope, double * out)
{
    const int D = s.D;
    double ss = 0;
    for (int d = 0; d < D; ++d) ss += (double) x[d] * x[d];
    const double r = w ? 1.0 / std::sqrt(ss / D + s.eps) : 1.0;
    std::vector<double> a(D);
    for (int d = 0; d < D; ++d) a[d] = x[d] * r * (w ? w[d] : 1.0);
    for (int i = 0; i < D / 2; ++i) {
        const double ang = rope ? pos * std::pow((double) s.rope_base, -2.0 * i / D) : 0.0;
        out[i] = a[i] * std::cos(ang) - a[i + D / 2] * std::sin(ang);
        out[i + D / 2] = a[i + D / 2] * std::cos(ang) + a[i] * std::sin(ang);
    }
}

}  // namespace

int main()
{
    const GqaShape s{ 32, 8, 64, 1e7f, 1e-5f };
    const int H = s.H, HKV = s.HKV, D = s.D, G = H / HKV;
    const int n_ctx = 6144;
    std::mt19937 rng(11);
    const std::vector<float> qn = randn(D, rng), kn = randn(D, rng);
    float * d_qn = up(qn), * d_kn = up(kn);
    half * kc, * vc;
    TRUSS_CUDA(cudaMalloc(&kc, sizeof(half) * n_ctx * HKV * D));
    TRUSS_CUDA(cudaMalloc(&vc, sizeof(half) * n_ctx * HKV * D));
    std::vector<double> Kr((size_t) n_ctx * HKV * D), Vr((size_t) n_ctx * HKV * D);   // reference cache
    const size_t ws_bytes = truss::attn::gqa_workspace_bytes(s, 32, n_ctx);
    void * ws;
    TRUSS_CUDA(cudaMalloc(&ws, std::max<size_t>(ws_bytes, 16)));
    int fails = 0;
    // chunks: (pos0, T); the long gap after 1000 is filled by an unchecked chunk so decode sees a 5K-cell cache
    const int plan[][3] = { { 0, 600, 1 }, { 600, 400, 1 }, { 1000, 4096, 0 }, { 5096, 1, 1 }, { 5097, 1, 1 }, { 5098, 5, 1 } };
    for (const auto & c : plan) {
        const int pos0 = c[0], T = c[1];
        const std::vector<float> q = randn((size_t) T * H * D, rng), k = randn((size_t) T * HKV * D, rng),
                                 v = randn((size_t) T * HKV * D, rng);
        float * dq = up(q), * dk = up(k), * dv = up(v);
        half * q16, * out16;
        TRUSS_CUDA(cudaMalloc(&q16, sizeof(half) * T * H * D));
        TRUSS_CUDA(cudaMalloc(&out16, sizeof(half) * T * H * D));
        TRUSS_CUDA(cudaMemset(out16, 0xff, sizeof(half) * T * H * D));   // NaN
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0), cudaEventCreate(&e1);
        truss::attn::gqa_prepare(s, dq, dk, dv, d_qn, d_kn, pos0, T, q16, kc, vc, nullptr);
        cudaEventRecord(e0);
        truss::attn::gqa_attention(s, q16, kc, vc, pos0, T, out16, ws, ws_bytes, nullptr);
        cudaEventRecord(e1);
        TRUSS_CUDA(cudaDeviceSynchronize());
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        // reference: cache rows, then attention per (t, h)
        std::vector<double> qr((size_t) T * H * D);
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < H; ++h) prep(s, &q[((size_t) t * H + h) * D], qn.data(), pos0 + t, true, &qr[((size_t) t * H + h) * D]);
            for (int h = 0; h < HKV; ++h) {
                prep(s, &k[((size_t) t * HKV + h) * D], kn.data(), pos0 + t, true, &Kr[((size_t) (pos0 + t) * HKV + h) * D]);
                for (int d = 0; d < D; ++d) Vr[((size_t) (pos0 + t) * HKV + h) * D + d] = v[((size_t) t * HKV + h) * D + d];
            }
        }
        if (!c[2]) {
            std::printf("pos0 %5d T %4d: cache fill (unchecked), %.3f ms\n", pos0, T, ms);
            continue;
        }
        std::vector<half> got((size_t) T * H * D);
        TRUSS_CUDA(cudaMemcpy(got.data(), out16, got.size() * sizeof(half), cudaMemcpyDeviceToHost));
        double num = 0, den = 0;
        bool nan = false;
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < H; ++h) {
                const int hk = h / G, n = pos0 + t + 1;
                std::vector<double> sc(n);
                double mx = -1e300;
                for (int j = 0; j < n; ++j) {
                    double a = 0;
                    for (int d = 0; d < D; ++d) a += qr[((size_t) t * H + h) * D + d] * Kr[((size_t) j * HKV + hk) * D + d];
                    sc[j] = a / std::sqrt((double) D), mx = std::max(mx, sc[j]);
                }
                double l = 0;
                for (double & x : sc) l += x = std::exp(x - mx);
                for (int d = 0; d < D; ++d) {
                    double o = 0;
                    for (int j = 0; j < n; ++j) o += sc[j] * Vr[((size_t) j * HKV + hk) * D + d];
                    o /= l;
                    const float g = __half2float(got[((size_t) t * H + h) * D + d]);
                    nan |= std::isnan(g);
                    num += (g - o) * (g - o), den += o * o;
                }
            }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        const bool ok = !nan && rel <= 3e-3;
        fails += !ok;
        std::printf("pos0 %5d T %4d: rel %.2e %s, %.3f ms  %s\n", pos0, T, rel, nan ? "(NaN!)" : "", ms, ok ? "PASS" : "FAIL");
        cudaFree(dq), cudaFree(dk), cudaFree(dv), cudaFree(q16), cudaFree(out16);
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
