// gdn::delta_rule vs ref::gated_delta_rule (same math, token-sequential fp32; only summation order differs) on
// random inputs shaped like Flash-Next's GDN (16 key heads, 48 value heads, head 128): unit-norm q, k (as after
// the l2 norm), decays exp(g) in [0.1, 1], beta in (0, 1), a nonzero start state. Checks outputs and final state.
// usage: gdn_prefill_test [T ...]
#include "core/cuda_check.h"
#include "kernels/gdn/gdn_prefill.cuh"
#include "kernels/reference/ref.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace truss;

namespace {

constexpr int HK = 16, HV = 48, S = 128;

float * upload(const std::vector<float> & v)
{
    float * d;
    TRUSS_CUDA(cudaMalloc(&d, v.size() * 4));
    TRUSS_CUDA(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

double rel(const float * a, const float * b, size_t n)
{
    std::vector<float> x(n), y(n);
    TRUSS_CUDA(cudaMemcpy(x.data(), a, n * 4, cudaMemcpyDeviceToHost));
    TRUSS_CUDA(cudaMemcpy(y.data(), b, n * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) return INFINITY;
        num += (x[i] - (double) y[i]) * (x[i] - (double) y[i]);
        den += (double) y[i] * y[i];
    }
    return std::sqrt(num / den);
}

int run(int T, std::mt19937 & rng)
{
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    std::vector<float> q((size_t) T * HK * S), k(q.size()), v((size_t) T * HV * S), g((size_t) T * HV), b(g.size()),
        st((size_t) HV * S * S);
    for (auto * a : { &q, &k }) {
        for (auto & x : *a) x = nd(rng);
        for (size_t r = 0; r < a->size() / S; ++r) {   // unit norm per head
            double n2 = 0;
            for (int i = 0; i < S; ++i) n2 += (*a)[r * S + i] * (*a)[r * S + i];
            for (int i = 0; i < S; ++i) (*a)[r * S + i] /= (float) std::sqrt(n2);
        }
    }
    for (auto & x : v) x = nd(rng);
    for (auto & x : g) x = std::log(0.1f + 0.9f * ud(rng));
    for (auto & x : b) x = ud(rng);
    for (auto & x : st) x = 0.1f * nd(rng);
    float * dq = upload(q), * dk = upload(k), * dv = upload(v), * dg = upload(g), * db = upload(b);
    float * s_ref = upload(st), * s_fast = upload(st), * o_ref, * o_fast;
    TRUSS_CUDA(cudaMalloc(&o_ref, v.size() * 4));
    TRUSS_CUDA(cudaMalloc(&o_fast, v.size() * 4));
    TRUSS_CUDA(cudaMemset(o_fast, 0xff, v.size() * 4));

    ref::gated_delta_rule(dq, dk, dv, dg, db, s_ref, o_ref, T, HK, HV, S, nullptr);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    gdn::delta_rule(dq, dk, dv, dg, db, s_fast, o_fast, T, HK, HV, nullptr);
    cudaEventRecord(e1);
    TRUSS_CUDA(cudaEventSynchronize(e1));
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    const double ro = rel(o_fast, o_ref, v.size()), rs = rel(s_fast, s_ref, st.size());
    const bool ok = ro <= 1e-5 && rs <= 1e-5;
    std::printf("T=%-5d out rel %.1e state rel %.1e %s | %.3f ms, %.2f us/token/layer\n", T, ro, rs, ok ? "PASS" : "FAIL",
                ms, ms * 1e3 / T);
    for (float * p : { dq, dk, dv, dg, db, s_ref, s_fast, o_ref, o_fast }) cudaFree(p);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv)
{
    std::vector<int> Ts;
    for (int i = 1; i < argc; ++i) Ts.push_back(std::atoi(argv[i]));
    if (Ts.empty()) Ts = { 1, 12, 512, 2048, 8192 };
    try {
        std::mt19937 rng(7);
        int fails = 0;
        for (int T : Ts) fails += run(T, rng);
        std::printf("%s\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
