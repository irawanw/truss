// Speculative sampling is lossless: over many trials, the first token a window emits (the accepted draft, or the
// residual draw when it is rejected) must be distributed as p_0, the target row's kept probabilities; the
// acceptance rate must match sum_v min(p_0(v), q(v)). Chi-square on a toy vocabulary, for several sampler settings.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "kernels/sampling/spec.cuh"

using namespace truss::sampling;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s\n", cudaGetErrorString(e_)); return 1; } } while (0)

static int run(const char * name, SampleParams sp, float draft_shift, int V, int trials, unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<float> tgt(2 * V), drf(V);   // target rows 0 (judged) and 1 (bonus); draft row: target + noise
    for (int v = 0; v < V; ++v) tgt[v] = nd(rng), tgt[V + v] = nd(rng);
    for (int v = 0; v < V; ++v) drf[v] = tgt[v] + draft_shift * nd(rng);

    float *d_tgt, *d_p, *d_drf, *d_q;
    int *d_x, *d_out;
    CK(cudaMalloc(&d_tgt, sizeof(float) * 2 * V));
    CK(cudaMalloc(&d_p, sizeof(float) * 2 * V));
    CK(cudaMalloc(&d_drf, sizeof(float) * V));
    CK(cudaMalloc(&d_q, sizeof(float) * V));
    CK(cudaMalloc(&d_x, sizeof(int)));
    CK(cudaMalloc(&d_out, sizeof(int) * 2));
    CK(cudaMemcpy(d_tgt, tgt.data(), sizeof(float) * 2 * V, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_drf, drf.data(), sizeof(float) * V, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_p, d_tgt, sizeof(float) * 2 * V, cudaMemcpyDeviceToDevice));
    probs_inplace(d_p, V, 2, sp, 0);
    std::vector<float> p(V), q(V);
    CK(cudaMemcpy(p.data(), d_p, sizeof(float) * V, cudaMemcpyDeviceToHost));
    draft_sample(d_drf, V, nullptr, V, sp, 0, d_q, d_x, 0);
    CK(cudaMemcpy(q.data(), d_q, sizeof(float) * V, cudaMemcpyDeviceToHost));
    double overlap = 0;
    for (int v = 0; v < V; ++v) overlap += std::min(p[v], q[v]);

    std::vector<long> cnt(V, 0);
    long acc = 0;
    std::vector<int> xs(1), outs(2);
    for (int t = 0; t < trials; ++t) {
        draft_sample(d_drf, V, nullptr, V, sp, (uint64_t) t * 7, d_q, d_x, 0);
        spec_accept(d_p, d_q, d_x, 1, V, sp.seed, (uint64_t) t * 7 + 3, d_out, 0);
        CK(cudaMemcpy(xs.data(), d_x, sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(outs.data(), d_out, sizeof(int) * 2, cudaMemcpyDeviceToHost));
        const int first = outs[0] >= 1 ? xs[0] : outs[1];
        if (first < 0 || first >= V) { std::printf("%s: bad token %d\n", name, first); return 1; }
        ++cnt[first];
        acc += outs[0] >= 1;
    }
    double chi = 0;
    int bins = 0;
    for (int v = 0; v < V; ++v) {
        const double e = p[v] * trials;
        if (e < 5) {
            if (cnt[v] > 0 && p[v] == 0.f) { std::printf("%s: token %d emitted %ld times with p = 0\n", name, v, cnt[v]); return 1; }
            continue;
        }
        chi += (cnt[v] - e) * (cnt[v] - e) / e, ++bins;
    }
    const int dof = bins - 1;
    const double lim = dof + 5.0 * std::sqrt(2.0 * dof);   // ~5 sigma
    const double ar = (double) acc / trials;
    const bool ok = chi < lim && std::fabs(ar - overlap) < 5.0 * std::sqrt(overlap * (1 - overlap) / trials) + 1e-3;
    std::printf("%-22s V %d trials %d: chi2 %.1f (dof %d, limit %.1f)  accept %.4f vs overlap %.4f  %s\n", name, V,
                trials, chi, dof, lim, ar, overlap, ok ? "ok" : "FAIL");
    cudaFree(d_tgt), cudaFree(d_p), cudaFree(d_drf), cudaFree(d_q), cudaFree(d_x), cudaFree(d_out);
    return ok ? 0 : 1;
}

int main()
{
    int bad = 0;
    SampleParams plain;
    plain.temperature = 1.f, plain.seed = 11;
    bad += run("T1 plain", plain, 0.5f, 64, 200000, 1);
    SampleParams served;   // the brain's request settings
    served.temperature = 0.8f, served.top_p = 0.95f, served.top_k = 40, served.min_p = 0.05f, served.seed = 12;
    bad += run("T0.8 p0.95 k40 mp0.05", served, 0.5f, 256, 200000, 2);
    bad += run("served, far draft", served, 2.0f, 256, 200000, 3);
    SampleParams cold;
    cold.temperature = 0.3f, cold.top_k = 5, cold.seed = 13;
    bad += run("T0.3 k5", cold, 0.7f, 128, 200000, 4);
    std::printf(bad ? "spec_test: FAIL\n" : "spec_test: all ok\n");
    return bad ? 1 : 0;
}
