// truss::moe::prefill: correctness vs moe::window (tested against llama-paw, moe_window_test) run 8 tokens at a
// time on the same random trellis weights and routing, and timing per layer.
// Shape: Flash-Next layer. Env: TRUSS_KFIX (0 = mixed 1..4 per expert), TRUSS_REPS.
// usage: moe_prefill_test [n_tokens ...]
#include "core/cuda_check.h"
#include "kernels/moe/moe_prefill.cuh"
#include "kernels/moe/moe_window.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace truss;
using Shape = moe::FlashNext;

namespace {

int64_t env_int(const char * n, int64_t d) { const char * e = getenv(n); return e ? atoll(e) : d; }

template <class T> T * upload(const std::vector<T> & v)
{
    T * d;
    TRUSS_CUDA(cudaMalloc(&d, v.size() * sizeof(T)));
    TRUSS_CUDA(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

struct RandomLayer {
    moe::Weights W;
    std::vector<void *> bufs;
    std::vector<int> k_of[3];

    RandomLayer(std::mt19937 & rng, int n_expert, int kfix)
    {
        W.n_expert = n_expert;
        std::uniform_real_distribution<float> mag(0.25f, 1.0f);
        for (int p = 0; p < 3; ++p) {
            const int in = p == 2 ? Shape::D_FF : Shape::D_MODEL, out = p == 2 ? Shape::D_MODEL : Shape::D_FF;
            const int64_t ntiles = (in / 16) * (out / 16);
            std::vector<int32_t> meta(2 * n_expert);
            int64_t words = 0;
            for (int e = 0; e < n_expert; ++e) {
                meta[2 * e] = kfix ? kfix : 1 + (e + p) % 4;
                meta[2 * e + 1] = (int32_t) words;
                words += 16 * meta[2 * e] * ntiles;
                k_of[p].push_back(meta[2 * e]);
            }
            std::vector<uint16_t> tw(words);
            for (auto & w : tw) w = (uint16_t) (rng() & 0xffff);
            std::vector<half> suh((size_t) n_expert * in), svh((size_t) n_expert * out);
            for (auto & h : suh) h = __float2half(((rng() & 1) ? 0.05f : -0.05f) * mag(rng));
            for (auto & h : svh) h = __float2half(((rng() & 1) ? 0.05f : -0.05f) * mag(rng));
            auto * a = upload(tw); auto * b = upload(meta); auto * c = upload(suh); auto * d = upload(svh);
            bufs.insert(bufs.end(), { a, b, c, d });
            W.proj[p] = { a, b, c, d };
        }
    }
    ~RandomLayer() { for (void * b : bufs) cudaFree(b); }
};

int run_case(int T, const RandomLayer & L, std::mt19937 & rng)
{
    constexpr int D = Shape::D_MODEL, TOPK = Shape::TOPK;
    const int E = L.W.n_expert;
    std::vector<int32_t> ids((size_t) T * TOPK);
    std::vector<float> w((size_t) T * TOPK), x((size_t) T * D);
    std::uniform_real_distribution<float> mag(0.25f, 1.0f);
    for (int t = 0; t < T; ++t) {
        std::vector<int32_t> pick;
        while ((int) pick.size() < TOPK) {
            const int32_t e = (int32_t) (rng() % E);
            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
        }
        float sum = 0;
        for (int s = 0; s < TOPK; ++s) ids[t * TOPK + s] = pick[s], w[t * TOPK + s] = mag(rng), sum += w[t * TOPK + s];
        for (int s = 0; s < TOPK; ++s) w[t * TOPK + s] /= sum;
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = nd(rng);
    int * d_ids = upload(ids);
    float * d_w = upload(w), * d_x = upload(x), * d_out, * d_ref;
    TRUSS_CUDA(cudaMalloc(&d_out, sizeof(float) * T * D));
    TRUSS_CUDA(cudaMalloc(&d_ref, sizeof(float) * T * D));
    void * ws, * wws;
    const size_t ws_bytes = moe::prefill_workspace_bytes<Shape>(T);
    TRUSS_CUDA(cudaMalloc(&ws, ws_bytes));
    TRUSS_CUDA(cudaMalloc(&wws, moe::workspace_bytes<Shape>()));
    // NaN-fill workspace and output: anything the op skips fails the check
    TRUSS_CUDA(cudaMemset(ws, 0xff, ws_bytes));
    TRUSS_CUDA(cudaMemset(d_out, 0xff, sizeof(float) * T * D));
    moe::workspace_init<Shape>(wws, nullptr);

    moe::prefill<Shape>(L.W, d_x, d_ids, d_w, T, d_out, ws, T, nullptr);
    for (int t0 = 0; t0 < T; t0 += moe::MAX_ROWS)
        moe::window<Shape>(L.W, d_x + (size_t) t0 * D, d_ids + t0 * TOPK, d_w + t0 * TOPK,
                           std::min(moe::MAX_ROWS, T - t0), d_ref + (size_t) t0 * D, wws, nullptr);
    TRUSS_CUDA(cudaDeviceSynchronize());
    std::vector<float> a((size_t) T * D), b((size_t) T * D);
    TRUSS_CUDA(cudaMemcpy(a.data(), d_out, a.size() * 4, cudaMemcpyDeviceToHost));
    TRUSS_CUDA(cudaMemcpy(b.data(), d_ref, b.size() * 4, cudaMemcpyDeviceToHost));
    double worst = 0, num = 0, den = 0;
    int64_t nonfinite = 0;
    for (int t = 0; t < T; ++t) {
        double dd = 0, rr = 0;
        for (int i = 0; i < D; ++i) {
            const double av = a[(size_t) t * D + i], bv = b[(size_t) t * D + i];
            nonfinite += !std::isfinite(av);
            dd += (av - bv) * (av - bv);
            rr += bv * bv;
        }
        worst = std::max(worst, std::sqrt(dd / std::max(rr, 1e-30)));
        num += dd;
        den += rr;
    }
    const double rel = std::sqrt(num / den);
    // moe_window accumulates in fp16 per 2 k slices (TRACKER #16); the gate covers its rounding, not a wrong formula
    // prefix invariance: a row's output must not depend on the other rows of the chunk (chunked prefill == one shot)
    int64_t prefix_diff = 0;
    const int Th = T / 2;
    if (Th > 0) {
        TRUSS_CUDA(cudaMemset(d_ref, 0xff, sizeof(float) * T * D));
        moe::prefill<Shape>(L.W, d_x, d_ids, d_w, Th, d_ref, ws, T, nullptr);
        std::vector<float> h((size_t) Th * D);
        TRUSS_CUDA(cudaMemcpy(h.data(), d_ref, h.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < h.size(); ++i) prefix_diff += h[i] != a[i];
    }
    // skipped pairs (TRACKER #117: id -1, the CPU tier computes them): equal to the same pairs at weight 0
    int64_t skip_diff = 0;
    {
        std::vector<int32_t> ids_m = ids;
        std::vector<float> w0 = w;
        for (size_t p = 0; p < ids.size(); ++p)
            if (rng() % 3 == 0) ids_m[p] = -1, w0[p] = 0.f;
        int * d_ids_m = upload(ids_m);
        float * d_w0 = upload(w0);
        TRUSS_CUDA(cudaMemset(d_ref, 0xff, sizeof(float) * T * D));
        moe::prefill<Shape>(L.W, d_x, d_ids_m, d_w, T, d_ref, ws, T, nullptr);
        std::vector<float> m((size_t) T * D), z((size_t) T * D);
        TRUSS_CUDA(cudaMemcpy(m.data(), d_ref, m.size() * 4, cudaMemcpyDeviceToHost));
        moe::prefill<Shape>(L.W, d_x, d_ids, d_w0, T, d_ref, ws, T, nullptr);
        TRUSS_CUDA(cudaMemcpy(z.data(), d_ref, z.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < m.size(); ++i) skip_diff += !(m[i] == z[i]);
        cudaFree(d_ids_m); cudaFree(d_w0);
    }
    const bool ok = !nonfinite && rel <= 5e-3 && worst <= 2e-2 && prefix_diff == 0 && skip_diff == 0;

    const int reps = (int) env_int("TRUSS_REPS", 10);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    moe::prefill<Shape>(L.W, d_x, d_ids, d_w, T, d_out, ws, T, nullptr);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) moe::prefill<Shape>(L.W, d_x, d_ids, d_w, T, d_out, ws, T, nullptr);
    cudaEventRecord(e1);
    TRUSS_CUDA(cudaEventSynchronize(e1));
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    ms /= reps;
    const double flop = 2.0 * T * TOPK * 3.0 * D * Shape::D_FF;
    std::printf("T=%-5d rel %.2e worst token %.2e nonfinite %lld, first T/2 rows alone differ in %lld values, skip %lld %s | "
                "%.2f ms/layer, %.1f TFLOPS, %.1f us/token/layer\n",
                T, rel, worst, (long long) nonfinite, (long long) prefix_diff, (long long) skip_diff, ok ? "PASS" : "FAIL", ms, flop / (ms * 1e-3) / 1e12,
                ms * 1e3 / T);
    cudaFree(d_ids); cudaFree(d_w); cudaFree(d_x); cudaFree(d_out); cudaFree(d_ref); cudaFree(ws); cudaFree(wws);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv)
{
    std::vector<int> Ts;
    for (int i = 1; i < argc; ++i) Ts.push_back(atoi(argv[i]));
    if (Ts.empty()) Ts = { 64, 512, 2048, 8192 };
    try {
        std::mt19937 rng(4321);
        const RandomLayer L(rng, 512, (int) env_int("TRUSS_KFIX", 0));
        int fails = 0;
        for (int T : Ts) fails += run_case(T, L, rng);
        std::printf("%s\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
