// The doorbell's routing and publish must give the plain three launches' BYTES, not close values: TRUSS_ROUTE_PUBLISH
// claims to move work (route + route(pred) + publish -> one launch), not arithmetic, and the driver starts reading the
// mapped buffers the moment it sees the doorbell word - so a fence that does not fence is a wrong-answer bug, not a
// race nobody notices. On the engine's own shapes (E 512, k 10, d_model 2560, rows 1..8), hint on and off, with and
// without rows to publish, and at every copy-CTA count it may launch:
//   ids and weights on device, ids / prediction / weights in mapped memory and the FFN rows in mapped memory are
//   identical to the plain sequence, the doorbell carries seq, no byte outside the published rows moves (poison past
//   the end of each mapped buffer), and one arrival counter survives back-to-back launches at different row counts
//   and CTA counts - it has to reset itself, since the CTA count is not constant.
// usage: route_publish_test
#include "core/cuda_check.h"
#include "kernels/ffn/ffn_ops.cuh"
#include "runtime/doorbell.cuh"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace truss;

namespace {

constexpr int E = 512, K = 10, D = 2560, R = 8;   // the router, its top-k, d_model, cpu::MAX_ROWS
constexpr int POISON = 32;                        // floats of guard past the published rows
constexpr float POISON_V = -1234.5f;

void * mapped(size_t bytes)
{
    void * p = nullptr;
    TRUSS_CUDA(cudaHostAlloc(&p, bytes, cudaHostAllocMapped));
    return p;
}

// One layer's publish target: the mapped buffers the driver reads, each with a guard tail, plus the device outputs.
struct Side {
    int * ids = nullptr, * pred = nullptr;      // mapped [R][k]
    float * w = nullptr;                        // mapped [R][k]
    float * x = nullptr;                        // mapped [R][d] + POISON guard
    int * bell = nullptr;                       // mapped word
    int * d_ids = nullptr;                      // device [R][k]
    float * d_w = nullptr;                      // device [R][k]
    unsigned * arrive = nullptr;                // device word (the fused launch only)
};

Side make_side(bool arrive)
{
    Side s;
    s.ids = (int *) mapped(sizeof(int) * R * K);
    s.pred = (int *) mapped(sizeof(int) * R * K);
    s.w = (float *) mapped(sizeof(float) * R * K);
    s.x = (float *) mapped(sizeof(float) * (size_t) R * D + sizeof(float) * POISON);
    s.bell = (int *) mapped(sizeof(int));
    std::memset(s.ids, 0, sizeof(int) * R * K);
    std::memset(s.pred, 0, sizeof(int) * R * K);
    std::memset(s.w, 0, sizeof(float) * R * K);
    std::memset(s.x, 0, sizeof(float) * (size_t) R * D);
    for (int i = 0; i < POISON; ++i) s.x[(size_t) R * D + i] = POISON_V;
    *s.bell = 0;
    TRUSS_CUDA(cudaMalloc(&s.d_ids, sizeof(int) * R * K));
    TRUSS_CUDA(cudaMalloc(&s.d_w, sizeof(float) * R * K));
    TRUSS_CUDA(cudaMemset(s.d_ids, 0, sizeof(int) * R * K));
    TRUSS_CUDA(cudaMemset(s.d_w, 0, sizeof(float) * R * K));
    if (arrive) {
        TRUSS_CUDA(cudaMalloc(&s.arrive, sizeof(unsigned)));
        TRUSS_CUDA(cudaMemset(s.arrive, 0, sizeof(unsigned)));
    }
    return s;
}

int fails = 0;

void check(bool ok, const char * what, int T, int hint, int n_x, int ctas)
{
    if (!ok) std::printf("  FAIL %s (T %d, hint %d, n_x %d, ctas %d)\n", what, T, hint, n_x, ctas);
    fails += !ok;
}

// Bytes the driver and the window kernel end up with, compared against the plain three launches.
bool same(const Side & a, const Side & b, int T, bool hint, int n_x)
{
    std::vector<int> ai((size_t) T * K), bi((size_t) T * K);
    std::vector<float> af((size_t) T * K), bf((size_t) T * K);
    TRUSS_CUDA(cudaMemcpy(ai.data(), a.d_ids, sizeof(int) * (size_t) T * K, cudaMemcpyDeviceToHost));
    TRUSS_CUDA(cudaMemcpy(bi.data(), b.d_ids, sizeof(int) * (size_t) T * K, cudaMemcpyDeviceToHost));
    TRUSS_CUDA(cudaMemcpy(af.data(), a.d_w, sizeof(float) * (size_t) T * K, cudaMemcpyDeviceToHost));
    TRUSS_CUDA(cudaMemcpy(bf.data(), b.d_w, sizeof(float) * (size_t) T * K, cudaMemcpyDeviceToHost));
    if (std::memcmp(ai.data(), bi.data(), sizeof(int) * (size_t) T * K) != 0) return false;
    if (std::memcmp(af.data(), bf.data(), sizeof(float) * (size_t) T * K) != 0) return false;
    if (std::memcmp(a.ids, b.ids, sizeof(int) * (size_t) T * K) != 0) return false;
    if (std::memcmp(a.w, b.w, sizeof(float) * (size_t) T * K) != 0) return false;
    if (hint && std::memcmp(a.pred, b.pred, sizeof(int) * (size_t) T * K) != 0) return false;
    if (n_x && std::memcmp(a.x, b.x, sizeof(float) * (size_t) n_x) != 0) return false;
    return true;
}

bool guard_clean(const Side & s)
{
    for (int i = 0; i < POISON; ++i)
        if (s.x[(size_t) R * D + i] != POISON_V) return false;
    return true;
}

}  // namespace

int main()
{
    std::mt19937 g(7);
    std::normal_distribution<float> nd(0.f, 1.f);

    std::vector<float> hlogits((size_t) R * E), hpred((size_t) R * E), hx((size_t) R * D);
    for (auto & v : hlogits) v = nd(g);
    for (auto & v : hpred) v = nd(g);
    for (auto & v : hx) v = nd(g);
    float *logits = nullptr, *predl = nullptr, *x = nullptr;
    float * pw = nullptr;   // the plain hint route's weights: written, never read
    TRUSS_CUDA(cudaMalloc(&logits, sizeof(float) * (size_t) R * E));
    TRUSS_CUDA(cudaMalloc(&predl, sizeof(float) * (size_t) R * E));
    TRUSS_CUDA(cudaMalloc(&x, sizeof(float) * (size_t) R * D));
    TRUSS_CUDA(cudaMalloc(&pw, sizeof(float) * R * K));
    TRUSS_CUDA(cudaMemcpy(logits, hlogits.data(), sizeof(float) * (size_t) R * E, cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(predl, hpred.data(), sizeof(float) * (size_t) R * E, cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(x, hx.data(), sizeof(float) * (size_t) R * D, cudaMemcpyHostToDevice));

    int * d_pred = nullptr;   // the plain path publishes from a device buffer; the fused one has no such thing
    TRUSS_CUDA(cudaMalloc(&d_pred, sizeof(int) * R * K));

    Side ref = make_side(false), got = make_side(true);
    cudaStream_t s;
    TRUSS_CUDA(cudaStreamCreate(&s));

    for (int T = 1; T <= R; ++T)
        for (int hint = 0; hint <= 1; ++hint)
            for (int rows = 0; rows <= 1; ++rows) {
                const int n_x = rows ? T * D : 0;
                for (int ctas : {1, 2, 8, 16}) {
                    const int seq = 5;   // both sides get the same word; the fused one must ring it too
                    // the plain three launches
                    ffn::route(logits, T, E, K, ref.d_ids, ref.d_w, s);
                    if (hint) ffn::route(predl, T, E, K, d_pred, pw, s);
                    runtime::publish(ref.d_ids, hint ? d_pred : nullptr, ref.d_w, T * K, x, n_x, ref.ids,
                                     hint ? ref.pred : nullptr, ref.w, ref.x, ref.bell, seq, s);
                    // the fused launch
                    ffn::route_publish(logits, hint ? predl : nullptr, T, E, K, got.d_ids, got.d_w, x, n_x, got.ids,
                                       hint ? got.pred : nullptr, got.w, got.x, got.bell, seq, got.arrive, ctas, s);
                    TRUSS_CUDA(cudaStreamSynchronize(s));
                    check(*ref.bell == seq, "plain doorbell", T, hint, n_x, ctas);
                    check(*got.bell == seq, "fused doorbell", T, hint, n_x, ctas);
                    check(same(ref, got, T, hint, n_x), "published bytes differ", T, hint, n_x, ctas);
                    check(guard_clean(ref) && guard_clean(got), "wrote past the published rows", T, hint, n_x, ctas);
                    unsigned left = 1;
                    TRUSS_CUDA(cudaMemcpy(&left, got.arrive, sizeof(unsigned), cudaMemcpyDeviceToHost));
                    check(left == 0, "arrival counter left behind", T, hint, n_x, ctas);
                }
            }
    std::printf("route+publish == route, route, publish: %s (%d rows cases x hint x ctas)\n",
                fails ? "FAIL" : "ok", R * 2 * 2 * 4);

    // One bell, one counter, launches that differ in rows and in CTA count - what a pass actually does when a window
    // shrinks mid-run. Every launch must still ring, and ring with its own data.
    for (int i = 1; i <= 200 && !fails; ++i) {
        const int T = i % 3 ? R : 1;
        const int ctas = i % 2 ? 8 : 1;
        const int n_x = T * D;
        const int seq = i;
        ffn::route(logits, T, E, K, ref.d_ids, ref.d_w, s);
        ffn::route(predl, T, E, K, d_pred, pw, s);
        runtime::publish(ref.d_ids, d_pred, ref.d_w, T * K, x, n_x, ref.ids, ref.pred, ref.w, ref.x, ref.bell, seq, s);
        ffn::route_publish(logits, predl, T, E, K, got.d_ids, got.d_w, x, n_x, got.ids, got.pred, got.w, got.x,
                           got.bell, seq, got.arrive, ctas, s);
        TRUSS_CUDA(cudaStreamSynchronize(s));
        check(*got.bell == seq, "seq", T, 1, n_x, ctas);
        check(same(ref, got, T, true, n_x), "published bytes differ", T, 1, n_x, ctas);
        unsigned left = 1;
        TRUSS_CUDA(cudaMemcpy(&left, got.arrive, sizeof(unsigned), cudaMemcpyDeviceToHost));
        check(left == 0, "arrival counter left behind", T, 1, n_x, ctas);
    }
    std::printf("200 back-to-back launches, rows and CTA count alternating: %s\n", fails ? "FAIL" : "ok");

    std::printf("route_publish_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
