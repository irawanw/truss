// The doorbell's routing and publish must give the plain three launches' BYTES, not close values: TRUSS_ROUTE_PUBLISH
// claims to move work (route + route(pred) + publish -> one launch), not arithmetic, and the driver starts reading the
// mapped buffers the moment it sees the doorbell word - so a fence that does not fence is a wrong-answer bug, not a
// race nobody notices. On the engine's own shapes (E 512, k 10, d_model 2560, rows 1..8), hint on and off, with and
// without rows to publish, and at every copy-CTA count it may launch:
//   ids and weights on device, ids / prediction / weights in mapped memory and the FFN rows in mapped memory are
//   identical to the plain sequence, the doorbell carries seq, no byte outside the published rows moves, and one
//   arrival counter survives back-to-back launches at different row counts and CTA counts - it has to reset itself,
//   since the CTA count is not constant.
// Two things here are not "compare the two paths" and exist because comparing the two paths is not enough:
//   - the routing scratch is EXACTLY T*K entries with a guard pattern after it, the way the engine's arena hands it
//     out. A kernel that runs a warp past the last row overruns it, and two paths that overrun the same oversized
//     buffer agree with each other: route_kernel lost its `t >= T` guard in the refactor that moved the top-k into a
//     shared device function, the base path passed this test as written, and the served model came back with
//     "ExpertStore::fetch: expert id 1065353216" - 0x3F800000, a neighbouring scratch float read as an expert id.
//   - a host top-k of the same logits, independent of both device paths, so arithmetic that moved on BOTH sides at
//     once (a different k, a renormalisation, a tie going the other way) still has something to disagree with.
// usage: route_publish_test
#include "core/cuda_check.h"
#include "kernels/ffn/ffn_ops.cuh"
#include "runtime/doorbell.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace truss;

namespace {

constexpr int E = 512, K = 10, D = 2560, R = 8;   // the router, its top-k, d_model, cpu::MAX_ROWS
constexpr int POISON = 32;                        // guard entries past the published rows
constexpr float POISON_V = -1234.5f;
constexpr int GUARD = 0x5A5A5A5A;                 // in the device scratch, where an id would be

void * mapped(size_t bytes)
{
    void * p = nullptr;
    TRUSS_CUDA(cudaHostAlloc(&p, bytes, cudaHostAllocMapped));
    return p;
}

// One device layer's routing outputs, sized the way the engine's scratch arena sizes them: exactly T*K ints and
// T*K floats, back to back, each followed by POISON guard words. Nothing here is padded for a kernel's convenience.
struct Scratch {
    int * ids = nullptr;
    float * w = nullptr;
    int rows = 0;
};

Scratch make_scratch(int T)
{
    Scratch sc;
    sc.rows = T;
    const size_t n = (size_t) T * K, block = n + POISON;
    int * base = nullptr;
    TRUSS_CUDA(cudaMalloc(&base, sizeof(int) * 2 * block));
    const std::vector<int> pat(2 * block, GUARD);   // poison first: a launch that writes its own rows and nothing
    TRUSS_CUDA(cudaMemcpy(base, pat.data(), sizeof(int) * 2 * block, cudaMemcpyHostToDevice));          // else is fine
    sc.ids = base;
    sc.w = reinterpret_cast<float *>(base + block);
    return sc;
}

bool scratch_clean(const Scratch & sc)
{
    const size_t n = (size_t) sc.rows * K, block = n + POISON;
    std::vector<int> got(2 * block, 0);
    TRUSS_CUDA(cudaMemcpy(got.data(), sc.ids, sizeof(int) * 2 * block, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < POISON; ++i)
        if (got[n + i] != GUARD || got[block + n + i] != GUARD) return false;
    return true;
}

// One layer's publish target: the mapped buffers the driver reads, each with a guard tail, plus the bell and the
// arrival counter. The device routing is NOT here - see Scratch.
struct Side {
    int * ids = nullptr, * pred = nullptr;      // mapped [R][k]
    float * w = nullptr;                        // mapped [R][k]
    float * x = nullptr;                        // mapped [R][d] + POISON guard
    int * bell = nullptr;                       // mapped word
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
bool same(const Side & a, const Scratch & as, const Side & b, const Scratch & bs, int T, bool hint, int n_x)
{
    std::vector<int> ai((size_t) T * K), bi((size_t) T * K);
    std::vector<float> af((size_t) T * K), bf((size_t) T * K);
    TRUSS_CUDA(cudaMemcpy(ai.data(), as.ids, sizeof(int) * (size_t) T * K, cudaMemcpyDeviceToHost));
    TRUSS_CUDA(cudaMemcpy(bi.data(), bs.ids, sizeof(int) * (size_t) T * K, cudaMemcpyDeviceToHost));
    TRUSS_CUDA(cudaMemcpy(af.data(), as.w, sizeof(float) * (size_t) T * K, cudaMemcpyDeviceToHost));
    TRUSS_CUDA(cudaMemcpy(bf.data(), bs.w, sizeof(float) * (size_t) T * K, cudaMemcpyDeviceToHost));
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

// What the router means, on the host: softmax over the row's logits, the k largest probabilities in descending
// order (a tie to the lower expert), renormalised by their sum clamped at 6.103515625e-5. The ids must match bit for
// bit; the weights to a relative 1e-6, because the kernel sums the softmax butterfly-wide and the host in a line, so
// the last bit of a weight is not defined by the algorithm. Random logits, so no two probabilities are that close.
void ref_route(const std::vector<float> & hl, int T, int k, std::vector<int> & ids, std::vector<float> & w)
{
    ids.assign((size_t) T * k, -1);
    w.assign((size_t) T * k, 0.f);
    for (int t = 0; t < T; ++t) {
        const float * l = hl.data() + (size_t) t * E;
        float mx = l[0], sum = 0.f;
        for (int e = 0; e < E; ++e) mx = std::max(mx, l[e]);
        std::vector<float> p((size_t) E);
        for (int e = 0; e < E; ++e) sum += p[e] = std::exp(l[e] - mx);
        for (int e = 0; e < E; ++e) p[e] /= sum;
        std::vector<int> order((size_t) E);
        for (int e = 0; e < E; ++e) order[e] = e;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
        float kept = 0.f;
        for (int s = 0; s < k; ++s) kept += p[order[s]];
        const float den = std::fmax(kept, 6.103515625e-5f);
        for (int s = 0; s < k; ++s) {
            ids[(size_t) t * k + s] = order[s];
            w[(size_t) t * k + s] = p[order[s]] / den;
        }
    }
}

bool near(const std::vector<float> & got, const std::vector<float> & want)
{
    for (size_t i = 0; i < want.size(); ++i)
        if (std::fabs(got[i] - want[i]) > 1e-6f * std::fabs(want[i])) return false;
    return true;
}

// A device path's routing, read back, against the host's. Pass device outputs, or mapped ids with w_dev null (the
// fused hint prediction: ids only, its weights were always dead).
void check_against_host(const std::vector<float> & hl, int T, const int * ids_dev, const float * w_dev,
                        const int * ids_mapped, const char * what)
{
    std::vector<int> want_ids, got_ids((size_t) T * K, -1);
    std::vector<float> want_w, got_w((size_t) T * K, 0.f);
    ref_route(hl, T, K, want_ids, want_w);
    if (ids_dev) {
        TRUSS_CUDA(cudaMemcpy(got_ids.data(), ids_dev, sizeof(int) * (size_t) T * K, cudaMemcpyDeviceToHost));
        if (w_dev) TRUSS_CUDA(cudaMemcpy(got_w.data(), w_dev, sizeof(float) * (size_t) T * K, cudaMemcpyDeviceToHost));
    } else {
        std::copy(ids_mapped, ids_mapped + (size_t) T * K, got_ids.begin());
    }
    check(got_ids == want_ids, what, T, 0, 0, 0);
    if (w_dev) check(near(got_w, want_w), what, T, 0, 0, 0);
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
    TRUSS_CUDA(cudaMalloc(&logits, sizeof(float) * (size_t) R * E));
    TRUSS_CUDA(cudaMalloc(&predl, sizeof(float) * (size_t) R * E));
    TRUSS_CUDA(cudaMalloc(&x, sizeof(float) * (size_t) R * D));
    TRUSS_CUDA(cudaMemcpy(logits, hlogits.data(), sizeof(float) * (size_t) R * E, cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(predl, hpred.data(), sizeof(float) * (size_t) R * E, cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(x, hx.data(), sizeof(float) * (size_t) R * D, cudaMemcpyHostToDevice));

    Side ref = make_side(false), got = make_side(true);
    // The plain path routes the hint into device memory before publishing it; the fused one has no such buffer. Same
    // exact-size-with-guard treatment, because it is the same kind of launch.
    std::vector<Scratch> ref_sc, got_sc, hint_sc;
    for (int T = 0; T <= R; ++T) {
        ref_sc.push_back(make_scratch(T));
        got_sc.push_back(make_scratch(T));
        hint_sc.push_back(make_scratch(T));
    }
    cudaStream_t s;
    TRUSS_CUDA(cudaStreamCreate(&s));

    for (int T = 1; T <= R; ++T) {
        // Both device paths' routing against an independent host top-k of the same logits, before the two paths are
        // compared against each other - and against their scratch guards, which is the check the missing row guard
        // would have failed.
        ffn::route(logits, T, E, K, ref_sc[T].ids, ref_sc[T].w, s);
        ffn::route(predl, T, E, K, hint_sc[T].ids, hint_sc[T].w, s);
        ffn::route_publish(logits, predl, T, E, K, got_sc[T].ids, got_sc[T].w, nullptr, 0, got.ids, got.pred, got.w,
                           nullptr, got.bell, 1, got.arrive, 0, s);
        TRUSS_CUDA(cudaStreamSynchronize(s));
        check_against_host(hlogits, T, ref_sc[T].ids, ref_sc[T].w, nullptr, "route (plain) != host top-k");
        check_against_host(hpred, T, hint_sc[T].ids, hint_sc[T].w, nullptr, "route (plain, hint) != host top-k");
        check_against_host(hlogits, T, got_sc[T].ids, got_sc[T].w, nullptr, "route+publish != host top-k");
        check_against_host(hpred, T, nullptr, nullptr, got.pred, "route+publish (hint ids) != host top-k");
        check(scratch_clean(ref_sc[T]) && scratch_clean(hint_sc[T]) && scratch_clean(got_sc[T]),
              "wrote past the routing scratch", T, 0, 0, 0);
        check(guard_clean(got), "wrote past the published rows", T, 1, 0, 0);
        for (int hint = 0; hint <= 1; ++hint)
            for (int rows = 0; rows <= 1; ++rows) {
                const int n_x = rows ? T * D : 0;
                for (int ctas : {1, 2, 8, 16}) {
                    const int seq = 5;   // both sides get the same word; the fused one must ring it too
                    // the plain three launches
                    ffn::route(logits, T, E, K, ref_sc[T].ids, ref_sc[T].w, s);
                    if (hint) ffn::route(predl, T, E, K, hint_sc[T].ids, hint_sc[T].w, s);
                    runtime::publish(ref_sc[T].ids, hint ? hint_sc[T].ids : nullptr, ref_sc[T].w, T * K, x, n_x,
                                     ref.ids, hint ? ref.pred : nullptr, ref.w, ref.x, ref.bell, seq, s);
                    // the fused launch
                    ffn::route_publish(logits, hint ? predl : nullptr, T, E, K, got_sc[T].ids, got_sc[T].w, x, n_x,
                                       got.ids, hint ? got.pred : nullptr, got.w, got.x, got.bell, seq, got.arrive,
                                       ctas, s);
                    TRUSS_CUDA(cudaStreamSynchronize(s));
                    check(*ref.bell == seq, "plain doorbell", T, hint, n_x, ctas);
                    check(*got.bell == seq, "fused doorbell", T, hint, n_x, ctas);
                    check(same(ref, ref_sc[T], got, got_sc[T], T, hint, n_x), "published bytes differ", T, hint, n_x,
                          ctas);
                    check(guard_clean(ref) && guard_clean(got), "wrote past the published rows", T, hint, n_x, ctas);
                    check(scratch_clean(ref_sc[T]) && scratch_clean(got_sc[T]) && (!hint || scratch_clean(hint_sc[T])),
                          "wrote past the routing scratch", T, hint, n_x, ctas);
                    unsigned left = 1;
                    TRUSS_CUDA(cudaMemcpy(&left, got.arrive, sizeof(unsigned), cudaMemcpyDeviceToHost));
                    check(left == 0, "arrival counter left behind", T, hint, n_x, ctas);
                }
            }
    }
    std::printf("route+publish == route, route, publish: %s (%d rows cases x hint x ctas)\n",
                fails ? "FAIL" : "ok", R * 2 * 2 * 4);

    // One bell, one counter, launches that differ in rows and in CTA count - what a pass actually does when a window
    // shrinks mid-run. Every launch must still ring, and ring with its own data, and stay inside its scratch.
    for (int i = 1; i <= 200 && !fails; ++i) {
        const int T = i % 3 ? R : 1;
        const int ctas = i % 2 ? 8 : 1;
        const int n_x = T * D;
        const int seq = i;
        ffn::route(logits, T, E, K, ref_sc[T].ids, ref_sc[T].w, s);
        ffn::route(predl, T, E, K, hint_sc[T].ids, hint_sc[T].w, s);
        runtime::publish(ref_sc[T].ids, hint_sc[T].ids, ref_sc[T].w, T * K, x, n_x, ref.ids, ref.pred, ref.w, ref.x,
                         ref.bell, seq, s);
        ffn::route_publish(logits, predl, T, E, K, got_sc[T].ids, got_sc[T].w, x, n_x, got.ids, got.pred, got.w,
                           got.x, got.bell, seq, got.arrive, ctas, s);
        TRUSS_CUDA(cudaStreamSynchronize(s));
        check(*got.bell == seq, "seq", T, 1, n_x, ctas);
        check(same(ref, ref_sc[T], got, got_sc[T], T, true, n_x), "published bytes differ", T, 1, n_x, ctas);
        check(scratch_clean(ref_sc[T]) && scratch_clean(got_sc[T]) && scratch_clean(hint_sc[T]),
              "wrote past the routing scratch", T, 1, n_x, ctas);
        unsigned left = 1;
        TRUSS_CUDA(cudaMemcpy(&left, got.arrive, sizeof(unsigned), cudaMemcpyDeviceToHost));
        check(left == 0, "arrival counter left behind", T, 1, n_x, ctas);
    }
    std::printf("200 back-to-back launches, rows and CTA count alternating: %s\n", fails ? "FAIL" : "ok");

    std::printf("route_publish_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
