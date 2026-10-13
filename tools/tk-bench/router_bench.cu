// The router and the doorbell publish at decode shapes, timed on an idle GPU: the top-k, the publish (a layer's FFN
// rows - 80 KB at 8 rows - into mapped host memory, from ONE 1024-thread CTA today), the three launches a hinted
// decode layer runs today, and the fused launch at every copy-CTA count. Phase 0's in-pass numbers for comparison
// (reports/LOG.md, device 0, 8-row passes): route 7.3 us x 45 launches, publish 18.0 us x 24. The last row is a
// cudaMemcpyAsync of the same bytes into pinned memory: what the copy engine can do, which the doorbell cannot use
// because the ring has to happen behind the data on the device side.
// usage: tk-bench-router [reps=200]
#include "core/cuda_check.h"
#include "kernels/ffn/ffn_ops.cuh"
#include "runtime/doorbell.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace truss;

namespace {

constexpr int E = 512, K = 10, D = 2560, R = 8;

struct Row {
    const char * name;
    float us;
    float gb;   // mapped/pinned bytes moved per call, for the copy rows
};

// The span of one stream, per call. The events go on the stream being measured: a variant that moves work to a
// second stream must be judged by what the compute stream waits for, and the device sync at the end keeps one
// row's backlog out of the next row's measurement.
template <class F>
float bench(int reps, cudaStream_t st, F && call)
{
    cudaEvent_t e0, e1;
    TRUSS_CUDA(cudaEventCreate(&e0));
    TRUSS_CUDA(cudaEventCreate(&e1));
    for (int w = 0; w < 20; ++w) call(w);   // first touch of the buffers and of the module is not the cost
    TRUSS_CUDA(cudaDeviceSynchronize());
    TRUSS_CUDA(cudaEventRecord(e0, st));
    for (int it = 0; it < reps; ++it) call(it);
    TRUSS_CUDA(cudaEventRecord(e1, st));
    TRUSS_CUDA(cudaEventSynchronize(e1));
    float ms;
    TRUSS_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0), cudaEventDestroy(e1);
    TRUSS_CUDA(cudaDeviceSynchronize());
    return ms * 1e3f / reps;
}

}  // namespace

int main(int argc, char ** argv)
{
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
    std::mt19937 g(11);
    std::normal_distribution<float> nd(0.f, 1.f);

    std::vector<float> hl((size_t) R * E), hp((size_t) R * E), hx((size_t) R * D);
    for (auto & v : hl) v = nd(g);
    for (auto & v : hp) v = nd(g);
    for (auto & v : hx) v = nd(g);
    float *logits = nullptr, *predl = nullptr, *x = nullptr, *pw = nullptr;
    int * d_pred = nullptr;
    TRUSS_CUDA(cudaMalloc(&logits, sizeof(float) * (size_t) R * E));
    TRUSS_CUDA(cudaMalloc(&predl, sizeof(float) * (size_t) R * E));
    TRUSS_CUDA(cudaMalloc(&x, sizeof(float) * (size_t) R * D));
    TRUSS_CUDA(cudaMalloc(&pw, sizeof(float) * R * K));
    TRUSS_CUDA(cudaMalloc(&d_pred, sizeof(int) * R * K));
    TRUSS_CUDA(cudaMemcpy(logits, hl.data(), sizeof(float) * (size_t) R * E, cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(predl, hp.data(), sizeof(float) * (size_t) R * E, cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(x, hx.data(), sizeof(float) * (size_t) R * D, cudaMemcpyHostToDevice));

    auto mapped = [](size_t bytes) {
        void * p = nullptr;
        TRUSS_CUDA(cudaHostAlloc(&p, bytes, cudaHostAllocMapped));
        std::memset(p, 0, bytes);
        return p;
    };
    int * m_ids = (int *) mapped(sizeof(int) * R * K), * m_pred = (int *) mapped(sizeof(int) * R * K);
    float * m_w = (float *) mapped(sizeof(float) * R * K), * m_x = (float *) mapped(sizeof(float) * (size_t) R * D);
    int * bell = (int *) mapped(sizeof(int));
    void * pin = nullptr;   // for the copy-engine reference row
    TRUSS_CUDA(cudaHostAlloc(&pin, sizeof(float) * (size_t) R * D, cudaHostAllocDefault));

    int * d_ids = nullptr;
    float * d_w = nullptr;
    unsigned * arrive = nullptr;
    TRUSS_CUDA(cudaMalloc(&d_ids, sizeof(int) * R * K));
    TRUSS_CUDA(cudaMalloc(&d_w, sizeof(float) * R * K));
    TRUSS_CUDA(cudaMalloc(&arrive, sizeof(unsigned)));
    TRUSS_CUDA(cudaMemset(arrive, 0, sizeof(unsigned)));

    cudaStream_t s, s_copy;
    cudaEvent_t ev_x, ev_r;
    TRUSS_CUDA(cudaStreamCreate(&s));
    TRUSS_CUDA(cudaStreamCreateWithFlags(&s_copy, cudaStreamNonBlocking));
    TRUSS_CUDA(cudaEventCreateWithFlags(&ev_x, cudaEventDisableTiming));   // the FFN rows are ready
    TRUSS_CUDA(cudaEventCreateWithFlags(&ev_r, cudaEventDisableTiming));   // the routing words are in host memory
    int * pin_seq = nullptr, * pin_bell = nullptr;   // the ring travels as a 4-byte copy behind the rows
    TRUSS_CUDA(cudaHostAlloc(&pin_seq, sizeof(int), cudaHostAllocDefault));
    TRUSS_CUDA(cudaHostAlloc(&pin_bell, sizeof(int), cudaHostAllocDefault));

    std::printf("reps %d, E %d, k %d, d_model %d\n", reps, E, K, D);
    std::printf("hint (the next layer's router on this input) is on at 8 rows and off below: TRUSS_HINT_MIN_ROWS=8"
                " in the served config, so the 1-row baseline is route + publish, not three launches.\n");
    for (int T : {R, 1}) {
        const bool hint = T >= 8;
        const int n_x = T * D;
        std::printf("\n%d rows (%s hint): us per layer\n", T, hint ? "with" : "no");
        Row rows[] = {
                { "route (this layer)", 0, 0 },
                { "publish alone, 1 CTA (today)", 0, (float) n_x * 4 },
                { "TODAY: route(+hint) + publish", 0, (float) n_x * 4 },
                { "fused, 1 copy CTA", 0, (float) n_x * 4 },
                { "fused, 4 copy CTAs", 0, (float) n_x * 4 },
                { "fused, 16 copy CTAs", 0, (float) n_x * 4 },
                { "fused, no rows (no CPU tier)", 0, 0 },
                { "SPLIT: rows + ring on a copy stream", 0, (float) n_x * 4 },
                { "  the copy stream itself: rows", 0, (float) n_x * 4 },
                { "  the copy stream itself: + ring", 0, (float) (n_x * 4 + 4) },
                { "copy engine, rows on s (serial)", 0, (float) n_x * 4 },
        };
        int seq = 0;
        rows[0].us = bench(reps, s, [&](int i) { ffn::route(logits, T, E, K, d_ids, d_w, s); });
        rows[1].us = bench(reps, s, [&](int i) {
            runtime::publish(d_ids, hint ? d_pred : nullptr, d_w, T * K, x, n_x, m_ids, hint ? m_pred : nullptr, m_w,
                             m_x, bell, ++seq, s);
        });
        rows[2].us = bench(reps, s, [&](int i) {
            ffn::route(logits, T, E, K, d_ids, d_w, s);
            if (hint) ffn::route(predl, T, E, K, d_pred, pw, s);
            runtime::publish(d_ids, hint ? d_pred : nullptr, d_w, T * K, x, n_x, m_ids, hint ? m_pred : nullptr, m_w,
                             m_x, bell, ++seq, s);
        });
        const int cta_opts[] = {1, 4, 16};
        for (int c = 0; c < 3; ++c)
            rows[3 + c].us = bench(reps, s, [&](int i) {
                ffn::route_publish(logits, hint ? predl : nullptr, T, E, K, d_ids, d_w, x, n_x, m_ids,
                                   hint ? m_pred : nullptr, m_w, m_x, bell, ++seq, arrive, cta_opts[c], s);
            });
        rows[6].us = bench(reps, s, [&](int i) {
            ffn::route_publish(logits, hint ? predl : nullptr, T, E, K, d_ids, d_w, nullptr, 0, m_ids,
                               hint ? m_pred : nullptr, m_w, nullptr, bell, ++seq, arrive, 0, s);
        });
        // the routing kernel on the compute stream; the rows and the ring go down the copy stream, the ring behind
        // the rows, so the compute stream never waits for 80 KB to cross the link
        rows[7].us = bench(reps, s, [&](int i) {
            *pin_seq = ++seq;
            TRUSS_CUDA(cudaEventRecord(ev_x, s));
            ffn::route_publish(logits, hint ? predl : nullptr, T, E, K, d_ids, d_w, nullptr, 0, m_ids,
                               hint ? m_pred : nullptr, m_w, nullptr, nullptr, seq, nullptr, 0, s);
            TRUSS_CUDA(cudaEventRecord(ev_r, s));
            TRUSS_CUDA(cudaStreamWaitEvent(s_copy, ev_x, 0));
            TRUSS_CUDA(cudaMemcpyAsync(pin, x, sizeof(float) * (size_t) n_x, cudaMemcpyDeviceToHost, s_copy));
            TRUSS_CUDA(cudaStreamWaitEvent(s_copy, ev_r, 0));
            TRUSS_CUDA(cudaMemcpyAsync(pin_bell, pin_seq, sizeof(int), cudaMemcpyHostToDevice, s_copy));
        });
        rows[8].us = bench(reps, s_copy, [&](int i) {
            TRUSS_CUDA(cudaMemcpyAsync(pin, x, sizeof(float) * (size_t) n_x, cudaMemcpyDeviceToHost, s_copy));
        });
        rows[9].us = bench(reps, s_copy, [&](int i) {
            *pin_seq = ++seq;
            TRUSS_CUDA(cudaMemcpyAsync(pin, x, sizeof(float) * (size_t) n_x, cudaMemcpyDeviceToHost, s_copy));
            TRUSS_CUDA(cudaMemcpyAsync(pin_bell, pin_seq, sizeof(int), cudaMemcpyHostToDevice, s_copy));
        });
        rows[10].us = bench(reps, s, [&](int i) {
            ffn::route_publish(logits, hint ? predl : nullptr, T, E, K, d_ids, d_w, nullptr, 0, m_ids,
                               hint ? m_pred : nullptr, m_w, nullptr, nullptr, ++seq, nullptr, 0, s);
            TRUSS_CUDA(cudaMemcpyAsync(pin, x, sizeof(float) * (size_t) n_x, cudaMemcpyDeviceToHost, s));
        });
        for (const Row & r : rows) {
            std::printf("  %-36s %7.2f us", r.name, r.us);
            if (r.gb > 0) std::printf("   %6.2f GB/s", r.gb / 1e9f / (r.us * 1e-6f));
            std::printf("\n");
        }
    }
    std::fflush(stdout);
    return 0;
}
