// Timeline of one MoE window launch from the kernel's own trace (truss::moe::TraceEvent): where the time goes
// between dependency waits, the three item kinds, and the tail. Random trellis weights; correctness is the unit
// test's job (tests/unit/moe_window_test).
// usage: tk-bench-moe-trace [n_rows] [K: 0 = mixed 2..4]
#include "kernels/moe/moe_window.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

using Shape = truss::moe::FlashNext;
using truss::moe::TraceEvent;

__global__ void fill_random(uint32_t * p, size_t n, uint32_t seed)
{
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t) i * 0x9E3779B1u ^ seed;
        h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12;
        p[i] = h;
    }
}

__global__ void fill_scale(half * p, size_t n)
{
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        p[i] = __float2half((i & 1) ? 0.03f : -0.04f);
}

int main(int argc, char ** argv)
{
    const int n_rows = argc > 1 ? atoi(argv[1]) : 4, kfix = argc > 2 ? atoi(argv[2]) : 2;
    const int D = Shape::D_MODEL, F = Shape::D_FF, TOPK = Shape::TOPK, E = 512;
    truss::moe::Weights W;
    W.n_expert = E;
    for (int p = 0; p < 3; ++p) {
        const int in = p == 2 ? F : D, out = p == 2 ? D : F;
        std::vector<int32_t> meta(2 * E);
        size_t words = 0;
        for (int e = 0; e < E; ++e) {
            meta[2 * e] = kfix ? kfix : 2 + (e + p) % 3;
            meta[2 * e + 1] = (int32_t) words;
            words += (size_t) in * out * meta[2 * e] / 16;
        }
        uint16_t * tr; int32_t * dm; half *suh, *svh;
        CK(cudaMalloc(&tr, words * 2));
        fill_random<<<1024, 256>>>((uint32_t *) tr, words / 2, 17u + p);
        CK(cudaMalloc(&dm, meta.size() * 4));
        CK(cudaMemcpy(dm, meta.data(), meta.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&suh, (size_t) E * in * 2));
        CK(cudaMalloc(&svh, (size_t) E * out * 2));
        fill_scale<<<1024, 256>>>(suh, (size_t) E * in);
        fill_scale<<<1024, 256>>>(svh, (size_t) E * out);
        W.proj[p] = { tr, dm, suh, svh };
    }
    // routing like a verify window: distinct experts per row, some reuse of the previous row
    std::mt19937 rng(4321);
    std::vector<int> ids(n_rows * TOPK);
    std::vector<float> wts(n_rows * TOPK, 0.1f);
    for (int t = 0; t < n_rows; ++t)
        for (int s = 0; s < TOPK; ++s) {
            int e;
            do e = (t > 0 && rng() % 8 == 0) ? ids[(t - 1) * TOPK + rng() % TOPK] : (int) (rng() % E);
            while (std::find(ids.begin() + t * TOPK, ids.begin() + t * TOPK + s, e) != ids.begin() + t * TOPK + s);
            ids[t * TOPK + s] = e;
        }
    int * d_ids; float *d_w, *d_x, *d_out; void * ws; TraceEvent * d_tr;
    CK(cudaMalloc(&d_ids, ids.size() * 4));
    CK(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_w, wts.size() * 4));
    CK(cudaMemcpy(d_w, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_x, (size_t) n_rows * D * 4));
    CK(cudaMemset(d_x, 0, (size_t) n_rows * D * 4));
    CK(cudaMalloc(&d_out, (size_t) n_rows * D * 4));
    CK(cudaMalloc(&ws, truss::moe::workspace_bytes<Shape>()));
    CK(cudaMalloc(&d_tr, sizeof(TraceEvent) * truss::moe::MAX_TRACE_EVENTS));
    truss::moe::workspace_init<Shape>(ws, 0);

    // warm, then time untraced and traced launches
    for (int r = 0; r < 20; ++r) truss::moe::window<Shape>(W, d_x, d_ids, d_w, n_rows, d_out, ws, 0);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int r = 0; r < 100; ++r) truss::moe::window<Shape>(W, d_x, d_ids, d_w, n_rows, d_out, ws, 0);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaMemset(d_tr, 0, sizeof(TraceEvent) * truss::moe::MAX_TRACE_EVENTS));
    truss::moe::window<Shape>(W, d_x, d_ids, d_w, n_rows, d_out, ws, 0, d_tr);
    CK(cudaDeviceSynchronize());
    std::vector<TraceEvent> tr(truss::moe::MAX_TRACE_EVENTS);
    CK(cudaMemcpy(tr.data(), d_tr, sizeof(TraceEvent) * tr.size(), cudaMemcpyDeviceToHost));
    tr.erase(std::remove_if(tr.begin(), tr.end(), [] (const TraceEvent & e) { return e.t_end == 0; }), tr.end());

    unsigned long long t0 = ~0ull, t1 = 0;
    for (auto & e : tr) { t0 = std::min(t0, e.t_start); t1 = std::max(t1, e.t_end); }
    printf("rows %d K %s: %.1f us per launch (untraced, back to back); traced span first item start -> last end %.1f us\n",
           n_rows, kfix ? std::to_string(kfix).c_str() : "mix", ms * 10.0, (t1 - t0) / 1e3);
    const char * kname[3] = { "H", "gate/up", "down" };
    for (int k = 0; k < 3; ++k) {
        double busy = 0, wait = 0, first = 1e30, last = 0;
        std::vector<double> dur;
        for (auto & e : tr) if (e.kind == k) {
            dur.push_back((e.t_end - e.t_ready) / 1e3);
            busy += (e.t_end - e.t_ready) / 1e3;
            wait += (e.t_ready - e.t_start) / 1e3;
            first = std::min(first, (e.t_start - t0) / 1e3);
            last = std::max(last, (e.t_end - t0) / 1e3);
        }
        if (dur.empty()) continue;
        std::sort(dur.begin(), dur.end());
        printf("  %-8s items %4zu | work %7.1f us-blocks (median %5.2f, max %5.2f us) | waiting %7.1f us-blocks | "
               "window %5.1f..%5.1f us\n", kname[k], dur.size(), busy, dur[dur.size() / 2], dur.back(), wait, first, last);
    }
    // per-block view: how long each block was busy, and when the last one finished
    std::vector<double> busy_b(4096, 0), end_b(4096, 0);
    int nb = 0;
    for (auto & e : tr) {
        busy_b[e.block] += (e.t_end - e.t_start) / 1e3;
        end_b[e.block] = std::max(end_b[e.block], (e.t_end - t0) / 1e3);
        nb = std::max(nb, e.block + 1);
    }
    std::vector<double> ends(end_b.begin(), end_b.begin() + nb);
    std::sort(ends.begin(), ends.end());
    double busy = 0;
    for (int b = 0; b < nb; ++b) busy += busy_b[b];
    // prologue: kernel entry (per block) to its first item
    std::vector<double> pro(nb, 1e30);
    unsigned long long enter0 = ~0ull;
    for (auto & e : tr) {
        pro[e.block] = std::min(pro[e.block], (e.t_start - e.t_enter) / 1e3);
        enter0 = std::min(enter0, e.t_enter);
    }
    std::sort(pro.begin(), pro.end());
    printf("  prologue (entry -> first item): median %.1f, max %.1f us; first entry -> first item start %.1f us\n",
           pro[nb / 2], pro.back(), (t0 - enter0) / 1e3);
    printf("  blocks %d: busy %.0f%% of span; block finish times p10 %.1f, p50 %.1f, p90 %.1f, max %.1f us\n", nb,
           100.0 * busy / (nb * (t1 - t0) / 1e3), ends[nb / 10], ends[nb / 2], ends[nb * 9 / 10], ends.back());
    return 0;
}
