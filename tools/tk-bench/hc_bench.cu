// The launches of one hyper-connection mix at decode shapes, timed on an idle GPU: norm / combine_norm, the
// down+inject multi gemv, silu, the up gemv, collapse - each alone and the whole sequence. The engine's decode
// hc_mix is the five of combine_norm + down + silu + up + collapse (the layer's closing combine is folded into the
// next mix's norm); `norm` alone is listed for the first mix of a stage. The weights rotate through enough copies
// to miss L2 (> 96 MB), the way a decode pass reads a fresh layer's weights. Compare with the in-pass per-kernel
// numbers from the nsys trace (reports/LOG.md Phase 0: at 8 rows combine_norm 7.9 us, down 9.1, up 7.5,
// collapse 3.0, silu 1.2): the difference is what the pass costs a kernel beyond its own work.
// usage: tk-bench-hc [reps=200]
#include "core/cuda_check.h"
#include "kernels/dense/q8_gemm.cuh"
#include "kernels/hc/hc_prefill.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace truss;

namespace {

constexpr int HC = 4, D = 2560, RANK = 320;
constexpr float EPS = 1e-6f;

struct Weights {
    dense::Q8Matrix down, inj, up;   // [10240 -> 320], [10240 -> 4], [320 -> 10240]
};

struct Row {
    const char * name;
    float us, mb;
};

template <class F>
float bench(int reps, F && call)
{
    cudaEvent_t e0, e1;
    TRUSS_CUDA(cudaEventCreate(&e0));
    TRUSS_CUDA(cudaEventCreate(&e1));
    for (int w = 0; w < 20; ++w) call(w);
    TRUSS_CUDA(cudaEventRecord(e0));
    for (int it = 0; it < reps; ++it) call(it);
    TRUSS_CUDA(cudaEventRecord(e1));
    TRUSS_CUDA(cudaEventSynchronize(e1));
    float ms;
    TRUSS_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0), cudaEventDestroy(e1);
    return ms * 1e3f / reps;
}

}  // namespace

int main(int argc, char ** argv)
{
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
    const int R = dense::GEMV_ROWS;
    std::mt19937 g(5);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    std::normal_distribution<float> nd(0.f, 1.f);

    // one layer's weights are 6.6 MB; rotate through > 96 MB so no call hits L2
    const size_t wbytes = ((size_t) (HC * D) * RANK + (size_t) (HC * D) / 32 * RANK * 2 +
                           (size_t) (HC * D) * HC + (size_t) (HC * D) / 32 * HC * 2 +
                           (size_t) RANK * (HC * D) + (size_t) RANK / 32 * (HC * D) * 2);
    const int copies = (int) std::max<size_t>(2, (96u << 20) / wbytes + 1);
    std::printf("hc_mix at decode shapes: hc %d, d %d, rank %d, %.2f MB of weights per layer, %d copies, %d reps\n",
                HC, D, RANK, wbytes / 1e6, copies, reps);

    std::vector<Weights> W(copies);
    {
        std::vector<int8_t> q((size_t) (HC * D) * RANK), qi((size_t) (HC * D) * HC), qu((size_t) RANK * (HC * D));
        std::vector<half> d((size_t) (HC * D) / 32 * RANK), di((size_t) (HC * D) / 32 * HC),
            du((size_t) RANK / 32 * (HC * D));
        for (auto & v : q) v = (int8_t) qd(g);
        for (auto & v : qi) v = (int8_t) qd(g);
        for (auto & v : qu) v = (int8_t) qd(g);
        for (auto & v : d) v = __float2half(sd(g));
        for (auto & v : di) v = __float2half(sd(g));
        for (auto & v : du) v = __float2half(sd(g));
        auto put = [](const std::vector<int8_t> & qq, const std::vector<half> & dd, int in, int out,
                      dense::Q8Matrix & m) {
            int8_t * dq;
            half * dh;
            TRUSS_CUDA(cudaMalloc(&dq, qq.size()));
            TRUSS_CUDA(cudaMalloc(&dh, dd.size() * 2));
            TRUSS_CUDA(cudaMemcpy(dq, qq.data(), qq.size(), cudaMemcpyHostToDevice));
            TRUSS_CUDA(cudaMemcpy(dh, dd.data(), dd.size() * 2, cudaMemcpyHostToDevice));
            m = { dq, dh, in, out };
        };
        for (int c = 0; c < copies; ++c) {
            put(q, d, HC * D, RANK, W[c].down);
            put(qi, di, HC * D, HC, W[c].inj);
            put(qu, du, RANK, HC * D, W[c].up);
        }
    }

    // activations and scratch, sized for 8 rows
    auto alloc = [](size_t n) {
        float * p;
        TRUSS_CUDA(cudaMalloc(&p, n * 4));
        return p;
    };
    half * xn16, * lo16, * mixed16;
    TRUSS_CUDA(cudaMalloc(&xn16, (size_t) R * HC * D * 2));
    TRUSS_CUDA(cudaMalloc(&lo16, (size_t) R * RANK * 2));
    TRUSS_CUDA(cudaMalloc(&mixed16, (size_t) R * D * 2));
    float * res = alloc((size_t) R * HC * D), * out = alloc((size_t) R * D), * gin = alloc((size_t) R * HC);
    float * gamma = alloc((size_t) HC * D), * rstd = alloc((size_t) R * HC), * lo = alloc((size_t) R * RANK);
    float * gate = alloc((size_t) R * HC * D), * mixed = alloc((size_t) R * D);
    {
        std::vector<float> h((size_t) R * HC * D);
        for (auto & v : h) v = nd(g);
        TRUSS_CUDA(cudaMemcpy(res, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        std::vector<float> o((size_t) R * D);
        for (auto & v : o) v = nd(g);
        TRUSS_CUDA(cudaMemcpy(out, o.data(), o.size() * 4, cudaMemcpyHostToDevice));
        std::vector<float> gm((size_t) HC * D);
        for (auto & v : gm) v = sd(g);
        TRUSS_CUDA(cudaMemcpy(gamma, gm.data(), gm.size() * 4, cudaMemcpyHostToDevice));
        std::vector<float> ij((size_t) R * HC);
        for (auto & v : ij) v = nd(g);
        TRUSS_CUDA(cudaMemcpy(gin, ij.data(), ij.size() * 4, cudaMemcpyHostToDevice));
        std::vector<float> gt((size_t) R * HC * D);
        for (auto & v : gt) v = nd(g);
        TRUSS_CUDA(cudaMemcpy(gate, gt.data(), gt.size() * 4, cudaMemcpyHostToDevice));
    }

    for (int T : { 1, 2, 4, 8 }) {
        std::printf("\n%d row%s\n", T, T == 1 ? "" : "s");
        const float mb_down = (float) (((size_t) (HC * D) * (RANK + HC)) * 1.0625) / 1e6f;
        const float mb_up = (float) ((size_t) RANK * (HC * D) * 1.0625) / 1e6f;
        const float mb_norm = (float) ((size_t) T * HC * D * (4 + 4 + 2) + (size_t) T * D * 4) / 1e6f;  // res r+w, xn16, out
        const float mb_collapse = (float) ((size_t) T * HC * D * 8 + (size_t) T * D * 6) / 1e6f;        // res, gate, mixed
        const float mb_silu = (float) ((size_t) T * RANK * 6) / 1e6f;
        Row rows[10];
        rows[0] = {"hc::norm", bench(reps, [&](int) { hc::norm(res, gamma, T, HC, D, EPS, xn16, rstd, nullptr); }),
                   mb_norm};
        rows[1] = {"hc::combine_norm",
                   bench(reps, [&](int) {
                       hc::combine_norm(res, out, gin, gamma, T, HC, D, EPS, xn16, rstd, nullptr);
                   }),
                   mb_norm};
        rows[2] = {"down+inject gemv",
                   bench(reps, [&](int i) {
                       const dense::Q8Matrix * w[2] = { &W[i % copies].down, &W[i % copies].inj };
                       float * y[2] = { lo, gin };
                       dense::q8_gemv_multi_a16(w, y, 2, xn16, T, nullptr);
                   }),
                   mb_down};
        rows[3] = {"hc::silu", bench(reps, [&](int) { hc::silu(lo, T * RANK, 1.f / HC, lo16, nullptr); }), mb_silu};
        rows[4] = {"up gemv",
                   bench(reps, [&](int i) { dense::q8_gemv_a16(W[i % copies].up, lo16, T, gate, nullptr); }), mb_up};
        rows[5] = {"hc::collapse",
                   bench(reps, [&](int) {
                       hc::collapse(res, rstd, gamma, gate, T, HC, D, mixed, mixed16, nullptr);
                   }),
                   mb_collapse};
        rows[6] = {"SEQUENCE (5 launches)",
                   bench(reps, [&](int i) {
                       const dense::Q8Matrix * w[2] = { &W[i % copies].down, &W[i % copies].inj };
                       float * y[2] = { lo, gin };
                       hc::combine_norm(res, out, gin, gamma, T, HC, D, EPS, xn16, rstd, nullptr);
                       dense::q8_gemv_multi_a16(w, y, 2, xn16, T, nullptr);
                       hc::silu(lo, T * RANK, 1.f / HC, lo16, nullptr);
                       dense::q8_gemv_a16(W[i % copies].up, lo16, T, gate, nullptr);
                       hc::collapse(res, rstd, gamma, gate, T, HC, D, mixed, mixed16, nullptr);
                   }),
                   mb_norm + mb_down + mb_silu + mb_up + mb_collapse};
        // the two fusions: same inputs, same arithmetic, fewer launches
        rows[7] = {"down+silu (fused)",
                   bench(reps, [&](int i) {
                       const dense::Q8Matrix * w[2] = { &W[i % copies].down, &W[i % copies].inj };
                       float * y[2] = { nullptr, gin };
                       half * o16[2] = { lo16, nullptr };
                       const float s16[2] = { 1.f / HC, 0.f };
                       dense::q8_gemv_multi_a16_silu(w, y, o16, s16, 2, xn16, T, nullptr);
                   }),
                   mb_down};
        rows[8] = {"up+collapse (fused)",
                   bench(reps, [&](int i) {
                       dense::q8_gemv_a16_up_collapse(W[i % copies].up, lo16, T, res, rstd, gamma, mixed, mixed16, HC,
                                                      D, nullptr);
                   }),
                   mb_up + (float) ((size_t) T * HC * D * 4 + (size_t) T * D * 6) / 1e6f};
        rows[9] = {"SEQUENCE fused (3)",
                   bench(reps, [&](int i) {
                       const dense::Q8Matrix * w[2] = { &W[i % copies].down, &W[i % copies].inj };
                       float * y[2] = { nullptr, gin };
                       half * o16[2] = { lo16, nullptr };
                       const float s16[2] = { 1.f / HC, 0.f };
                       hc::combine_norm(res, out, gin, gamma, T, HC, D, EPS, xn16, rstd, nullptr);
                       dense::q8_gemv_multi_a16_silu(w, y, o16, s16, 2, xn16, T, nullptr);
                       dense::q8_gemv_a16_up_collapse(W[i % copies].up, lo16, T, res, rstd, gamma, mixed, mixed16, HC,
                                                      D, nullptr);
                   }),
                   mb_norm + mb_down + mb_up + (float) ((size_t) T * HC * D * 4 + (size_t) T * D * 6) / 1e6f};
        float sum = 0;
        for (int k = 1; k < 6; ++k) sum += rows[k].us;   // the five of one mix (combine_norm, not norm)
        for (const Row & r : rows)
            std::printf("  %-22s %6.2f us  %6.2f MB  %5.0f GB/s\n", r.name, r.us, r.mb, r.mb / r.us * 1e3f);
        std::printf("  %-22s %6.2f us  (sum of those five alone: %5.2f us; the sequence is %s by %4.2f us)\n", "",
                    rows[6].us, sum, rows[6].us > sum ? "slower" : "faster", std::fabs(rows[6].us - sum));
        std::printf("  %-22s %6.2f us  vs %6.2f us unfused: %5.1f%% of the mix\n", "", rows[9].us, rows[6].us,
                    rows[9].us / rows[6].us * 100.f);
    }
    for (auto & c : W)
        cudaFree((void *) c.down.q), cudaFree((void *) c.down.d), cudaFree((void *) c.inj.q),
            cudaFree((void *) c.inj.d), cudaFree((void *) c.up.q), cudaFree((void *) c.up.d);
    return 0;
}
