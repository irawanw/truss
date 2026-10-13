// The two hyper-connection fusions must give the plain sequence's BYTES, not close values: they claim to move work,
// not arithmetic. On the engine's own hc shapes (down 10240 -> 320, inject 10240 -> 4, up 320 -> 10240, hc = 4
// streams of d_model 2560) and every decode row count 1..8:
//   TRUSS_HC_SILU   the gemv's own lo16 == hc::silu applied to the fp32 lo the plain launch wrote, bit for bit, and
//                   the inject matrix's fp32 output identical fused, unfused, and alone (fusion-invariance);
//   TRUSS_HC_COLLAPSE  the up projection's launch gives mixed / mixed16 == hc::collapse over the fp32 gate the
//                   plain up gemv wrote, bit for bit (same k split, same warp order, same stream order).
// usage: hc_fuse_test
#include "core/cuda_check.h"
#include "kernels/dense/q8_gemm.cuh"
#include "kernels/hc/hc_prefill.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace truss;

namespace {

constexpr int HC = 4, D = 2560, RANK = 320;

struct Weights {
    int8_t * q;
    half * d;
    dense::Q8Matrix m;
};

Weights make(int in, int out, std::mt19937 & g)
{
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    std::vector<int8_t> q((size_t) in * out);
    std::vector<half> d((size_t) in / 32 * out);
    for (auto & v : q) v = (int8_t) qd(g);
    for (auto & v : d) v = __float2half(sd(g));
    Weights w;
    TRUSS_CUDA(cudaMalloc(&w.q, q.size()));
    TRUSS_CUDA(cudaMalloc(&w.d, d.size() * 2));
    TRUSS_CUDA(cudaMemcpy(w.q, q.data(), q.size(), cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(w.d, d.data(), d.size() * 2, cudaMemcpyHostToDevice));
    w.m = { w.q, w.d, in, out };
    return w;
}

float * put(const std::vector<float> & h)
{
    float * p;
    TRUSS_CUDA(cudaMalloc(&p, h.size() * 4));
    TRUSS_CUDA(cudaMemcpy(p, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return p;
}

half * put16(const std::vector<half> & h)
{
    half * p;
    TRUSS_CUDA(cudaMalloc(&p, h.size() * 2));
    TRUSS_CUDA(cudaMemcpy(p, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    return p;
}

std::vector<float> get(float * p, size_t n)
{
    std::vector<float> h(n);
    TRUSS_CUDA(cudaMemcpy(h.data(), p, n * 4, cudaMemcpyDeviceToHost));
    return h;
}

std::vector<half> get16(half * p, size_t n)
{
    std::vector<half> h(n);
    TRUSS_CUDA(cudaMemcpy(h.data(), p, n * 2, cudaMemcpyDeviceToHost));
    return h;
}

// the first element that differs, or -1; compared as raw bits
long diff(const std::vector<float> & a, const std::vector<float> & b)
{
    if (a.size() != b.size()) return -2;
    for (size_t i = 0; i < a.size(); ++i)
        if (memcmp(&a[i], &b[i], 4) != 0) return (long) i;
    return -1;
}
long diff16(const std::vector<half> & a, const std::vector<half> & b)
{
    if (a.size() != b.size()) return -2;
    for (size_t i = 0; i < a.size(); ++i)
        if (memcmp(&a[i], &b[i], 2) != 0) return (long) i;
    return -1;
}

}  // namespace

int main()
{
    std::mt19937 g(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    if (!dense::mma_on()) { std::printf("SKIP: the fused epilogues live in the mma gemv (TRUSS_GEMV_MMA=0)\n"); return 0; }
    if (!dense::up_collapse_ok(RANK, HC * D, HC, D)) {
        std::printf("FAIL: the engine's own up shape is not collapse-fusable (k split left the block)\n");
        return 1;
    }

    const Weights down = make(HC * D, RANK, g), inj = make(HC * D, HC, g), up = make(RANK, HC * D, g);
    const int R = dense::GEMV_ROWS;
    std::vector<float> fres((size_t) R * HC * D), fout((size_t) R * D), fgamma((size_t) HC * D), frstd((size_t) R * HC),
        finject((size_t) R * HC);
    for (auto & v : fres) v = nd(g);
    for (auto & v : fout) v = nd(g);
    for (auto & v : fgamma) v = sd(g);
    for (auto & v : frstd) v = sd(g);
    for (auto & v : finject) v = nd(g);
    float * res = put(fres), * out = put(fout), * gamma = put(fgamma), * rstd = put(frstd), * inj_in = put(finject);

    int fails = 0;
    for (int T = 1; T <= R; ++T) {
        // the activations the down projection reads (an RMSNorm output; any fp16 values will do)
        std::vector<half> hx((size_t) T * HC * D);
        for (auto & v : hx) v = __float2half(nd(g));
        half * x = put16(hx);

        // the plain five launches
        float * lo = put(std::vector<float>((size_t) T * RANK, 0.f)), * inj_ref = put(std::vector<float>((size_t) T * HC, 0.f));
        half * lo16_ref = put16(std::vector<half>((size_t) T * RANK, __float2half(0.f)));
        float * gate = put(std::vector<float>((size_t) T * HC * D, 0.f));
        float * mixed_ref = put(std::vector<float>((size_t) T * D, 0.f));
        half * mixed16_ref = put16(std::vector<half>((size_t) T * D, __float2half(0.f)));
        const dense::Q8Matrix * w2[2] = { &down.m, &inj.m };
        float * y2[2] = { lo, inj_ref };
        dense::q8_gemv_multi_a16(w2, y2, 2, x, T, nullptr);
        hc::silu(lo, T * RANK, 1.f / HC, lo16_ref, nullptr);
        dense::q8_gemv_a16(up.m, lo16_ref, T, gate, nullptr);
        hc::collapse(res, rstd, gamma, gate, T, HC, D, mixed_ref, mixed16_ref, nullptr);

        // the same with the two epilogues
        half * lo16 = put16(std::vector<half>((size_t) T * RANK, __float2half(0.f)));
        float * inj_fus = put(std::vector<float>((size_t) T * HC, 0.f));
        float * mixed = put(std::vector<float>((size_t) T * D, 0.f));
        half * mixed16 = put16(std::vector<half>((size_t) T * D, __float2half(0.f)));
        float * y2f[2] = { nullptr, inj_fus };
        half * o2f[2] = { lo16, nullptr };
        const float s2f[2] = { 1.f / HC, 0.f };
        dense::q8_gemv_multi_a16_silu(w2, y2f, o2f, s2f, 2, x, T, nullptr);
        dense::q8_gemv_a16_up_collapse(up.m, lo16, T, res, rstd, gamma, mixed, mixed16, HC, D, nullptr);

        // and the down projection alone (a layer without an inject projection fuses nothing)
        half * lo16_alone = put16(std::vector<half>((size_t) T * RANK, __float2half(0.f)));
        const dense::Q8Matrix * w1[1] = { &down.m };
        float * y1[1] = { nullptr };
        half * o1[1] = { lo16_alone };
        const float s1[1] = { 1.f / HC };
        dense::q8_gemv_multi_a16_silu(w1, y1, o1, s1, 1, x, T, nullptr);

        const long d_silu = diff16(get16(lo16_ref, (size_t) T * RANK), get16(lo16, (size_t) T * RANK));
        const long d_alone = diff16(get16(lo16, (size_t) T * RANK), get16(lo16_alone, (size_t) T * RANK));
        const long d_inj = diff(get(inj_ref, (size_t) T * HC), get(inj_fus, (size_t) T * HC));
        const long d_mix = diff(get(mixed_ref, (size_t) T * D), get(mixed, (size_t) T * D));
        const long d_m16 = diff16(get16(mixed16_ref, (size_t) T * D), get16(mixed16, (size_t) T * D));
        char line[256];
        std::snprintf(line, sizeof line, "  %d rows: silu epilogue %s, alone == fused %s, inject %s, collapse %s / %s", T,
                      d_silu < 0 ? "bit-identical" : "DIFFERS", d_alone < 0 ? "ok" : "DIFFERS",
                      d_inj < 0 ? "ok" : "DIFFERS", d_mix < 0 ? "bit-identical" : "DIFFERS",
                      d_m16 < 0 ? "bit-identical" : "DIFFERS");
        std::printf("%s\n", line);
        if (d_silu >= 0 || d_alone >= 0 || d_inj >= 0 || d_mix >= 0 || d_m16 >= 0) {
            std::printf("    first difference at %ld: ref %g vs fused %g (silu %ld, alone %ld, inj %ld, mix %ld, mix16 %ld)\n",
                        std::max(std::max(d_silu, d_alone), std::max(d_inj, std::max(d_mix, d_m16))), 0.f, 0.f, d_silu,
                        d_alone, d_inj, d_mix, d_m16);
            ++fails;
        }
        cudaFree(x), cudaFree(lo), cudaFree(inj_ref), cudaFree(lo16_ref), cudaFree(gate), cudaFree(mixed_ref),
            cudaFree(mixed16_ref), cudaFree(lo16), cudaFree(inj_fus), cudaFree(mixed), cudaFree(mixed16),
            cudaFree(lo16_alone);
    }
    std::printf("hc_fuse_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
