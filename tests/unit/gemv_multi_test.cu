// dense::q8_gemv_multi / f32_gemv_multi (decode: several matrices on the same activations, one launch, TRACKER #83)
// against an fp64 CPU reference on the same int8 / fp32 inputs, on the engine's decode groups; each row's result must
// not depend on the row count (bit-identical row 0 for rows 1 .. 8: the router and the spec windows rely on it) nor
// on fusion (each matrix alone gives the fused result bit for bit; "row-invariant" covers both);
// then the time of one fused launch vs one launch per matrix.
// usage: gemv_multi_test
#include "core/cuda_check.h"
#include "kernels/dense/q8_gemm.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace truss;

namespace {

struct Mat {
    int in, out;
    std::vector<int8_t> q;
    std::vector<half> d;
    int8_t * dq = nullptr;
    half * dd = nullptr;
    dense::Q8Matrix W{};
};

Mat make(int in, int out, std::mt19937 & g)
{
    Mat m{ in, out };
    m.q.resize((size_t) in * out), m.d.resize((size_t) in / 32 * out);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    for (auto & v : m.q) v = (int8_t) qd(g);
    for (auto & v : m.d) v = __float2half(sd(g));
    TRUSS_CUDA(cudaMalloc(&m.dq, m.q.size()));
    TRUSS_CUDA(cudaMalloc(&m.dd, m.d.size() * 2));
    TRUSS_CUDA(cudaMemcpy(m.dq, m.q.data(), m.q.size(), cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(m.dd, m.d.data(), m.d.size() * 2, cudaMemcpyHostToDevice));
    m.W = { m.dq, m.dd, in, out };
    return m;
}

float elapsed(cudaEvent_t a, cudaEvent_t b)
{
    float ms;
    TRUSS_CUDA(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

// one group: matrices sharing `in`; returns false on a failure
bool q8_group(const char * name, int in, std::vector<int> outs, std::mt19937 & g)
{
    const int n = (int) outs.size(), R = dense::GEMV_ROWS;
    std::vector<Mat> ms;
    for (int o : outs) ms.push_back(make(in, o, g));
    std::vector<int8_t> xq((size_t) R * in);
    std::vector<half> xd((size_t) R * in / 32);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.005f, 0.05f);
    for (auto & v : xq) v = (int8_t) qd(g);
    for (auto & v : xd) v = __float2half(sd(g));
    int8_t * dxq;
    half * dxd;
    TRUSS_CUDA(cudaMalloc(&dxq, xq.size()));
    TRUSS_CUDA(cudaMalloc(&dxd, xd.size() * 2));
    TRUSS_CUDA(cudaMemcpy(dxq, xq.data(), xq.size(), cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(dxd, xd.data(), xd.size() * 2, cudaMemcpyHostToDevice));
    std::vector<float *> dy(n);
    for (int i = 0; i < n; ++i) TRUSS_CUDA(cudaMalloc(&dy[i], (size_t) R * outs[i] * 4));
    std::vector<const dense::Q8Matrix *> Ws(n);
    for (int i = 0; i < n; ++i) Ws[i] = &ms[i].W;

    double worst = 0;
    bool invariant = true;
    std::vector<std::vector<float>> row0(n);
    for (int rows = 1; rows <= R; ++rows) {
        dense::q8_gemv_multi(Ws.data(), dy.data(), n, dxq, dxd, rows, nullptr);
        TRUSS_CUDA(cudaDeviceSynchronize());
        for (int i = 0; i < n; ++i) {
            std::vector<float> y((size_t) rows * outs[i]);
            TRUSS_CUDA(cudaMemcpy(y.data(), dy[i], y.size() * 4, cudaMemcpyDeviceToHost));
            if (rows == 1) row0[i].assign(y.begin(), y.begin() + outs[i]);
            else invariant &= std::memcmp(row0[i].data(), y.data(), outs[i] * 4) == 0;
            double e = 0, r = 0;
            for (int t = 0; t < rows; ++t)
                for (int o = 0; o < outs[i]; ++o) {
                    double ref = 0;
                    for (int b = 0; b < in / 32; ++b) {
                        long s = 0;
                        for (int j = 0; j < 32; ++j)
                            s += (long) ms[i].q[(size_t) o * in + b * 32 + j] * xq[(size_t) t * in + b * 32 + j];
                        ref += (double) s * ((double) __half2float(ms[i].d[(size_t) o * (in / 32) + b]) *
                                             __half2float(xd[(size_t) t * (in / 32) + b]));
                    }
                    const double v = y[(size_t) t * outs[i] + o];
                    e += (v - ref) * (v - ref), r += ref * ref;
                }
            worst = std::max(worst, std::sqrt(e / r));
        }
    }
    // each matrix alone (n = 1) must give the fused result bit for bit (mapping from `in` only)
    bool alone = true;
    dense::q8_gemv_multi(Ws.data(), dy.data(), n, dxq, dxd, 3, nullptr);
    std::vector<std::vector<float>> fusedv(n);
    for (int i = 0; i < n; ++i) {
        fusedv[i].resize((size_t) 3 * outs[i]);
        TRUSS_CUDA(cudaMemcpy(fusedv[i].data(), dy[i], fusedv[i].size() * 4, cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < n; ++i) {
        dense::q8_gemv(ms[i].W, dxq, dxd, 3, dy[i], nullptr);
        std::vector<float> a(fusedv[i].size());
        TRUSS_CUDA(cudaMemcpy(a.data(), dy[i], a.size() * 4, cudaMemcpyDeviceToHost));
        alone &= std::memcmp(a.data(), fusedv[i].data(), a.size() * 4) == 0;
    }
    invariant &= alone;
    // time: rows 4 (a spec window), one fused launch vs one launch per matrix
    cudaEvent_t e0, e1, e2;
    TRUSS_CUDA(cudaEventCreate(&e0));
    TRUSS_CUDA(cudaEventCreate(&e1));
    TRUSS_CUDA(cudaEventCreate(&e2));
    const int reps = 200;
    TRUSS_CUDA(cudaEventRecord(e0));
    for (int k = 0; k < reps; ++k) dense::q8_gemv_multi(Ws.data(), dy.data(), n, dxq, dxd, 4, nullptr);
    TRUSS_CUDA(cudaEventRecord(e1));
    for (int k = 0; k < reps; ++k)
        for (int i = 0; i < n; ++i) dense::q8_gemv(ms[i].W, dxq, dxd, 4, dy[i], nullptr);
    TRUSS_CUDA(cudaEventRecord(e2));
    TRUSS_CUDA(cudaEventSynchronize(e2));
    double mb = 0;
    for (int o : outs) mb += (double) in * o * (1 + 2.0 / 32) / 1e6;
    const float fused = elapsed(e0, e1) * 1e3f / reps, apart = elapsed(e1, e2) * 1e3f / reps;
    const bool pass = worst < 1e-6 && invariant;
    std::string shape;
    for (int o : outs) shape += std::to_string(in) + "->" + std::to_string(o) + " ";
    std::printf("%-12s %-44s rel %.1e, row-invariant %s | 4 rows: fused %.1f us (%.0f GB/s), %d launches %.1f us  %s\n", name,
                shape.c_str(), worst, invariant ? "yes" : "NO", fused, mb / fused * 1e3, n, apart, pass ? "ok" : "FAIL");
    for (auto & m : ms) cudaFree(m.dq), cudaFree(m.dd);
    for (float * p : dy) cudaFree(p);
    cudaFree(dxq), cudaFree(dxd);
    return pass;
}

bool f32_group(const char * name, int in, std::vector<int> outs, std::mt19937 & g)
{
    const int n = (int) outs.size(), R = 8;
    std::normal_distribution<float> nd;
    std::vector<std::vector<float>> W(n);
    std::vector<float *> dW(n), dy(n);
    for (int i = 0; i < n; ++i) {
        W[i].resize((size_t) in * outs[i]);
        for (auto & v : W[i]) v = nd(g) * 0.02f;
        TRUSS_CUDA(cudaMalloc(&dW[i], W[i].size() * 4));
        TRUSS_CUDA(cudaMemcpy(dW[i], W[i].data(), W[i].size() * 4, cudaMemcpyHostToDevice));
        TRUSS_CUDA(cudaMalloc(&dy[i], (size_t) R * outs[i] * 4));
    }
    std::vector<float> x((size_t) R * in);
    for (auto & v : x) v = nd(g);
    float * dx;
    TRUSS_CUDA(cudaMalloc(&dx, x.size() * 4));
    TRUSS_CUDA(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    std::vector<const float *> Wc(dW.begin(), dW.end());
    double worst = 0;
    bool invariant = true;
    std::vector<std::vector<float>> row0(n);
    for (int rows = 1; rows <= R; ++rows) {
        dense::f32_gemv_multi(Wc.data(), outs.data(), dy.data(), n, in, dx, rows, nullptr);
        TRUSS_CUDA(cudaDeviceSynchronize());
        for (int i = 0; i < n; ++i) {
            std::vector<float> y((size_t) rows * outs[i]);
            TRUSS_CUDA(cudaMemcpy(y.data(), dy[i], y.size() * 4, cudaMemcpyDeviceToHost));
            if (rows == 1) row0[i].assign(y.begin(), y.begin() + outs[i]);
            else invariant &= std::memcmp(row0[i].data(), y.data(), outs[i] * 4) == 0;
            double e = 0, r = 0;
            for (int t = 0; t < rows; ++t)
                for (int o = 0; o < outs[i]; ++o) {
                    double ref = 0;
                    for (int k = 0; k < in; ++k) ref += (double) W[i][(size_t) o * in + k] * x[(size_t) t * in + k];
                    const double v = y[(size_t) t * outs[i] + o];
                    e += (v - ref) * (v - ref), r += ref * ref;
                }
            worst = std::max(worst, std::sqrt(e / r));
        }
    }
    {   // each matrix alone gives the fused result bit for bit
        dense::f32_gemv_multi(Wc.data(), outs.data(), dy.data(), n, in, dx, 3, nullptr);
        for (int i = 0; i < n; ++i) {
            std::vector<float> f((size_t) 3 * outs[i]), a(f.size());
            TRUSS_CUDA(cudaMemcpy(f.data(), dy[i], f.size() * 4, cudaMemcpyDeviceToHost));
            dense::f32_gemv_multi(&Wc[i], &outs[i], &dy[i], 1, in, dx, 3, nullptr);
            TRUSS_CUDA(cudaMemcpy(a.data(), dy[i], a.size() * 4, cudaMemcpyDeviceToHost));
            invariant &= std::memcmp(a.data(), f.data(), f.size() * 4) == 0;
        }
    }
    cudaEvent_t e0, e1, e2;
    TRUSS_CUDA(cudaEventCreate(&e0));
    TRUSS_CUDA(cudaEventCreate(&e1));
    TRUSS_CUDA(cudaEventCreate(&e2));
    const int reps = 200;
    TRUSS_CUDA(cudaEventRecord(e0));
    for (int k = 0; k < reps; ++k) dense::f32_gemv_multi(Wc.data(), outs.data(), dy.data(), n, in, dx, 4, nullptr);
    TRUSS_CUDA(cudaEventRecord(e1));
    for (int k = 0; k < reps; ++k)
        for (int i = 0; i < n; ++i) dense::f32_gemv(dW[i], in, outs[i], dx, 4, dy[i], nullptr);
    TRUSS_CUDA(cudaEventRecord(e2));
    TRUSS_CUDA(cudaEventSynchronize(e2));
    const float fused = elapsed(e0, e1) * 1e3f / reps, apart = elapsed(e1, e2) * 1e3f / reps;
    const bool pass = worst < 1e-6 && invariant;
    std::printf("%-12s f32 in %d, %d matrices %-26s rel %.1e, row-invariant %s | 4 rows: fused %.1f us, %d launches %.1f us  %s\n",
                name, in, n, "", worst, invariant ? "yes" : "NO", fused, n, apart, pass ? "ok" : "FAIL");
    for (int i = 0; i < n; ++i) cudaFree(dW[i]), cudaFree(dy[i]);
    cudaFree(dx);
    return pass;
}

}  // namespace

int main()
{
    std::mt19937 g(11);
    bool ok = true;
    // the engine's decode groups (Flash-Next shapes)
    ok &= q8_group("hc down+inj", 10240, { 320, 4 }, g);
    ok &= q8_group("hc up", 320, { 10240 }, g);
    ok &= q8_group("gdn in", 2560, { 10240, 6144, 48, 48 }, g);
    ok &= q8_group("dsa in", 2560, { 12288, 512, 512, 512, 128 }, g);
    ok &= q8_group("shexp g+u", 2560, { 640, 640 }, g);
    ok &= q8_group("shexp down", 640, { 2560 }, g);
    ok &= q8_group("out-proj", 6144, { 2560 }, g);
    ok &= f32_group("router+hint", 2560, { 512, 512, 1 }, g);
    ok &= f32_group("router", 2560, { 512, 1 }, g);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
