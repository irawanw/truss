// dense::q8_gemv_grouped_a16 (the grouped tensor-core gemv, DS41 wo_a at decode rows) vs a host fp64 reference of the
// block-diagonal product and vs dense::q8_gemm_a16_grouped (dequant + cuBLAS, the path it replaces), rows 1..8, on
// DeepSeek-V4.1's wo_a shape (G 8, R 1024, K 4096) and a small one; timing of both at 4 rows.
// Gate: rel. error vs fp64 <= 1e-4 (int8 x fp16 products are exact in the mma, fp32 accumulation), and not worse than
// the cuBLAS path's (which rounds each dequantized weight to fp16).
// usage: q8_grouped_gemv_test
#include "core/cublas_check.h"
#include "core/cuda_check.h"
#include "kernels/dense/q8_gemm.cuh"

#include <cublas_v2.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace truss;

namespace {

int run(int G, int R, int K, std::mt19937 & rng, cublasHandle_t cublas)
{
    const int out = G * R, nb = K / 32;
    std::vector<int8_t> q((size_t) out * K);
    std::vector<half> d((size_t) out * nb);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    for (auto & v : q) v = (int8_t) qd(rng);
    for (auto & v : d) v = __float2half(sd(rng));
    const int RM = dense::GEMV_ROWS;
    std::vector<half> x((size_t) RM * G * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = __float2half(nd(rng));
    int8_t * dq;
    half * dd, * dx, * w16;
    float * y1, * y2;
    TRUSS_CUDA(cudaMalloc(&dq, q.size()));
    TRUSS_CUDA(cudaMalloc(&dd, d.size() * 2));
    TRUSS_CUDA(cudaMalloc(&dx, x.size() * 2));
    TRUSS_CUDA(cudaMalloc(&w16, q.size() * 2));
    TRUSS_CUDA(cudaMalloc(&y1, (size_t) RM * out * 4));
    TRUSS_CUDA(cudaMalloc(&y2, (size_t) RM * out * 4));
    TRUSS_CUDA(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(dd, d.data(), d.size() * 2, cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    const dense::Q8Matrix W{ dq, dd, K, out };
    int bad = 0;
    for (int rows = 1; rows <= RM; ++rows) {
        TRUSS_CUDA(cudaMemset(y1, 0xff, (size_t) RM * out * 4));
        dense::q8_gemv_grouped_a16(W, G, dx, rows, y1, nullptr);
        dense::q8_gemm_a16_grouped(W, G, dx, rows, y2, w16, cublas, nullptr);
        std::vector<float> a((size_t) rows * out), b(a.size());
        TRUSS_CUDA(cudaMemcpy(a.data(), y1, a.size() * 4, cudaMemcpyDeviceToHost));
        TRUSS_CUDA(cudaMemcpy(b.data(), y2, b.size() * 4, cudaMemcpyDeviceToHost));
        double na = 0, nb2 = 0, den = 0;
        for (int t = 0; t < rows; ++t)
            for (int o = 0; o < out; o += (o % 97 == 0 ? 1 : 13)) {   // a strided sample of outputs, all groups
                const int g = o / R;
                double ref = 0;
                for (int k = 0; k < K; ++k)
                    ref += (double) q[(size_t) o * K + k] * __half2float(d[(size_t) o * nb + k / 32]) *
                           __half2float(x[(size_t) t * G * K + (size_t) g * K + k]);
                const double ea = a[(size_t) t * out + o] - ref, eb = b[(size_t) t * out + o] - ref;
                na += ea * ea, nb2 += eb * eb, den += ref * ref;
            }
        const double ra = std::sqrt(na / den), rb = std::sqrt(nb2 / den);
        const bool ok = std::isfinite(ra) && ra <= 1e-4 && ra <= rb * 1.01 + 1e-7;
        bad += !ok;
        std::printf("G %d R %d K %d rows %d: gemv rel %.2e, cublas path rel %.2e  %s\n", G, R, K, rows, ra, rb,
                    ok ? "PASS" : "FAIL");
    }
    auto time = [&](auto && f) {
        for (int i = 0; i < 5; ++i) f();
        TRUSS_CUDA(cudaDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 50; ++i) f();
        TRUSS_CUDA(cudaDeviceSynchronize());
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 50;
    };
    const double tg = time([&] { dense::q8_gemv_grouped_a16(W, G, dx, 4, y1, nullptr); });
    const double tc = time([&] { dense::q8_gemm_a16_grouped(W, G, dx, 4, y2, w16, cublas, nullptr); });
    std::printf("G %d R %d K %d, 4 rows: grouped gemv %.1f us, dequant + cuBLAS %.1f us (%.1f MB int8: %.0f GB/s)\n", G,
                R, K, tg, tc, q.size() / 1e6, q.size() / tg / 1e3);
    cudaFree(dq), cudaFree(dd), cudaFree(dx), cudaFree(w16), cudaFree(y1), cudaFree(y2);
    return bad;
}

}  // namespace

int main()
{
    std::mt19937 rng(7);
    cublasHandle_t cublas;
    TRUSS_CUBLAS(cublasCreate(&cublas));
    int bad = run(8, 1024, 4096, rng, cublas);   // DeepSeek-V4.1-Flash wo_a
    bad += run(4, 64, 256, rng, cublas);
    cublasDestroy(cublas);
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
