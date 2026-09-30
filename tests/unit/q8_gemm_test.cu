// dense::q8_gemm (Q8_1 activations) vs ref::linear with llama numerics (same math: gate 1e-5, summation order), and
// dense::q8_gemm_a16 (fp16 activations) vs ref::linear fp32 on the same fp16-exact inputs (gate 3e-4: the one fp16
// rounding of each dequantized weight, <= 2^-12 relative).
// Random weights; timing on the Flash-Next dense prefill shapes.
// usage: q8_gemm_test [rows]      (default 8192 timing rows; correctness always on 300 rows)
#include "core/cuda_check.h"
#include "kernels/dense/q8_gemm.cuh"
#include "kernels/reference/ref.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace truss;

namespace {

struct Shape { int in, out; const char * name; };

int run(const Shape & sh, int rows, int time_rows, std::mt19937 & rng, cublasHandle_t cublas)
{
    const int nb = sh.in / 32 * sh.out;
    std::vector<uint8_t> blocks((size_t) nb * 34);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    for (int i = 0; i < nb; ++i) {
        const half d = __float2half(sd(rng));
        std::memcpy(&blocks[(size_t) i * 34], &d, 2);
        for (int j = 0; j < 32; ++j) blocks[(size_t) i * 34 + 2 + j] = (uint8_t) (int8_t) qd(rng);
    }
    const int R = std::max(rows, time_rows);
    std::vector<float> x((size_t) R * sh.in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = __half2float(__float2half(nd(rng)));   // fp16-exact, so the a16 path sees the same x
    std::vector<half> xh(x.size());
    for (size_t i = 0; i < x.size(); ++i) xh[i] = __float2half(x[i]);

    void * d_blocks;
    int8_t * wq, * xq;
    half * wd, * xd;
    float * d_x, * y, * yr;
    half * d_xh, * w16;
    TRUSS_CUDA(cudaMalloc(&d_blocks, blocks.size()));
    TRUSS_CUDA(cudaMalloc(&wq, (size_t) sh.in * sh.out));
    TRUSS_CUDA(cudaMalloc(&wd, (size_t) nb * 2));
    TRUSS_CUDA(cudaMalloc(&xq, (size_t) R * sh.in));
    TRUSS_CUDA(cudaMalloc(&xd, (size_t) R * sh.in / 32 * 2));
    TRUSS_CUDA(cudaMalloc(&d_x, x.size() * 4));
    TRUSS_CUDA(cudaMalloc(&d_xh, x.size() * 2));
    TRUSS_CUDA(cudaMalloc(&w16, (size_t) sh.in * sh.out * 2));
    TRUSS_CUDA(cudaMemcpy(d_xh, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMalloc(&y, (size_t) R * sh.out * 4));
    TRUSS_CUDA(cudaMalloc(&yr, (size_t) rows * sh.out * 4));
    TRUSS_CUDA(cudaMemcpy(d_blocks, blocks.data(), blocks.size(), cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    TRUSS_CUDA(cudaMemset(y, 0xff, (size_t) R * sh.out * 4));

    dense::q8_repack(d_blocks, sh.in, sh.out, wq, wd, nullptr);
    const dense::Q8Matrix W{ wq, wd, sh.in, sh.out };
    dense::q8_quantize_act(d_x, rows, sh.in, xq, xd, nullptr);
    dense::q8_gemm(W, xq, xd, rows, y, nullptr);
    DTensor T;
    T.data = d_blocks;
    T.type = gguf::Type::Q8_0;
    T.ne = { sh.in, sh.out, 1, 1 };
    ref::linear(T, d_x, yr, rows, nullptr, ref::Numerics::LLAMA);
    TRUSS_CUDA(cudaDeviceSynchronize());
    std::vector<float> a((size_t) rows * sh.out), b(a.size());
    TRUSS_CUDA(cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost));
    TRUSS_CUDA(cudaMemcpy(b.data(), yr, b.size() * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0;
    bool finite = true;
    for (size_t i = 0; i < a.size(); ++i) {
        finite &= std::isfinite(a[i]);
        num += (a[i] - (double) b[i]) * (a[i] - (double) b[i]);
        den += (double) b[i] * b[i];
    }
    const double rel = std::sqrt(num / den);

    auto rel_of = [&](const float * dev_a, const float * dev_b) {
        std::vector<float> u((size_t) rows * sh.out), v(u.size());
        TRUSS_CUDA(cudaMemcpy(u.data(), dev_a, u.size() * 4, cudaMemcpyDeviceToHost));
        TRUSS_CUDA(cudaMemcpy(v.data(), dev_b, v.size() * 4, cudaMemcpyDeviceToHost));
        double n2 = 0, d2 = 0;
        bool fin = true;
        for (size_t i = 0; i < u.size(); ++i) {
            fin &= std::isfinite(u[i]);
            n2 += (u[i] - (double) v[i]) * (u[i] - (double) v[i]);
            d2 += (double) v[i] * v[i];
        }
        return fin ? std::sqrt(n2 / d2) : INFINITY;
    };
    TRUSS_CUDA(cudaMemset(y, 0xff, (size_t) R * sh.out * 4));
    dense::q8_gemm_a16(W, d_xh, rows, y, w16, cublas, nullptr);
    ref::linear(T, d_x, yr, rows, nullptr, ref::Numerics::FP32);
    TRUSS_CUDA(cudaDeviceSynchronize());
    const double rel16 = rel_of(y, yr);
    const bool ok = finite && rel <= 1e-5 && rel16 <= 3e-4;

    dense::q8_quantize_act(d_x, time_rows, sh.in, xq, xd, nullptr);
    cudaEvent_t e0, e1, e2;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    const int reps = 20;
    dense::q8_gemm(W, xq, xd, time_rows, y, nullptr);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) dense::q8_gemm(W, xq, xd, time_rows, y, nullptr);
    cudaEventRecord(e1);
    for (int r = 0; r < reps; ++r) dense::q8_quantize_act(d_x, time_rows, sh.in, xq, xd, nullptr);
    cudaEventRecord(e2);
    for (int r = 0; r < reps; ++r) dense::q8_gemm_a16(W, d_xh, time_rows, y, w16, cublas, nullptr);
    cudaEvent_t e3;
    cudaEventCreate(&e3);
    cudaEventRecord(e3);
    TRUSS_CUDA(cudaEventSynchronize(e3));
    float ms, msq, ms16;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventElapsedTime(&msq, e1, e2);
    cudaEventElapsedTime(&ms16, e2, e3);
    ms /= reps;
    msq /= reps;
    ms16 /= reps;
    const double ops = 2.0 * time_rows * sh.in * sh.out;
    std::printf("%-9s %5d x %5d rel q8 %.1e a16 %.1e %s | rows %d: q8 %.3f ms %3.0f TOPS (+quantize %.3f) | a16 %.3f ms "
                "%3.0f TFLOPS\n", sh.name, sh.in, sh.out, rel, rel16, ok ? "PASS" : "FAIL", time_rows, ms, ops / ms / 1e9,
                msq, ms16, ops / ms16 / 1e9);
    cudaFree(d_blocks); cudaFree(wq); cudaFree(wd); cudaFree(xq); cudaFree(xd); cudaFree(d_x); cudaFree(d_xh); cudaFree(w16); cudaFree(y);
    cudaFree(yr);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv)
{
    const int time_rows = argc > 1 ? std::atoi(argv[1]) : 8192;
    // Flash-Next dense matmuls (tk-pack-inspect): GDN qkv / gate / out, DSA q / out, hyper-connection down / up,
    // shared expert gate|up / down
    const Shape shapes[] = { { 2560, 10240, "gdn qkv" }, { 2560, 6144, "gdn gate" }, { 6144, 2560, "gdn out" },
                             { 2560, 12288, "dsa q" }, { 10240, 320, "hc down" }, { 320, 10240, "hc up" },
                             { 2560, 640, "shexp gu" }, { 640, 2560, "shexp dn" } };
    try {
        std::mt19937 rng(99);
        cublasHandle_t cublas;
        if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate");
        int fails = 0;
        for (const auto & s : shapes) fails += run(s, 300, time_rows, rng, cublas);
        cublasDestroy(cublas);
        std::printf("%s\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
