// dense::q8_gemm: correctness vs ref::linear with llama numerics (Q8_0 x Q8_1, the fp32 reference implementation of
// the same math) on random weights, and timing on the Flash-Next dense prefill shapes.
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

int run(const Shape & sh, int rows, int time_rows, std::mt19937 & rng)
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
    for (auto & v : x) v = nd(rng);

    void * d_blocks;
    int8_t * wq, * xq;
    half * wd, * xd;
    float * d_x, * y, * yr;
    TRUSS_CUDA(cudaMalloc(&d_blocks, blocks.size()));
    TRUSS_CUDA(cudaMalloc(&wq, (size_t) sh.in * sh.out));
    TRUSS_CUDA(cudaMalloc(&wd, (size_t) nb * 2));
    TRUSS_CUDA(cudaMalloc(&xq, (size_t) R * sh.in));
    TRUSS_CUDA(cudaMalloc(&xd, (size_t) R * sh.in / 32 * 2));
    TRUSS_CUDA(cudaMalloc(&d_x, x.size() * 4));
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
    const bool ok = finite && rel <= 1e-5;   // same integer dots and scales; only the fp32 block-sum order differs

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
    TRUSS_CUDA(cudaEventSynchronize(e2));
    float ms, msq;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventElapsedTime(&msq, e1, e2);
    ms /= reps;
    msq /= reps;
    std::printf("%-10s %5d x %5d  rel %.1e %s | rows %d: gemm %.3f ms %.0f TOPS, act quantize %.3f ms\n", sh.name,
                sh.in, sh.out, rel, ok ? "PASS" : "FAIL", time_rows, ms, 2.0 * time_rows * sh.in * sh.out / ms / 1e9,
                msq);
    cudaFree(d_blocks); cudaFree(wq); cudaFree(wd); cudaFree(xq); cudaFree(xd); cudaFree(d_x); cudaFree(y);
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
        int fails = 0;
        for (const auto & s : shapes) fails += run(s, 300, time_rows, rng);
        std::printf("%s\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
