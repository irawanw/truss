// Is the dense GEMM bit-identical when its rows are split into pieces? The DS41 prompt path processes a chunk in
// row pieces (the attention's and the MoE's per-row scratch must not scale with the chunk), so the pieces change
// the GEMM's row count M. cuBLAS picks its kernel by shape, and a different M can mean a different k-split and
// therefore different last bits (TRACKER #60: the router's SGEMM). This test measures it for the DS41 attention's
// shapes: M = 4096 (one call) vs the same rows in pieces of 1024 / 2048, for dense::q8_gemm_a16 (fp16 activations)
// and dense::q8_gemm_a16_grouped (the grouped output LoRA).
// usage: gemm_piece_test [rows=4096]
#include "core/cuda_check.h"
#include "kernels/dense/q8_gemm.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace truss;

namespace {

struct Shape {
    int in, out;
    const char * name;
};

// a Q8_0 matrix (random), repacked like the engine does
void make_q8(int in, int out, std::mt19937 & rng, void *& blocks, int8_t *& wq, half *& wd)
{
    const int nb = in / 32 * out;
    std::vector<uint8_t> h((size_t) nb * 34);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    for (int i = 0; i < nb; ++i) {
        const half d = __float2half(sd(rng));
        std::memcpy(&h[(size_t) i * 34], &d, 2);
        for (int j = 0; j < 32; ++j) h[(size_t) i * 34 + 2 + j] = (uint8_t) (int8_t) qd(rng);
    }
    TRUSS_CUDA(cudaMalloc(&blocks, h.size()));
    TRUSS_CUDA(cudaMalloc(&wq, (size_t) in * out));
    TRUSS_CUDA(cudaMalloc(&wd, (size_t) nb * 2));
    TRUSS_CUDA(cudaMemcpy(blocks, h.data(), h.size(), cudaMemcpyHostToDevice));
    dense::q8_repack(blocks, in, out, wq, wd, nullptr);
}

}  // namespace

int main(int argc, char ** argv)
{
    const int rows = argc > 1 ? std::atoi(argv[1]) : 4096;
    // the DS41 attention's projections: q_a (5120->1280), q_b (1280->32768), kv (5120->512), out_b (8192->5120)
    const Shape shapes[] = { { 5120, 1280, "q_a" }, { 1280, 32768, "q_b" }, { 5120, 512, "kv" }, { 8192, 5120, "out_b" } };
    try {
        std::mt19937 rng(7);
        cublasHandle_t cublas;
        if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate");
        int fails = 0;
        for (const Shape & sh : shapes) {
            void * blocks;
            int8_t * wq;
            half * wd;
            make_q8(sh.in, sh.out, rng, blocks, wq, wd);
            const dense::Q8Matrix W{ wq, wd, sh.in, sh.out };
            half * x, * w16;
            float * y;
            TRUSS_CUDA(cudaMalloc(&x, (size_t) rows * sh.in * 2));
            TRUSS_CUDA(cudaMalloc(&w16, (size_t) sh.in * sh.out * 2));
            TRUSS_CUDA(cudaMalloc(&y, (size_t) rows * sh.out * 4));
            std::vector<half> hx((size_t) rows * sh.in);
            std::uniform_real_distribution<float> ud(-1.f, 1.f);
            for (auto & v : hx) v = __float2half(ud(rng));
            TRUSS_CUDA(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
            dense::q8_gemm_a16(W, x, rows, y, w16, cublas, nullptr);
            std::vector<float> full((size_t) rows * sh.out);
            TRUSS_CUDA(cudaMemcpy(full.data(), y, full.size() * 4, cudaMemcpyDeviceToHost));
            for (int piece : { 512, 1024, 2048 }) {
                if (piece >= rows) continue;
                TRUSS_CUDA(cudaMemset(y, 0, (size_t) rows * sh.out * 4));
                for (int r0 = 0; r0 < rows; r0 += piece) {
                    const int rb = std::min(piece, rows - r0);
                    dense::q8_gemm_a16(W, x + (size_t) r0 * sh.in, rb, y + (size_t) r0 * sh.out, w16, cublas, nullptr);
                }
                std::vector<float> part((size_t) rows * sh.out);
                TRUSS_CUDA(cudaMemcpy(part.data(), y, part.size() * 4, cudaMemcpyDeviceToHost));
                long diff = 0;
                double num = 0, den = 0;
                for (size_t i = 0; i < part.size(); ++i) {
                    diff += part[i] != full[i];
                    num += (double) (part[i] - full[i]) * (part[i] - full[i]);
                    den += (double) full[i] * full[i];
                }
                const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
                std::printf("%-6s %5d x %6d rows %d in pieces of %4d: %8ld values differ, rel %.2e %s\n", sh.name, sh.in,
                            sh.out, rows, piece, diff, rel, diff == 0 ? "IDENTICAL" : "DIFFER");
                if (diff) fails++;
            }
            cudaFree(blocks); cudaFree(wq); cudaFree(wd); cudaFree(x); cudaFree(w16); cudaFree(y);
        }
        // the grouped output LoRA: W [G * R][K], row t's group g reads x[t][g * K ..) and writes y[t][g * R ..)
        {
            const int G = 8, R = 1024, K = 4096;
            void * blocks;
            int8_t * wq;
            half * wd;
            make_q8(K, G * R, rng, blocks, wq, wd);
            const dense::Q8Matrix W{ wq, wd, K, G * R };
            half * x, * w16;
            float * y;
            TRUSS_CUDA(cudaMalloc(&x, (size_t) rows * G * K * 2));
            TRUSS_CUDA(cudaMalloc(&w16, (size_t) K * G * R * 2));
            TRUSS_CUDA(cudaMalloc(&y, (size_t) rows * G * R * 4));
            std::vector<half> hx((size_t) rows * G * K);
            std::uniform_real_distribution<float> ud(-1.f, 1.f);
            for (auto & v : hx) v = __float2half(ud(rng));
            TRUSS_CUDA(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
            dense::q8_gemm_a16_grouped(W, G, x, rows, y, w16, cublas, nullptr);
            std::vector<float> full((size_t) rows * G * R);
            TRUSS_CUDA(cudaMemcpy(full.data(), y, full.size() * 4, cudaMemcpyDeviceToHost));
            for (int piece : { 1024, 2048 }) {
                if (piece >= rows) continue;
                TRUSS_CUDA(cudaMemset(y, 0, (size_t) rows * G * R * 4));
                for (int r0 = 0; r0 < rows; r0 += piece) {
                    const int rb = std::min(piece, rows - r0);
                    dense::q8_gemm_a16_grouped(W, G, x + (size_t) r0 * G * K, rb, y + (size_t) r0 * G * R, w16, cublas,
                                               nullptr);
                }
                std::vector<float> part((size_t) rows * G * R);
                TRUSS_CUDA(cudaMemcpy(part.data(), y, part.size() * 4, cudaMemcpyDeviceToHost));
                long diff = 0;
                double num = 0, den = 0;
                for (size_t i = 0; i < part.size(); ++i) {
                    diff += part[i] != full[i];
                    num += (double) (part[i] - full[i]) * (part[i] - full[i]);
                    den += (double) full[i] * full[i];
                }
                const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
                std::printf("grouped G=%d R=%d K=%d rows %d in pieces of %4d: %8ld values differ, rel %.2e %s\n", G, R, K,
                            rows, piece, diff, rel, diff == 0 ? "IDENTICAL" : "DIFFER");
                if (diff) fails++;
            }
            cudaFree(blocks); cudaFree(wq); cudaFree(wd); cudaFree(x); cudaFree(w16); cudaFree(y);
        }
        // the grouped int8 GEMM vs the dense product of the stacked rows (the path the attention used): the same
        // values (the int32 block dots and their fp32 folding are per output element)
        {
            const int G = 8, R = 1024, K = 4096, dm = 5120;
            void * blocks;
            int8_t * wq;
            half * wd;
            make_q8(K, G * R, rng, blocks, wq, wd);
            const dense::Q8Matrix W{ wq, wd, K, G * R };
            int8_t * xq;
            half * xd;
            half * x;
            float * y, * ya;
            TRUSS_CUDA(cudaMalloc(&xq, (size_t) rows * G * K));
            TRUSS_CUDA(cudaMalloc(&xd, (size_t) rows * G * K / 32 * 2));
            TRUSS_CUDA(cudaMalloc(&x, (size_t) rows * G * K * 2));
            TRUSS_CUDA(cudaMalloc(&y, (size_t) rows * G * R * 4));
            TRUSS_CUDA(cudaMalloc(&ya, (size_t) rows * G * G * R * 4));
            std::vector<half> hx((size_t) rows * G * K);
            std::uniform_real_distribution<float> ud(-1.f, 1.f);
            for (auto & v : hx) v = __float2half(ud(rng));
            TRUSS_CUDA(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
            dense::q8_quantize_act(x, rows * G, K, xq, xd, nullptr);
            dense::q8_gemm(W, xq, xd, rows * G, ya, nullptr);
            dense::q8_gemm_grouped(W, G, xq, xd, rows, y, nullptr);
            std::vector<float> b((size_t) rows * G * R), a((size_t) rows * G * G * R);
            TRUSS_CUDA(cudaMemcpy(a.data(), ya, a.size() * 4, cudaMemcpyDeviceToHost));
            TRUSS_CUDA(cudaMemcpy(b.data(), y, b.size() * 4, cudaMemcpyDeviceToHost));
            long diff = 0;
            double num = 0, den = 0;
            for (int t = 0; t < rows; ++t)
                for (int gg = 0; gg < G; ++gg)
                    for (int c = 0; c < R; ++c) {
                        const size_t i = (size_t) t * G * R + gg * R + c;
                        const float va = a[(size_t) (t * G + gg) * G * R + gg * R + c];
                        diff += va != b[i];
                        num += (double) (va - b[i]) * (va - b[i]);
                        den += (double) b[i] * b[i];
                    }
            const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
            std::printf("grouped int8 G=%d R=%d K=%d rows %d vs the dense stacked rows: %8ld values differ, rel %.2e %s\n",
                        G, R, K, rows, diff, rel, diff == 0 ? "IDENTICAL" : "DIFFER");
            if (diff) fails++;
            cudaFree(blocks); cudaFree(wq); cudaFree(wd); cudaFree(xq); cudaFree(xd); cudaFree(x); cudaFree(y);
            cudaFree(ya);
        }
        cublasDestroy(cublas);
        std::printf("%s\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
