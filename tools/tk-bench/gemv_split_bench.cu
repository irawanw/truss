// W8A16 decode gemv: time every (S warps per tile, KB blocks per tile) split on the engine's decode groups, weights
// rotated through enough copies to miss L2 as in a decode pass (each matrix is read once per layer). Prints us and
// GB/s per split for 1 and 8 rows, the default (dense::mma_split) marked, and the relative difference of each split's
// result from S=1 KB=1 (a different fp32 summation order only).
// usage: tk-bench-gemv-split [reps] [group name filter]
#include "core/cuda_check.h"
#include "kernels/dense/q8_gemm.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace truss;

namespace {

struct Group {
    const char * name;
    int in;
    std::vector<int> outs;
};

}  // namespace

int main(int argc, char ** argv)
{
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
    const std::vector<Group> groups = {
        { "hc down+inj", 10240, { 320, 4 } },  { "hc up", 320, { 10240 } },
        { "shexp g+u", 2560, { 640, 640 } },   { "shexp down", 640, { 2560 } },
        { "out-proj", 6144, { 2560 } },        { "mtp eh", 5120, { 2560 } },
        { "gdn in", 2560, { 10240, 6144, 48, 48 } }, { "dsa in", 2560, { 12288, 512, 512, 512, 128 } },
        { "draft head", 2560, { 40528 } },
    };
    std::mt19937 g(5);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int R = dense::GEMV_ROWS;
    cudaEvent_t e0, e1;
    TRUSS_CUDA(cudaEventCreate(&e0));
    TRUSS_CUDA(cudaEventCreate(&e1));
    for (const Group & G : groups) {
        if (argc > 2 && !std::strstr(G.name, argv[2])) continue;
        const int n = (int) G.outs.size();
        size_t bytes = 0;
        for (int o : G.outs) bytes += (size_t) G.in * o + (size_t) G.in / 32 * o * 2;
        const int copies = (int) std::max<size_t>(2, (96u << 20) / bytes + 1);   // > 96 MB: no L2 reuse
        std::vector<std::vector<dense::Q8Matrix>> W(copies, std::vector<dense::Q8Matrix>(n));
        for (int i = 0; i < n; ++i) {
            const int o = G.outs[i];
            std::vector<int8_t> q((size_t) G.in * o);
            std::vector<half> d((size_t) G.in / 32 * o);
            for (auto & v : q) v = (int8_t) qd(g);
            for (auto & v : d) v = __float2half(sd(g));
            for (int c = 0; c < copies; ++c) {
                int8_t * dq;
                half * dd;
                TRUSS_CUDA(cudaMalloc(&dq, q.size()));
                TRUSS_CUDA(cudaMalloc(&dd, d.size() * 2));
                TRUSS_CUDA(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
                TRUSS_CUDA(cudaMemcpy(dd, d.data(), d.size() * 2, cudaMemcpyHostToDevice));
                W[c][i] = { dq, dd, G.in, o };
            }
        }
        std::vector<half> x((size_t) R * G.in);
        for (auto & v : x) v = __float2half(nd(g));
        half * dx;
        TRUSS_CUDA(cudaMalloc(&dx, x.size() * 2));
        TRUSS_CUDA(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        std::vector<float *> dy(n);
        int total = 0;
        for (int i = 0; i < n; ++i) {
            TRUSS_CUDA(cudaMalloc(&dy[i], (size_t) R * G.outs[i] * 4));
            total += G.outs[i];
        }
        std::vector<std::vector<const dense::Q8Matrix *>> Wp(copies, std::vector<const dense::Q8Matrix *>(n));
        for (int c = 0; c < copies; ++c)
            for (int i = 0; i < n; ++i) Wp[c][i] = &W[c][i];
        auto result = [&](dense::MmaSplit s) {
            dense::q8_gemv_multi_a16_cfg(Wp[0].data(), dy.data(), n, dx, R, s, nullptr);
            std::vector<float> all;
            for (int i = 0; i < n; ++i) {
                std::vector<float> y((size_t) R * G.outs[i]);
                TRUSS_CUDA(cudaMemcpy(y.data(), dy[i], y.size() * 4, cudaMemcpyDeviceToHost));
                all.insert(all.end(), y.begin(), y.end());
            }
            return all;
        };
        const std::vector<float> ref = result({ 1, 1 });
        const dense::MmaSplit def = dense::mma_split(G.in, total);
        std::printf("%s: in %d, %d outputs, %.2f MB, %d copies; default S %d KB %d\n", G.name, G.in, total, bytes / 1e6,
                    copies, def.S, def.KB);
        for (int S : { 1, 2, 4, 8, 16 }) {
            if (S > G.in / 32) continue;
            for (int KB : { 1, 2, 3, 4, 6, 8, 12, 16 }) {
                if (S * KB > G.in / 32) continue;
                const std::vector<float> y = result({ S, KB });
                double e = 0, r = 0;
                for (size_t i = 0; i < y.size(); ++i) e += (y[i] - ref[i]) * (double) (y[i] - ref[i]), r += ref[i] * (double) ref[i];
                float t[2];
                int k = 0;
                for (int rows : { 1, 8 }) {
                    for (int w = 0; w < 20; ++w) dense::q8_gemv_multi_a16_cfg(Wp[w % copies].data(), dy.data(), n, dx, rows, { S, KB }, nullptr);
                    TRUSS_CUDA(cudaEventRecord(e0));
                    for (int it = 0; it < reps; ++it)
                        dense::q8_gemv_multi_a16_cfg(Wp[it % copies].data(), dy.data(), n, dx, rows, { S, KB }, nullptr);
                    TRUSS_CUDA(cudaEventRecord(e1));
                    TRUSS_CUDA(cudaEventSynchronize(e1));
                    float ms;
                    TRUSS_CUDA(cudaEventElapsedTime(&ms, e0, e1));
                    t[k++] = ms * 1e3f / reps;
                }
                std::printf("  S %2d KB %2d: 1 row %6.1f us (%4.0f GB/s), 8 rows %6.1f us (%4.0f GB/s), rel diff %.1e%s\n", S,
                            KB, t[0], bytes / t[0] / 1e3, t[1], bytes / t[1] / 1e3, std::sqrt(e / r),
                            S == def.S && KB == def.KB ? "  <- default" : "");
            }
        }
        for (auto & c : W)
            for (auto & m : c) cudaFree((void *) m.q), cudaFree((void *) m.d);
        for (auto * p : dy) cudaFree(p);
        cudaFree(dx);
    }
    return 0;
}
