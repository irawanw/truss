// Decode throughput of qwen4exp::Forward on a model file: a prompt (N random tokens, or @file of int32 ids, e.g. a
// real code-agent prompt from flashnext_truss_prompt_ids.py: routing, hence expert misses, depend on it), then greedy decode steps (one
// token per run(), device argmax, token copied back like the server does). One warm-up series, then a timed one.
// Prints ms per step and the split into steps; run under nsys for the per-kernel split.
// usage: tk-bench-decode <model.gguf> [prompt=512|@ids.i32] [steps=64] [n_ctx=65536] [chunk=8192] [usage file|-]
//                        [expert budget MiB, 0 = auto] [ring MiB]
#include "core/cuda_check.h"
#include "kernels/sampling/argmax.cuh"
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/forward.h"
#include "model/qwen4exp/weights.h"
#include "runtime/expert_store.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace truss;
namespace q = truss::qwen4exp;

int main(int argc, char ** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [prompt] [steps] [n_ctx] [chunk] [usage|-] [budget MiB]\n", argv[0]);
        return 2;
    }
    try {
        const auto file = gguf::File::open(argv[1]);
        const q::Config c = q::Config::from_gguf(*file);
        const q::Weights w = q::bind(*file, c);
        std::vector<int32_t> tok;
        if (argc > 2 && argv[2][0] == '@') {
            FILE * fp = std::fopen(argv[2] + 1, "rb");
            if (!fp) throw std::runtime_error(std::string("cannot open ") + (argv[2] + 1));
            int32_t t;
            while (std::fread(&t, 4, 1, fp) == 1) tok.push_back(t);
            std::fclose(fp);
        } else {
            std::mt19937 rng(1);
            tok.resize(argc > 2 ? std::atoi(argv[2]) : 512);
            for (auto & t : tok) t = (int32_t) (rng() % 150000);
        }
        const int n = (int) tok.size(), steps = argc > 3 ? std::atoi(argv[3]) : 64;
        const int n_ctx = argc > 4 ? std::atoi(argv[4]) : 65536, chunk = argc > 5 ? std::atoi(argv[5]) : 8192;
        q::Forward::Options o;
        if (argc > 6 && std::string(argv[6]) != "-")
            o.expert_usage = runtime::ExpertStore::load_usage(argv[6], c.n_layer, c.n_expert);
        o.expert_budget = argc > 7 ? (size_t) std::atoll(argv[7]) << 20 : 0;
        if (argc > 8) o.ring_bytes = (size_t) std::atoll(argv[8]) << 20;
        float * logits;
        int * next_d, * next_h;
        TRUSS_CUDA(cudaMalloc(&logits, sizeof(float) * c.n_vocab));
        TRUSS_CUDA(cudaMalloc(&next_d, sizeof(int)));
        TRUSS_CUDA(cudaMallocHost(&next_h, sizeof(int)));
        q::Forward f(c, w, n_ctx, chunk, o);
        std::printf("experts: %d of %d resident (%s), ring %.2f GB\n", f.hot_experts(), c.n_expert * c.n_layer,
                    o.expert_usage.empty() ? "index order" : "usage-ranked", f.experts().ring_bytes() / 1e9);
        for (int pass = 0; pass < 2; ++pass) {
            f.reset();
            for (int s = 0; s < n; s += chunk) f.run(tok.data() + s, std::min(chunk, n - s));
            f.head((n - 1) % chunk, 1, logits);   // the prompt's last row, in the last chunk
            sampling::argmax(logits, c.n_vocab, next_d, f.stream());
            TRUSS_CUDA(cudaMemcpyAsync(next_h, next_d, 4, cudaMemcpyDeviceToHost, f.stream()));
            TRUSS_CUDA(cudaStreamSynchronize(f.stream()));
            const runtime::ExpertStore::Stats st0 = f.experts().stats();
            std::vector<double> ms;
            for (int i = 0; i < steps; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                const int32_t t = *next_h;
                f.run(&t, 1);
                f.head(0, 1, logits);
                sampling::argmax(logits, c.n_vocab, next_d, f.stream());
                TRUSS_CUDA(cudaMemcpyAsync(next_h, next_d, 4, cudaMemcpyDeviceToHost, f.stream()));
                TRUSS_CUDA(cudaStreamSynchronize(f.stream()));
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            std::vector<double> sorted = ms;
            std::sort(sorted.begin(), sorted.end());
            double sum = 0;
            for (double x : ms) sum += x;
            std::printf("%s: prompt %d, %d steps: mean %.2f ms (%.1f tok/s), median %.2f, min %.2f, max %.2f\n",
                        pass ? "timed" : "warm-up", n, steps, sum / steps, 1e3 * steps / sum, sorted[steps / 2], sorted[0],
                        sorted[steps - 1]);
            const runtime::ExpertStore::Stats & st = f.experts().stats();
            std::printf("  per step: %.1f cold experts routed, %.1f fetched (%.1f MB)\n",
                        (double) (st.experts_asked - st0.experts_asked) / steps, (double) (st.misses - st0.misses) / steps,
                        (st.bytes - st0.bytes) / 1e6 / steps);
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
