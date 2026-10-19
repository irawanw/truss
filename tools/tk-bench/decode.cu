// Decode throughput of engine::Pipeline on a model file: a prompt (N random tokens, or @file of int32 ids, e.g. a
// real code-agent prompt from flashnext_truss_prompt_ids.py: routing, hence expert misses, depend on it), then greedy decode steps (one
// token per run(), device argmax, token copied back like the server does). One warm-up series, then a timed one.
// Prints ms per step and the split into steps; run under nsys for the per-kernel split.
// usage: tk-bench-decode <model.gguf> [prompt=512|@ids.i32] [steps=64] [n_ctx=65536] [chunk=8192] [usage file|-]
//                        [expert budget MiB, 0 = auto] [ring MiB]
#include "core/cuda_check.h"
#include "kernels/sampling/argmax.cuh"
#include "model/qwen4exp/config.h"
#include "engine/graph_step.h"
#include "engine/model.h"
#include "engine/options.h"
#include "engine/pipeline.h"
#include "model/qwen4exp/weights.h"
#include "runtime/expert_store.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace truss;

int main(int argc, char ** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [prompt] [steps] [n_ctx] [chunk] [usage|-] [budget MiB]\n", argv[0]);
        return 2;
    }
    try {
        const auto file = gguf::File::open(argv[1]);
        if (const char * ov = std::getenv("TRUSS_EXPERT_OVERLAY")) file->overlay_list(ov);   // X3.1 experts (#118)
        const auto fam = engine::open_family(*file, {});
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
        engine::Options o;
        engine::apply_env(o);   // TRUSS_LAYER_SPLIT (two GPUs), TRUSS_PREFILL_ROWS, decode-tier knobs; the args below win
        if (argc > 6 && std::string(argv[6]) != "-")
            o.expert_usage = runtime::ExpertStore::load_usage(argv[6], fam->n_layer(), fam->n_expert());
        o.moe.expert_budget = argc > 7 ? (size_t) std::atoll(argv[7]) << 20 : 0;
        if (argc > 8) o.moe.ring_bytes = (size_t) std::atoll(argv[8]) << 20;
        float * logits;
        int * next_d, * next_h;
        TRUSS_CUDA(cudaMalloc(&logits, sizeof(float) * fam->n_vocab()));
        TRUSS_CUDA(cudaMalloc(&next_d, sizeof(int)));
        TRUSS_CUDA(cudaMallocHost(&next_h, sizeof(int)));
        engine::Pipeline f(*fam, n_ctx, chunk, o);
        std::printf("experts: %d of %d resident (%s), ring %.2f GB\n", f.hot_experts(), fam->n_expert() * fam->n_layer(),
                    o.expert_usage.empty() ? "index order" : "usage-ranked", f.experts()->ring_bytes() / 1e9);
        // TRUSS_DECODE_GRAPH=1: B1 CUDA-graph decode (one graph per 128-block, pos0 refreshed per step)
        const bool use_graph = std::getenv("TRUSS_DECODE_GRAPH") && std::atoi(std::getenv("TRUSS_DECODE_GRAPH"));
        std::unique_ptr<engine::GraphStep> gs;
        if (use_graph) gs = std::make_unique<engine::GraphStep>(f, fam->n_vocab());
        for (int pass = 0; pass < 2; ++pass) {
            f.reset();
            const auto p0 = std::chrono::steady_clock::now();
            // the server's prompt path: a layer split runs its stages a chunk apart (#155); TRUSS_BENCH_SERIAL=1 = one
            // chunk at a time through every stage (the numbers before 10-19)
            if (std::getenv("TRUSS_BENCH_SERIAL") && std::atoi(std::getenv("TRUSS_BENCH_SERIAL")))
                for (int s = 0; s < n; s += chunk) f.run(tok.data() + s, std::min(chunk, n - s));
            else
                f.run_prompt(tok.data(), n, chunk);
            f.head((n - 1) % chunk, 1, logits);   // the prompt's last row, in the last chunk
            sampling::argmax(logits, fam->n_vocab(), next_d, f.stream());
            TRUSS_CUDA(cudaMemcpyAsync(next_h, next_d, 4, cudaMemcpyDeviceToHost, f.stream()));
            TRUSS_CUDA(cudaStreamSynchronize(f.stream()));
            const double pms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - p0).count();
            std::printf("  prefill %d tokens: %.0f ms (%.0f tok/s)\n", n, pms, 1e3 * n / pms);
            const runtime::ExpertStore::Stats st0 = f.experts()->stats();
            std::vector<double> ms;
            int pos = n;
            for (int i = 0; i < steps; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                const int32_t t = *next_h;
                if (use_graph) *next_h = gs->step(pos++, t, logits, next_d);
                else {
                    f.run(&t, 1);
                    f.head(0, 1, logits);
                    sampling::argmax(logits, fam->n_vocab(), next_d, f.stream());
                    TRUSS_CUDA(cudaMemcpyAsync(next_h, next_d, 4, cudaMemcpyDeviceToHost, f.stream()));
                    TRUSS_CUDA(cudaStreamSynchronize(f.stream()));
                }
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            std::vector<double> sorted = ms;
            std::sort(sorted.begin(), sorted.end());
            double sum = 0;
            for (double x : ms) sum += x;
            std::printf("%s: prompt %d, %d steps: mean %.2f ms (%.1f tok/s), median %.2f, min %.2f, max %.2f\n",
                        pass ? "timed" : "warm-up", n, steps, sum / steps, 1e3 * steps / sum, sorted[steps / 2], sorted[0],
                        sorted[steps - 1]);
            const runtime::ExpertStore::Stats & st = f.experts()->stats();
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
