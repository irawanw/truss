// Forward throughput of qwen4exp::Forward on a model file (all weights resident): a prompt of random tokens in
// chunks, one warm-up pass, then one timed pass. Run under nsys for the per-kernel split.
// usage: tk-bench-prefill <model.gguf> [tokens=4096] [chunk=2048] [expert budget MiB, 0 = all free memory] [q8|fp16]
#include "core/cuda_check.h"
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/forward.h"
#include "model/qwen4exp/weights.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace truss;
namespace q = truss::qwen4exp;

int main(int argc, char ** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [tokens] [chunk]\n", argv[0]);
        return 2;
    }
    try {
        const auto file = gguf::File::open(argv[1]);
        if (const char * ov = std::getenv("TRUSS_EXPERT_OVERLAY")) {   // X3.1 experts over the served GGUF (#118)
            file->overlay(ov);
            std::fprintf(stderr, "expert overlay %s: %zu tensors replaced\n", ov, file->overlaid());
        }
        const q::Config c = q::Config::from_gguf(*file);
        const q::Weights w = q::bind(*file, c);
        const int n = argc > 2 ? std::atoi(argv[2]) : 4096, chunk = argc > 3 ? std::atoi(argv[3]) : 2048;
        const size_t budget = argc > 4 ? (size_t) std::atoll(argv[4]) << 20 : 0;
        const q::Activations act = argc > 5 && std::string(argv[5]) == "fp16" ? q::Activations::FP16 : q::Activations::Q8_1;
        std::mt19937 rng(1);
        std::vector<int32_t> tok(n);
        for (auto & t : tok) t = (int32_t) (rng() % 150000);
        for (int pass = 0; pass < 2; ++pass) {
            q::Forward p(c, w, n, chunk, { budget, act });
            if (!pass) std::printf("experts: %d of %d resident, %.2f GB streamed per chunk\n", p.hot_experts(),
                                   c.n_expert * c.n_layer, p.cold_bytes() / 1e9);
            const auto t0 = std::chrono::steady_clock::now();
            for (int s = 0; s < n; s += chunk) {
                const int len = std::min(chunk, n - s);
                const bool nx = s + len < n;   // PLE lookahead: the next chunk's reads start during this one
                p.run(tok.data() + s, len, nullptr, nx ? tok.data() + s + chunk : nullptr,
                      nx ? std::min(chunk, n - s - chunk) : 0);
            }
            TRUSS_CUDA(cudaStreamSynchronize(p.stream()));
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("%s: %d tokens, chunks of %d, %d layers: %.1f ms = %.0f tok/s (%.2f us/token/layer)\n",
                        pass ? "timed" : "warm-up", n, chunk, c.n_layer, sec * 1e3, n / sec, sec * 1e6 / n / c.n_layer);
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
