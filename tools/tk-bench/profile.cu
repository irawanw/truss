// Routing profile for the expert hot set (the "calibration" step, like Strata's make_profile): runs each prompt
// through qwen4exp::Forward and generates greedy tokens after it, counting every routed (layer, expert); writes the
// counts as the usage file ExpertStore::load_usage reads (n_layer x n_expert float32). Decode tokens are counted
// too: they are what the hot set serves.
// usage: tk-profile <model.gguf> <out.f32> <generate per prompt> @prompt1.i32 [@prompt2.i32 ...]
#include "core/cuda_check.h"
#include "kernels/sampling/argmax.cuh"
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/forward.h"
#include "model/qwen4exp/weights.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace truss;
namespace q = truss::qwen4exp;

static std::vector<int32_t> read_ids(const char * arg)
{
    if (arg[0] != '@') throw std::runtime_error(std::string("prompt must be @file.i32: ") + arg);
    FILE * f = std::fopen(arg + 1, "rb");
    if (!f) throw std::runtime_error(std::string("cannot open ") + (arg + 1));
    std::vector<int32_t> v;
    int32_t t;
    while (std::fread(&t, 4, 1, f) == 1) v.push_back(t);
    std::fclose(f);
    return v;
}

int main(int argc, char ** argv)
{
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <model.gguf> <out.f32> <generate> @prompt.i32 ...\n", argv[0]);
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
        const int gen = std::atoi(argv[3]), chunk = 8192;
        float * logits;
        int * next_d, * next_h;
        TRUSS_CUDA(cudaMalloc(&logits, sizeof(float) * c.n_vocab));
        TRUSS_CUDA(cudaMalloc(&next_d, sizeof(int)));
        TRUSS_CUDA(cudaMallocHost(&next_h, sizeof(int)));
        q::Forward f(c, w, 65536, chunk);
        f.profile_routes(true);
        long tokens = 0;
        for (int a = 4; a < argc; ++a) {
            const std::vector<int32_t> tok = read_ids(argv[a]);
            const int n = (int) tok.size();
            f.reset();
            for (int s = 0; s < n; s += chunk) f.run(tok.data() + s, std::min(chunk, n - s));
            f.head((n - 1) % chunk, 1, logits);
            for (int i = 0; i < gen; ++i) {
                sampling::argmax(logits, c.n_vocab, next_d, f.stream());
                TRUSS_CUDA(cudaMemcpyAsync(next_h, next_d, 4, cudaMemcpyDeviceToHost, f.stream()));
                TRUSS_CUDA(cudaStreamSynchronize(f.stream()));
                const int32_t t = *next_h;
                f.run(&t, 1);
                f.head(0, 1, logits);
            }
            tokens += n + gen;
            std::printf("%s: %d prompt + %d generated\n", argv[a] + 1, n, gen);
        }
        const std::vector<float> counts = f.route_counts();
        FILE * out = std::fopen(argv[2], "wb");
        if (!out || std::fwrite(counts.data(), 4, counts.size(), out) != counts.size())
            throw std::runtime_error(std::string("cannot write ") + argv[2]);
        std::fclose(out);
        std::printf("%s: %ld tokens profiled\n", argv[2], tokens);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
