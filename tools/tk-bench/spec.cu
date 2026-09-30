// Speculative (MTP) greedy decode vs plain greedy decode on the full model: the same prompt, N generated tokens each
// way; the token sequences must be identical (verify windows, rollback and the MTP cache must not change the
// output), and the speculative run's speed and tokens per verify pass are printed. Exit code 1 on a mismatch.
// usage: tk-bench-spec <model.gguf> <mtp.gguf> @prompt.i32 [tokens=128] [drafts=3] [usage file|-] [n_ctx=65536]
//                      [chunk=8192]
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
#include <stdexcept>
#include <string>
#include <vector>

using namespace truss;
namespace q = truss::qwen4exp;

namespace {

std::vector<int32_t> read_ids(const char * arg)
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

struct Greedy {   // argmax of logit rows on the device, copied back
    int vocab;
    float * logits;
    int * d_ids;
    int * h_ids;
    explicit Greedy(int v) : vocab(v)
    {
        TRUSS_CUDA(cudaMalloc(&logits, sizeof(float) * v * 9));
        TRUSS_CUDA(cudaMalloc(&d_ids, sizeof(int) * 9));
        TRUSS_CUDA(cudaMallocHost(&h_ids, sizeof(int) * 9));
    }
    void rows(q::Forward & f, int first, int n, int32_t * out)
    {
        f.head(first, n, logits);
        for (int r = 0; r < n; ++r) sampling::argmax(logits + (size_t) r * vocab, vocab, d_ids + r, f.stream());
        TRUSS_CUDA(cudaMemcpyAsync(h_ids, d_ids, sizeof(int) * n, cudaMemcpyDeviceToHost, f.stream()));
        TRUSS_CUDA(cudaStreamSynchronize(f.stream()));
        std::copy(h_ids, h_ids + n, out);
    }
};

void prompt(q::Forward & f, const std::vector<int32_t> & tok, int chunk)
{
    for (size_t s = 0; s < tok.size(); s += chunk) f.run(tok.data() + s, (int) std::min<size_t>(chunk, tok.size() - s));
}

}  // namespace

int main(int argc, char ** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <mtp.gguf> @prompt.i32 [tokens] [drafts] [usage|-] [n_ctx] [chunk]\n",
                     argv[0]);
        return 2;
    }
    try {
        const auto file = gguf::File::open(argv[1]);
        const auto mfile = gguf::File::open(argv[2]);
        const q::Config c = q::Config::from_gguf(*file);
        const q::Weights w = q::bind(*file, c);
        const q::Mtp mtp = q::bind_mtp(*mfile, c);
        const std::vector<int32_t> tok = read_ids(argv[3]);
        const int N = argc > 4 ? std::atoi(argv[4]) : 128, nd = argc > 5 ? std::atoi(argv[5]) : 3;
        const int n_ctx = argc > 7 ? std::atoi(argv[7]) : 65536, chunk = argc > 8 ? std::atoi(argv[8]) : 8192;
        q::Forward::Options o;
        o.mtp = &mtp;
        o.spec_rows = nd + 1;
        if (argc > 6 && std::string(argv[6]) != "-")
            o.expert_usage = runtime::ExpertStore::load_usage(argv[6], c.n_layer, c.n_expert);   // MTP: Forward adds
                                                                                               // the mean layer
        Greedy g(c.n_vocab);   // before the Forward: its expert budget takes the memory left
        q::Forward f(c, w, n_ctx, chunk, o);
        std::printf("experts: %d resident, ring %.2f GB; drafts %d\n", f.hot_experts(), f.experts().ring_bytes() / 1e9, nd);

        // plain greedy
        std::vector<int32_t> ref;
        prompt(f, tok, chunk);
        int32_t t;
        g.rows(f, (int) (tok.size() - 1) % chunk, 1, &t);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) {
            ref.push_back(t);
            f.run(&t, 1);
            g.rows(f, 0, 1, &t);
        }
        const double plain = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        // speculative
        f.reset();
        prompt(f, tok, chunk);
        g.rows(f, (int) (tok.size() - 1) % chunk, 1, &t);
        std::vector<int32_t> got;
        const runtime::ExpertStore::Stats st0 = f.experts().stats();
        int passes = 0;
        long drafted = 0, accepted = 0;
        t0 = std::chrono::steady_clock::now();
        while ((int) got.size() < N) {
            int32_t win[9], best[9];
            win[0] = t;
            f.draft(t, nd, win + 1);
            f.verify(win, nd + 1);
            g.rows(f, 0, nd + 1, best);
            int j = 0;
            while (j < nd && best[j] == win[j + 1]) ++j;
            f.accept(j + 1);
            for (int i = 0; i <= j; ++i) got.push_back(win[i]);
            t = best[j];
            ++passes, drafted += nd, accepted += j;
        }
        const double spec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        got.resize(N);
        int same = 0;
        while (same < N && got[same] == ref[same]) ++same;
        std::printf("plain: %d tokens in %.2f s = %.1f tok/s\n", N, plain, N / plain);
        std::printf("spec : %d tokens in %.2f s = %.1f tok/s, %d passes (%.2f tokens/pass), drafts accepted %ld of %ld\n",
                    (int) got.size(), spec, N / spec, passes, (double) N / passes, accepted, drafted);
        const runtime::ExpertStore::Stats & st = f.experts().stats();
        std::printf("       per pass: %.1f cold experts routed, %.1f fetched on demand (%.1f MB), %.1f prefetched (%.1f MB)\n",
                    (double) (st.experts_asked - st0.experts_asked) / passes, (double) (st.misses - st0.misses) / passes,
                    (st.bytes - st0.bytes) / 1e6 / passes, (double) (st.hinted - st0.hinted) / passes,
                    (st.hint_bytes - st0.hint_bytes) / 1e6 / passes);
        std::printf("tokens identical to plain greedy: %d of %d  %s\n", same, N, same == N ? "PASS" : "FAIL");
        return same == N ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
