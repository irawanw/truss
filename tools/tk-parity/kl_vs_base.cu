// Full-model logits of qwen4exp::Forward vs a llama-perplexity --kl-divergence-base file (the Q8 teacher's
// log-probs), with llama-perplexity's own formulas: per chunk of n_ctx tokens (a fresh sequence each, no BOS for
// qwen4exp), positions n_ctx/2 .. n_ctx-2 predict the next token; KL(base || ours) over the base's tokens with log
// prob > -16, top-1 agreement, and both perplexities.
//
// File: "_logits_", int32 n_ctx, n_vocab, n_chunk, int32 tokens [n_chunk * n_ctx], then per chunk and scored
// position nv = 2 ((n_vocab + 1) / 2) + 4 uint16: float scale, float min_log_prob, n_vocab quantized log probs.
// usage: tk-parity-kl <model.gguf> <base.logits> [chunks=8] [first chunk=0] [q8|fp16] [step=0] [usage|-] [cpu dir|-]
//   step > 0: the first half of each chunk as one prompt, then the scored half in runs of `step` tokens (<= 32: the
//   decode path, with expert fetch, the ring and the CPU tier); usage / cpu dir as Forward::Options.
#include "core/cuda_check.h"
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/forward.h"
#include "model/qwen4exp/weights.h"
#include "runtime/expert_store.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace truss;
namespace q = truss::qwen4exp;

namespace {

struct Stats {
    double kld = 0, kld2 = 0, nll = 0, nll_base = 0;
    long count = 0, same_top = 0;
    void add(const Stats & o)
    {
        kld += o.kld, kld2 += o.kld2, nll += o.nll, nll_base += o.nll_base, count += o.count, same_top += o.same_top;
    }
};

// one scored position, as llama-perplexity's log_softmax(n_vocab, logits, base_log_prob, tok, kld)
void score(int n_vocab, const float * logits, const uint16_t * base, int tok, Stats & st)
{
    float max_logit = logits[0];
    int imax = 0;
    for (int i = 1; i < n_vocab; ++i)
        if (logits[i] > max_logit) max_logit = logits[i], imax = i;
    double sum_exp = 0;
    for (int i = 0; i < n_vocab; ++i) sum_exp += expf(logits[i] - max_logit);
    const float log_sum_exp = (float) std::log(sum_exp);
    float scale, min_log_prob;
    std::memcpy(&scale, base, 4);
    std::memcpy(&min_log_prob, base + 2, 4);
    const uint16_t * lp = base + 4;
    st.nll += max_logit + log_sum_exp - logits[tok];
    st.nll_base += -(scale * lp[tok] + min_log_prob);
    const float top = max_logit + log_sum_exp;
    double sum = 0;
    int imax_base = -1;
    float best = 0;
    for (int i = 0; i < n_vocab; ++i) {
        const float p_log_base = scale * lp[i] + min_log_prob;
        if (i == 0 || p_log_base > best) best = p_log_base, imax_base = i;
        if (p_log_base > -16.f) sum += expf(p_log_base) * (p_log_base - logits[i] + top);
    }
    st.kld += sum, st.kld2 += sum * sum;
    ++st.count;
    st.same_top += imax == imax_base;
}

}  // namespace

int main(int argc, char ** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <base.logits> [chunks]\n", argv[0]);
        return 2;
    }
    try {
        std::ifstream in(argv[2], std::ios::binary);
        char magic[8];
        int32_t n_ctx, n_vocab, n_chunk;
        in.read(magic, 8);
        in.read((char *) &n_ctx, 4), in.read((char *) &n_vocab, 4), in.read((char *) &n_chunk, 4);
        if (!in || std::memcmp(magic, "_logits_", 8)) throw std::runtime_error("not a llama-perplexity logits file");
        std::vector<int32_t> tokens((size_t) n_chunk * n_ctx);
        in.read((char *) tokens.data(), tokens.size() * 4);
        const int first_chunk = argc > 4 ? std::atoi(argv[4]) : 0;
        const int chunks = std::min(n_chunk - first_chunk, argc > 3 ? std::atoi(argv[3]) : 8);
        const q::Activations act = argc > 5 && std::string(argv[5]) == "fp16" ? q::Activations::FP16 : q::Activations::Q8_1;
        const int first = n_ctx / 2, n_scored = n_ctx - 1 - first;
        const size_t nv = 2 * ((n_vocab + 1) / 2) + 4;
        std::printf("base: n_ctx %d, n_vocab %d, %d chunks in file, scoring %d\n", n_ctx, n_vocab, n_chunk, chunks);

        const auto file = gguf::File::open(argv[1]);
        const q::Config c = q::Config::from_gguf(*file);
        const q::Weights w = q::bind(*file, c);
        if (c.n_vocab != n_vocab) throw std::runtime_error("vocabulary size differs from the base file");
        float * d_logits;   // before the engine, whose expert budget takes the memory left
        TRUSS_CUDA(cudaMalloc(&d_logits, (size_t) n_scored * n_vocab * 4));
        const int step = argc > 6 ? std::atoi(argv[6]) : 0;
        q::Forward::Options o;
        o.act = act;
        if (argc > 7 && std::string(argv[7]) != "-") o.expert_usage = runtime::ExpertStore::load_usage(argv[7], c.n_layer, c.n_expert);
        if (argc > 8 && std::string(argv[8]) != "-") o.cpu_dir = argv[8];
        q::Forward p(c, w, n_ctx, (n_ctx + 3) / 4 * 4, o);
        std::printf("experts: %d of %d resident, %.2f GB streamed per chunk\n", p.hot_experts(), c.n_expert * c.n_layer,
                    p.cold_bytes() / 1e9);
        std::vector<float> logits((size_t) n_scored * n_vocab);
        std::vector<uint16_t> base((size_t) n_scored * nv);
        Stats total;
        const int n_threads = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        in.seekg((std::streamoff) first_chunk * n_scored * nv * 2, std::ios::cur);
        for (int ch = first_chunk; ch < first_chunk + chunks; ++ch) {
            const auto t0 = std::chrono::steady_clock::now();
            const int32_t * tok = tokens.data() + (size_t) ch * n_ctx;
            p.reset();
            if (step <= 0) {
                p.run(tok, n_ctx);
                p.head(first, n_scored, d_logits);
            } else {   // prompt, then the scored positions a few tokens at a time
                p.run(tok, first);
                for (int pos = first; pos < first + n_scored; pos += step) {
                    const int T = std::min(step, first + n_scored - pos);
                    p.run(tok + pos, T);
                    p.head(0, T, d_logits + (size_t) (pos - first) * n_vocab);
                }
            }
            TRUSS_CUDA(cudaMemcpy(logits.data(), d_logits, logits.size() * 4, cudaMemcpyDeviceToHost));
            in.read((char *) base.data(), base.size() * 2);
            if (!in) throw std::runtime_error("base file ends early");
            std::vector<Stats> part(n_threads);
            std::vector<std::thread> pool;
            for (int th = 0; th < n_threads; ++th)
                pool.emplace_back([&, th] {
                    for (int i = th; i < n_scored; i += n_threads)
                        score(n_vocab, logits.data() + (size_t) i * n_vocab, base.data() + (size_t) i * nv,
                              tok[first + i + 1], part[th]);
                });
            for (auto & t : pool) t.join();
            Stats st;
            for (const Stats & s : part) st.add(s);
            total.add(st);
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("chunk %2d: KL %.5f, top-1 %.2f%%, PPL ours %.3f base %.3f | running KL %.5f top-1 %.2f%% "
                        "(%.1f s)\n", ch, st.kld / st.count, 100.0 * st.same_top / st.count, std::exp(st.nll / st.count),
                        std::exp(st.nll_base / st.count), total.kld / total.count,
                        100.0 * total.same_top / total.count, sec);
            std::fflush(stdout);
        }
        const double mean = total.kld / total.count;
        const double sd = std::sqrt(std::max(0.0, total.kld2 / total.count - mean * mean) / (total.count - 1));
        std::printf("TOTAL %ld tokens: KL %.5f +- %.5f, top-1 %.2f%%, PPL ours %.4f base %.4f\n", total.count, mean, sd,
                    100.0 * total.same_top / total.count, std::exp(total.nll / total.count),
                    std::exp(total.nll_base / total.count));
        cudaFree(d_logits);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
