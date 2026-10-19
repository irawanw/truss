// Full-model KL of engine::Pipeline on DeepSeek-V4.1 against references from the PoC's Python stream
// (20261018_poc a5_stream.py ref_logits*.pt, exported to raw bf16 [n_seq][ctx][n_vocab] by ds41_kl_export.py):
// per sequence one prompt of ctx tokens, logits of every position, KL(ref || TRUSS) over the full vocab per token
// (a5_stream's definition), top-1 agreement; mean / median / p99 over all tokens.
// Several references at once (e.g. the unpruned FP4 model and the same pack run in Python: the second isolates the
// engine from the pack).
// usage: TRUSS_EXPERT_OVERLAY=<experts>,<engram L1>,<engram L14> ds41_kl <dense.gguf> <ids.i32> <n_seq> <ctx>
//          <ref.bf16> [<ref2.bf16> ...]
// TRUSS_KL_DECODE=P: each sequence's first P tokens run as one prompt, the rest one token per run (the decode path:
// CPU tier, tail / weight-share skip act only there); KL and top-1 over positions >= P only.
#include "core/cuda_check.h"
#include "engine/model.h"
#include "engine/options.h"
#include "engine/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace truss;

int main(int argc, char ** argv)
{
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s <dense.gguf> <ids.i32> <n_seq> <ctx> <ref.bf16> [ref2 ...]\n", argv[0]);
        return 2;
    }
    try {
        const int n_seq = std::atoi(argv[3]), ctx = std::atoi(argv[4]);
        const int dec = std::getenv("TRUSS_KL_DECODE") ? std::atoi(std::getenv("TRUSS_KL_DECODE")) : 0;
        if (dec < 0 || dec >= ctx) throw std::runtime_error("TRUSS_KL_DECODE must be in [0, ctx)");
        std::vector<int32_t> ids((size_t) n_seq * ctx);
        std::ifstream(argv[2], std::ios::binary).read((char *) ids.data(), ids.size() * 4);
        const auto file = gguf::File::open(argv[1]);
        if (const char * ov = std::getenv("TRUSS_EXPERT_OVERLAY")) file->overlay_list(ov);
        const auto fam = engine::open_family(*file, {});
        const int V = fam->n_vocab();
        float * d_logits;
        TRUSS_CUDA(cudaMalloc(&d_logits, (size_t) ctx * V * 4));
        engine::Options o;
        o.act = blocks::Activations::FP16;
        o.prefill_rows = ctx;
        engine::apply_env(o);
        engine::Pipeline p(*fam, ctx, ctx, o);
        file->release_pages();
        std::printf("experts: %d of %d resident, %.2f GB streamed per chunk\n", p.hot_experts(),
                    fam->n_expert() * fam->n_layer(), p.cold_bytes() / 1e9);
        const int n_ref = argc - 5;
        std::vector<std::ifstream> refs;
        for (int r = 0; r < n_ref; ++r) refs.emplace_back(argv[5 + r], std::ios::binary);
        std::vector<std::vector<float>> kl(n_ref);
        std::vector<long> agree(n_ref, 0);
        std::vector<float> lg((size_t) ctx * V);
        std::vector<uint16_t> rb((size_t) ctx * V);
        for (int s = 0; s < n_seq; ++s) {
            p.reset();
            const int32_t * seq = ids.data() + (size_t) s * ctx;
            if (dec > 0) {
                p.run(seq, dec);
                for (int t = dec; t < ctx; ++t) {
                    p.run(seq + t, 1);
                    p.head(0, 1, d_logits + (size_t) t * V);
                }
            } else {
                p.run(seq, ctx);
                p.head(0, ctx, d_logits);
            }
            TRUSS_CUDA(cudaMemcpyAsync(lg.data(), d_logits, lg.size() * 4, cudaMemcpyDeviceToHost, p.stream()));
            TRUSS_CUDA(cudaStreamSynchronize(p.stream()));
            for (int r = 0; r < n_ref; ++r) {
                refs[r].read((char *) rb.data(), rb.size() * 2);
                if (!refs[r]) throw std::runtime_error(std::string("short reference ") + argv[5 + r]);
                std::vector<float> ks(ctx);
                std::vector<char> ag(ctx);
#pragma omp parallel for
                for (int t = dec; t < ctx; ++t) {
                    const float * q = lg.data() + (size_t) t * V;
                    const uint16_t * pr = rb.data() + (size_t) t * V;
                    auto bf = [] (uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; std::memcpy(&f, &u, 4); return f; };
                    float mr = -INFINITY, mq = -INFINITY;
                    int ar = 0, aq = 0;
                    for (int v = 0; v < V; ++v) {
                        const float a = bf(pr[v]);
                        if (a > mr) mr = a, ar = v;
                        if (q[v] > mq) mq = q[v], aq = v;
                    }
                    double sr = 0, sq = 0;
                    for (int v = 0; v < V; ++v) sr += std::exp((double) bf(pr[v]) - mr), sq += std::exp((double) q[v] - mq);
                    const double lsr = mr + std::log(sr), lsq = mq + std::log(sq);
                    double k = 0;
                    for (int v = 0; v < V; ++v) {
                        const double lr = bf(pr[v]) - lsr;
                        k += std::exp(lr) * (lr - (q[v] - lsq));
                    }
                    ks[t] = (float) k, ag[t] = ar == aq;
                }
                for (int t = dec; t < ctx; ++t) kl[r].push_back(ks[t]), agree[r] += ag[t];
            }
            double m0 = 0;
            for (float v : kl[0]) m0 += v;
            std::printf("seq %d/%d: running KL vs ref 0 %.4f\n", s + 1, n_seq, m0 / kl[0].size());
            std::fflush(stdout);
        }
        for (int r = 0; r < n_ref; ++r) {
            auto v = kl[r];
            double m = 0;
            for (float x : v) m += x;
            std::sort(v.begin(), v.end());
            std::printf("vs %s: KL mean %.4f median %.4f p99 %.4f, top-1 agreement %.4f (%zu tokens)\n", argv[5 + r],
                        m / v.size(), v[v.size() / 2], v[(size_t) (v.size() * 0.99)], (double) agree[r] / v.size(),
                        v.size());
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
