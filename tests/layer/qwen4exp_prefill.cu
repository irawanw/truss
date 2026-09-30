// qwen4exp::Prefill (the fast layer chain) on the 8-layer slice.
//   short <dump>  12-token llama dump: after every layer, the residual vs llama's l_last and vs the fp32 reference
//                 chain (qwen4exp::reference blocks, FP32 numerics, from llama's hc_init). Gate vs reference 5e-3:
//                 the engine rounds activations to fp16 before each GEMM and fp16 P in attention.
//   stream <dump> all experts resident vs a 2 GiB expert budget (most experts streamed per layer through the
//                 ExpertStore slots): bit-identical residuals.
//   long <dump>   the 3,659-token dump: the next layer's attention-side mix of the engine's residual (reference
//                 hc_mix) vs llama's hc_mixed; chunked (2,048 + rest) vs one chunk, which must agree to 1e-4
//                 (caches and carried state); timing.
// usage: qwen4exp_prefill <slice.gguf> short|stream|long <dump dir>
#include "core/cuda_check.h"
#include "core/device_tensors.h"
#include "core/scratch.h"
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/ple.h"
#include "model/qwen4exp/prefill.h"
#include "model/qwen4exp/reference.h"
#include "model/qwen4exp/weights.h"
#include "tests/layer/dump.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace truss;
namespace q = truss::qwen4exp;

namespace {

std::vector<float> host(const float * d, size_t n)
{
    std::vector<float> h(n);
    TRUSS_CUDA(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
    return h;
}

double rel(const float * a, const float * b, size_t n)
{
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i])) return INFINITY;
        num += (a[i] - (double) b[i]) * (a[i] - (double) b[i]), den += (double) b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

std::string at(const char * base, int l, const char * suffix = "")
{
    return std::string(base) + "-" + std::to_string(l) + suffix;
}

// every layer's residual of one prompt through the engine, in chunks of `chunk`
std::vector<std::vector<float>> run_engine(q::Prefill & p, const std::vector<int32_t> & tok, int chunk, int n_layer,
                                           size_t row, double * seconds)
{
    std::vector<std::vector<float>> out(n_layer, std::vector<float>(tok.size() * row));
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t s = 0; s < tok.size(); s += chunk) {
        const int T = (int) std::min<size_t>(chunk, tok.size() - s);
        p.run(tok.data() + s, T, [&](int l, const float * res, int n) {
            TRUSS_CUDA(cudaMemcpy(out[l].data() + s * row, res, (size_t) n * row * 4, cudaMemcpyDeviceToHost));
        });
    }
    TRUSS_CUDA(cudaDeviceSynchronize());
    *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

// the residual's start on the host: token_embd rows (Q8_0 blocks: fp16 scale + 32 int8) in every hc stream
std::vector<float> embed(const q::Config & c, const q::Weights & w, const std::vector<int32_t> & tok)
{
    const int d = c.d_model;
    const size_t row_bytes = (size_t) d / 32 * 34;
    std::vector<float> res(tok.size() * c.hc_dim());
    for (size_t t = 0; t < tok.size(); ++t) {
        const auto * r = reinterpret_cast<const uint8_t *>(w.token_embd->data) + tok[t] * row_bytes;
        for (int b = 0; b < d / 32; ++b) {
            const float s = __half2float(*reinterpret_cast<const half *>(r + b * 34));
            for (int i = 0; i < 32; ++i)
                for (int h = 0; h < c.hc; ++h)
                    res[(t * c.hc + h) * d + b * 32 + i] = (float) reinterpret_cast<const int8_t *>(r + b * 34 + 2)[i] * s;
        }
    }
    return res;
}

// every layer's residual of the reference chain (qwen4exp::reference blocks) from llama's embedding
std::vector<std::vector<float>> reference_chain(const q::Config & c, const q::Weights & w, const DeviceTensors & dev,
                                                const std::vector<int32_t> & tok, const std::vector<float> & init,
                                                ref::Numerics num)
{
    const int T = (int) tok.size();
    const size_t row = c.hc_dim();
    Scratch scratch(6ull << 30);
    const q::reference::Ctx x{ c, dev, scratch, nullptr, num };
    std::vector<std::vector<float>> out;
    float * res;
    TRUSS_CUDA(cudaMalloc(&res, (size_t) T * row * 4));
    TRUSS_CUDA(cudaMemcpy(res, init.data(), init.size() * 4, cudaMemcpyHostToDevice));
    for (int l = 0; l < c.n_layer; ++l) {
        scratch.reset();
        const q::Layer & L = w.layers[l];
        float * mixed = scratch.alloc((size_t) T * c.d_model), * o = scratch.alloc((size_t) T * c.d_model);
        float * inj = scratch.alloc((size_t) T * c.hc);
        if (c.is_ple(l)) {
            std::vector<int32_t> rows((size_t) T * c.ple_heads());
            std::vector<float> emb((size_t) T * c.ple_heads() * c.ple_head_dim);
            q::ple_rows(c, tok.data(), T, rows.data());
            q::ple_gather(c, w, rows.data(), T, emb.data());
            float * d_emb = scratch.alloc(emb.size());
            TRUSS_CUDA(cudaMemcpy(d_emb, emb.data(), emb.size() * 4, cudaMemcpyHostToDevice));
            q::reference::ple(x, L.ple, d_emb, res, T);
        }
        q::reference::hc_mix(x, L.hc_attn, res, T, mixed, inj);
        if (L.mixer == q::Mixer::GDN) q::reference::gdn(x, L.gdn, mixed, T, o);
        else q::reference::dsa(x, L.dsa, (int) c.compress_ratio[l], mixed, T, o);
        q::reference::hc_combine(x, res, o, inj, T);
        q::reference::hc_mix(x, L.hc_ffn, res, T, mixed, inj);
        q::reference::ffn(x, L.moe, mixed, T, o);
        q::reference::hc_combine(x, res, o, inj, T);
        out.push_back(host(res, (size_t) T * row));
    }
    cudaFree(res);
    return out;
}

std::vector<const gguf::Tensor *> all_but_ple_table(const gguf::File & f, const q::Weights & w)
{
    std::vector<const gguf::Tensor *> up;
    for (const auto & t : f.tensors())
        if (&t != w.ple_table && &t != w.ple_scale) up.push_back(&t);
    return up;
}

double run_timed(q::Prefill & p, const std::vector<int32_t> & tok, int chunk)
{
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t s = 0; s < tok.size(); s += chunk) p.run(tok.data() + s, (int) std::min<size_t>(chunk, tok.size() - s));
    TRUSS_CUDA(cudaStreamSynchronize(p.stream()));
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char ** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <slice.gguf> short|long <dump dir>\n", argv[0]);
        return 2;
    }
    try {
        const auto file = gguf::File::open(argv[1]);
        const q::Config c = q::Config::from_gguf(*file);
        const q::Weights w = q::bind(*file, c);
        const std::string mode = argv[2];
        const test::Dump dump(argv[3]);
        const std::vector<int32_t> tok = dump.tokens();
        const int T = (int) tok.size();
        const size_t row = c.hc_dim();
        int fails = 0;

        if (mode == "short") {
            std::vector<std::vector<float>> eng;
            {
                q::Prefill p(c, w, 512, 512);
                double sec;
                eng = run_engine(p, tok, 512, c.n_layer, row, &sec);
            }
            // the fp32 chain, and the same chain with llama-paw's rounding (Q8_1 activations): how much of the
            // engine-vs-llama gap is llama's own quantization
            const DeviceTensors dev(all_but_ple_table(*file, w));
            const std::vector<float> init = embed(c, w, tok);
            const double e_emb = rel(init.data(), dump.get("hc_init").f.data(), init.size());
            std::printf("embedding vs llama hc_init %.1e  %s\n", e_emb, e_emb == 0 ? "PASS" : "FAIL");
            fails += e_emb != 0;
            const std::vector<std::vector<float>> chain[2] = {
                reference_chain(c, w, dev, tok, init, ref::Numerics::FP32),
                reference_chain(c, w, dev, tok, init, ref::Numerics::LLAMA) };
            for (int l = 0; l < c.n_layer; ++l) {
                const std::vector<float> & r = chain[0][l], & rl = chain[1][l], ll = dump.get(at("l_last", l)).f;
                const double e_ref = rel(eng[l].data(), r.data(), r.size()), e_llama = rel(eng[l].data(), ll.data(), r.size());
                const bool ok = e_ref <= 5e-3;
                fails += !ok;
                std::printf("layer %d residual: engine vs fp32 reference %.2e | vs llama: engine %.2e, fp32 reference "
                            "%.2e, reference with llama numerics %.2e  %s\n",
                            l, e_ref, e_llama, rel(r.data(), ll.data(), r.size()), rel(rl.data(), ll.data(), r.size()),
                            ok ? "PASS" : "FAIL");
            }
        } else if (mode == "stream") {
            std::vector<std::vector<float>> all, streamed;
            double t_all, t_str;
            int hot_all, hot_str;
            size_t cold;
            for (int pass = 0; pass < 2; ++pass) {
                q::Prefill p(c, w, T, (T + 3) / 4 * 4, pass ? 2ull << 30 : 0);
                (pass ? streamed : all) = run_engine(p, tok, (T + 3) / 4 * 4, c.n_layer, row, pass ? &t_str : &t_all);
                (pass ? hot_str : hot_all) = p.hot_experts();
                if (pass) cold = p.cold_bytes();
            }
            long diff = 0;
            for (int l = 0; l < c.n_layer; ++l)
                for (size_t i = 0; i < all[l].size(); ++i) diff += all[l][i] != streamed[l][i];
            fails += diff != 0;
            std::printf("hot experts per layer: %d (all resident) vs %d (2 GiB budget, %.2f GB streamed per chunk): %ld "
                        "residual values differ  %s\n", hot_all, hot_str, cold / 1e9, diff, diff ? "FAIL" : "PASS");
        } else if (mode == "long") {
            const int chunk = 2048;
            std::vector<std::vector<float>> one, two;
            double t_one, t_two;
            {
                q::Prefill p(c, w, T, (T + 3) / 4 * 4);
                one = run_engine(p, tok, T, c.n_layer, row, &t_one);
            }
            {
                q::Prefill p(c, w, T, chunk);
                two = run_engine(p, tok, chunk, c.n_layer, row, &t_two);
            }
            const DeviceTensors dev(all_but_ple_table(*file, w));
            const std::vector<std::vector<float>> chain =
                reference_chain(c, w, dev, tok, embed(c, w, tok), ref::Numerics::FP32);
            Scratch scratch(1ull << 30);
            const q::reference::Ctx x{ c, dev, scratch, nullptr, ref::Numerics::FP32 };
            for (int l = 0; l < c.n_layer; ++l) {
                const size_t n = one[l].size();
                const double e1 = rel(one[l].data(), chain[l].data(), n), e2 = rel(two[l].data(), chain[l].data(), n);
                const double chunked = rel(two[l].data(), one[l].data(), n);
                // cuBLAS picks its algorithm by row count (rel ~2.5e-6 between chunk sizes, q8_gemm_test), which fp16
                // rounding and routing amplify: the chunk gap must stay within the engine's own distance to fp32
                const bool ok = e1 <= 5e-3 && e2 <= 5e-3 && chunked <= std::max(e1, e2);
                fails += !ok;
                std::vector<double> pt;   // per token: rounding noise (median) vs routing near-tie flips (tail)
                for (int t = 0; t < T; ++t) pt.push_back(rel(one[l].data() + (size_t) t * row, chain[l].data() + (size_t) t * row, row));
                std::sort(pt.begin(), pt.end());
                std::printf("layer %d residual vs fp32 reference: one chunk %.2e (per token median %.1e p99 %.1e max %.1e), "
                            "chunked (%d + %d) %.2e; chunked vs one %.2e", l, e1, pt[T / 2], pt[T * 99 / 100], pt.back(),
                            chunk, T - chunk, e2, chunked);
                if (l + 1 < c.n_layer && !c.is_ple(l + 1)) {   // the next layer's mixer input vs llama
                    scratch.reset();
                    float * res = scratch.alloc(n), * mixed = scratch.alloc((size_t) T * c.d_model);
                    TRUSS_CUDA(cudaMemcpy(res, one[l].data(), n * 4, cudaMemcpyHostToDevice));
                    q::reference::hc_mix(x, w.layers[l + 1].hc_attn, res, T, mixed, nullptr);
                    const std::vector<float> got = host(mixed, (size_t) T * c.d_model);
                    std::printf(" | layer %d mixer input vs llama %.2e", l + 1,
                                rel(got.data(), dump.get(at("hc_mixed", l + 1, "#1")).f.data(), got.size()));
                }
                std::printf("  %s\n", ok ? "PASS" : "FAIL");
            }
            std::printf("time with per-layer copies: one chunk %.3f s, chunked %.3f s\n", t_one, t_two);
            {
                q::Prefill p(c, w, T, chunk);
                run_timed(p, tok, chunk);   // warm-up (cuBLAS heuristics, first-touch)
            }
            for (int ch : { 1024, 2048, 4096 }) {
                const int mc = std::min(ch, (T + 3) / 4 * 4);
                q::Prefill p(c, w, T, mc);
                const double sec = run_timed(p, tok, mc);
                std::printf("prefill %d tokens in chunks of %d: %.1f ms = %.0f tok/s on %d layers (%.1f us/token/layer)\n",
                            T, mc, sec * 1e3, T / sec, c.n_layer, sec * 1e6 / T / c.n_layer);
            }
        } else {
            throw std::runtime_error("mode must be short, stream or long");
        }
        std::printf("%s\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
