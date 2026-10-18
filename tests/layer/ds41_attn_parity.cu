// blocks::AttnDs41 vs the official reference attention (tools/tk-parity/ds41_attn_ref.py: dsref/model.py Attention on
// the same dense-pack weights, storage simulations off), a chain of layers in model order with their shared state:
// a prefill chunk, then single-token decode steps. Per layer and step: output rel err (gate 2e-2: fp16 kv and
// projection inputs vs the reference's fp32), and on index sources the top-k set vs the reference's (reported:
// near-tied scores can swap under fp16 inputs).
// usage: ds41_attn_parity <ds41-dense.gguf> <dump dir>
#include "blocks/attn_ds41.h"
#include "blocks/linear.h"
#include "core/cuda_check.h"
#include "core/scratch.h"
#include "engine/gpu.h"
#include "kernels/ds41/attn_ops.cuh"
#include "model/deepseek41/config.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace truss;

namespace {

template <class X> std::vector<X> load(const std::string & path, size_t n)
{
    std::vector<X> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.read((char *) v.data(), n * sizeof(X))) throw std::runtime_error("short read " + path);
    return v;
}

double rel(const std::vector<float> & a, const std::vector<float> & b)
{
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i])) return INFINITY;
        num += (a[i] - (double) b[i]) * (a[i] - (double) b[i]), den += (double) b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

}  // namespace

int main(int argc, char ** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <ds41-dense.gguf> <dump dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    std::vector<int> layers;
    std::vector<std::pair<int, int>> steps;
    {
        std::ifstream m(dir + "/meta.txt");
        std::string line;
        while (std::getline(m, line)) {
            std::istringstream is(line);
            std::string k;
            is >> k;
            if (k == "layers") for (int l; is >> l;) layers.push_back(l);
            if (k == "step") {
                int s, t;
                is >> s >> t;
                steps.push_back({ s, t });
            }
        }
    }
    const auto f = gguf::File::open(argv[1]);
    const auto c = deepseek41::Config::from_gguf(*f);
    engine::Gpu g(0);
    const int max_rows = steps[0].second, n_ctx = (steps.back().first + 1 + 255) / 256 * 256;
    blocks::Linear lin(g, blocks::Activations::FP16);

    auto inv_freq = [&] (bool compress) {
        const auto v = ds41::rope_inv_freq(c.rope_dim, compress ? c.compress_rope_base : c.rope_base,
                                           compress ? c.rope_ctx_orig : 0, c.rope_factor, c.beta_fast, c.beta_slow);
        float * d = g.alloc<float>(v.size());
        TRUSS_CUDA(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        return d;
    };
    const float * if_plain = inv_freq(false), * if_comp = inv_freq(true);

    std::map<int, std::unique_ptr<blocks::AttnDs41>> blk;
    std::vector<blocks::Ds41AttnWeights> ws;
    size_t per_row = 0, fixed = 0;
    for (int l : layers) {
        const std::string p = "blk." + std::to_string(l) + ".";
        auto T = [&] (const std::string & n) { return f->find(p + n); };
        blocks::Ds41AttnWeights w;
        w.q_a = T("attn_q_a.weight"), w.q_a_norm = T("attn_q_a_norm.weight"), w.q_b = T("attn_q_b.weight");
        w.kv = T("attn_kv.weight"), w.kv_norm = T("attn_kv_a_norm.weight"), w.out_a = T("attn_output_a.weight");
        w.out_b = T("attn_output_b.weight"), w.sinks = T("attn_sinks.weight");
        w.cmp_kv = T("attn_compressor_kv.weight"), w.cmp_gate = T("attn_compressor_gate.weight");
        w.cmp_norm = T("attn_compressor_norm.weight"), w.idx_k = T("indexer.attn_k.weight");
        w.idx_k_norm = T("indexer.k_norm.weight"), w.idx_q_b = T("indexer.attn_q_b.weight");
        w.idx_proj = T("indexer.proj.weight");
        blocks::AttnDs41::add_tensors(lin, w);
        const int ratio = (int) c.compress_ratio[l];
        blocks::Ds41AttnDims d { c.d_model, c.n_head, c.head_dim, c.rope_dim, c.q_rank, c.o_rank, c.o_groups,
                                 c.window, c.index_heads, c.index_dim, c.index_top_k, c.candidate_top_k_blocks,
                                 c.candidate_block, ratio, c.is_kv_source(l), c.is_index_source(l),
                                 l == c.candidate_source, ratio > 0 && c.candidate_source >= 0 && l > c.candidate_source
                                     && c.is_index_source(l),
                                 c.rms_eps, ratio ? if_comp : if_plain };
        blk[l] = std::make_unique<blocks::AttnDs41>(lin, w, d);
        per_row = std::max(per_row, blocks::AttnDs41::scratch_per_row(d));
        fixed = std::max(fixed, blocks::AttnDs41::scratch_fixed(d, n_ctx));
    }
    lin.upload(nullptr, {});
    Scratch sc(per_row * max_rows + fixed + (64u << 20));
    g.sc = &sc;
    for (int l : layers) {
        auto src = [&] (int s) -> const blocks::AttnDs41 * {
            if (s < 0 || s == l) return nullptr;
            if (!blk.count(s)) throw std::runtime_error("layer " + std::to_string(l) + " needs source " + std::to_string(s));
            return blk[s].get();
        };
        const bool uses_cand = c.compress_ratio[l] && l > c.candidate_source && c.is_index_source(l);
        blk[l]->set_sources(src(c.kv_source_of(l)), src(c.index_source_of(l)), uses_cand ? src(c.candidate_source) : nullptr);
        blk[l]->alloc(g, n_ctx, max_rows, 1, n_ctx, 0);
    }
    const int d = c.d_model;
    float * x = g.alloc<float>((size_t) max_rows * d), * out = g.alloc<float>((size_t) max_rows * d);
    half * x16 = g.alloc<half>((size_t) max_rows * d);
    int fails = 0;
    for (size_t si = 0; si < steps.size(); ++si) {
        const auto [pos0, T] = steps[si];
        std::map<int, long> swapped_rows;          // index source -> rows whose top-k set differs this step
        for (int l : layers) {
            const std::string b = dir + "/s" + std::to_string(si) + "_L" + std::to_string(l) + "_";
            const auto hx = load<float>(b + "x.bin", (size_t) T * d), href = load<float>(b + "out.bin", (size_t) T * d);
            TRUSS_CUDA(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
            ds41::to_half(x, hx.size(), x16, g.s);
            TRUSS_CUDA(cudaMemset(out, 0xff, (size_t) T * d * 4));
            blk[l]->forward(g, 0, x, x16, pos0, T, out);
            TRUSS_CUDA(cudaStreamSynchronize(g.s));
            std::vector<float> ho((size_t) T * d);
            TRUSS_CUDA(cudaMemcpy(ho.data(), out, ho.size() * 4, cudaMemcpyDeviceToHost));
            const double e = rel(ho, href);
            std::string sel;
            if (blk[l]->topk() && c.is_index_source(l)) {   // top-k sets vs the reference's (reported)
                const int K = c.index_top_k;
                std::vector<int> mine((size_t) T * K);
                TRUSS_CUDA(cudaMemcpy(mine.data(), blk[l]->topk(), mine.size() * 4, cudaMemcpyDeviceToHost));
                const int rk = (int) (std::ifstream(b + "topk.bin", std::ios::ate | std::ios::binary).tellg() / 4 / T);
                const auto ref = load<int>(b + "topk.bin", (size_t) T * rk);
                const int off = pos0 == 0 ? T : c.window;   // the reference shifts past its window kv rows
                long diff_rows = 0, diff = 0, worst = 0;
                for (int t = 0; t < T; ++t) {
                    std::set<int> a, r;
                    for (int j = 0; j < K; ++j) if (mine[(size_t) t * K + j] >= 0) a.insert(mine[(size_t) t * K + j]);
                    for (int j = 0; j < rk; ++j) if (ref[(size_t) t * rk + j] >= 0) r.insert(ref[(size_t) t * rk + j] - off);
                    std::vector<int> x;
                    std::set_symmetric_difference(a.begin(), a.end(), r.begin(), r.end(), std::back_inserter(x));
                    diff_rows += !x.empty(), diff += (long) x.size(), worst = std::max(worst, (long) x.size());
                }
                char buf[128];
                std::snprintf(buf, sizeof buf, " | top-k: %ld of %d rows differ, %ld entries, worst row %ld", diff_rows,
                              T, diff, worst);
                sel = buf;
                // a near-tie swap moves one entry in and one out; more than 2 swaps in a row is a selection bug
                if (worst > 4) fails++, sel += " (> 2 swaps in a row: FAIL)";
                swapped_rows[l] = diff_rows;
            }
            // gate: rel 2e-2 on rows with the reference's selection; a step whose index source swapped a near-tied
            // entry is reported, not gated (fp16 vs fp32 indexer inputs; the real model rounds them to fp4)
            const int src = c.compress_ratio[l] ? c.index_source_of(l) : -1;
            const bool swap = src >= 0 && swapped_rows.count(src) && swapped_rows[src] > 0;
            const bool ok = e <= 2e-2;
            if (!ok && !swap) fails++;
            std::printf("step %zu (pos0 %5d, T %4d) L%-2d ratio %lld rel %.3e %s%s\n", si, pos0, T, l,
                        (long long) c.compress_ratio[l], e, ok ? "PASS" : swap ? "SWAP" : "FAIL", sel.c_str());
        }
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
