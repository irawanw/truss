// DSA on a long prompt (> idx_top_k + ratio - 1 tokens), where the indexer drops blocks: the reference vs llama-paw's
// dump (tools/tk-parity/llama_dump with bases hc_mixed, Qcur, Kcur, Vcur, indexer_q, indexer_k, indexer_top_k,
// attn_pregate).
//   A. rule: ref::qsa_select on llama's own indexer queries and keys. TRUSS takes the top idx_top_k / ratio whole
//      blocks plus the tail; llama-paw the top idx_top_k + ratio - 1 cells, i.e. the same set plus
//      (ratio - 1 - tail) cells of the next block when the tail is short (TRACKER #47). llama ranks at ~fp16
//      precision, so near-tied blocks can swap (TRACKER #48). Gate: llama's extras number exactly ratio - 1 - tail,
//      and every other difference is a swap whose relative block-score gap (fp64 from llama's tensors) is <= 1e-3.
//   B. attention: the reference's q, k, v vs llama's (gate 1e-3), then ref::masked_attention on llama's exact cell
//      sets vs llama's output. llama's flash-attention kernel accumulates P.V in fp16, which puts it ~1e-3..2e-3
//      from fp32 at ~2K cells (TRACKER #48); the noise scale is measured as Numerics::LLAMA vs FP32 on the same
//      inputs. Gate: median and max per-query error <= 2x that noise.
//   C. end to end (reported): the reference's own indexer; its scores differ from llama's by ~1e-4 (Q8_1
//      projections), which can swap more near-tied blocks.
//   D. the fast op (dsa::select + dsa::attention, fp16 inputs) on the reference's own tensors: selection vs the
//      reference's, swaps only within a 2e-3 relative score gap (fp16 indexer inputs); gated attention vs
//      ref::masked_attention on the fast op's own selection, rel <= 1e-3, worst query <= 3e-3.
// usage: qwen4exp_dsa_long <slice.gguf> <dump dir>
#include "core/cuda_check.h"

#include <cuda_fp16.h>
#include "core/device_tensors.h"
#include "kernels/dsa/dsa_prefill.cuh"
#include "core/scratch.h"
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/reference.h"
#include "model/qwen4exp/weights.h"
#include "tests/layer/dump.h"

#include <algorithm>
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

__global__ void to_half_kernel(const float * f, half * h, size_t n)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = __float2half(f[i]);
}

double rel(const float * a, const float * b, size_t n)
{
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) num += (a[i] - (double) b[i]) * (a[i] - (double) b[i]), den += (double) b[i] * b[i];
    return std::sqrt(num / std::max(den, 1e-30));
}

}  // namespace

int main(int argc, char ** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <slice.gguf> <dump dir>\n", argv[0]);
        return 2;
    }
    try {
        const auto file = gguf::File::open(argv[1]);
        const q::Config c = q::Config::from_gguf(*file);
        const q::Weights w = q::bind(*file, c);
        std::vector<const gguf::Tensor *> up;
        for (int l = 0; l < c.n_layer; ++l) {
            const q::Dsa & a = w.layers[l].dsa;
            if (w.layers[l].mixer != q::Mixer::DSA) continue;
            for (auto * t : { a.q, a.k, a.v, a.out, a.q_norm, a.k_norm, a.idx_q, a.idx_k, a.idx_q_norm, a.idx_k_norm })
                up.push_back(t);
        }
        const DeviceTensors dev(up);
        Scratch scratch(3ull << 30);
        const test::Dump dump(argv[2]);
        const q::reference::Ctx x{ c, dev, scratch, nullptr, ref::Numerics::LLAMA };
        const int T = (int) dump.get("hc_mixed-0#1").ne[1];
        int fails = 0;
        for (int l = 0; l < c.n_layer; ++l) {
            if (w.layers[l].mixer != q::Mixer::DSA) continue;
            scratch.reset();
            const std::string L = std::to_string(l);
            const int H = c.n_head, Hkv = c.n_head_kv, D = c.head_dim, r = (int) c.compress_ratio[l];
            const int width = c.idx_top_k + r - 1, nb = T / r, Hi = c.idx_heads, Di = c.idx_head_dim;
            auto upload = [&](const std::vector<float> & v, size_t n) {
                float * d = scratch.alloc(n);
                TRUSS_CUDA(cudaMemcpy(d, v.data(), n * 4, cudaMemcpyHostToDevice));
                return d;
            };
            float * d_in = upload(dump.get("hc_mixed-" + L + "#1").f, (size_t) T * c.d_model);
            float * out = scratch.alloc((size_t) T * c.d_model), * pre = scratch.alloc((size_t) T * H * D);
            q::reference::DsaTrace tr;
            tr.pregate = pre;
            tr.q = scratch.alloc((size_t) T * H * D);
            tr.k = scratch.alloc((size_t) T * Hkv * D);
            tr.v = scratch.alloc((size_t) T * Hkv * D);
            tr.sel = scratch.alloc<uint8_t>((size_t) T * T);
            tr.gate = scratch.alloc((size_t) T * H * D);
            tr.gated = scratch.alloc((size_t) T * H * D);
            tr.idx_q = scratch.alloc((size_t) T * Hi * Di);
            tr.idx_k = scratch.alloc((size_t) nb * Di);
            q::reference::dsa(x, w.layers[l].dsa, r, d_in, T, out, &tr);

            // llama's cell sets (future picks are masked cells and do not count) as a mask
            const test::DumpTensor topk = dump.get("indexer_top_k-" + L);
            std::vector<uint8_t> theirs((size_t) T * T, 0);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < width; ++j) {
                    const int id = topk.i[(size_t) t * width + j];
                    if (id >= 0 && id <= t) theirs[(size_t) t * T + id] = 1;
                }
            // A: our rule on llama's indexer tensors
            const test::DumpTensor dq_idx = dump.get("indexer_q-" + L), dk_idx = dump.get("indexer_k-" + L);
            uint8_t * d_rule = scratch.alloc<uint8_t>((size_t) T * T);
            float * iq = upload(dq_idx.f, (size_t) T * Hi * Di);
            float * ik = upload(dk_idx.f, (size_t) nb * Di);   // complete blocks come first
            ref::qsa_select(iq, ik, T, Hi, Di, r, c.idx_top_k / r, d_rule, nullptr);
            std::vector<uint8_t> rule((size_t) T * T), own((size_t) T * T);
            TRUSS_CUDA(cudaMemcpy(rule.data(), d_rule, rule.size(), cudaMemcpyDeviceToHost));
            TRUSS_CUDA(cudaMemcpy(own.data(), tr.sel, own.size(), cudaMemcpyDeviceToHost));
            auto block_score = [&](int t, int b) {
                double s = 0;
                for (int h = 0; h < Hi; ++h) {
                    double dot = 0;
                    for (int i = 0; i < Di; ++i)
                        dot += (double) dq_idx.f[((size_t) t * Hi + h) * Di + i] * dk_idx.f[(size_t) b * Di + i];
                    s += std::max(dot, 0.0);
                }
                return s;
            };
            long rule_outside = 0, rule_bad_extra = 0, own_outside = 0, swaps = 0, far_swaps = 0;
            double worst_gap = 0;
            for (int t = 0; t < T; ++t) {
                int n_rule = 0, n_theirs = 0;
                for (int j = 0; j <= t; ++j) {
                    const size_t ij = (size_t) t * T + j;
                    n_rule += rule[ij];
                    n_theirs += theirs[ij];
                    rule_outside += rule[ij] && !theirs[ij];
                    own_outside += own[ij] && !theirs[ij];
                }
                const int expect_extra = t + 1 > width ? r - 1 - (t + 1) % r : 0;
                rule_bad_extra += n_theirs - n_rule != expect_extra;
                // blocks we keep that llama does not keep whole, vs blocks llama keeps (in part) that we drop
                double ours_min = INFINITY, theirs_max = -INFINITY;
                const int seen = (t + 1) / r;
                for (int b = 0; b < seen; ++b) {
                    const uint8_t * rr = &rule[(size_t) t * T + b * r], * th = &theirs[(size_t) t * T + b * r];
                    int in_rule = 0, in_theirs = 0;
                    for (int e = 0; e < r; ++e) in_rule += rr[e], in_theirs += th[e];
                    if (in_rule && in_theirs < r) ours_min = std::min(ours_min, block_score(t, b));
                    if (!in_rule && in_theirs) theirs_max = std::max(theirs_max, block_score(t, b));
                }
                if (ours_min == INFINITY) continue;   // no swap (llama's partial next block is the extra count)
                // llama kept a block we rank lower: the gap between our weakest dropped-by-llama block and its best
                const double gap = theirs_max == -INFINITY ? INFINITY : (ours_min - theirs_max) / ours_min;
                ++swaps;
                worst_gap = std::max(worst_gap, std::abs(gap));
                far_swaps += std::abs(gap) > 1e-3;
            }

            // B: inputs, then attention on llama's exact sets
            const float scale = 1.f / std::sqrt((float) D);
            const std::vector<float> hq = host(tr.q, (size_t) T * H * D), hk = host(tr.k, (size_t) T * Hkv * D),
                                     hv = host(tr.v, (size_t) T * Hkv * D);
            const double rq = rel(hq.data(), dump.get("Qcur-" + L).f.data(), hq.size()),
                         rk = rel(hk.data(), dump.get("Kcur-" + L + "#2").f.data(), hk.size()),
                         rv = rel(hv.data(), dump.get("Vcur-" + L).f.data(), hv.size());
            uint8_t * d_theirs = scratch.alloc<uint8_t>((size_t) T * T);
            TRUSS_CUDA(cudaMemcpy(d_theirs, theirs.data(), theirs.size(), cudaMemcpyHostToDevice));
            float * pre_b = scratch.alloc((size_t) T * H * D), * pre_n = scratch.alloc((size_t) T * H * D);
            ref::masked_attention(tr.q, tr.k, tr.v, d_theirs, pre_b, T, H, Hkv, D, scale, nullptr);
            ref::masked_attention(tr.q, tr.k, tr.v, d_theirs, pre_n, T, H, Hkv, D, scale, nullptr, ref::Numerics::LLAMA);
            const test::DumpTensor want = dump.get("attn_pregate-" + L);
            const std::vector<float> got_b = host(pre_b, (size_t) T * H * D), got_n = host(pre_n, (size_t) T * H * D),
                                     got_c = host(pre, (size_t) T * H * D);
            std::vector<double> eb, en;
            double worst_c = 0;
            for (int t = 0; t < T; ++t) {
                const size_t o = (size_t) t * H * D, n = (size_t) H * D;
                eb.push_back(rel(&got_b[o], &want.f[o], n));
                en.push_back(rel(&got_n[o], &got_b[o], n));
                worst_c = std::max(worst_c, rel(&got_c[o], &want.f[o], n));
            }
            std::sort(eb.begin(), eb.end());
            std::sort(en.begin(), en.end());
            const bool ok_a = rule_bad_extra == 0 && far_swaps == 0;
            const bool ok_b = std::max({ rq, rk, rv }) <= 1e-3 && eb[T / 2] <= 2 * en[T / 2] && eb.back() <= 2 * en.back();
            fails += !(ok_a && ok_b);
            std::printf("layer %d, T=%d\n"
                        "  A rule on llama's indexer: extra-count mismatches %ld, cells outside llama's set %ld from %ld "
                        "swaps, worst relative score gap %.1e (gate 1e-3)  %s\n"
                        "  B q/k/v vs llama %.1e/%.1e/%.1e; attention on llama's sets vs llama: median %.2e max %.2e | "
                        "fp16-accumulation noise (LLAMA vs FP32): median %.2e max %.2e  %s\n"
                        "  C end to end: cells outside llama's set %ld, worst query rel %.2e\n",
                        l, T, rule_bad_extra, rule_outside, swaps, worst_gap, ok_a ? "PASS" : "FAIL", rq, rk, rv,
                        eb[T / 2], eb.back(), en[T / 2], en.back(), ok_b ? "PASS" : "FAIL", own_outside, worst_c);

            // D: the fast op (dsa::select + dsa::attention) on the reference's own tensors, in fp16
            {
                using S = dsa::FlashNext;
                auto to16 = [&](const float * f, size_t n) {
                    half * h = scratch.alloc<half>(n);
                    to_half_kernel<<<(unsigned) ((n + 255) / 256), 256>>>(f, h, n);
                    return h;
                };
                half * q16 = to16(tr.q, (size_t) T * H * D), * k16 = to16(tr.k, (size_t) T * Hkv * D),
                     * v16 = to16(tr.v, (size_t) T * Hkv * D), * iq16 = to16(tr.idx_q, (size_t) T * Hi * Di),
                     * ik16 = to16(tr.idx_k, (size_t) nb * Di);
                int * blocks = scratch.alloc<int>((size_t) T * S::TOP_BLOCKS), * n_blocks = scratch.alloc<int>(T);
                half * fast = scratch.alloc<half>((size_t) T * H * D);
                const size_t ws_bytes = dsa::select_workspace_bytes<S>(T, T);
                void * ws = scratch.alloc<unsigned char>(ws_bytes);
                dsa::select<S>(iq16, ik16, 0, T, blocks, n_blocks, ws, ws_bytes, nullptr);
                dsa::attention<S>(q16, tr.gate, k16, v16, blocks, n_blocks, 0, T, fast, nullptr, 0, nullptr);
                std::vector<int> hb((size_t) T * S::TOP_BLOCKS), hn(T);
                TRUSS_CUDA(cudaMemcpy(hb.data(), blocks, hb.size() * 4, cudaMemcpyDeviceToHost));
                TRUSS_CUDA(cudaMemcpy(hn.data(), n_blocks, hn.size() * 4, cudaMemcpyDeviceToHost));
                const std::vector<float> hiq = host(tr.idx_q, (size_t) T * Hi * Di), hik = host(tr.idx_k, (size_t) nb * Di);
                auto score = [&](int t, int b) {
                    double s = 0;
                    for (int h = 0; h < Hi; ++h) {
                        double dot = 0;
                        for (int i = 0; i < Di; ++i) dot += (double) hiq[((size_t) t * Hi + h) * Di + i] * hik[(size_t) b * Di + i];
                        s += std::max(dot, 0.0);
                    }
                    return s;
                };
                std::vector<uint8_t> mine((size_t) T * T, 0);
                long d_swaps = 0;
                double d_gap = 0;
                for (int t = 0; t < T; ++t) {
                    const int seen = (t + 1) / r;
                    uint8_t * row = &mine[(size_t) t * T];
                    for (int i = 0; i < hn[t]; ++i)
                        for (int e = 0; e < r; ++e) row[r * hb[(size_t) t * S::TOP_BLOCKS + i] + e] = 1;
                    for (int j = r * seen; j <= t; ++j) row[j] = 1;
                    double ours_min = INFINITY, ref_max = -INFINITY;
                    for (int b = 0; b < seen; ++b) {
                        const bool o = row[r * b], w2 = own[(size_t) t * T + r * b];
                        if (o && !w2) ours_min = std::min(ours_min, score(t, b));
                        if (!o && w2) ref_max = std::max(ref_max, score(t, b));
                    }
                    if (ours_min == INFINITY && ref_max == -INFINITY) continue;
                    ++d_swaps;
                    d_gap = std::max(d_gap, ours_min == INFINITY || ref_max == -INFINITY
                                                ? INFINITY
                                                : std::abs(ref_max - ours_min) / std::max(ours_min, ref_max));
                }
                // reference attention (fp32 inputs) on the fast op's selection, gated
                uint8_t * d_mine = scratch.alloc<uint8_t>(mine.size());
                TRUSS_CUDA(cudaMemcpy(d_mine, mine.data(), mine.size(), cudaMemcpyHostToDevice));
                ref::masked_attention(tr.q, tr.k, tr.v, d_mine, pre_b, T, H, Hkv, D, scale, nullptr);
                const std::vector<float> ra = host(pre_b, (size_t) T * H * D), gt = host(tr.gate, (size_t) T * H * D);
                std::vector<half> fh((size_t) T * H * D);
                TRUSS_CUDA(cudaMemcpy(fh.data(), fast, fh.size() * 2, cudaMemcpyDeviceToHost));
                std::vector<float> fw(fh.size()), rw(fh.size());
                for (size_t i = 0; i < fh.size(); ++i)
                    fw[i] = __half2float(fh[i]), rw[i] = ra[i] / (1.f + std::exp(-gt[i]));
                double d_worst = 0;
                for (int t = 0; t < T; ++t)
                    d_worst = std::max(d_worst, rel(&fw[(size_t) t * H * D], &rw[(size_t) t * H * D], (size_t) H * D));
                const double d_all = rel(fw.data(), rw.data(), fw.size());
                const bool ok_d = d_gap <= 2e-3 && d_all <= 1e-3 && d_worst <= 3e-3;
                fails += !ok_d;
                std::printf("  D fast op vs reference: select swaps %ld (worst gap %.1e, gate 2e-3); gated attention "
                            "rel %.1e, worst query %.1e  %s\n", d_swaps, d_gap, d_all, d_worst, ok_d ? "PASS" : "FAIL");
            }
        }
        return fails ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
