// DSA on a long prompt (> idx_top_k + ratio - 1 tokens), where the indexer drops blocks: the reference vs llama-paw's
// dump (tools/tk-parity/llama_dump with bases hc_mixed, indexer_q, indexer_k, indexer_top_k, attn_pregate).
//   A. rule: ref::qsa_select on llama's own indexer queries and keys. TRUSS takes the top idx_top_k / ratio whole
//      blocks plus the tail; llama-paw the top idx_top_k + ratio - 1 cells, i.e. the same set plus
//      (ratio - 1 - tail) cells of the next block when the tail is short (TRACKER #47). Gate: every TRUSS cell is
//      in llama's set, and llama's extras number exactly ratio - 1 - tail.
//   B. attention arithmetic: ref::masked_attention with the reference's q, k, v on llama's exact cell set vs
//      llama's output (gate 1e-3).
//   C. end to end (reported): the reference's own indexer. Its scores differ from llama's by ~1e-4 (Q8_1
//      projections), which can swap near-tied blocks at the budget boundary.
// usage: qwen4exp_dsa_long <slice.gguf> <dump dir>
#include "core/cuda_check.h"

#include <cuda_fp16.h>
#include "core/device_tensors.h"
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

__global__ void round16_kernel(float * v, size_t n)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = __half2float(__float2half(v[i]));
}

std::vector<float> host(const float * d, size_t n)
{
    std::vector<float> h(n);
    TRUSS_CUDA(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
    return h;
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
            uint8_t * d_rule = scratch.alloc<uint8_t>((size_t) T * T);
            float * iq = upload(dump.get("indexer_q-" + L).f, (size_t) T * Hi * Di);
            float * ik = upload(dump.get("indexer_k-" + L).f, (size_t) nb * Di);   // complete blocks come first
            ref::qsa_select(iq, ik, T, Hi, Di, r, c.idx_top_k / r, d_rule, nullptr);
            // B: attention on llama's exact sets
            uint8_t * d_theirs = scratch.alloc<uint8_t>((size_t) T * T);
            TRUSS_CUDA(cudaMemcpy(d_theirs, theirs.data(), theirs.size(), cudaMemcpyHostToDevice));
            float * pre_b = scratch.alloc((size_t) T * H * D), * pre_16 = scratch.alloc((size_t) T * H * D);
            ref::masked_attention(tr.q, tr.k, tr.v, d_theirs, pre_b, T, H, Hkv, D, 1.f / std::sqrt((float) D), nullptr);
            const std::pair<float *, size_t> qkv[3] = { { tr.q, (size_t) T * H * D }, { tr.k, (size_t) T * Hkv * D },
                                                        { tr.v, (size_t) T * Hkv * D } };
            for (const auto & [p, n] : qkv) round16_kernel<<<(unsigned) ((n + 255) / 256), 256>>>(p, n);
            ref::masked_attention(tr.q, tr.k, tr.v, d_theirs, pre_16, T, H, Hkv, D, 1.f / std::sqrt((float) D), nullptr);
            TRUSS_CUDA(cudaDeviceSynchronize());

            std::vector<uint8_t> rule((size_t) T * T), own((size_t) T * T);
            TRUSS_CUDA(cudaMemcpy(rule.data(), d_rule, rule.size(), cudaMemcpyDeviceToHost));
            TRUSS_CUDA(cudaMemcpy(own.data(), tr.sel, own.size(), cudaMemcpyDeviceToHost));
            const test::DumpTensor want = dump.get("attn_pregate-" + L);
            const std::vector<float> got_b = host(pre_b, (size_t) T * H * D), got_c = host(pre, (size_t) T * H * D);
            const std::vector<float> got_16 = host(pre_16, (size_t) T * H * D);
            std::vector<double> eb, e16;
            long rule_outside = 0, rule_bad_extra = 0, own_outside = 0;
            double worst_b = 0, worst_c = 0;
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
                const size_t o = (size_t) t * H * D;
                eb.push_back(rel(&got_b[o], &want.f[o], (size_t) H * D));
                e16.push_back(rel(&got_16[o], &want.f[o], (size_t) H * D));
                worst_b = std::max(worst_b, eb.back());
                worst_c = std::max(worst_c, rel(&got_c[o], &want.f[o], (size_t) H * D));
            }
            std::sort(eb.begin(), eb.end());
            std::sort(e16.begin(), e16.end());
            std::printf("  B detail: fp32 q/k/v median %.2e p99 %.2e max %.2e | fp16-rounded q/k/v median %.2e p99 %.2e "
                        "max %.2e\n", eb[T / 2], eb[T * 99 / 100], eb.back(), e16[T / 2], e16[T * 99 / 100], e16.back());
            const bool ok = rule_outside == 0 && rule_bad_extra == 0 && worst_b <= 1e-3;
            fails += !ok;
            std::printf("layer %d, T=%d: A rule on llama's indexer: cells outside llama's set %ld, extra-count "
                        "mismatches %ld | B attention on llama's sets: worst query rel %.2e | C end to end: cells "
                        "outside %ld, worst query rel %.2e  %s\n",
                        l, T, rule_outside, rule_bad_extra, worst_b, own_outside, worst_c, ok ? "PASS" : "FAIL");
        }
        return fails ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
