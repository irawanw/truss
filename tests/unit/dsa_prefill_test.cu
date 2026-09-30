// dsa::select and dsa::attention vs ref::qsa_select and ref::masked_attention (FP32) on random fp16-exact inputs
// shaped like Flash-Next's DSA layers. A chunk is positions pos0 .. pos0 + T - 1 of an n_ctx-token sequence whose
// caches hold every position; the reference runs the whole sequence and the chunk's rows are compared.
//   select     the same blocks, except swaps of blocks whose fp64 scores tie within 1e-5 (the two paths sum in
//              different orders)
//   attention  on the fast op's own selection (so the check is independent of swaps), gated by sigmoid(gate):
//              rel <= 5e-4 overall, <= 2e-3 per query (fp16 P and fp16 output)
// Long contexts are timed only (the reference is O(n_ctx^2 D) in fp64).
// usage: dsa_prefill_test
#include "core/cuda_check.h"
#include "kernels/dsa/dsa_prefill.cuh"
#include "kernels/reference/ref.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace truss;
using Shape = dsa::FlashNext;

namespace {

constexpr int H = Shape::H, HKV = Shape::HKV, D = Shape::D, IH = Shape::IH, ID = Shape::ID, R = Shape::RATIO;
constexpr int TOP = Shape::TOP_BLOCKS;

template <class T> T * dalloc(size_t n)
{
    T * p;
    TRUSS_CUDA(cudaMalloc(&p, n * sizeof(T)));
    TRUSS_CUDA(cudaMemset(p, 0xff, n * sizeof(T)));   // NaN / -1 until written
    return p;
}

struct Buf {   // fp16-exact random values, as fp32 (reference) and fp16 (fast op)
    std::vector<float> h;
    float * f = nullptr;
    half * x = nullptr;
    Buf(size_t n, std::mt19937 & rng, float sd)
    {
        std::normal_distribution<float> nd(0.f, sd);
        h.resize(n);
        std::vector<half> hh(n);
        for (size_t i = 0; i < n; ++i) hh[i] = __float2half(nd(rng)), h[i] = __half2float(hh[i]);
        f = dalloc<float>(n);
        x = dalloc<half>(n);
        TRUSS_CUDA(cudaMemcpy(f, h.data(), n * 4, cudaMemcpyHostToDevice));
        TRUSS_CUDA(cudaMemcpy(x, hh.data(), n * 2, cudaMemcpyHostToDevice));
    }
    ~Buf() { cudaFree(f), cudaFree(x); }
};

double rel(const float * a, const float * b, size_t n)
{
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i])) return INFINITY;
        num += (a[i] - (double) b[i]) * (a[i] - (double) b[i]), den += (double) b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

int run(int n_ctx, int pos0, int T, bool check, std::mt19937 & rng)
{
    const int nb = n_ctx / R;
    Buf q((size_t) n_ctx * H * D, rng, 1.f), k((size_t) n_ctx * HKV * D, rng, 1.f), v(k.h.size(), rng, 1.f);
    Buf gate((size_t) n_ctx * H * D, rng, 1.f), iq((size_t) n_ctx * IH * ID, rng, 1.f), ik((size_t) nb * ID, rng, 1.f);
    int * blocks = dalloc<int>((size_t) T * TOP), * n_blocks = dalloc<int>(T);
    half * out = dalloc<half>((size_t) T * H * D);
    const size_t ws_bytes = dsa::select_workspace_bytes<Shape>(T, n_ctx);
    void * ws = dalloc<unsigned char>(ws_bytes);
    const size_t aws_bytes = dsa::attention_workspace_bytes<Shape>(T);   // split path (T <= SPLIT_ROWS)
    void * aws = aws_bytes ? dalloc<unsigned char>(aws_bytes) : nullptr;
    const half * cq = q.x + (size_t) pos0 * H * D, * ciq = iq.x + (size_t) pos0 * IH * ID;
    const float * cg = gate.f + (size_t) pos0 * H * D;

    cudaEvent_t e[3];
    for (auto & x : e) cudaEventCreate(&x);
    dsa::select<Shape>(ciq, ik.x, pos0, T, blocks, n_blocks, ws, ws_bytes, nullptr);   // warm-up
    dsa::attention<Shape>(cq, cg, k.x, v.x, blocks, n_blocks, pos0, T, out, aws, aws_bytes, nullptr);
    cudaEventRecord(e[0]);
    dsa::select<Shape>(ciq, ik.x, pos0, T, blocks, n_blocks, ws, ws_bytes, nullptr);
    cudaEventRecord(e[1]);
    dsa::attention<Shape>(cq, cg, k.x, v.x, blocks, n_blocks, pos0, T, out, aws, aws_bytes, nullptr);
    cudaEventRecord(e[2]);
    TRUSS_CUDA(cudaEventSynchronize(e[2]));
    float ms_sel, ms_att;
    cudaEventElapsedTime(&ms_sel, e[0], e[1]);
    cudaEventElapsedTime(&ms_att, e[1], e[2]);
    std::printf("n_ctx=%-6d pos0=%-6d T=%-5d select %.2f ms, attention %.2f ms = %.2f us/token/layer", n_ctx, pos0, T,
                ms_sel, ms_att, (ms_sel + ms_att) * 1e3 / T);

    bool ok = true;
    if (check) {
        std::vector<int> hb((size_t) T * TOP), hn(T);
        TRUSS_CUDA(cudaMemcpy(hb.data(), blocks, hb.size() * 4, cudaMemcpyDeviceToHost));
        TRUSS_CUDA(cudaMemcpy(hn.data(), n_blocks, hn.size() * 4, cudaMemcpyDeviceToHost));
        // our cell sets as a mask for the reference; rows before the chunk stay empty
        std::vector<uint8_t> mask((size_t) n_ctx * n_ctx, 0), want((size_t) n_ctx * n_ctx);
        uint8_t * d_sel = dalloc<uint8_t>(mask.size());
        ref::qsa_select(iq.f, ik.f, n_ctx, IH, ID, R, TOP, d_sel, nullptr);
        TRUSS_CUDA(cudaMemcpy(want.data(), d_sel, want.size(), cudaMemcpyDeviceToHost));
        auto score = [&](int p, int b) {
            double s = 0;
            for (int h = 0; h < IH; ++h) {
                double dot = 0;
                for (int i = 0; i < ID; ++i) dot += (double) iq.h[((size_t) p * IH + h) * ID + i] * ik.h[(size_t) b * ID + i];
                s += std::max(dot, 0.0);
            }
            return s;
        };
        long bad_count = 0, swaps = 0;
        double worst_gap = 0;
        for (int t = 0; t < T; ++t) {
            const int p = pos0 + t, seen = (p + 1) / R;
            bad_count += hn[t] != std::min(seen, TOP);
            uint8_t * row = &mask[(size_t) p * n_ctx];
            for (int i = 0; i < hn[t]; ++i) {
                const int b = hb[(size_t) t * TOP + i];
                if (b < 0 || b >= seen || (i && b <= hb[(size_t) t * TOP + i - 1])) { ++bad_count; continue; }
                for (int e2 = 0; e2 < R; ++e2) row[R * b + e2] = 1;
            }
            for (int j = R * seen; j <= p; ++j) row[j] = 1;
            double ours_min = INFINITY, theirs_max = -INFINITY;   // blocks only we keep vs only the reference keeps
            for (int b = 0; b < seen; ++b) {
                const bool o = row[R * b], w = want[(size_t) p * n_ctx + R * b];
                if (o && !w) ours_min = std::min(ours_min, score(p, b));
                if (!o && w) theirs_max = std::max(theirs_max, score(p, b));
            }
            if (ours_min == INFINITY && theirs_max == -INFINITY) continue;
            ++swaps;
            worst_gap = std::max(worst_gap, ours_min == INFINITY || theirs_max == -INFINITY
                                                ? INFINITY
                                                : std::abs(theirs_max - ours_min) / std::max(ours_min, theirs_max));
        }
        TRUSS_CUDA(cudaMemcpy(d_sel, mask.data(), mask.size(), cudaMemcpyHostToDevice));
        float * ref_out = dalloc<float>((size_t) n_ctx * H * D);
        ref::masked_attention(q.f, k.f, v.f, d_sel, ref_out, n_ctx, H, HKV, D, 1.f / std::sqrt((float) D), nullptr);
        std::vector<float> want_o((size_t) T * H * D), got((size_t) T * H * D);
        std::vector<half> got_h(got.size());
        TRUSS_CUDA(cudaMemcpy(want_o.data(), ref_out + (size_t) pos0 * H * D, want_o.size() * 4, cudaMemcpyDeviceToHost));
        TRUSS_CUDA(cudaMemcpy(got_h.data(), out, got_h.size() * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < got.size(); ++i) {
            got[i] = __half2float(got_h[i]);
            want_o[i] /= 1.f + std::exp(-gate.h[(size_t) pos0 * H * D + i]);
        }
        double worst = 0;
        for (int t = 0; t < T; ++t)
            worst = std::max(worst, rel(&got[(size_t) t * H * D], &want_o[(size_t) t * H * D], (size_t) H * D));
        const double all = rel(got.data(), want_o.data(), got.size());
        ok = bad_count == 0 && worst_gap <= 1e-5 && all <= 5e-4 && worst <= 2e-3;
        std::printf(" | select: bad %ld, swaps %ld (worst gap %.1e) | attention rel %.1e, worst query %.1e", bad_count,
                    swaps, worst_gap, all, worst);
        cudaFree(d_sel);
        cudaFree(ref_out);
    }
    std::printf("  %s\n", check ? (ok ? "PASS" : "FAIL") : "(timed)");
    for (auto & x : e) cudaEventDestroy(x);
    cudaFree(blocks), cudaFree(n_blocks), cudaFree(out), cudaFree(ws), cudaFree(aws);
    return ok ? 0 : 1;
}

}  // namespace

int main()
{
    try {
        std::mt19937 rng(11);
        int fails = 0;
        fails += run(12, 0, 12, true, rng);
        fails += run(3001, 0, 3001, true, rng);      // dense and sparse queries, a tail on most
        fails += run(4096, 3072, 1024, true, rng);   // a later chunk
        fails += run(3001, 2999, 1, true, rng);      // split path: a decode step past the top-k (sparse)
        fails += run(600, 587, 13, true, rng);       // split path: a window, dense (every block kept)
        fails += run(4100, 4068, 32, true, rng);     // split path: the largest split chunk
        fails += run(4096, 0, 4096, false, rng);
        fails += run(8192, 0, 8192, false, rng);
        fails += run(32768, 24576, 8192, false, rng);
        fails += run(32768, 32767, 1, false, rng);    // decode step at 32K
        fails += run(32768, 32764, 4, false, rng);    // 4-token verify window at 32K
        std::printf("%s\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
