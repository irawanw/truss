// DSpark draft: one block forward (always block_size rows: the draft attention is noncausal over the whole block),
// the main output head, then the markov head's per-row bias and sample. Returns the first n drafts.
#include "model/deepseek41/dspark.h"

#include "engine/stage.h"
#include "kernels/dense/q8_gemm.cuh"
#include "kernels/ds41/attn_ops.cuh"
#include "kernels/hc/hc_sinkhorn.cuh"
#include "kernels/sampling/argmax.cuh"
#include "kernels/sampling/spec.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace truss::deepseek41 {

namespace {

__global__ void add_bias_kernel(float * x, const float * bias, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += bias[i];
}

}  // namespace

int DSpark::draft(engine::Stage & st, int slot, int32_t next, int pos, int n, int32_t * out, const Sampling & smp)
{
    engine::require(pos >= 1, "DSpark draft needs a committed position");
    engine::require(n >= 1, "draft count");
    if (!sl_[slot].seeded) return 0;
    n = std::min<int>(n, d_.block_size);
    const int T = d_.block_size, d = d_.d_model, NM = 24;
    (void) d;
    engine::Gpu & g = st.gpu();
    st.use_small();
    g.sc->reset();
    Scratch & sc = *g.sc;
    const auto * ds_p = static_cast<const sampling::SampleParams *>(smp.params);
    // [next, noise, noise, noise, noise] at positions pos .. pos+block-1
    ids_host_[0] = next;
    for (int i = 1; i < T; ++i) ids_host_[i] = d_.noise_token;
    TRUSS_CUDA(cudaMemcpyAsync(ids_, ids_host_, sizeof(int32_t) * T, cudaMemcpyHostToDevice, g.s));
    dense::q8_rows(lin_.q8(w_.token_embd), ids_, T, emb_, g.s);
    hc::sk_expand(emb_, T, sk_dims(), res_, coef_pre_, g.s);
    float * y = sc.alloc((size_t) T * d), * x = sc.alloc((size_t) T * d), * o = sc.alloc((size_t) T * d);
    half * x16 = sc.alloc<half>((size_t) T * d), * y16 = sc.alloc<half>((size_t) T * d);
    float * pre = coef_pre_;   // the carry: layer 0's attention input is the identity pre-mix; then each layer's F
    for (int i = 0; i < d_.n_layer; ++i) {
        hc_attn_[i]->collapse(g, res_, pre, NM, T, y, nullptr);
        attn_norm_[i]->forward(g, y, T, x, x16);
        attn_[i]->forward(g, slot, x16, pos, T, o);
        hc_attn_[i]->mix(g, res_, T, coef_a_);
        hc_attn_[i]->post(g, res_, o, coef_a_, NM, T);
        hc_ffn_[i]->collapse(g, res_, coef_a_, NM, T, y, nullptr);
        ffn_norm_[i]->forward(g, y, T, x, x16);
        moe_[i]->forward(g, x, x16, T, o, false, nullptr, 0);
        hc_ffn_[i]->mix(g, res_, T, coef_f_);
        hc_ffn_[i]->post(g, res_, o, coef_f_, NM, T);
        pre = coef_f_;
    }
    // head: last layer's collapse coef, dspark.2.norm, the main output head
    hc_ffn_.back()->collapse(g, res_, coef_f_, NM, T, y, y16);
    last_norm_->forward(g, y, T, nullptr, y16);
    lin_.lin(w_.output, y16, T, logits_);
    // markov: bias from the previous draft token's embedding, then sample this row
    int made = 0;
    for (int i = 0; i < n; ++i) {
        dense::q8_rows(lin_.q8(dw_.layers.back().markov_embed), ids_ + i, 1, me_, g.s);
        ds41::to_half(me_, d_.markov_rank, m16_, g.s);
        lin_.lin(dw_.layers.back().markov_head, m16_, 1, mbias_);
        add_bias_kernel<<<(d_.n_vocab + 255) / 256, 256, 0, g.s>>>(logits_ + (size_t) i * d_.n_vocab, mbias_, d_.n_vocab);
        sampling::argmax_prob(logits_ + (size_t) i * d_.n_vocab, d_.n_vocab, nullptr, draft_ids_ + i + 1,
                              draft_prob_ + i, g.s);
        if (ds_p)
            sampling::draft_sample(logits_ + (size_t) i * d_.n_vocab, d_.n_vocab, nullptr, d_.n_vocab, *ds_p,
                                   smp.counter + i, smp.q + (size_t) i * d_.n_vocab, draft_ids_ + i + 1, g.s);
        TRUSS_CUDA(cudaMemcpyAsync(ids_ + i + 1, draft_ids_ + i + 1, sizeof(int32_t), cudaMemcpyDeviceToDevice, g.s));
        if (min_p_ > 0.f) {   // one host round trip per draft; an unsure guess ends the block
            float p = 0.f;
            TRUSS_CUDA(cudaMemcpyAsync(&p, draft_prob_ + i, sizeof(float), cudaMemcpyDeviceToHost, g.s));
            g.sync();
            if (p < min_p_) break;
        }
        ++made;
    }
    if (made) TRUSS_CUDA(cudaMemcpyAsync(draft_host_, draft_ids_ + 1, sizeof(int32_t) * made, cudaMemcpyDeviceToHost, g.s));
    g.sync();
    std::copy(draft_host_, draft_host_ + made, out);
    {   // TRUSS_DSPARK_DEBUG=<file>: one record per draft call (parity with the Python reference)
        static FILE * f = std::getenv("TRUSS_DSPARK_DEBUG") ? std::fopen(std::getenv("TRUSS_DSPARK_DEBUG"), "ab") : nullptr;
        if (f) {
            std::vector<float> lg((size_t) d_.n_vocab);
            TRUSS_CUDA(cudaMemcpyAsync(lg.data(), logits_, sizeof(float) * d_.n_vocab, cudaMemcpyDeviceToHost, g.s));
            g.sync();
            std::vector<int32_t> idx((size_t) d_.n_vocab);
            for (int i = 0; i < d_.n_vocab; ++i) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                              [&](int32_t a, int32_t b) { return lg[a] > lg[b]; });
            const int32_t hdr[4] = { pos, next, n, made };
            std::fwrite(hdr, 4, 4, f);
            std::fwrite(draft_host_, 4, made, f);
            for (int i = 0; i < 5; ++i) {
                const int32_t t = idx[i];
                const float v = lg[t];
                std::fwrite(&t, 4, 1, f), std::fwrite(&v, 4, 1, f);
            }
            std::fflush(f);
        }
    }
    return made;
}

}  // namespace truss::deepseek41
