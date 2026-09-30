#include "model/qwen4exp/prefill.h"

#include "core/cublas_check.h"
#include "core/cuda_check.h"
#include "core/device_tensors.h"
#include "core/scratch.h"
#include "kernels/dense/q8_gemm.cuh"
#include "kernels/dsa/dsa_prefill.cuh"
#include "kernels/dsa/dsa_prepare.cuh"
#include "kernels/ffn/ffn_ops.cuh"
#include "kernels/gdn/gdn_prefill.cuh"
#include "kernels/gdn/gdn_prepare.cuh"
#include "kernels/hc/hc_prefill.cuh"
#include "kernels/moe/moe_prefill.cuh"
#include "kernels/ple/ple_prefill.cuh"
#include "model/qwen4exp/ple.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace truss::qwen4exp {

using DsaShape = dsa::FlashNext;
using MoeShape = moe::FlashNext;

namespace {

void require(bool ok, const std::string & what)
{
    if (!ok) throw std::runtime_error("qwen4exp::Prefill: " + what);
}

template <class X> X * dmalloc(size_t n, bool zero = true)
{
    X * p;
    TRUSS_CUDA(cudaMalloc(&p, n * sizeof(X)));
    if (zero) TRUSS_CUDA(cudaMemset(p, 0, n * sizeof(X)));
    return p;
}

}  // namespace

struct Prefill::Impl {
    const Config & c;
    const Weights & w;
    const int n_ctx, max_chunk;
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    std::unique_ptr<DeviceTensors> dev;                  // every tensor except Q8_0 matrices and the PLE table
    std::unordered_map<T, dense::Q8Matrix> q8;           // Q8_0 matrices, repacked
    std::vector<void *> owned;                           // cudaFree at exit
    Scratch scratch;                                     // per-chunk temporaries, reset per layer
    half * w16 = nullptr;                                // q8_gemm_a16's dequantized weight
    void * moe_ws = nullptr;
    float * res = nullptr;                               // [max_chunk][hc][d]

    struct LayerState {
        half * k = nullptr, * v = nullptr, * idx_k = nullptr;   // DSA caches
        float * state = nullptr, * conv = nullptr;              // GDN recurrence, conv rows
        float * ple_hist = nullptr;
    };
    std::vector<LayerState> st;
    std::vector<int32_t> tail;                           // the last ple_ngram - 1 tokens seen (PLE window)

    Impl(const Config & cc, const Weights & ww, int nc, int mc)
        : c(cc), w(ww), n_ctx(nc), max_chunk(mc), scratch(scratch_bytes(cc, mc, nc))
    {
        check_shapes();
        TRUSS_CUDA(cudaStreamCreate(&s));
        TRUSS_CUBLAS(cublasCreate(&blas));
        upload();
        res = alloc<float>((size_t) max_chunk * c.hc_dim());
        moe_ws = alloc<unsigned char>(moe::prefill_workspace_bytes<MoeShape>(max_chunk));
        st.resize(c.n_layer);
        const int R = DsaShape::RATIO;
        for (int l = 0; l < c.n_layer; ++l) {
            LayerState & L = st[l];
            if (c.mixer[l] == Mixer::DSA) {
                L.k = alloc<half>((size_t) n_ctx * c.n_head_kv * c.head_dim);
                L.v = alloc<half>((size_t) n_ctx * c.n_head_kv * c.head_dim);
                L.idx_k = alloc<half>((size_t) (n_ctx / R) * c.idx_head_dim);
            } else {
                L.state = alloc<float>((size_t) c.ssm_v_heads * c.ssm_state * c.ssm_state);
                L.conv = alloc<float>((size_t) (c.ssm_conv - 1) * c.conv_dim());
            }
            if (c.is_ple(l)) L.ple_hist = alloc<float>((size_t) (c.ple_conv - 1) * c.ple_ngram * c.hc_dim());
        }
    }

    ~Impl()
    {
        for (void * p : owned) cudaFree(p);
        if (blas) cublasDestroy(blas);
        if (s) cudaStreamDestroy(s);
    }

    template <class X> X * alloc(size_t n)
    {
        X * p = dmalloc<X>(n);
        owned.push_back(p);
        return p;
    }

    // peak per-layer temporaries: the layer's own buffers plus the largest of hc mix / PLE / GDN / DSA / FFN
    static size_t scratch_bytes(const Config & c, int T, int n_ctx)
    {
        const size_t hcd = c.hc_dim(), dm = c.d_model, qd = (size_t) c.n_head * c.head_dim;
        const size_t base = dm * (4 + 4 + 2) + c.hc * 4;
        const size_t mix = hcd * (2 + 4) + c.hc * 4 + c.hc_rank * (4 + 2);
        const size_t ple = (size_t) c.ple_heads() * c.ple_head_dim * 2 + hcd * 8 + dm * 4 + c.hc * 4;
        const size_t gdn = (size_t) c.conv_dim() * 4 + (size_t) c.value_dim() * (4 + 4 + 4 + 2) + (size_t) c.key_dim() * 8 +
                           (size_t) c.ssm_v_heads * 16;
        const size_t dsa = qd * (8 + 2 + 4 + 2) + (size_t) c.n_head_kv * c.head_dim * 8 +
                           (size_t) c.idx_heads * c.idx_head_dim * 6 + c.idx_head_dim * 4 + DsaShape::ROPE_DIMS * 4 +
                           DsaShape::TOP_BLOCKS * 4 + 4;
        const size_t ffn = (size_t) c.n_expert * 4 + c.n_expert_used * 8 + dm * 8 + (size_t) c.d_ff_shexp * 10 + 4;
        const size_t per_token = base + std::max({ mix, ple, gdn, dsa, ffn });
        const size_t fixed = dsa::select_workspace_bytes<DsaShape>(T, n_ctx) + (64ull << 20);   // + alignment slack
        return (size_t) T * per_token + fixed;
    }

    void check_shapes() const
    {
        require(c.d_model == MoeShape::D_MODEL && c.d_ff_exp == MoeShape::D_FF && c.n_expert_used == MoeShape::TOPK,
                "routed-expert shape differs from moe::FlashNext");
        require(c.n_head == DsaShape::H && c.n_head_kv == DsaShape::HKV && c.head_dim == DsaShape::D &&
                    c.idx_heads == DsaShape::IH && c.idx_head_dim == DsaShape::ID && c.rope_dims == DsaShape::ROPE_DIMS &&
                    c.idx_top_k == DsaShape::TOP_BLOCKS * DsaShape::RATIO,
                "DSA shape differs from dsa::FlashNext");
        for (int l = 0; l < c.n_layer; ++l)
            require(c.mixer[l] != Mixer::DSA || c.compress_ratio[l] == DsaShape::RATIO, "DSA ratio differs");
        require(max_chunk % DsaShape::RATIO == 0, "max_chunk must be a multiple of the DSA ratio");
    }

    // Q8_0 matrices are repacked (one staging buffer); everything else is uploaded as stored
    void upload()
    {
        std::vector<const gguf::Tensor *> plain, q8_list;
        auto add = [&](T t) {
            if (!t) return;
            (t->type == gguf::Type::Q8_0 ? q8_list : plain).push_back(t);
        };
        auto add_hc = [&](const HyperConnection & h) { add(h.norm), add(h.down), add(h.up), add(h.inject); };
        auto add_table = [&](const formats::ExpertTable & e) { add(e.trellis), add(e.meta), add(e.suh), add(e.svh); };
        add(w.token_embd);
        add_hc(w.hc_head);
        for (const Layer & L : w.layers) {
            add_hc(L.hc_attn), add_hc(L.hc_ffn);
            for (T t : { L.gdn.qkv, L.gdn.gate, L.gdn.conv1d, L.gdn.dt_bias, L.gdn.a, L.gdn.beta, L.gdn.alpha,
                         L.gdn.norm, L.gdn.out })
                add(t);
            for (T t : { L.dsa.q, L.dsa.k, L.dsa.v, L.dsa.out, L.dsa.q_norm, L.dsa.k_norm, L.dsa.idx_q, L.dsa.idx_k,
                         L.dsa.idx_q_norm, L.dsa.idx_k_norm })
                add(t);
            for (T t : { L.ple.key, L.ple.value, L.ple.norm_key, L.ple.norm_query, L.ple.norm_conv, L.ple.conv1d })
                add(t);
            add(L.moe.router), add(L.moe.shexp_gate_inp), add(L.moe.shexp_gate), add(L.moe.shexp_up),
                add(L.moe.shexp_down);
            add_table(L.moe.gate), add_table(L.moe.up), add_table(L.moe.down);
        }
        dev = std::make_unique<DeviceTensors>(plain);

        size_t biggest = 0, total_q = 0, total_d = 0;
        for (T t : q8_list) {
            biggest = std::max<size_t>(biggest, t->bytes);
            total_q += (size_t) t->elements();
            total_d += (size_t) t->elements() / 32;
        }
        int8_t * qs = alloc<int8_t>(total_q);
        half * ds = alloc<half>(total_d);
        void * stage = dmalloc<unsigned char>(biggest, false);
        size_t max_w = 0;
        for (T t : q8_list) {
            if (q8.count(t)) continue;
            const int in = (int) t->shape[0], out = (int) t->elements() / in;
            TRUSS_CUDA(cudaMemcpy(stage, t->data, t->bytes, cudaMemcpyHostToDevice));
            dense::q8_repack(stage, in, out, qs, ds, s);
            q8[t] = { qs, ds, in, out };
            qs += (size_t) in * out;
            ds += (size_t) in * out / 32;
            if (t != w.token_embd) max_w = std::max(max_w, (size_t) in * out);
        }
        TRUSS_CUDA(cudaStreamSynchronize(s));
        cudaFree(stage);
        w16 = alloc<half>(max_w);
    }

    const DTensor & d(T t) const { return (*dev)(t); }
    const float * f32(T t) const { return d(t).as<float>(); }

    // y [rows][out] = W x, fp16 activations
    void lin(T t, const half * x, int rows, float * y) { dense::q8_gemm_a16(q8.at(t), x, rows, y, w16, blas, s); }

    // y [rows][out] = W x, fp32 weights and activations (router, shared-expert gate)
    void lin32(T t, const float * x, int rows, float * y)
    {
        const DTensor & W = d(t);
        const int in = (int) W.ne[0], out = (int) W.ne[1];
        const float one = 1.f, zero = 0.f;
        TRUSS_CUBLAS(cublasSetStream(blas, s));
        TRUSS_CUBLAS(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, &one, W.as<float>(), in, x, in, &zero,
                                 y, out));
    }

    // hyper-connection mix: mixed [T][d] (fp32 and fp16), inject [T][hc] when h.inject
    void hc_mix(const HyperConnection & h, int T, float * mixed, half * mixed16, float * inject)
    {
        const size_t m = scratch.mark();
        half * xn16 = scratch.alloc<half>((size_t) T * c.hc_dim());
        float * rstd = scratch.alloc((size_t) T * c.hc);
        float * lo = scratch.alloc((size_t) T * c.hc_rank);
        half * lo16 = scratch.alloc<half>((size_t) T * c.hc_rank);
        float * gate = scratch.alloc((size_t) T * c.hc_dim());
        hc::norm(res, f32(h.norm), T, c.hc, c.d_model, c.rms_eps, xn16, rstd, s);
        lin(h.down, xn16, T, lo);
        hc::silu(lo, T * c.hc_rank, 1.f / c.hc, lo16, s);
        lin(h.up, lo16, T, gate);
        hc::collapse(res, rstd, f32(h.norm), gate, T, c.hc, c.d_model, mixed, mixed16, s);
        if (h.inject) lin(h.inject, xn16, T, inject);
        scratch.release(m);
    }

    void ple(const Ple & p, LayerState & L, const int32_t * tokens, int T)
    {
        // n-gram window across chunks: hash [tail | chunk] and keep the chunk's rows
        const int H = c.ple_heads(), E = H * c.ple_head_dim, n_prev = (int) tail.size();
        std::vector<int32_t> seq(tail);
        seq.insert(seq.end(), tokens, tokens + T);
        std::vector<int32_t> rows(seq.size() * H);
        ple_rows(c, seq.data(), (int) seq.size(), rows.data());
        std::vector<float> emb((size_t) T * E);
        ple_gather(c, w, rows.data() + (size_t) n_prev * H, T, emb.data());
        std::vector<half> emb16(emb.size());
        for (size_t i = 0; i < emb.size(); ++i) emb16[i] = __float2half(emb[i]);

        const size_t m = scratch.mark();
        half * e16 = scratch.alloc<half>(emb16.size());
        float * key = scratch.alloc((size_t) T * c.hc_dim()), * value = scratch.alloc((size_t) T * c.d_model);
        float * gate = scratch.alloc((size_t) T * c.hc), * normed = scratch.alloc((size_t) T * c.hc_dim());
        TRUSS_CUDA(cudaMemcpyAsync(e16, emb16.data(), emb16.size() * 2, cudaMemcpyHostToDevice, s));
        lin(p.key, e16, T, key);
        lin(p.value, e16, T, value);
        ple::gate(key, res, f32(p.norm_key), f32(p.norm_query), T, c.hc, c.d_model, c.rms_eps, gate, s);
        ple::apply(value, gate, f32(p.norm_conv), d(p.conv1d).as<half>(), L.ple_hist, T, c.hc, c.d_model, c.ple_conv,
                   c.ple_ngram, c.rms_eps, normed, res, s);
        TRUSS_CUDA(cudaStreamSynchronize(s));   // emb16 is a host temporary
        scratch.release(m);
    }

    void gdn(const Gdn & g, LayerState & L, const half * in16, int T, float * out)
    {
        const size_t m = scratch.mark();
        const gdn::Dims dm{ c.ssm_groups, c.ssm_v_heads, c.ssm_state, c.ssm_conv };
        const int kd = c.key_dim(), vd = c.value_dim(), Hv = c.ssm_v_heads;
        float * qkv = scratch.alloc((size_t) T * c.conv_dim()), * z = scratch.alloc((size_t) T * vd);
        float * alpha = scratch.alloc((size_t) T * Hv), * beta_raw = scratch.alloc((size_t) T * Hv);
        float * q = scratch.alloc((size_t) T * kd), * k = scratch.alloc((size_t) T * kd), * v = scratch.alloc((size_t) T * vd);
        float * gt = scratch.alloc((size_t) T * Hv), * beta = scratch.alloc((size_t) T * Hv);
        float * core = scratch.alloc((size_t) T * vd);
        half * o16 = scratch.alloc<half>((size_t) T * vd);
        lin(g.qkv, in16, T, qkv);
        lin(g.gate, in16, T, z);
        lin(g.alpha, in16, T, alpha);
        lin(g.beta, in16, T, beta_raw);
        gdn::prepare(dm, qkv, f32(g.conv1d), L.conv, alpha, beta_raw, f32(g.dt_bias), f32(g.a), T, c.rms_eps, q, k, v,
                     gt, beta, s);
        gdn::delta_rule(q, k, v, gt, beta, L.state, core, T, c.ssm_groups, Hv, s);
        gdn::output_norm(dm, core, z, f32(g.norm), T, c.rms_eps, o16, s);
        lin(g.out, o16, T, out);
        scratch.release(m);
    }

    void dsa(const Dsa & a, LayerState & L, const half * in16, int pos0, int T, float * out)
    {
        const size_t m = scratch.mark();
        const int H = c.n_head, D = c.head_dim, Hkv = c.n_head_kv, IH = c.idx_heads, ID = c.idx_head_dim;
        float * qfull = scratch.alloc((size_t) T * H * 2 * D);
        float * k = scratch.alloc((size_t) T * Hkv * D), * v = scratch.alloc((size_t) T * Hkv * D);
        float * iq = scratch.alloc((size_t) T * IH * ID), * ik = scratch.alloc((size_t) T * ID);
        float2 * cs = scratch.alloc<float2>((size_t) T * DsaShape::ROPE_DIMS / 2);
        half * q16 = scratch.alloc<half>((size_t) T * H * D), * iq16 = scratch.alloc<half>((size_t) T * IH * ID);
        float * gate = scratch.alloc((size_t) T * H * D);
        int * blocks = scratch.alloc<int>((size_t) T * DsaShape::TOP_BLOCKS), * n_blocks = scratch.alloc<int>(T);
        half * att16 = scratch.alloc<half>((size_t) T * H * D);
        lin(a.q, in16, T, qfull);
        lin(a.k, in16, T, k);
        lin(a.v, in16, T, v);
        lin(a.idx_q, in16, T, iq);
        lin(a.idx_k, in16, T, ik);
        dsa::rope_table<DsaShape>(pos0, T, c.rope_base, cs, s);
        dsa::prepare_qkv<DsaShape>(qfull, k, v, f32(a.q_norm), f32(a.k_norm), cs, pos0, T, c.rms_eps, q16, gate, L.k,
                                   L.v, s);
        dsa::prepare_index<DsaShape>(iq, ik, f32(a.idx_q_norm), f32(a.idx_k_norm), cs, pos0, T, c.rms_eps, iq16,
                                     L.idx_k, s);
        const size_t ws_bytes = dsa::select_workspace_bytes<DsaShape>(T, pos0 + T);
        void * ws = scratch.alloc<unsigned char>(ws_bytes);
        dsa::select<DsaShape>(iq16, L.idx_k, pos0, T, blocks, n_blocks, ws, ws_bytes, s);
        dsa::attention<DsaShape>(q16, gate, L.k, L.v, blocks, n_blocks, pos0, T, att16, s);
        lin(a.out, att16, T, out);
        scratch.release(m);
    }

    void ffn(const Moe & mo, const float * in, const half * in16, int T, float * out)
    {
        const size_t m = scratch.mark();
        const int E = c.n_expert, K = c.n_expert_used, F = c.d_ff_shexp, dm = c.d_model;
        float * logits = scratch.alloc((size_t) T * E), * wts = scratch.alloc((size_t) T * K);
        int * ids = scratch.alloc<int>((size_t) T * K);
        float * routed = scratch.alloc((size_t) T * dm), * y = scratch.alloc((size_t) T * dm);
        float * g = scratch.alloc((size_t) T * F), * u = scratch.alloc((size_t) T * F), * sg = scratch.alloc(T);
        half * mid16 = scratch.alloc<half>((size_t) T * F);
        lin32(mo.router, in, T, logits);
        ffn::route(logits, T, E, K, ids, wts, s);
        const moe::Weights mw{ { view(mo.gate), view(mo.up), view(mo.down) }, E };
        moe::prefill<MoeShape>(mw, in, ids, wts, T, routed, moe_ws, max_chunk, s);
        lin(mo.shexp_gate, in16, T, g);
        lin(mo.shexp_up, in16, T, u);
        ffn::swiglu(g, u, T * F, mid16, s);
        lin(mo.shexp_down, mid16, T, y);
        lin32(mo.shexp_gate_inp, in, T, sg);
        ffn::shared_add(routed, y, sg, T, dm, out, s);
        scratch.release(m);
    }

    moe::ProjView view(const formats::ExpertTable & t) const
    {
        return { d(t.trellis).as<uint16_t>(), d(t.meta).as<int32_t>(), d(t.suh).as<half>(), d(t.svh).as<half>() };
    }

    void run(const int32_t * tokens, int pos0, int T, const LayerHook & hook)
    {
        require(T > 0 && T <= max_chunk, "chunk of " + std::to_string(T) + " tokens (max " + std::to_string(max_chunk) + ")");
        require(pos0 + T <= n_ctx, "sequence longer than n_ctx");
        require(pos0 % DsaShape::RATIO == 0, "a chunk after one whose length is not a multiple of the DSA ratio");
        scratch.reset();
        int * ids = scratch.alloc<int>(T);
        float * emb = scratch.alloc((size_t) T * c.d_model);
        TRUSS_CUDA(cudaMemcpyAsync(ids, tokens, (size_t) T * 4, cudaMemcpyHostToDevice, s));
        dense::q8_rows(q8.at(w.token_embd), ids, T, emb, s);
        hc::expand(emb, T, c.hc, c.d_model, res, s);
        const size_t base = scratch.mark();
        for (int l = 0; l < c.n_layer; ++l) {
            scratch.release(base);
            const Layer & L = w.layers[l];
            float * mixed = scratch.alloc((size_t) T * c.d_model), * out = scratch.alloc((size_t) T * c.d_model);
            float * inject = scratch.alloc((size_t) T * c.hc);
            half * mixed16 = scratch.alloc<half>((size_t) T * c.d_model);
            if (c.is_ple(l)) ple(L.ple, st[l], tokens, T);
            hc_mix(L.hc_attn, T, mixed, mixed16, inject);
            if (L.mixer == Mixer::GDN) gdn(L.gdn, st[l], mixed16, T, out);
            else dsa(L.dsa, st[l], mixed16, pos0, T, out);
            hc::combine(res, out, inject, T, c.hc, c.d_model, s);
            hc_mix(L.hc_ffn, T, mixed, mixed16, inject);
            ffn(L.moe, mixed, mixed16, T, out);
            hc::combine(res, out, inject, T, c.hc, c.d_model, s);
            if (hook) {
                TRUSS_CUDA(cudaStreamSynchronize(s));
                hook(l, res, T);
            }
        }
        if (!c.ple_layers.empty()) {   // keep the window's predecessors for the next chunk
            std::vector<int32_t> seq(tail);
            seq.insert(seq.end(), tokens, tokens + T);
            const size_t keep = std::min<size_t>(seq.size(), c.ple_ngram - 1);
            tail.assign(seq.end() - keep, seq.end());
        }
    }
};

Prefill::Prefill(const Config & c, const Weights & w, int n_ctx, int max_chunk)
    : m_(std::make_unique<Impl>(c, w, n_ctx, max_chunk))
{
}

Prefill::~Prefill() = default;

cudaStream_t Prefill::stream() const { return m_->s; }

void Prefill::run(const int32_t * tokens, int T, const LayerHook & hook)
{
    m_->run(tokens, pos_, T, hook);
    pos_ += T;
}

}  // namespace truss::qwen4exp
