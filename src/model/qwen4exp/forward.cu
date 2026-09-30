#include "model/qwen4exp/forward.h"

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
#include "kernels/moe/moe_window.cuh"
#include "kernels/mtp/mtp_ops.cuh"
#include "kernels/ple/ple_prefill.cuh"
#include "kernels/sampling/argmax.cuh"
#include "kernels/spec/rollback.cuh"
#include "model/qwen4exp/ple.h"
#include "runtime/expert_store.h"

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
    if (!ok) throw std::runtime_error("qwen4exp::Forward: " + what);
}

template <class X> X * dmalloc(size_t n, bool zero = true)
{
    X * p;
    TRUSS_CUDA(cudaMalloc(&p, n * sizeof(X)));
    if (zero) TRUSS_CUDA(cudaMemset(p, 0, n * sizeof(X)));
    return p;
}

}  // namespace

struct Forward::Impl {
    const Config & c;
    const Weights & w;
    const int n_ctx, max_chunk;
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    std::unique_ptr<DeviceTensors> dev;                  // every tensor except Q8_0 matrices and the PLE table
    std::unordered_map<T, dense::Q8Matrix> q8;           // Q8_0 matrices, repacked
    std::vector<void *> owned;                           // cudaFree at exit
    // Per-chunk buffers come in two sizes. Chunks of up to FETCH_ROWS rows (decode steps, verify windows) use small
    // resident ones; longer prompt chunks borrow the ExpertStore ring's spare region (stream mode), so the VRAM a
    // prompt needs for scratch, MoE workspace and residual holds cached experts during decode (TRACKER #57).
    struct Buffers {
        Scratch * scratch;                               // temporaries, reset per layer
        void * moe_ws;                                   // moe::prefill
        int moe_rows;                                    // rows moe_ws was sized for
        float * res;                                     // [rows][hc][d]
        float * mh, * mres;                              // MTP block: input hidden rows, its residual [rows][hc][d]
    };
    Scratch small_scratch;
    Buffers small{}, big{}, cur{};
    std::unique_ptr<Scratch> big_scratch;
    Scratch * sc = nullptr;                              // cur.scratch
    float * res = nullptr;                               // cur.res (persists after run() for head())
    half * w16 = nullptr;                                // q8_gemm_a16's dequantized weight
    size_t w16_elems = 0;
    void * window_ws = nullptr;                          // moe::window (decode steps)
    int * ids_host = nullptr;                            // pinned: a decode step's routing, for the expert fetch

    struct LayerState {
        half * k = nullptr, * v = nullptr, * idx_k = nullptr;   // DSA caches
        float * idx_partial = nullptr;                          // raw indexer keys of the open block
        float * state_buf[2] = {};                              // GDN recurrence (two with speculation: a verify
        int cur = 0;                                            // window writes the other one)
        float * conv = nullptr;                                 // GDN conv rows
        float * ple_hist = nullptr;
        // verify-window rollback (Options::spec_rows): snapshots before the window and the window's raw rows
        float * conv_snap = nullptr, * raw_qkv = nullptr, * raw_alpha = nullptr, * raw_beta = nullptr;
        float * partial_snap = nullptr, * raw_ik = nullptr;
        float * hist_snap = nullptr, * normed_rows = nullptr;
        float * state() const { return state_buf[cur]; }
    };
    std::vector<LayerState> st;
    LayerState mst;                                      // the MTP block's DSA caches
    const Mtp * mtp = nullptr;
    const int spec_rows;
    float * pending_h = nullptr;                         // MTP: hidden state [hc][d] of the last committed row
    bool has_pending = false;
    float * mtp_partial_snap = nullptr;                  // MTP indexer open block before a draft chain
    float * mtp_logits = nullptr;                        // [n_vocab]
    int * draft_ids = nullptr;                           // device [spec_rows]
    int32_t * draft_host = nullptr;                      // pinned
    struct Verify {                                      // the window verify() left uncommitted
        bool open = false;
        int pos0 = 0, T = 0;
        std::vector<int32_t> tokens;
    } vw;
    std::unique_ptr<runtime::ExpertStore> experts;
    int n_hot = 0;                                       // resident experts, all layers
    std::vector<int32_t> tail;                           // the last ple_ngram - 1 tokens seen (PLE window)

    const Activations act;
    const int hint_k;                                    // Options::hint_k
    int n_store = 0;                                     // ExpertStore layers: n_layer (+ 1 with the MTP block)

    Impl(const Config & cc, const Weights & ww, int nc, int mc, const Options & o)
        : c(cc), w(ww), n_ctx(nc), max_chunk(mc), small_scratch(scratch_bytes(cc, std::min(mc, FETCH_ROWS), nc)),
          spec_rows(o.spec_rows), act(o.act), hint_k(o.hint_k)
    {
        check_shapes();
        mtp = o.mtp;
        TRUSS_CUDA(cudaStreamCreate(&s));
        TRUSS_CUBLAS(cublasCreate(&blas));
        upload();
        const int small_rows = std::min(max_chunk, FETCH_ROWS);
        mtp = o.mtp;
        require(spec_rows >= 0 && spec_rows <= moe::MAX_ROWS, "spec_rows must be 0.." + std::to_string(moe::MAX_ROWS));
        small = { &small_scratch, alloc<unsigned char>(moe::prefill_workspace_bytes<MoeShape>(small_rows)), small_rows,
                  alloc<float>((size_t) small_rows * c.hc_dim()), nullptr, nullptr };
        if (mtp) {
            small.mh = alloc<float>((size_t) small_rows * c.hc_dim());
            small.mres = alloc<float>((size_t) small_rows * c.hc_dim());
            pending_h = alloc<float>(c.hc_dim());
            mtp_partial_snap = alloc<float>((size_t) (DsaShape::RATIO - 1) * c.idx_head_dim);
            mtp_logits = alloc<float>(c.n_vocab);
            draft_ids = alloc<int>(moe::MAX_ROWS + 1);
            TRUSS_CUDA(cudaMallocHost(&draft_host, sizeof(int32_t) * (moe::MAX_ROWS + 1)));
        }
        use(small);
        window_ws = alloc<unsigned char>(moe::workspace_bytes<MoeShape>());
        moe::workspace_init<MoeShape>(window_ws, s);
        TRUSS_CUDA(cudaMallocHost(&ids_host, sizeof(int) * 2 * FETCH_ROWS * MoeShape::TOPK));   // routing + prediction
        st.resize(c.n_layer);
        const int R = DsaShape::RATIO;
        const int W = spec_rows;
        auto dsa_state = [&](LayerState & L) {
            L.k = alloc<half>((size_t) n_ctx * c.n_head_kv * c.head_dim);
            L.v = alloc<half>((size_t) n_ctx * c.n_head_kv * c.head_dim);
            L.idx_k = alloc<half>((size_t) (n_ctx / R) * c.idx_head_dim);
            L.idx_partial = alloc<float>((size_t) (R - 1) * c.idx_head_dim);
        };
        for (int l = 0; l < c.n_layer; ++l) {
            LayerState & L = st[l];
            if (c.mixer[l] == Mixer::DSA) {
                dsa_state(L);
                if (W) L.partial_snap = alloc<float>((size_t) (R - 1) * c.idx_head_dim), L.raw_ik = alloc<float>((size_t) W * c.idx_head_dim);
            } else {
                const size_t sz = (size_t) c.ssm_v_heads * c.ssm_state * c.ssm_state;
                L.state_buf[0] = alloc<float>(sz);
                if (W) L.state_buf[1] = alloc<float>(sz);
                L.conv = alloc<float>((size_t) (c.ssm_conv - 1) * c.conv_dim());
                if (W) {
                    L.conv_snap = alloc<float>((size_t) (c.ssm_conv - 1) * c.conv_dim());
                    L.raw_qkv = alloc<float>((size_t) W * c.conv_dim());
                    L.raw_alpha = alloc<float>((size_t) W * c.ssm_v_heads);
                    L.raw_beta = alloc<float>((size_t) W * c.ssm_v_heads);
                }
            }
            if (c.is_ple(l)) {
                const size_t hist = (size_t) (c.ple_conv - 1) * c.ple_ngram * c.hc_dim();
                L.ple_hist = alloc<float>(hist);
                if (W) L.hist_snap = alloc<float>(hist), L.normed_rows = alloc<float>((size_t) W * c.hc_dim());
            }
        }
        if (mtp) dsa_state(mst);
        std::vector<runtime::ExpertLayer> tables;
        for (const Layer & L : w.layers) tables.push_back({ &L.moe.gate, &L.moe.up, &L.moe.down });
        if (mtp) tables.push_back({ &mtp->layer.moe.gate, &mtp->layer.moe.up, &mtp->layer.moe.down });
        n_store = (int) tables.size();
        std::vector<float> usage = o.expert_usage;
        if (mtp && usage.size() == (size_t) c.n_layer * c.n_expert) {
            // no MTP profile: the MTP block runs once per draft (3-4 times per verify pass), so its experts rank
            // above every layer's (all resident when they fit; TRACKER #58)
            const float top = *std::max_element(usage.begin(), usage.end());
            usage.insert(usage.end(), c.n_expert, 4.f * top + 1.f);
        }
        size_t expert_budget = o.expert_budget;
        if (!expert_budget) {
            size_t free_b, total_b;
            TRUSS_CUDA(cudaMemGetInfo(&free_b, &total_b));
            constexpr size_t MARGIN = 768ull << 20;   // cuBLAS workspaces, the CUDA context's growth
            require(free_b > MARGIN, "no device memory left for the routed experts");
            expert_budget = free_b - MARGIN;
        }
        runtime::ExpertStore::Sizes z;
        z.ring_bytes = o.ring_bytes;
        if (max_chunk > FETCH_ROWS) z.stream_extra = big_bytes();
        const runtime::ExpertStore::HotSet hot = runtime::ExpertStore::plan(tables, expert_budget, z, usage);
        for (const auto & h : hot) n_hot += (int) std::count(h.begin(), h.end(), 1);
        experts = std::make_unique<runtime::ExpertStore>(tables, hot, z);
        if (max_chunk > FETCH_ROWS) {   // the prompt path's buffers, carved from the ring's spare region
            auto * p = static_cast<unsigned char *>(experts->spare());
            const size_t sb = scratch_bytes(c, max_chunk, n_ctx), wb = align(moe::prefill_workspace_bytes<MoeShape>(max_chunk));
            big_scratch = std::make_unique<Scratch>(p, sb);
            const size_t rb = align(sizeof(float) * max_chunk * c.hc_dim());
            big = { big_scratch.get(), p + align(sb), max_chunk, reinterpret_cast<float *>(p + align(sb) + wb), nullptr,
                    nullptr };
            if (mtp) {
                big.mh = reinterpret_cast<float *>(p + align(sb) + wb + rb);
                big.mres = reinterpret_cast<float *>(p + align(sb) + wb + 2 * rb);
            }
        }
    }

    ~Impl()
    {
        for (void * p : owned) cudaFree(p);
        if (ids_host) cudaFreeHost(ids_host);
        if (draft_host) cudaFreeHost(draft_host);
        if (blas) cublasDestroy(blas);
        if (s) cudaStreamDestroy(s);
    }

    static size_t align(size_t x) { return (x + 255) / 256 * 256; }

    // bytes of the prompt path's buffers (scratch, moe::prefill workspace, residual) at max_chunk rows
    size_t big_bytes() const
    {
        return align(scratch_bytes(c, max_chunk, n_ctx)) + align(moe::prefill_workspace_bytes<MoeShape>(max_chunk)) +
               (mtp ? 3 : 1) * align(sizeof(float) * max_chunk * c.hc_dim());
    }

    void copy(float * dst, const float * src, size_t n)
    {
        TRUSS_CUDA(cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, s));
    }

    void use(const Buffers & b)
    {
        cur = b;
        sc = b.scratch;
        res = b.res;
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
        const size_t base = dm * 4 + 4 + dm * (4 + 4 + 2) + c.hc * 4;   // embedding + ids, then the layer's own
        const size_t mix = hcd * (2 + 4) + c.hc * 4 + c.hc_rank * (4 + 2);
        const size_t ple = (size_t) c.ple_heads() * c.ple_head_dim * 2 + hcd * 8 + dm * 4 + c.hc * 4;
        const size_t gdn = (size_t) c.conv_dim() * 4 + (size_t) c.value_dim() * (4 + 4 + 4 + 2) + (size_t) c.key_dim() * 8 +
                           (size_t) c.ssm_v_heads * 16;
        const size_t dsa = qd * (8 + 2 + 4 + 2) + (size_t) c.n_head_kv * c.head_dim * 8 +
                           (size_t) c.idx_heads * c.idx_head_dim * 6 + c.idx_head_dim * 4 + DsaShape::ROPE_DIMS * 4 +
                           DsaShape::TOP_BLOCKS * 4 + 4;
        const size_t ffn = (size_t) c.n_expert * 4 + c.n_expert_used * 8 + dm * 8 + (size_t) c.d_ff_shexp * 10 + 4;
        // MTP join: hn16, rstd, [e | hn] fp16 and its Q8_1 form (the widest GEMM input when present)
        const size_t mtp = hcd * 2 + c.hc * 4 + (size_t) c.hc * 2 * dm * (2 + 1) + (size_t) c.hc * 2 * dm / 16;
        const size_t q8_act = hcd + hcd / 16;   // Activations::Q8_1: the widest GEMM input, quantized
        const size_t per_token = base + std::max({ mix, ple, gdn, dsa, ffn, mtp }) + q8_act;
        const size_t fixed = dsa::select_workspace_bytes<DsaShape>(T, n_ctx) + dsa::attention_workspace_bytes<DsaShape>(T) +
                             (64ull << 20);   // + alignment slack
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
        add(w.token_embd), add(w.output);
        add_hc(w.hc_head);
        std::vector<const Layer *> layers;
        for (const Layer & L : w.layers) layers.push_back(&L);
        if (mtp) {
            layers.push_back(&mtp->layer);
            add(mtp->eh_proj), add(mtp->enorm), add(mtp->hnorm);
        }
        for (const Layer * Lp : layers) {
            const Layer & L = *Lp;
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
            if (t != w.token_embd && t != w.output) max_w = std::max(max_w, (size_t) in * out);
        }
        TRUSS_CUDA(cudaStreamSynchronize(s));
        cudaFree(stage);
        w16_elems = max_w;
        w16 = alloc<half>(max_w);
    }

    const DTensor & d(T t) const { return (*dev)(t); }
    const float * f32(T t) const { return d(t).as<float>(); }

    // y [rows][out] = W x, fp16 activations
    void lin(T t, const half * x, int rows, float * y)
    {
        const dense::Q8Matrix & W = q8.at(t);
        if (act == Activations::FP16) {
            dense::q8_gemm_a16(W, x, rows, y, w16, blas, s);
            return;
        }
        const size_t m = sc->mark();
        int8_t * xq = sc->alloc<int8_t>((size_t) rows * W.in);
        half * xd = sc->alloc<half>((size_t) rows * W.in / 32);
        dense::q8_quantize_act(x, rows, W.in, xq, xd, s);
        if (rows <= dense::GEMV_ROWS) dense::q8_gemv(W, xq, xd, rows, y, s);
        else dense::q8_gemm(W, xq, xd, rows, y, s);
        sc->release(m);
    }

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
    void hc_mix(const HyperConnection & h, const float * res, int T, float * mixed, half * mixed16, float * inject)
    {
        const size_t m = sc->mark();
        half * xn16 = sc->alloc<half>((size_t) T * c.hc_dim());
        float * rstd = sc->alloc((size_t) T * c.hc);
        float * lo = sc->alloc((size_t) T * c.hc_rank);
        half * lo16 = sc->alloc<half>((size_t) T * c.hc_rank);
        float * gate = sc->alloc((size_t) T * c.hc_dim());
        hc::norm(res, f32(h.norm), T, c.hc, c.d_model, c.rms_eps, xn16, rstd, s);
        lin(h.down, xn16, T, lo);
        hc::silu(lo, T * c.hc_rank, 1.f / c.hc, lo16, s);
        lin(h.up, lo16, T, gate);
        hc::collapse(res, rstd, f32(h.norm), gate, T, c.hc, c.d_model, mixed, mixed16, s);
        if (h.inject) lin(h.inject, xn16, T, inject);
        sc->release(m);
    }

    void ple(const Ple & p, LayerState & L, const int32_t * tokens, int T, bool tentative)
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

        const size_t m = sc->mark();
        half * e16 = sc->alloc<half>(emb16.size());
        float * key = sc->alloc((size_t) T * c.hc_dim()), * value = sc->alloc((size_t) T * c.d_model);
        float * gate = sc->alloc((size_t) T * c.hc), * normed = sc->alloc((size_t) T * c.hc_dim());
        TRUSS_CUDA(cudaMemcpyAsync(e16, emb16.data(), emb16.size() * 2, cudaMemcpyHostToDevice, s));
        lin(p.key, e16, T, key);
        lin(p.value, e16, T, value);
        ple::gate(key, res, f32(p.norm_key), f32(p.norm_query), T, c.hc, c.d_model, c.rms_eps, gate, s);
        const size_t hist = (size_t) (c.ple_conv - 1) * c.ple_ngram * c.hc_dim();
        if (tentative) copy(L.hist_snap, L.ple_hist, hist);
        ple::apply(value, gate, f32(p.norm_conv), d(p.conv1d).as<half>(), L.ple_hist, T, c.hc, c.d_model, c.ple_conv,
                   c.ple_ngram, c.rms_eps, normed, res, s);
        if (tentative) copy(L.normed_rows, normed, (size_t) T * c.hc_dim());
        TRUSS_CUDA(cudaStreamSynchronize(s));   // emb16 is a host temporary
        sc->release(m);
    }

    void gdn(const Gdn & g, LayerState & L, const half * in16, int T, float * out, bool tentative)
    {
        const size_t m = sc->mark();
        const gdn::Dims dm{ c.ssm_groups, c.ssm_v_heads, c.ssm_state, c.ssm_conv };
        const int kd = c.key_dim(), vd = c.value_dim(), Hv = c.ssm_v_heads;
        float * qkv = sc->alloc((size_t) T * c.conv_dim()), * z = sc->alloc((size_t) T * vd);
        float * alpha = sc->alloc((size_t) T * Hv), * beta_raw = sc->alloc((size_t) T * Hv);
        float * q = sc->alloc((size_t) T * kd), * k = sc->alloc((size_t) T * kd), * v = sc->alloc((size_t) T * vd);
        float * gt = sc->alloc((size_t) T * Hv), * beta = sc->alloc((size_t) T * Hv);
        float * core = sc->alloc((size_t) T * vd);
        half * o16 = sc->alloc<half>((size_t) T * vd);
        lin(g.qkv, in16, T, qkv);
        lin(g.gate, in16, T, z);
        lin(g.alpha, in16, T, alpha);
        lin(g.beta, in16, T, beta_raw);
        if (tentative) {   // accept() may redo the accepted rows from these
            copy(L.conv_snap, L.conv, (size_t) (c.ssm_conv - 1) * c.conv_dim());
            copy(L.raw_qkv, qkv, (size_t) T * c.conv_dim());
            copy(L.raw_alpha, alpha, (size_t) T * Hv);
            copy(L.raw_beta, beta_raw, (size_t) T * Hv);
        }
        gdn::prepare(dm, qkv, f32(g.conv1d), L.conv, alpha, beta_raw, f32(g.dt_bias), f32(g.a), T, c.rms_eps, q, k, v,
                     gt, beta, s);
        gdn::delta_rule(q, k, v, gt, beta, L.state(), tentative ? L.state_buf[1 - L.cur] : L.state(), core, T,
                        c.ssm_groups, Hv, s);
        gdn::output_norm(dm, core, z, f32(g.norm), T, c.rms_eps, o16, s);
        lin(g.out, o16, T, out);
        sc->release(m);
    }

    void dsa(const Dsa & a, LayerState & L, const half * in16, int pos0, int T, float * out, bool tentative)
    {
        const size_t m = sc->mark();
        const int H = c.n_head, D = c.head_dim, Hkv = c.n_head_kv, IH = c.idx_heads, ID = c.idx_head_dim;
        float * qfull = sc->alloc((size_t) T * H * 2 * D);
        float * k = sc->alloc((size_t) T * Hkv * D), * v = sc->alloc((size_t) T * Hkv * D);
        float * iq = sc->alloc((size_t) T * IH * ID), * ik = sc->alloc((size_t) T * ID);
        float2 * cs = sc->alloc<float2>((size_t) T * DsaShape::ROPE_DIMS / 2);
        half * q16 = sc->alloc<half>((size_t) T * H * D), * iq16 = sc->alloc<half>((size_t) T * IH * ID);
        float * gate = sc->alloc((size_t) T * H * D);
        int * blocks = sc->alloc<int>((size_t) T * DsaShape::TOP_BLOCKS), * n_blocks = sc->alloc<int>(T);
        half * att16 = sc->alloc<half>((size_t) T * H * D);
        lin(a.q, in16, T, qfull);
        lin(a.k, in16, T, k);
        lin(a.v, in16, T, v);
        lin(a.idx_q, in16, T, iq);
        lin(a.idx_k, in16, T, ik);
        if (tentative) {
            copy(L.partial_snap, L.idx_partial, (size_t) (DsaShape::RATIO - 1) * c.idx_head_dim);
            copy(L.raw_ik, ik, (size_t) T * c.idx_head_dim);
        }
        dsa::rope_table<DsaShape>(pos0, T, c.rope_base, cs, s);
        dsa::prepare_qkv<DsaShape>(qfull, k, v, f32(a.q_norm), f32(a.k_norm), cs, pos0, T, c.rms_eps, q16, gate, L.k,
                                   L.v, s);
        dsa::prepare_index<DsaShape>(iq, ik, f32(a.idx_q_norm), f32(a.idx_k_norm), cs, c.rope_base, pos0, T, c.rms_eps,
                                     L.idx_partial, iq16, L.idx_k, s);
        const size_t ws_bytes = dsa::select_workspace_bytes<DsaShape>(T, pos0 + T);
        void * ws = sc->alloc<unsigned char>(ws_bytes);
        dsa::select<DsaShape>(iq16, L.idx_k, pos0, T, blocks, n_blocks, ws, ws_bytes, s);
        const size_t aws_bytes = dsa::attention_workspace_bytes<DsaShape>(T);
        void * aws = aws_bytes ? sc->alloc<unsigned char>(aws_bytes) : nullptr;
        dsa::attention<DsaShape>(q16, gate, L.k, L.v, blocks, n_blocks, pos0, T, att16, aws, aws_bytes, s);
        lin(a.out, att16, T, out);
        sc->release(m);
    }

    // Chunks of up to FETCH_ROWS rows (decode steps, verify windows, short prompts) fetch only the cold experts
    // their routing uses; longer chunks use most experts, so whole layers stream ahead of the compute instead.
    // A 67-token prompt took 2.1 s streaming ~24 GB (TRACKER #55). The window kernel serves up to moe::MAX_ROWS rows.
    static constexpr int FETCH_ROWS = 32;
    static bool fetch_mode(int T) { return T <= FETCH_ROWS; }

    // stream: this chunk streams whole layers (prompt chunk, stream mode); else it fetches (decode, windows)
    void ffn(int l, const Moe & mo, const float * in, const half * in16, int T, float * out, bool stream)
    {
        const size_t m = sc->mark();
        const int E = c.n_expert, K = c.n_expert_used, F = c.d_ff_shexp, dm = c.d_model;
        float * logits = sc->alloc((size_t) T * E), * wts = sc->alloc((size_t) T * K);
        int * ids = sc->alloc<int>((size_t) T * K);
        float * routed = sc->alloc((size_t) T * dm), * y = sc->alloc((size_t) T * dm);
        float * g = sc->alloc((size_t) T * F), * u = sc->alloc((size_t) T * F), * sg = sc->alloc(T);
        half * mid16 = sc->alloc<half>((size_t) T * F);
        lin32(mo.router, in, T, logits);
        ffn::route(logits, T, E, K, ids, wts, s);
        if (route_counts) ffn::count(ids, T * K, route_counts + (size_t) l * E, s);
        if (!stream) {   // fetch the few cold experts this chunk routes to
            // pre-gating: the next layer's router on this layer's input predicts 72% of its experts (TRACKER #58);
            // their copies start behind this layer's, one sync for both id sets
            const bool hint = hint_k > 0 && l + 1 < c.n_layer;
            int * pred = nullptr;
            if (hint) {
                float * pl = sc->alloc((size_t) T * E), * pw = sc->alloc((size_t) T * K);
                pred = sc->alloc<int>((size_t) T * K);
                lin32(w.layers[l + 1].moe.router, in, T, pl);
                ffn::route(pl, T, E, K, pred, pw, s);
                TRUSS_CUDA(cudaMemcpyAsync(ids_host + T * K, pred, sizeof(int) * T * K, cudaMemcpyDeviceToHost, s));
            }
            TRUSS_CUDA(cudaMemcpyAsync(ids_host, ids, sizeof(int) * T * K, cudaMemcpyDeviceToHost, s));
            TRUSS_CUDA(cudaStreamSynchronize(s));
            experts->fetch(l, ids_host, T * K, s);
            if (hint) {   // each row's first hint_k guesses (ids are in descending probability)
                std::vector<int> h;
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < std::min(K, hint_k); ++i) h.push_back(ids_host[T * K + t * K + i]);
                experts->prefetch_hint(l, h.data(), (int) h.size());
            }
            experts->acquire(l, s);
            if (T <= moe::MAX_ROWS) moe::window<MoeShape>(experts->weights(l), in, ids, wts, T, routed, window_ws, s);
            else moe::prefill<MoeShape>(experts->weights(l), in, ids, wts, T, routed, cur.moe_ws, cur.moe_rows, s);
            experts->release(l, s);
        } else {                // whole layers stream ahead (prefetch(l + 2) while this layer computes)
            experts->acquire(l, s);
            moe::prefill<MoeShape>(experts->weights(l), in, ids, wts, T, routed, cur.moe_ws, cur.moe_rows, s);
            experts->release(l, s);
            if (l + 2 < n_store) experts->prefetch(l + 2);   // the MTP block streams after the last layer
        }
        lin(mo.shexp_gate, in16, T, g);
        lin(mo.shexp_up, in16, T, u);
        ffn::swiglu(g, u, T * F, mid16, s);
        lin(mo.shexp_down, mid16, T, y);
        lin32(mo.shexp_gate_inp, in, T, sg);
        ffn::shared_add(routed, y, sg, T, dm, out, s);
        sc->release(m);
    }

    int last_T = 0;                                      // rows of res from the last run()
    float * route_counts = nullptr;                      // [layer][expert] while profiling (Forward::profile_routes)

    void head(int first, int n, float * logits)
    {
        require(first >= 0 && n > 0 && first + n <= last_T, "head rows outside the last chunk");
        sc->reset();   // run()'s temporaries are dead; the residual is not in the scratch
        head_rows(res + (size_t) first * c.hc_dim(), n, logits);
    }

    // logits [n][vocab] of residual rows [n][hc][d]: the head hc mix, then the output projection
    void head_rows(const float * rows, int n, float * logits)
    {
        const size_t m = sc->mark();
        float * mixed = sc->alloc((size_t) n * c.d_model);
        half * mixed16 = sc->alloc<half>((size_t) n * c.d_model);
        hc_mix(w.hc_head, rows, n, mixed, mixed16, nullptr);
        if (act == Activations::Q8_1) {   // q8_gemv / q8_gemm write the whole vocab (decode: 0.9 ms vs 4 ms, #56)
            lin(w.output, mixed16, n, logits);
            sc->release(m);
            return;
        }
        // FP16: the output matrix in vocab tiles that fit the dequantized-weight scratch
        const dense::Q8Matrix & O = q8.at(w.output);
        const int tile = (int) std::min<size_t>(O.out, w16_elems / O.in) / 64 * 64;
        float * part = sc->alloc((size_t) n * tile);
        for (int v0 = 0; v0 < O.out; v0 += tile) {
            const int nv = std::min(tile, O.out - v0);
            const dense::Q8Matrix sub{ O.q + (size_t) v0 * O.in, O.d + (size_t) v0 * (O.in / 32), O.in, nv };
            dense::q8_gemm_a16(sub, mixed16, n, part, w16, blas, s);
            TRUSS_CUDA(cudaMemcpy2DAsync(logits + v0, (size_t) O.out * 4, part, (size_t) nv * 4, (size_t) nv * 4, n,
                                         cudaMemcpyDeviceToDevice, s));
        }
        sc->release(m);
    }

    void reset()
    {
        for (int l = 0; l < c.n_layer; ++l) {
            const LayerState & L = st[l];
            if (L.state()) TRUSS_CUDA(cudaMemsetAsync(L.state(), 0, sizeof(float) * c.ssm_v_heads * c.ssm_state * c.ssm_state, s));
            if (L.conv) TRUSS_CUDA(cudaMemsetAsync(L.conv, 0, sizeof(float) * (c.ssm_conv - 1) * c.conv_dim(), s));
            if (L.ple_hist)
                TRUSS_CUDA(cudaMemsetAsync(L.ple_hist, 0, sizeof(float) * (c.ple_conv - 1) * c.ple_ngram * c.hc_dim(), s));
        }
        tail.clear();
        last_T = 0;
        has_pending = false;
        vw.open = false;
    }

    // One chunk of the target model at positions pos0 .. pos0 + T - 1. tentative: a verify window (accept() commits).
    void run_chunk(const int32_t * tokens, int pos0, int T, const LayerHook & hook, bool tentative)
    {
        require(T > 0 && T <= max_chunk, "chunk of " + std::to_string(T) + " tokens (max " + std::to_string(max_chunk) + ")");
        require(pos0 + T <= n_ctx, "sequence longer than n_ctx");
        require(!vw.open, "run()/verify() before accept() of the previous verify()");
        const bool stream = !fetch_mode(T);
        if (!stream) {
            use(small);
        } else {   // the ring becomes the stream slots and this chunk's buffers
            experts->begin_stream(s);
            use(big);
        }
        sc->reset();
        int * ids = sc->alloc<int>(T);
        float * emb = sc->alloc((size_t) T * c.d_model);
        TRUSS_CUDA(cudaMemcpyAsync(ids, tokens, (size_t) T * 4, cudaMemcpyHostToDevice, s));
        dense::q8_rows(q8.at(w.token_embd), ids, T, emb, s);
        hc::expand(emb, T, c.hc, c.d_model, res, s);
        if (stream)
            for (int l = 0; l < std::min(2, n_store); ++l) experts->prefetch(l);
        const size_t base = sc->mark();
        for (int l = 0; l < c.n_layer; ++l) {
            sc->release(base);
            const Layer & L = w.layers[l];
            float * mixed = sc->alloc((size_t) T * c.d_model), * out = sc->alloc((size_t) T * c.d_model);
            float * inject = sc->alloc((size_t) T * c.hc);
            half * mixed16 = sc->alloc<half>((size_t) T * c.d_model);
            if (c.is_ple(l)) ple(L.ple, st[l], tokens, T, tentative);
            hc_mix(L.hc_attn, res, T, mixed, mixed16, inject);
            if (L.mixer == Mixer::GDN) gdn(L.gdn, st[l], mixed16, T, out, tentative);
            else dsa(L.dsa, st[l], mixed16, pos0, T, out, tentative);
            hc::combine(res, out, inject, T, c.hc, c.d_model, s);
            hc_mix(L.hc_ffn, res, T, mixed, mixed16, inject);
            ffn(l, L.moe, mixed, mixed16, T, out, stream);
            hc::combine(res, out, inject, T, c.hc, c.d_model, s);
            if (hook) {
                TRUSS_CUDA(cudaStreamSynchronize(s));
                hook(l, res, T);
            }
        }
        last_T = T;
        if (tentative) {
            vw = { true, pos0, T, std::vector<int32_t>(tokens, tokens + T) };
            return;
        }
        commit_tail(tokens, T);
        if (mtp) mtp_commit(tokens, pos0, T, stream);
    }

    void commit_tail(const int32_t * tokens, int T)   // the PLE window's predecessors for the next chunk
    {
        if (c.ple_layers.empty()) return;
        std::vector<int32_t> seq(tail);
        seq.insert(seq.end(), tokens, tokens + T);
        const size_t keep = std::min<size_t>(seq.size(), c.ple_ngram - 1);
        tail.assign(seq.end() - keep, seq.end());
    }

    // Commit the first n rows of the open verify window: every recurrent state as if only those rows had run.
    void accept(int n)
    {
        require(vw.open, "accept() without verify()");
        require(n >= 1 && n <= vw.T, "accept(" + std::to_string(n) + ") of a " + std::to_string(vw.T) + "-row window");
        vw.open = false;
        const bool all = n == vw.T;
        const size_t m = sc->mark();
        const gdn::Dims dm{ c.ssm_groups, c.ssm_v_heads, c.ssm_state, c.ssm_conv };
        for (int l = 0; l < c.n_layer; ++l) {
            LayerState & L = st[l];
            const Layer & W = w.layers[l];
            if (W.mixer == Mixer::GDN) {
                if (all) {   // the window's final state is the other buffer
                    L.cur ^= 1;
                } else {     // conv rows and state from before the window, then the accepted rows again
                    const int kd = c.key_dim(), vd = c.value_dim(), Hv = c.ssm_v_heads;
                    float * q = sc->alloc((size_t) n * kd), * k = sc->alloc((size_t) n * kd), * v = sc->alloc((size_t) n * vd);
                    float * gt = sc->alloc((size_t) n * Hv), * beta = sc->alloc((size_t) n * Hv), * core = sc->alloc((size_t) n * vd);
                    copy(L.conv, L.conv_snap, (size_t) (c.ssm_conv - 1) * c.conv_dim());
                    gdn::prepare(dm, L.raw_qkv, f32(W.gdn.conv1d), L.conv, L.raw_alpha, L.raw_beta, f32(W.gdn.dt_bias),
                                 f32(W.gdn.a), n, c.rms_eps, q, k, v, gt, beta, s);
                    gdn::delta_rule(q, k, v, gt, beta, L.state(), L.state(), core, n, c.ssm_groups, Hv, s);
                    sc->release(m);
                }
            } else if (!all) {   // the indexer's open block: the snapshot plus the accepted rows' raw keys
                copy(L.idx_partial, L.partial_snap, (size_t) (DsaShape::RATIO - 1) * c.idx_head_dim);
                dsa::carry_partial<DsaShape>(L.raw_ik, vw.pos0, n, L.idx_partial, s);
            }
            if (c.is_ple(l) && !all)
                spec::tail_rows(L.hist_snap, (c.ple_conv - 1) * c.ple_ngram, L.normed_rows, n, c.hc_dim(), L.ple_hist, s);
        }
        commit_tail(vw.tokens.data(), n);
        last_T = n;
        if (mtp) mtp_commit(vw.tokens.data(), vw.pos0, n, false);
    }

    // ---- MTP block (Options::mtp)

    // The MTP block on T rows: hidden states h [T][hc][d] (device), next tokens d_ids [T] (device) at MTP positions
    // pos0 .. pos0 + T - 1; its residual in cur.mres; logits of the last row into mtp_logits when asked.
    void mtp_rows(const float * h, const int * d_ids, int pos0, int T, bool stream, bool logits)
    {
        const Layer & L = mtp->layer;
        const int hc = c.hc, dm = c.d_model;
        float * mres = cur.mres;
        const size_t m = sc->mark();
        float * emb = sc->alloc((size_t) T * dm), * rstd = sc->alloc((size_t) T * hc);
        half * hn16 = sc->alloc<half>((size_t) T * c.hc_dim()), * cat16 = sc->alloc<half>((size_t) T * hc * 2 * dm);
        dense::q8_rows(q8.at(w.token_embd), d_ids, T, emb, s);
        hc::norm(h, f32(mtp->hnorm), T, hc, dm, c.rms_eps, hn16, rstd, s);
        mtp::join(emb, f32(mtp->enorm), hn16, T, hc, dm, c.rms_eps, cat16, s);
        lin(mtp->eh_proj, cat16, T * hc, mres);   // per stream: [2d] -> [d]
        const size_t base = sc->mark();
        float * mixed = sc->alloc((size_t) T * dm), * out = sc->alloc((size_t) T * dm);
        float * inject = sc->alloc((size_t) T * hc);
        half * mixed16 = sc->alloc<half>((size_t) T * dm);
        hc_mix(L.hc_attn, mres, T, mixed, mixed16, inject);
        dsa(L.dsa, mst, mixed16, pos0, T, out, false);
        hc::combine(mres, out, inject, T, hc, dm, s);
        hc_mix(L.hc_ffn, mres, T, mixed, mixed16, inject);
        ffn(c.n_layer, L.moe, mixed, mixed16, T, out, stream);
        hc::combine(mres, out, inject, T, hc, dm, s);
        (void) base;
        if (logits) head_rows(mres + (size_t) (T - 1) * c.hc_dim(), 1, mtp_logits);
        sc->release(m);
    }

    // After committing rows at positions pos0 .. pos0 + T - 1 (tokens, host): the MTP block at those positions, each
    // from the previous position's hidden state (pending for the first, the target's residual rows after), so its
    // cache matches a run over the true hidden states. Position 0 has no previous state and is skipped.
    void mtp_commit(const int32_t * tokens, int pos0, int T, bool stream)
    {
        const size_t row = c.hc_dim();
        const int skip = has_pending ? 0 : 1;   // only position 0 lacks a pending state
        const int n = T - skip;
        if (n > 0) {
            float * mh = cur.mh;
            if (!skip) copy(mh, pending_h, row);
            if (T > 1) copy(mh + (size_t) (1 - skip) * row, res, (size_t) (T - 1) * row);
            const size_t m = sc->mark();
            int * d_ids = sc->alloc<int>(n);
            TRUSS_CUDA(cudaMemcpyAsync(d_ids, tokens + skip, sizeof(int32_t) * n, cudaMemcpyHostToDevice, s));
            mtp_rows(mh, d_ids, pos0 + skip, n, stream, false);
            TRUSS_CUDA(cudaStreamSynchronize(s));   // tokens may be a host temporary
            sc->release(m);
        }
        copy(pending_h, res + (size_t) (T - 1) * row, row);
        has_pending = true;
    }

    // n greedy drafts after `next` (the token at position pos): a chain of single-row MTP steps, each from the
    // previous step's residual and argmax. The chain's cache rows and indexer block are tentative: the snapshot is
    // restored at the end, and mtp_commit rewrites those positions from the target's states.
    void draft(int32_t next, int pos, int n, int32_t * out)
    {
        require(mtp && has_pending, "draft() needs an MTP block and a committed position");
        require(n >= 1 && n <= moe::MAX_ROWS, "draft count");
        require(!vw.open, "draft() before accept()");
        use(small);
        sc->reset();
        const size_t row = c.hc_dim();
        copy(mtp_partial_snap, mst.idx_partial, (size_t) (DsaShape::RATIO - 1) * c.idx_head_dim);
        draft_host[0] = next;
        TRUSS_CUDA(cudaMemcpyAsync(draft_ids, draft_host, sizeof(int32_t), cudaMemcpyHostToDevice, s));
        copy(cur.mh, pending_h, row);
        for (int i = 0; i < n; ++i) {
            mtp_rows(cur.mh, draft_ids + i, pos + i, 1, false, true);
            sampling::argmax(mtp_logits, c.n_vocab, draft_ids + i + 1, s);
            if (i + 1 < n) copy(cur.mh, cur.mres, row);
        }
        copy(mst.idx_partial, mtp_partial_snap, (size_t) (DsaShape::RATIO - 1) * c.idx_head_dim);
        TRUSS_CUDA(cudaMemcpyAsync(draft_host, draft_ids + 1, sizeof(int32_t) * n, cudaMemcpyDeviceToHost, s));
        TRUSS_CUDA(cudaStreamSynchronize(s));
        std::copy(draft_host, draft_host + n, out);
    }
};

Forward::Forward(const Config & c, const Weights & w, int n_ctx, int max_chunk, const Options & o)
    : m_(std::make_unique<Impl>(c, w, n_ctx, max_chunk, o))
{
}

Forward::~Forward() = default;

cudaStream_t Forward::stream() const { return m_->s; }

int Forward::hot_experts() const { return m_->n_hot; }

size_t Forward::cold_bytes() const { return m_->experts->cold_bytes(); }

const runtime::ExpertStore & Forward::experts() const { return *m_->experts; }

void Forward::profile_routes(bool on)
{
    Impl & m = *m_;
    if (on && !m.route_counts) m.route_counts = m.alloc<float>((size_t) m.n_store * m.c.n_expert);   // zeroed
    if (!on && m.route_counts) {
        TRUSS_CUDA(cudaStreamSynchronize(m.s));
        TRUSS_CUDA(cudaFree(m.route_counts));
        m.owned.erase(std::find(m.owned.begin(), m.owned.end(), m.route_counts));
        m.route_counts = nullptr;
    }
}

std::vector<float> Forward::route_counts() const
{
    const Impl & m = *m_;
    std::vector<float> out((size_t) m.n_store * m.c.n_expert, 0.f);
    if (m.route_counts) {
        TRUSS_CUDA(cudaStreamSynchronize(m.s));
        TRUSS_CUDA(cudaMemcpy(out.data(), m.route_counts, out.size() * 4, cudaMemcpyDeviceToHost));
    }
    return out;
}

void Forward::head(int first, int n, float * logits) { m_->head(first, n, logits); }

void Forward::reset()
{
    m_->reset();
    pos_ = 0;
}

void Forward::run(const int32_t * tokens, int T, const LayerHook & hook)
{
    m_->run_chunk(tokens, pos_, T, hook, false);
    pos_ += T;
}

void Forward::verify(const int32_t * tokens, int T)
{
    if (!m_->spec_rows || T > m_->spec_rows)
        throw std::runtime_error("qwen4exp::Forward: verify() of " + std::to_string(T) + " rows (spec_rows " +
                                 std::to_string(m_->spec_rows) + ")");
    m_->run_chunk(tokens, pos_, T, nullptr, true);
}

void Forward::accept(int n)
{
    m_->accept(n);
    pos_ += n;
}

void Forward::draft(int32_t next, int n, int32_t * out) { m_->draft(next, pos_, n, out); }

}  // namespace truss::qwen4exp
