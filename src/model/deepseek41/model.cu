#include "model/deepseek41/model.h"

#include "engine/limits.h"
#include "engine/stage.h"
#include "kernels/dense/q8_gemm.cuh"
#include "kernels/ds41/attn_ops.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace truss::deepseek41 {

Family::Family(const gguf::File & f, const engine::DrafterFiles & d) : c_(Config::from_gguf(f)), w_(bind(f, c_))
{
    if (!d.path.empty()) {
        dspark_file_ = gguf::File::open(d.path);
        if (const char * e = std::getenv("TRUSS_DSPARK_EXPERT_OVERLAY")) dspark_file_->overlay_list(e);
        dspark_c_ = std::make_unique<DSparkConfig>(DSparkConfig::from_gguf(*dspark_file_, c_));
        dspark_w_ = std::make_unique<DSparkWeights>(bind_dspark(*dspark_file_, c_, *dspark_c_));
        drafter_min_p_ = d.min_p;
    }
    engine::require(c_.d_model == MoeShape::D_MODEL && c_.d_ff_exp == MoeShape::D_FF && c_.n_expert_used == MoeShape::TOPK
                        && c_.swiglu_limit == MoeShape::SWIGLU_LIMIT,
                    "deepseek41: routed-expert shape differs from moe::DsV41");
    for (int l = 0; l < c_.n_layer; ++l)
        if (c_.engram.index(l) >= 0)
            engine::require(w_.layers[l].engram.table != nullptr,
                            "deepseek41: no Engram table for layer " + std::to_string(l)
                                + " (add ds41-engram-L" + std::to_string(l) + ".gguf to TRUSS_EXPERT_OVERLAY)");
    if (const char * mm = std::getenv("TRUSS_MMPROJ"); mm && *mm) {   // image input: the text model's part of it
        const auto v = gguf::File::open(mm);
        engine::require(v->get_string("general.architecture") == "deepseek41-mmproj", std::string("TRUSS_MMPROJ ") + mm
                            + " is not a deepseek41-mmproj gguf (tools/tk-pack/ds41_mmproj.py)");
        image_token_ = (int32_t) v->get_int("deepseek41.vision.image_token_id");
        bias_vl_.resize((size_t) c_.n_layer * c_.n_expert);
        for (int l = 0; l < c_.n_layer; ++l) {
            const gguf::Tensor & t = v->at("blk." + std::to_string(l) + ".bias_vl");
            engine::require(t.type == gguf::Type::F32 && t.bytes % 4 == 0, "deepseek41: bias_vl is not f32");
            const float * b = (const float *) t.data;
            const int64_t n_src = (int64_t) (t.bytes / 4);
            for (int e = 0; e < c_.n_expert; ++e) {
                const int64_t src = c_.expert_keep[(size_t) l * c_.n_expert + e];
                engine::require(src >= 0 && src < n_src, "deepseek41: expert_keep id past bias_vl");
                bias_vl_[(size_t) l * c_.n_expert + e] = b[src];
            }
        }
        std::fprintf(stderr, "deepseek41: image input on (%s): image token %d, image router bias for %d layers\n", mm,
                     image_token_, c_.n_layer);
    }
}

std::vector<int> image_runs(const int32_t * tokens, int n, int32_t image)
{
    std::vector<int> v;
    if (image < 0) return v;
    for (int t = 0; t < n; ++t)
        if (tokens[t] == image && (t == 0 || tokens[t - 1] != image)) v.push_back(t);
        else if (tokens[t] != image && t > 0 && tokens[t - 1] == image) v.push_back(t);
    if (v.size() % 2) v.push_back(n);
    return v;
}

std::unique_ptr<engine::StageModel> Family::stage(int lb, int le, int n_ctx) const
{
    const Config & c = c_;
    for (int l = lb; l < le; ++l) {   // sources and consumers on one stage (blocks/attn_ds41.h)
        const int ks = c.kv_source_of(l), is = c.index_source_of(l);
        engine::require((ks < 0 || ks >= lb) && (is < 0 || is >= lb),
                        "deepseek41: stage [" + std::to_string(lb) + ", " + std::to_string(le)
                            + ") starts after an attention source its layers read; split at 2, 8, 14 or 20");
    }
    return std::make_unique<StageModel>(*this, lb, le, n_ctx);
}

StageModel::StageModel(const Family & fam, int lb, int le, int n_ctx)
    : fam_(fam), c_(fam.config()), w_(fam.weights()), lb_(lb), le_(le), n_ctx_(n_ctx)
{
}

StageModel::~StageModel() = default;

hc::SkDims StageModel::sk_dims() const
{
    hc::SkDims k { c_.hc, c_.d_model, c_.rms_eps, c_.hc_eps, c_.sinkhorn_iters };
    k.res_ld = c_.hc * c_.d_model + k.n_mix();
    k.coef_ld = k.n_mix();   // mix() writes a separate [T][24]; collapse / post take their coef_ld per call
    return k;
}

blocks::Ds41AttnDims StageModel::attn_dims(int l) const
{
    const int ratio = (int) c_.compress_ratio[l];
    const bool uses_cand = ratio > 0 && c_.candidate_source >= 0 && l > c_.candidate_source && c_.is_index_source(l);
    return { c_.d_model, c_.n_head, c_.head_dim, c_.rope_dim, c_.q_rank, c_.o_rank, c_.o_groups, c_.window,
             c_.index_heads, c_.index_dim, c_.index_top_k, c_.candidate_top_k_blocks, c_.candidate_block, ratio,
             ratio > 0 && c_.is_kv_source(l), ratio > 0 && c_.is_index_source(l), l == c_.candidate_source, uses_cand,
             c_.rms_eps, ratio ? inv_freq_comp_ : inv_freq_plain_ };
}

void StageModel::add_tensors(engine::Stage & st)
{
    blocks::Linear & lin = st.lin();
    if (lb_ == 0 || (le_ == c_.n_layer && fam_.has_drafter())) lin.add(w_.token_embd);
    if (le_ == c_.n_layer) lin.add(w_.final_norm), lin.add(w_.output);
    for (int l = lb_; l < le_; ++l) {
        const Layer & L = w_.layers[l];
        lin.add(L.attn_norm), lin.add(L.ffn_norm);
        blocks::HcSinkhorn::add_tensors(lin, L.hc_attn);
        blocks::HcSinkhorn::add_tensors(lin, L.hc_ffn);
        blocks::AttnDs41::add_tensors(lin, L.attn);
        blocks::Moe::add_tensors(lin, L.moe);
        if (L.engram.wkv) blocks::Engram::add_tensors(lin, L.engram);
    }
    if (le_ == c_.n_layer && fam_.has_drafter()) DSpark::add_tensors(lin, *fam_.dspark_weights());
}

std::vector<runtime::MoeLayer> StageModel::moe_layers() const
{
    std::vector<runtime::MoeLayer> out;
    for (int l = lb_; l < le_; ++l) {
        const blocks::MoeWeights & m = w_.layers[l].moe;
        out.push_back({ { &m.gate, &m.up, &m.down }, l, true, l + 1 < le_, true });   // CPU tier allowed (#191)
    }
    return out;
}

// The prompt path's peak temporaries at T rows. The attention's per-row scratch is the prompt path's (the decode
// path also stacks the grouped output LoRA's rows); its output buffer is a piece (engine::PREFILL_PIECE rows), not a
// row, so the chunk's rows do not scale the attention's memory (TRACKER #201: the prompt path's buffers are carved
// from the expert ring, and the ring's size decides how many experts stay resident).
size_t StageModel::scratch_bytes(int T, bool decode) const
{
    const size_t d = c_.d_model, rw = (size_t) c_.hc * d + 24;
    size_t attn = 0, fixed = 0, piece = 0;
    for (int l = lb_; l < le_; ++l) {
        const auto ad = attn_dims(l);
        attn = std::max(attn, decode ? blocks::AttnDs41::scratch_decode_per_row(ad)
                                      : blocks::AttnDs41::scratch_per_row(ad));
        fixed = std::max(fixed, blocks::AttnDs41::scratch_fixed(ad, n_ctx_));
        piece = std::max(piece, blocks::AttnDs41::scratch_piece(ad));
    }
    blocks::MoeDims md { c_.d_model, c_.n_expert, c_.n_expert_used, c_.d_ff_exp * c_.n_shexp };
    const blocks::EngramDims ed { c_.hc, c_.d_model, c_.engram.heads, c_.engram.head_dim, c_.engram.max_ngram, 0, 0,
                                  nullptr, nullptr, nullptr, nullptr, 0 };
    const size_t base = d * (4 + 4 + 2) + 2 * 24 * 4 + 64;   // collapse out fp32, sub-block out, normed fp16, coefs
    const size_t per_row = base + std::max({ attn, md.scratch_per_row(), blocks::Engram::scratch_per_row(ed),
                                             blocks::HcSinkhorn::scratch_per_row(sk_dims()) });
    const size_t rows = std::min((size_t) T, (size_t) engine::prefill_piece());
    (void) rw;
    return (size_t) T * per_row + fixed + piece * rows / engine::prefill_piece() + (64ull << 20);
}

void StageModel::build(engine::Stage & st)
{
    blocks::Linear & lin = st.lin();
    engine::Gpu & g = st.gpu();
    auto table = [&] (bool comp) {
        const auto v = ds41::rope_inv_freq(c_.rope_dim, comp ? c_.compress_rope_base : c_.rope_base,
                                           comp ? c_.rope_ctx_orig : 0, c_.rope_factor, c_.beta_fast, c_.beta_slow);
        float * p = g.alloc<float>(v.size());
        TRUSS_CUDA(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        return p;
    };
    inv_freq_plain_ = table(false), inv_freq_comp_ = table(true);
    const hc::SkDims k = sk_dims();
    for (int l = lb_; l < le_; ++l) {
        const Layer & L = w_.layers[l];
        attn_norm_.push_back(std::make_unique<blocks::RmsNorm>(lin, L.attn_norm, c_.d_model, c_.rms_eps));
        ffn_norm_.push_back(std::make_unique<blocks::RmsNorm>(lin, L.ffn_norm, c_.d_model, c_.rms_eps));
        hc_attn_.push_back(std::make_unique<blocks::HcSinkhorn>(lin, L.hc_attn, k));
        hc_ffn_.push_back(std::make_unique<blocks::HcSinkhorn>(lin, L.hc_ffn, k));
        attn_.push_back(std::make_unique<blocks::AttnDs41>(lin, L.attn, attn_dims(l)));
        const int ei = c_.engram.index(l);
        if (ei >= 0) {
            const auto & e = c_.engram;
            const int B = e.buckets();
            blocks::EngramDims ed { c_.hc, c_.d_model, e.heads, e.head_dim, e.max_ngram, e.pad_id, c_.rms_eps,
                                    e.multipliers.data() + (size_t) ei * e.max_ngram,
                                    e.primes.data() + (size_t) ei * B, e.offsets.data() + (size_t) ei * B,
                                    e.token_map.data(), (int64_t) e.token_map.size() };
            ed.dead_token = fam_.image_token();
            engram_.push_back(std::make_unique<blocks::Engram>(lin, L.engram, ed, *L.engram.table));
            engram_.back()->build(g);
        } else {
            engram_.push_back(nullptr);
        }
    }
    for (int l = lb_; l < le_; ++l) {   // attention sources -> consumers (same stage, Family::stage checked)
        auto at = [&] (int s) -> const blocks::AttnDs41 * { return s < 0 || s == l ? nullptr : attn_[s - lb_].get(); };
        const auto ad = attn_dims(l);
        attn_[l - lb_]->set_sources(ad.ratio ? at(c_.kv_source_of(l)) : nullptr, ad.ratio ? at(c_.index_source_of(l)) : nullptr,
                                    ad.uses_candidates ? at(c_.candidate_source) : nullptr);
    }
    if (le_ == c_.n_layer) {
        final_norm_ = std::make_unique<blocks::RmsNorm>(lin, w_.final_norm, c_.d_model, c_.rms_eps);
        head_hc_ = std::make_unique<blocks::HcSinkhorn>(lin, blocks::SkHcWeights {}, k);
        head_ = std::make_unique<blocks::Head>(g, lin, w_.output);
    }
    if (le_ == c_.n_layer && fam_.has_drafter()) {
        const DSparkConfig & dc = *fam_.dspark_config();
        for (int t : dc.target_layers)
            engine::require(t >= lb_ && t < le_, "deepseek41: DSpark target layer on another stage (split must put 37-39 last)");
        dspark_ = std::make_unique<DSpark>(st, c_, w_, dc, *fam_.dspark_weights(), fam_.drafter_min_p());
        fam_.release_drafter_pages();
    }
}

void StageModel::bind_experts(engine::Stage & st)
{
    blocks::MoeDims md { c_.d_model, c_.n_expert, c_.n_expert_used, c_.d_ff_exp * c_.n_shexp };
    md.swiglu_limit = c_.swiglu_limit;
    md.route_scale = c_.expert_scale;
    for (int l = lb_; l < le_; ++l) {
        const blocks::MoeWeights * next = l + 1 < le_ ? &w_.layers[l + 1].moe : nullptr;
        moe_.push_back(std::make_unique<blocks::Moe>(st.lin(), *st.moe(), l - lb_, l, w_.layers[l].moe, md, next,
                                                     &blocks::route_sqrtsoftplus_bias, false, &st.prof()));
        if (const float * bv = fam_.bias_vl(l)) {
            float * p = st.gpu().alloc<float>(c_.n_expert);
            TRUSS_CUDA(cudaMemcpy(p, bv, sizeof(float) * c_.n_expert, cudaMemcpyHostToDevice));
            moe_.back()->set_bias_vl(p);
        }
    }
}

void StageModel::embed(engine::Stage & st, const engine::Chunk & c, float * res)
{
    engine::Gpu & g = st.gpu();
    Scratch & sc = *g.sc;
    int * ids = sc.alloc<int>(c.T);
    float * emb = sc.alloc((size_t) c.T * c_.d_model);
    TRUSS_CUDA(cudaMemcpyAsync(ids, c.tokens, (size_t) c.T * 4, cudaMemcpyHostToDevice, g.s));
    dense::q8_rows(st.lin().q8(w_.token_embd), ids, c.T, emb, g.s);
    for (int i = 0; i < c.x.n_flat; ++i) {   // the reference's merge_image_embeddings: span rows = the tower's rows
        const engine::ImageIn & im = c.x.flat[i];
        engine::require(im.cols == c_.d_model, "deepseek41: image rows are not d_model wide");
        const int a = std::max(im.row0, c.x.flat_row0), b = std::min(im.row0 + im.n, c.x.flat_row0 + c.T);
        if (b <= a) continue;
        TRUSS_CUDA(cudaMemcpyAsync(emb + (size_t) (a - c.x.flat_row0) * c_.d_model,
                                   im.embeds + (size_t) (a - im.row0) * im.cols,
                                   sizeof(float) * (size_t) (b - a) * c_.d_model, cudaMemcpyHostToDevice, g.s));
    }
    engine::require(c.x.img_n == 0, "deepseek41: image rows must come as 1-D images (nh 1)");
    hc::SkDims k = sk_dims();
    k.coef_ld = k.res_ld;
    hc::sk_expand(emb, c.T, k, res, res + (size_t) c_.hc * c_.d_model, g.s);
}

// TRUSS_DS41_DUMP=<dir> (parity debugging, tests/layer/ds41_stage_dump.cu): append T x n fp32 rows of a layer's
// tensor to <dir>/<name>_L<nn>.f32 (synchronous; off unless the variable is set)
static void dump(engine::Gpu & g, const char * name, int l, const float * p, int T, int n, int ld)
{
    static const char * dir = std::getenv("TRUSS_DS41_DUMP");
    if (!dir) return;
    std::vector<float> h((size_t) T * n);
    TRUSS_CUDA(cudaMemcpy2DAsync(h.data(), (size_t) n * 4, p, (size_t) ld * 4, (size_t) n * 4, T, cudaMemcpyDeviceToHost, g.s));
    g.sync();
    char path[512];
    std::snprintf(path, sizeof path, "%s/%s_L%02d.f32", dir, name, l);
    if (FILE * f = std::fopen(path, "ab")) std::fwrite(h.data(), 4, h.size(), f), std::fclose(f);
}

// TRUSS_DUMP_DSPARK=<file> (E8 acceptance, debug; off unless set): one record per forward call of the stage that owns
// layers 37-39 (the DSpark target layers in config.json), as
//   int32 pos0, int32 T, int32 tokens[T], half hidden[T][3][d_model]
// where hidden[t][i] is the mean over the hc copies of the residual at the input of layer 37 + i. Host mean: this is
// a debug path and T is at most one prompt chunk, so the 3 x hc x d_model x T floats per call are cheap next to the
// forward. Records append, so one file holds both the warm-up and timed passes of a bench run.
namespace {

struct DsparkDump {
    FILE * f = nullptr;
    std::vector<half> hidden;   // [T][3][d_model]
    std::vector<float> acc, tmp;
};

DsparkDump & dspark_dump()
{
    static DsparkDump d;
    static bool init = false;
    if (!init) {
        init = true;
        if (const char * p = std::getenv("TRUSS_DUMP_DSPARK")) d.f = std::fopen(p, "ab");
    }
    return d;
}

void dump_dspark_layer(engine::Gpu & g, DsparkDump & d, const float * res, int W, int T, int hc, int dm, int li)
{
    d.acc.resize((size_t) T * dm);
    d.tmp.resize((size_t) T * dm);
    TRUSS_CUDA(cudaMemcpy2DAsync(d.acc.data(), (size_t) dm * 4, res, (size_t) W * 4, (size_t) dm * 4, T,
                                 cudaMemcpyDeviceToHost, g.s));
    g.sync();
    for (int c = 1; c < hc; ++c) {
        TRUSS_CUDA(cudaMemcpy2DAsync(d.tmp.data(), (size_t) dm * 4, res + (size_t) c * dm, (size_t) W * 4,
                                     (size_t) dm * 4, T, cudaMemcpyDeviceToHost, g.s));
        g.sync();
        for (size_t i = 0; i < d.acc.size(); ++i) d.acc[i] += d.tmp[i];
    }
    const float inv = 1.f / hc;
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < dm; ++j)
            d.hidden[((size_t) t * 3 + li) * dm + j] = __float2half(d.acc[(size_t) t * dm + j] * inv);
}

void dump_dspark_record(DsparkDump & d, const engine::Chunk & c, int dm)
{
    const int32_t pos0 = c.pos0, T = c.T;
    std::fwrite(&pos0, 4, 1, d.f);
    std::fwrite(&T, 4, 1, d.f);
    std::fwrite(c.tokens, 4, T, d.f);
    std::fwrite(d.hidden.data(), sizeof(half), (size_t) T * 3 * dm, d.f);
    std::fflush(d.f);
}

// the tail prefill (item 7): TRUSS_TAIL_PREFILL=<n> (rows) - the consumer layers run on the last n rows of a
// prompt chunk (the reference's "SWA bounded replay": 128 + 6 = 134). 0: off. Read per chunk (like
// TRUSS_FETCH_PROMPT), so a bench arm can change it on a loaded model.
int tail_rows_cfg()
{
    const char * e = std::getenv("TRUSS_TAIL_PREFILL");
    return e ? std::atoi(e) : 0;
}

}  // namespace

void StageModel::layers(engine::Stage & st, const engine::Chunk & c, float * res, const engine::LayerHook & hook)
{
    engine::Gpu & g = st.gpu();
    Scratch & sc = *g.sc;
    const int T = c.T, d = c_.d_model, W = c_.hc * d + 24, NM = 24;
    float * row_coef = res + (size_t) c_.hc * d;   // each row's next-collapse coef (W floats apart)
    float * A = sc.alloc((size_t) T * NM), * F = sc.alloc((size_t) T * NM);
    const size_t base = sc.mark();
    DsparkDump & dd = dspark_dump();
    const bool dspark = dd.f && lb_ <= 37 && le_ >= 40;   // the stage that owns the three DSpark target layers
    if (dspark) dd.hidden.resize((size_t) T * 3 * d);
    const int tail_n = tail_rows_cfg();   // the tail prefill (item 7): the consumer layers run on the last n rows
    const std::vector<int> vl_all = image_runs(c.tokens, T, fam_.image_token());   // image rows: their router bias
    const std::vector<int> vl_tail = tail_n > 0 && T > tail_n ? image_runs(c.tokens + (T - tail_n), tail_n, fam_.image_token())
                                                              : std::vector<int>();
    const auto is_consumer = [&](int l) {   // the layers that read the candidate source's compressed KV
        return c_.compress_ratio[l] > 0 && l > c_.candidate_source && c_.kv_source_of(l) == c_.candidate_source;
    };
    for (int l = lb_; l < le_; ++l) {
        sc.release(base);
        const int i = l - lb_;
        // the tail: the consumer layers (the compressed-KV readers after the candidate source) run on the last
        // tail_n rows of the prompt chunk; the rows before them are skipped (the SWA window is clipped at the
        // chunk's start, the reference's "SWA bounded replay").
        const bool tail = tail_n > 0 && c.stream && is_consumer(l) && T > tail_n;
        const int Tl = tail ? tail_n : T, sh = T - Tl;
        float * res_l = res + (size_t) sh * W, * coef_l = row_coef + (size_t) sh * W;
        const int32_t * tok_l = c.tokens + sh;
        const int pos_l = c.pos0 + sh;
        const int clip = sh > 0 ? c.pos0 : 0;   // the window's first position (the skipped rows' ring slots are stale)
        float * y = sc.alloc((size_t) T * d), * x = sc.alloc((size_t) T * d), * out = sc.alloc((size_t) T * d);
        half * x16 = sc.alloc<half>((size_t) T * d);
        st.prof().record(l, 0, g.s);
        dump(g, "res_in", l, res_l, Tl, c_.hc * d + NM, W);
        if (dspark && l >= 37 && l <= 39) dump_dspark_layer(g, dd, res_l, W, Tl, c_.hc, d, l - 37);
        if (engram_[i]) engram_[i]->forward(g, c.slot, tok_l, pos_l, Tl, res_l, W), dump(g, "res_eng", l, res_l, Tl, c_.hc * d + NM, W);
        if (dspark_) dspark_->capture(g, c.slot, l, res_l, W, Tl, sh);
        // attention
        hc_attn_[i]->mix(g, res_l, Tl, A);
        hc_attn_[i]->collapse(g, res_l, coef_l, W, Tl, y, nullptr);
        attn_norm_[i]->forward(g, y, Tl, x, x16);
        st.prof().record(l, 1, g.s);
        dump(g, "attn_in", l, x, Tl, d, d);
        attn_[i]->forward(g, c.slot, x, x16, pos_l, Tl, out, sh, clip);
        dump(g, "attn_out", l, out, Tl, d, d);
        st.prof().record(l, 2, g.s);
        hc_attn_[i]->post(g, res_l, out, A, NM, Tl);
        // FFN
        hc_ffn_[i]->mix(g, res_l, Tl, F);
        hc_ffn_[i]->collapse(g, res_l, A, NM, Tl, y, nullptr);
        ffn_norm_[i]->forward(g, y, Tl, x, x16);   // fp32 for the router, fp16 for the projections
        dump(g, "moe_in", l, x, Tl, d, d);
        const std::vector<int> & vl = tail ? vl_tail : vl_all;
        moe_[i]->forward(g, x, x16, Tl, out, c.stream, st.moe_ws(), st.moe_rows(), vl.data(), (int) vl.size() / 2);
        dump(g, "moe_out", l, out, Tl, d, d);
        hc_ffn_[i]->post(g, res_l, out, F, NM, Tl);
        TRUSS_CUDA(cudaMemcpy2DAsync(row_coef, (size_t) W * 4, F, (size_t) NM * 4, (size_t) NM * 4, T,
                                     cudaMemcpyDeviceToDevice, g.s));
        st.prof().record(l, 6, g.s);
        if (hook) {
            g.sync();
            hook(l, res, T);
        }
    }
    if (dspark) dump_dspark_record(dd, c, d);
}

void StageModel::head(engine::Stage & st, const float * rows, int n, float * logits, bool draft)
{
    engine::Gpu & g = st.gpu();
    Scratch & sc = *g.sc;
    const size_t m = sc.mark();
    const int d = c_.d_model, W = c_.hc * d + 24;
    float * y = sc.alloc((size_t) n * d);
    half * x16 = sc.alloc<half>((size_t) n * d);
    head_hc_->collapse(g, rows, rows + (size_t) c_.hc * d, W, n, y, nullptr);
    final_norm_->forward(g, y, n, nullptr, x16);
    head_->logits(x16, n, logits, draft);
    sc.release(m);
}

std::vector<engine::Stateful *> StageModel::states()
{
    std::vector<engine::Stateful *> out;
    for (size_t i = 0; i < attn_.size(); ++i) {
        if (engram_[i]) out.push_back(engram_[i].get());
        out.push_back(attn_[i].get());
    }
    return out;
}

}  // namespace truss::deepseek41
