// engine::Stage: construction (the memory plan), one chunk, and the per-slot sequence bookkeeping.
#include "engine/stage.h"

#include "engine/limits.h"

#include <algorithm>

namespace truss::engine {

namespace {
double vram_used()   // GiB in use on the device
{
    size_t f, t;
    TRUSS_CUDA(cudaMemGetInfo(&f, &t));
    return (t - f) / 1073741824.0;
}

// The prompt path's buffers are sized by prefill_rows, not max_chunk: they are carved from the ring's spare region,
// so every row above the prompt's real length holds cached experts out of VRAM for the whole run (TRACKER #70)
int prefill_rows_of(int mc, int rows, int align)
{
    if (rows <= 0 || rows >= mc) return mc;
    const int r = (rows + align - 1) / align * align;
    return std::min(std::max(r, FETCH_ROWS + 1), mc);
}
}  // namespace

Stage::Stage(const Family & fam, int n_ctx, int max_chunk, const Options & o, const StageSpec & sg)
    : fam_(fam), o_(o), g_(sg.device), lb_(sg.lb), le_(sg.le), first_(sg.lb == 0), last_(sg.le == fam.n_layer()),
      n_ctx_(n_ctx), max_chunk_(max_chunk),
      prefill_rows_(prefill_rows_of(max_chunk, o.prefill_rows, fam.chunk_align())), spec_rows_(o.spec_rows)
{
    load_mark("Stage: start");
    require(lb_ >= 0 && lb_ < le_ && le_ <= fam.n_layer(), "stage layer range");
    require(max_chunk_ % fam.chunk_align() == 0, "max_chunk must be a multiple of " + std::to_string(fam.chunk_align()));
    require(spec_rows_ >= 0 && spec_rows_ <= moe::MAX_ROWS, "spec_rows must be 0.." + std::to_string(moe::MAX_ROWS));
    model_ = fam.stage(lb_, le_, n_ctx_);
    lin_ = std::make_unique<blocks::Linear>(g_, o_.act);
    prof_ = std::make_unique<Profile>(fam.n_layer() + 1, g_.device);   // by model layer (a drafter's: n_layer)
    const double vram0 = vram_used();
    model_->add_tensors(*this);
    lin_->upload(model_->host_rows(), model_->no_w16());
    load_mark("dense weights uploaded");
    const double vram_w = vram_used();
    if (o_.after_upload) o_.after_upload();
    n_extra_ = model_->extra_buffers();
    alloc_buffers(std::min(max_chunk_, FETCH_ROWS));
    if (!last_) {
        TRUSS_CUDA(cudaMallocHost(&out_host_, sizeof(float) * (size_t) std::max(max_chunk_, prefill_rows_) * fam.res_width()));
        TRUSS_CUDA(cudaEventCreateWithFlags(&out_ready_, cudaEventDisableTiming));
    }
    if (!first_) TRUSS_CUDA(cudaEventCreateWithFlags(&in_taken_, cudaEventDisableTiming));
    // sequence slots: position-indexed caches are shared out as sub-ranges, recurrent state is per slot
    const int n_slots = std::max(1, o_.slots);
    seq_ctx_ = n_ctx_;
    if (n_slots > 1) {
        seq_ctx_ = n_ctx_ / n_slots / 256 * 256;
        require(seq_ctx_ >= 4096, "TRUSS_SLOTS: under 4,096 positions per slot");
    }
    slots_.resize(n_slots);
    model_->build(*this);
    states_ = model_->states();
    if (Drafter * d = model_->drafter()) states_.push_back(d->state());
    for (Stateful * s : states_) s->alloc_slots(*this, n_slots, seq_ctx_, spec_rows_);
    const double vram_st = vram_used();
    std::fprintf(stderr, "stage %d-%d: %.2f GiB before it, weights %.2f, buffers + state %.2f\n", lb_, le_ - 1, vram0,
                 vram_w - vram0, vram_st - vram_w);
    std::vector<runtime::MoeLayer> ml = model_->moe_layers();
    if (!ml.empty()) {   // routed experts: the store takes the memory left
        const int E = fam.n_expert();
        std::vector<float> usage;   // this stage's rows of [layer][expert] (and the drafter's when given)
        if (!o_.expert_usage.empty()) {
            require(o_.expert_usage.size() >= (size_t) fam.n_layer() * E, "expert_usage has fewer layers than the model");
            usage.assign(o_.expert_usage.begin() + (size_t) lb_ * E, o_.expert_usage.begin() + (size_t) le_ * E);
            const size_t want = ml.size() * (size_t) E;
            if (usage.size() < want && o_.expert_usage.size() >= (size_t) (fam.n_layer() + 1) * E)
                usage.insert(usage.end(), o_.expert_usage.begin() + (size_t) fam.n_layer() * E,
                             o_.expert_usage.begin() + (size_t) (fam.n_layer() + 1) * E);
            if (usage.size() < want) {
                // no drafter profile: the drafter runs once per draft (3-4 times per verify pass), so its experts
                // rank above every layer's (all resident when they fit; TRACKER #58)
                const float top = *std::max_element(usage.begin(), usage.end());
                usage.insert(usage.end(), want - usage.size(), 4.f * top + 1.f);
            }
        }
        runtime::MoeOptions mo = o_.moe;
        if (o_.ring_bytes_override) mo.ring_bytes = o_.ring_bytes_override;
        mo.budget_frac = sg.budget_frac;
        mo.last_stage = last_;
        const size_t extra = prefill_rows_ > FETCH_ROWS ? big_bytes() : 0;
        moe_ = std::make_unique<runtime::MoeExec>(g_, model_->moe_kernels(), std::move(ml), E,
                                                  model_->moe_kernels().topk, usage, extra, mo, max_chunk_, sg.pool,
                                                  sg.pool_mu, prof_.get());
        if (o_.kv_lend && n_slots == 1) {   // E6: every position-indexed array's tail
            std::vector<Stateful::Positional> pa;
            for (Stateful * s : states_) s->positional(pa);
            for (const auto & p : pa) moe_->lend(p.base, p.bytes, p.row_bytes);
            kv_reserve(0, true);
        }
    }
    model_->bind_experts(*this);
    // The store's constructor is the last reader of the shards: from here the mapping is dead weight (TRACKER #72)
    if (o_.after_upload) o_.after_upload();
    if (prefill_rows_ > FETCH_ROWS) {   // the prompt path's buffers: carved from the ring's spare region
        auto * p = static_cast<unsigned char *>(moe_ ? moe_->spare() : (void *) g_.alloc<unsigned char>(big_bytes()));
        const size_t sb = model_->scratch_bytes(prefill_rows_);
        const size_t wb = moe_ ? Gpu::align(model_->moe_kernels().prefill_ws(prefill_rows_)) : 0;
        const size_t rb = Gpu::align(sizeof(float) * prefill_rows_ * fam.res_width());
        big_scratch_ = std::make_unique<Scratch>(p, sb);
        big_.scratch = big_scratch_.get();
        big_.moe_ws = moe_ ? p + Gpu::align(sb) : nullptr;
        big_.moe_rows = prefill_rows_;
        big_.res = reinterpret_cast<float *>(p + Gpu::align(sb) + wb);
        for (int i = 0; i < n_extra_; ++i) big_.extra.push_back(reinterpret_cast<float *>(p + Gpu::align(sb) + wb + (1 + i) * rb));
    }
    if (moe_) moe_->start_pinning();   // last: it holds the driver lock for ~3 s
    for (SlotState & s : slots_) TRUSS_CUDA(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
    load_mark("Stage: done");
}

Stage::~Stage()
{
    if (g_.device >= 0) cudaSetDevice(g_.device);   // the owner's guard restores its device
    if (out_host_) cudaFreeHost(out_host_);
    if (out_ready_) cudaEventDestroy(out_ready_);
    if (in_taken_) cudaEventDestroy(in_taken_);
    for (SlotState & s : slots_) {
        if (s.done) cudaEventDestroy(s.done);
        if (s.ck.pinned) cudaFreeHost(s.ck.pinned);
    }
}

void Stage::alloc_buffers(int small_rows)
{
    small_scratch_ = std::make_unique<Scratch>(model_->scratch_bytes(small_rows));
    small_.scratch = small_scratch_.get();
    const runtime::MoeKernels k = model_->moe_kernels();
    if (k.prefill_ws) small_.moe_ws = g_.alloc<unsigned char>(k.prefill_ws(small_rows));
    small_.moe_rows = small_rows;
    small_.res = g_.alloc<float>((size_t) small_rows * fam_.res_width());
    for (int i = 0; i < n_extra_; ++i) small_.extra.push_back(g_.alloc<float>((size_t) small_rows * fam_.res_width()));
    use(small_);
}

// bytes of the prompt path's buffers (scratch, MoE prefill workspace, residual, extras) at prefill_rows rows
size_t Stage::big_bytes() const
{
    const runtime::MoeKernels k = model_->moe_kernels();
    return Gpu::align(model_->scratch_bytes(prefill_rows_)) + (k.prefill_ws ? Gpu::align(k.prefill_ws(prefill_rows_)) : 0) +
           (1 + n_extra_) * Gpu::align(sizeof(float) * prefill_rows_ * fam_.res_width());
}

void Stage::use(const Buffers & b)
{
    cur_ = b;
    g_.sc = b.scratch;
}

void Stage::kv_reserve(int upto, bool shrink)
{
    if (moe_) moe_->kv_reserve(upto, shrink, n_ctx_);
}

}  // namespace truss::engine
