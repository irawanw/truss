// DSpark drafter: its own all-resident MoE exec, the target-hidden capture, and the per-slot ring seeding.
// The draft block forward is in dspark_draft.cu.
#include "model/deepseek41/dspark.h"

#include "kernels/ds41/attn_ops.cuh"
#include "kernels/norm/rms_norm.cuh"
#include "engine/stage.h"
#include "runtime/expert_store_util.h"
#include "runtime/moe_kernels.h"

#include <algorithm>
#include <cstdlib>

namespace truss::deepseek41 {

namespace {

__global__ void capture_kernel(const float * res, int rw, int start, int hc, int dm, int li, int total, float * out)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) total) return;
    const int row = (int) (i / dm), j = (int) (i % dm);
    const float * p = res + (size_t) (start + row) * rw;
    float sum = p[j];
    for (int c = 1; c < hc; ++c) sum += p[(size_t) c * dm + j];
    out[((size_t) row * 3 + li) * dm + j] = sum / hc;
}

// fp16 copy of the captured rows for main_proj, each row scaled into the fp16 range: some rows (seen on a 21-token
// prompt; the long prompts keep only later rows) carry
// residual values far above 65504, which became inf and then NaN in the draft attention (10-21: short prompts
// crashed the drafter's MoE on garbage expert ids). main_proj has no bias and main_norm (RMSNorm) follows it, so a
// per-row positive scale leaves the result unchanged (up to eps).
__global__ void rows_to_half_scaled(const float * x, int cols, half * out)
{
    const float * r = x + (size_t) blockIdx.x * cols;
    __shared__ float red[256];
    float m = 0.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) m = fmaxf(m, fabsf(r[j]));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    const float s = red[0] > 16384.f ? 16384.f / red[0] : 1.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) out[(size_t) blockIdx.x * cols + j] = __float2half(r[j] * s);
}

size_t expert_budget(const DSparkWeights & dw, int n_expert)
{
    size_t b = 0;
    for (const auto & L : dw.layers) {
        const runtime::ExpertLayer t { &L.moe.gate, &L.moe.up, &L.moe.down };
        b += runtime::store_detail::fixed_bytes(t);
        for (int e = 0; e < n_expert; ++e) b += runtime::store_detail::expert_bytes(t, e);
    }
    return b + (32u << 20);   // slack for per-expert alignment: keep all 384 hot
}

}  // namespace

void DSpark::add_tensors(blocks::Linear & lin, const DSparkWeights & w)
{
    for (const auto & L : w.layers) {
        for (T t : { L.attn_norm, L.ffn_norm, L.bias_vl, L.main_proj, L.main_norm, L.norm, L.markov_embed,
                     L.markov_head, L.confidence_proj })
            lin.add(t);
        blocks::HcSinkhorn::add_tensors(lin, L.hc_attn);
        blocks::HcSinkhorn::add_tensors(lin, L.hc_ffn);
        DSparkAttn::add_tensors(lin, L.attn);
        blocks::Moe::add_tensors(lin, L.moe);
    }
}

hc::SkDims DSpark::sk_dims() const
{
    return hc::SkDims { d_.hc, d_.d_model, d_.rms_eps, d_.hc_eps, 20 };
}

DSpark::DSpark(engine::Stage & st, const Config & c, const Weights & w, const DSparkConfig & d,
               const DSparkWeights & dw, float min_p)
    : c_(c), w_(w), d_(d), dw_(dw), lin_(st.lin()), g_(st.gpu()), min_p_(min_p)
{
    const auto iv = ds41::rope_inv_freq(d_.rope_dim, d_.rope_base, 0, 1.f, 32.f, 1.f);
    inv_freq_ = g_.alloc<float>(iv.size());
    TRUSS_CUDA(cudaMemcpy(inv_freq_, iv.data(), iv.size() * 4, cudaMemcpyHostToDevice));

    std::vector<runtime::MoeLayer> ml;
    for (const auto & L : dw_.layers) {
        runtime::MoeLayer m;
        m.tables = { &L.moe.gate, &L.moe.up, &L.moe.down };
        m.model_layer = c_.n_layer;
        m.cpu_ok = false;
        m.next_in_stage = false;
        m.profiled = false;
        ml.push_back(m);
    }
    runtime::MoeOptions mo;
    mo.expert_budget = expert_budget(dw_, d_.n_expert);
    mo.ring_bytes = 0;
    mo.budget_frac = 1.0;
    mo.doorbell = false;
    mo.cpu_trellis = false;
    mo.last_stage = false;
    exec_ = std::make_unique<runtime::MoeExec>(g_, runtime::MoeKernels::of<moe::DSpark>(), ml, d_.n_expert,
                                               d_.n_expert_used, std::vector<float>(), 0, mo, st.max_rows(), nullptr,
                                               nullptr, nullptr);

    const DSparkAttnDims ad { d_.d_model, d_.n_head, d_.head_dim, d_.rope_dim, d_.q_rank, d_.o_rank, d_.o_groups,
                              d_.window, d_.rms_eps, inv_freq_ };
    const blocks::MoeDims md { d_.d_model, d_.n_expert, d_.n_expert_used, d_.d_ff, d_.swiglu_limit, d_.route_scale };
    for (int i = 0; i < d_.n_layer; ++i) {
        const auto & L = dw_.layers[i];
        attn_norm_.push_back(std::make_unique<blocks::RmsNorm>(lin_, L.attn_norm, d_.d_model, d_.rms_eps));
        ffn_norm_.push_back(std::make_unique<blocks::RmsNorm>(lin_, L.ffn_norm, d_.d_model, d_.rms_eps));
        hc_attn_.push_back(std::make_unique<blocks::HcSinkhorn>(lin_, L.hc_attn, sk_dims()));
        hc_ffn_.push_back(std::make_unique<blocks::HcSinkhorn>(lin_, L.hc_ffn, sk_dims()));
        attn_.push_back(std::make_unique<DSparkAttn>(lin_, L.attn, ad));
        moe_.push_back(std::make_unique<blocks::Moe>(lin_, *exec_, i, c_.n_layer, L.moe, md, nullptr,
                                                     &blocks::route_sqrtsoftplus_bias, false, nullptr));
    }
    last_norm_ = std::make_unique<blocks::RmsNorm>(lin_, dw_.layers.back().norm, d_.d_model, d_.rms_eps);
    const size_t n = CAP_ROWS, dm = d_.d_model;
    mproj_ = g_.alloc<float>(n * dm), mx_ = g_.alloc<float>(n * dm), kv_ = g_.alloc<float>(n * d_.head_dim);
    mx16_ = g_.alloc<half>(n * dm), kv16_ = g_.alloc<half>(n * d_.head_dim), hid16_ = g_.alloc<half>(n * 3 * dm);
    logits_ = g_.alloc<float>((size_t) d_.block_size * d_.n_vocab);
    me_ = g_.alloc<float>(d_.markov_rank), mbias_ = g_.alloc<float>(d_.n_vocab);
    m16_ = g_.alloc<half>(d_.markov_rank);
    // the confidence head: p_i = sigmoid(proj([hidden_i (before the final norm); markov_emb_i]) / T)
    xconf_ = g_.alloc<float>(d_.d_model + d_.markov_rank);
    conf_logit_ = g_.alloc<float>(d_.block_size);
    ids_ = g_.alloc<int>(d_.block_size + 1), draft_ids_ = g_.alloc<int>(d_.block_size + 1);
    TRUSS_CUDA(cudaMallocHost(reinterpret_cast<void **>(&ids_host_), (d_.block_size + 1) * sizeof(int32_t)));
    TRUSS_CUDA(cudaMallocHost(reinterpret_cast<void **>(&draft_host_), (d_.block_size + 1) * sizeof(int32_t)));
    draft_prob_ = g_.alloc<float>(d_.block_size), qfull_ = g_.alloc<float>((size_t) d_.block_size * d_.n_vocab);
    emb_ = g_.alloc<float>((size_t) d_.block_size * d_.d_model), res_ = g_.alloc<float>((size_t) d_.block_size * (c_.hc * c_.d_model + 24));
    coef_pre_ = g_.alloc<float>((size_t) d_.block_size * sk_dims().n_mix());
    coef_a_ = g_.alloc<float>((size_t) d_.block_size * sk_dims().n_mix());
    coef_f_ = g_.alloc<float>((size_t) d_.block_size * sk_dims().n_mix());
}

DSpark::~DSpark()
{
    if (ids_host_) cudaFreeHost(ids_host_);
    if (draft_host_) cudaFreeHost(draft_host_);
}

void DSpark::alloc_slots(engine::Stage & st, int n_slots, int, int)
{
    (void) st;
    sl_.assign(n_slots, Slot {});
    for (int s = 0; s < n_slots; ++s) sl_[s].cap = g_.alloc<float>((size_t) CAP_ROWS * 3 * d_.d_model);
    for (auto & a : attn_) a->alloc(g_, n_slots);
}

void DSpark::reset(engine::Stage & st, int slot)
{
    (void) st;
    for (auto & a : attn_)
        TRUSS_CUDA(cudaMemsetAsync(a->ring(slot), 0, (size_t) RING * d_.head_dim * 2, g_.s));
    sl_[slot].cap_n = 0;
    sl_[slot].cap_start = 0;
    sl_[slot].seeded = false;
}

void DSpark::ck_buffers(int slot, std::vector<std::pair<float *, size_t>> & out)
{
    for (auto & a : attn_) out.push_back({ reinterpret_cast<float *>(a->ring(slot)), (size_t) RING * d_.head_dim / 2 });
}

void DSpark::capture(engine::Gpu & g, int slot, int layer, const float * res, int W, int T, int row_off)
{
    int li = -1;
    for (int i = 0; i < (int) d_.target_layers.size(); ++i)
        if (d_.target_layers[i] == layer) li = i;
    if (li < 0) return;
    const int n = std::min(T, CAP_ROWS), start = T - n;
    const int total = n * d_.d_model;
    capture_kernel<<<(total + 255) / 256, 256, 0, g.s>>>(res, W, start, d_.hc, d_.d_model, li, total, sl_[slot].cap);
    sl_[slot].cap_n = n;
    sl_[slot].cap_start = row_off + start;   // the chunk's row of the first captured row
}

void DSpark::seed(engine::Stage & st, int slot, const float * hid, int n, int pos0)
{
    (void) st;
    if (n <= 0) return;
    Scratch & sc = *g_.sc;
    const size_t m = sc.mark();
    const int dm = d_.d_model;
    rows_to_half_scaled<<<n, 256, 0, g_.s>>>(hid, 3 * dm, hid16_);
    lin_.lin(dw_.layers[0].main_proj, hid16_, n, mproj_);
    norm::rms(mproj_, lin_.f32(dw_.layers[0].main_norm), n, dm, d_.rms_eps, mx_, nullptr, g_.s);
    ds41::to_half(mx_, (size_t) n * dm, mx16_, g_.s);
    for (auto & a : attn_) a->seed(g_, slot, mx16_, n, pos0);
    sc.release(m);
    sl_[slot].seeded = true;
}

void DSpark::on_commit(engine::Stage & st, const engine::Chunk & c)
{
    const int n = std::min(sl_[c.slot].cap_n, CAP_ROWS);
    if (n > 0) seed(st, c.slot, sl_[c.slot].cap, n, c.pos0 + sl_[c.slot].cap_start);
}

void DSpark::on_accept(engine::Stage & st, int slot, int n, const engine::Window & w)
{
    // the window's rows are captured from its first row (a window is at most spec_rows rows)
    const int k = std::min(n, sl_[slot].cap_n);
    if (k > 0 && sl_[slot].cap_start == 0) seed(st, slot, sl_[slot].cap, k, w.pos0);
}

}  // namespace truss::deepseek41
