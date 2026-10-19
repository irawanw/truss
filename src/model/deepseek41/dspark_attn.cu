// DSpark window-only attention (dsref DSparkAttention): q/kv from the draft rows, noncausal attention over the last
// min(window, pos) target latents plus the draft rows, inverse rope, grouped output LoRA. The ring stores each
// DSpark layer's target latents (seeded by DSpark::seed).
#include "model/deepseek41/dspark.h"

#include "kernels/dense/q8_gemm.cuh"
#include "kernels/ds41/attn_ops.cuh"
#include "kernels/ds41/sparse_attn.cuh"
#include "kernels/norm/rms_norm.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace truss::deepseek41 {

namespace {

__global__ void fill_idx_kernel(int * idx, int T, int R)
{
    const int t = blockIdx.x;
    for (int j = threadIdx.x; j < R + T; j += blockDim.x) idx[(size_t) t * (R + T) + j] = j;
}

}  // namespace

void DSparkAttn::add_tensors(blocks::Linear & lin, const blocks::Ds41AttnWeights & w)
{
    for (T t : { w.q_a, w.q_a_norm, w.q_b, w.kv, w.kv_norm, w.out_a, w.out_b, w.sinks }) lin.add(t);
}

void DSparkAttn::alloc(engine::Gpu & g, int n_slots)
{
    half * ring = g.alloc<half>((size_t) n_slots * DSpark::RING * d_.head_dim);
    sl_.resize(n_slots);
    for (int i = 0; i < n_slots; ++i) sl_[i] = ring + (size_t) i * DSpark::RING * d_.head_dim;
    comb_ = g.alloc<half>((size_t) (DSpark::RING + DSpark::MAX_DRAFT) * d_.head_dim);
    idx_ = g.alloc<int>((size_t) DSpark::MAX_DRAFT * (DSpark::RING + DSpark::MAX_DRAFT));
}

void DSparkAttn::seed(engine::Gpu & g, int slot, const half * main_x16, int n, int pos0)
{
    Scratch & sc = *g.sc;
    const size_t m = sc.mark();
    const int D = d_.head_dim, rd = d_.rope_dim;
    float * kv = sc.alloc((size_t) n * D), * kvn = sc.alloc((size_t) n * D);
    lin_.lin(w_.kv, main_x16, n, kv);
    norm::rms(kv, lin_.f32(w_.kv_norm), n, D, d_.eps, kvn, nullptr, g.s);
    ds41::rope_tail(kvn, n, 1, D, rd, pos0, 1, d_.inv_freq, false, g.s);
    ds41::ring_write(sl_[slot], DSpark::RING, kvn, D, pos0, n, g.s);
    sc.release(m);
}

void DSparkAttn::forward(engine::Gpu & g, int slot, const half * x16, int pos0, int T, float * out)
{
    Scratch & sc = *g.sc;
    const size_t m = sc.mark();
    const int D = d_.head_dim, H = d_.n_head, Q = d_.q_rank, G = d_.o_groups, R = d_.o_rank, rd = d_.rope_dim;
    engine::require(T <= DSpark::MAX_DRAFT, "dspark attention draft rows");
    const int nring = std::min(d_.window, pos0);
    engine::require(pos0 > 0 && nring + T <= DSpark::RING + DSpark::MAX_DRAFT, "dspark attention ring");
    // q
    float * qa = sc.alloc((size_t) T * Q), * q = sc.alloc((size_t) T * H * D);
    half * qr16 = sc.alloc<half>((size_t) T * Q);
    lin_.lin(w_.q_a, x16, T, qa);
    norm::rms(qa, lin_.f32(w_.q_a_norm), T, Q, d_.eps, nullptr, qr16, g.s);
    lin_.lin(w_.q_b, qr16, T, q);
    ds41::rope_tail(q, T, H, D, rd, pos0, 1, d_.inv_freq, false, g.s);
    // draft kv
    float * kv = sc.alloc((size_t) T * D), * kvn = sc.alloc((size_t) T * D);
    half * kv16 = sc.alloc<half>((size_t) T * D);
    lin_.lin(w_.kv, x16, T, kv);
    norm::rms(kv, lin_.f32(w_.kv_norm), T, D, d_.eps, kvn, nullptr, g.s);
    ds41::rope_tail(kvn, T, 1, D, rd, pos0, 1, d_.inv_freq, false, g.s);
    ds41::to_half(kvn, (size_t) T * D, kv16, g.s);
    // keys = ring slots 0..nring-1, then the draft rows
    TRUSS_CUDA(cudaMemcpy2DAsync(comb_, (size_t) D * 2, sl_[slot], (size_t) D * 2, (size_t) D * 2, nring,
                                 cudaMemcpyDeviceToDevice, g.s));
    TRUSS_CUDA(cudaMemcpy2DAsync(comb_ + (size_t) nring * D, (size_t) D * 2, kv16, (size_t) D * 2, (size_t) D * 2, T,
                                 cudaMemcpyDeviceToDevice, g.s));
    fill_idx_kernel<<<T, 256, 0, g.s>>>(idx_, T, nring);
    float * o = sc.alloc((size_t) T * H * D);
    ds41::SparseAttnArgs a;
    a.q = q, a.chunk_kv = nullptr, a.ring = nullptr, a.R = 0;
    a.comp = comb_, a.idx = idx_, a.k = nring + T;
    a.sinks = lin_.f32(w_.sinks), a.H = H, a.D = D, a.window = 0, a.pos0 = pos0, a.T = T;
    a.scale = 1.0f / std::sqrt((float) D), a.out = o;
    static const bool v1 = [] {
        const char * e = std::getenv("TRUSS_DSPARK_ATTN_V1");
        return e && std::atoi(e);
    }();
    if (!v1 && ds41::sparse_attn_tc_ok(a)) {
        const size_t n = ds41::sparse_attn_tc_ws_floats(a);
        ds41::sparse_attn_tc(a, n ? sc.alloc(n) : nullptr, g.s);
    } else {
        ds41::sparse_attn(a, g.s);
    }
    ds41::rope_tail(o, T, H, D, rd, pos0, 1, d_.inv_freq, true, g.s);
    // grouped output LoRA (as AttnDs41)
    half * o16 = sc.alloc<half>((size_t) T * H * D);
    ds41::to_half(o, (size_t) T * H * D, o16, g.s);
    float * yo = sc.alloc((size_t) T * G * R);
    half * yo16 = sc.alloc<half>((size_t) T * G * R);
    if (lin_.act() == blocks::Activations::FP16 && T * G > dense::GEMV_ROWS) {
        dense::q8_gemm_a16_grouped(lin_.q8(w_.out_a), G, o16, T, yo, lin_.w16(), g.blas, g.s);
    } else {
        float * ya = sc.alloc((size_t) T * G * G * R);
        lin_.lin(w_.out_a, o16, T * G, ya);
        ds41::pick_groups(ya, T, G, R, yo, g.s);
    }
    ds41::to_half(yo, (size_t) T * G * R, yo16, g.s);
    lin_.lin(w_.out_b, yo16, T, out);
    sc.release(m);
}

}  // namespace truss::deepseek41
