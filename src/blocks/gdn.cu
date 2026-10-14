#include "blocks/gdn.h"

#include "engine/stage.h"
#include "kernels/gdn/gdn_prefill.cuh"
#include "kernels/gdn/gdn_prepare.cuh"

namespace truss::blocks {

void Gdn::add_tensors(Linear & lin, const GdnWeights & w)
{
    for (T t : { w.qkv, w.gate, w.conv1d, w.dt_bias, w.a, w.beta, w.alpha, w.norm, w.out }) lin.add(t);
}

void Gdn::alloc_slots(engine::Stage & st, int n_slots, int, int spec_rows)
{
    engine::Gpu & g = st.gpu();
    const size_t W = spec_rows;
    s_.resize(n_slots);
    for (Slot & s : s_) {
        s.state_buf[0] = g.alloc<float>(d_.state_floats());
        if (W) s.state_buf[1] = g.alloc<float>(d_.state_floats());
        s.conv = g.alloc<float>(d_.conv_floats());
        if (W) {
            s.conv_snap = g.alloc<float>(d_.conv_floats());
            s.raw_qkv = g.alloc<float>(W * d_.conv_dim());
            s.raw_alpha = g.alloc<float>(W * d_.v_heads);
            s.raw_beta = g.alloc<float>(W * d_.v_heads);
        }
    }
}

void Gdn::forward(engine::Gpu & g, int slot, const half * in16, int T, float * out, bool tentative)
{
    Slot & L = s_[slot];
    Scratch & sc = *g.sc;
    const size_t m = sc.mark();
    const gdn::Dims dm{ d_.groups, d_.v_heads, d_.state, d_.conv };
    const int kd = d_.key_dim(), vd = d_.value_dim(), Hv = d_.v_heads;
    float * qkv = sc.alloc((size_t) T * d_.conv_dim()), * z = sc.alloc((size_t) T * vd);
    float * alpha = sc.alloc((size_t) T * Hv), * beta_raw = sc.alloc((size_t) T * Hv);
    float * q = sc.alloc((size_t) T * kd), * k = sc.alloc((size_t) T * kd), * v = sc.alloc((size_t) T * vd);
    float * gt = sc.alloc((size_t) T * Hv), * beta = sc.alloc((size_t) T * Hv);
    float * core = sc.alloc((size_t) T * vd);
    half * o16 = sc.alloc<half>((size_t) T * vd);
    const Linear::Act ia = lin_.quant(in16, T, d_.d_model);
    lin_.lin_multi({ { w_.qkv, qkv }, { w_.gate, z }, { w_.alpha, alpha }, { w_.beta, beta_raw } }, ia, T);
    if (tentative) {   // accept() may redo the accepted rows from these
        g.copy(L.conv_snap, L.conv, d_.conv_floats());
        g.copy(L.raw_qkv, qkv, (size_t) T * d_.conv_dim());
        g.copy(L.raw_alpha, alpha, (size_t) T * Hv);
        g.copy(L.raw_beta, beta_raw, (size_t) T * Hv);
    }
    gdn::prepare(dm, qkv, lin_.f32(w_.conv1d), L.conv, alpha, beta_raw, lin_.f32(w_.dt_bias), lin_.f32(w_.a), T, d_.eps,
                 q, k, v, gt, beta, g.s);
    gdn::delta_rule(q, k, v, gt, beta, L.state(), tentative ? L.state_buf[1 - L.cur] : L.state(), core, T, d_.groups, Hv,
                    g.s);
    gdn::output_norm(dm, core, z, lin_.f32(w_.norm), T, d_.eps, o16, g.s);
    lin_.lin(w_.out, o16, T, out);
    sc.release(m);
}

void Gdn::reset(engine::Stage & st, int slot)
{
    Slot & L = s_[slot];
    engine::Gpu & g = st.gpu();
    TRUSS_CUDA(cudaMemsetAsync(L.state(), 0, sizeof(float) * d_.state_floats(), g.s));
    TRUSS_CUDA(cudaMemsetAsync(L.conv, 0, sizeof(float) * d_.conv_floats(), g.s));
}

void Gdn::accept(engine::Stage & st, int slot, int n, const engine::Window & w)
{
    Slot & L = s_[slot];
    if (n == w.T) {   // the window's final state is the other buffer
        L.cur ^= 1;
        return;
    }
    engine::Gpu & g = st.gpu();
    g.copy(L.conv, L.conv_snap, d_.conv_floats());   // conv rows from before the window, then the accepted rows again
    if (n == 0) return;   // 0 rows would launch the prepare and the delta rule on an empty grid
    Scratch & sc = *g.sc;
    const size_t m = sc.mark();
    const gdn::Dims dm{ d_.groups, d_.v_heads, d_.state, d_.conv };
    const int kd = d_.key_dim(), vd = d_.value_dim(), Hv = d_.v_heads;
    float * q = sc.alloc((size_t) n * kd), * k = sc.alloc((size_t) n * kd), * v = sc.alloc((size_t) n * vd);
    float * gt = sc.alloc((size_t) n * Hv), * beta = sc.alloc((size_t) n * Hv), * core = sc.alloc((size_t) n * vd);
    gdn::prepare(dm, L.raw_qkv, lin_.f32(w_.conv1d), L.conv, L.raw_alpha, L.raw_beta, lin_.f32(w_.dt_bias),
                 lin_.f32(w_.a), n, d_.eps, q, k, v, gt, beta, g.s);
    gdn::delta_rule(q, k, v, gt, beta, L.state(), L.state(), core, n, d_.groups, Hv, g.s);
    sc.release(m);
}

void Gdn::ck_buffers(int slot, std::vector<std::pair<float *, size_t>> & out)
{
    out.push_back({ s_[slot].state(), d_.state_floats() });
    out.push_back({ s_[slot].conv, d_.conv_floats() });
}

}  // namespace truss::blocks
