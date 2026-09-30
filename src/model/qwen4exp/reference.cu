// Math follows llama-paw src/models/qwen4exp.cpp (build_hc_mix, build_hc_combine, ...): see each block.
#include "core/cuda_check.h"
#include "kernels/reference/ref.cuh"
#include "model/qwen4exp/reference.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>

namespace truss::qwen4exp::reference {

namespace {

__device__ float sigmoid(float v) { return 1.f / (1.f + expf(-v)); }

__global__ void silu_scale_kernel(float * v, int n, float scale)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float a = v[i] * scale;
        v[i] = a * sigmoid(a);
    }
}

// mixed[t][i] = (1/hc) * sum_c xn[t][c][i] * sigmoid(gate[t][c][i])
__global__ void hc_collapse_kernel(const float * xn, const float * gate, float * mixed, int T, int d, int hc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * d) return;
    const int t = i / d, k = i % d;
    float acc = 0.f;
    for (int c = 0; c < hc; ++c) {
        const size_t j = ((size_t) t * hc + c) * d + k;
        acc += xn[j] * sigmoid(gate[j]);
    }
    mixed[i] = acc / hc;
}

__global__ void hc_combine_kernel(float * res, const float * out, const float * inject, int T, int d, int hc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * hc * d) return;
    const int t = i / (hc * d), c = (i / d) % hc, k = i % d;
    res[i] += out[(size_t) t * d + k] * 2.f * sigmoid(inject[t * hc + c] / hc);
}

int blocks(int64_t n) { return (int) ((n + 255) / 256); }

}  // namespace

void hc_mix(const Ctx & x, const HyperConnection & h, const float * res, int T, float * mixed, float * inject)
{
    const Config & c = x.c;
    const int d = c.d_model, hc = c.hc;
    // grouped RMSNorm: each stream over d_model, gamma [hc][d] (the converter folded it to 1 + w)
    float * xn = x.scratch.alloc((size_t) T * hc * d);
    ref::rms_norm(res, x.dev(h.norm).as<float>(), hc, xn, T * hc, d, c.rms_eps, x.stream);
    // low-rank gate: up(silu(down(xn) / hc))
    float * lo = x.scratch.alloc((size_t) T * c.hc_rank);
    ref::linear(x.dev(h.down), xn, lo, T, x.stream, x.num);
    silu_scale_kernel<<<blocks((int64_t) T * c.hc_rank), 256, 0, x.stream>>>(lo, T * c.hc_rank, 1.f / hc);
    float * gate = x.scratch.alloc((size_t) T * hc * d);
    ref::linear(x.dev(h.up), lo, gate, T, x.stream, x.num);
    hc_collapse_kernel<<<blocks((int64_t) T * d), 256, 0, x.stream>>>(xn, gate, mixed, T, d, hc);
    if (h.inject && inject) ref::linear(x.dev(h.inject), xn, inject, T, x.stream, x.num);
    TRUSS_CUDA(cudaGetLastError());
}

void hc_combine(const Ctx & x, float * res, const float * block_out, const float * inject, int T)
{
    const int d = x.c.d_model, hc = x.c.hc;
    hc_combine_kernel<<<blocks((int64_t) T * hc * d), 256, 0, x.stream>>>(res, block_out, inject, T, d, hc);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::qwen4exp::reference

// ---------------------------------------------------------------------------------------------------------------
// FFN

namespace truss::qwen4exp::reference {

namespace {

// y = scale * x per column (scale fp16 [n]); LLAMA numerics: x rounded to fp16 and the product taken in fp16, as
// the PAW X3 op feeds its GEMV
__global__ void prescale_kernel(const float * x, const half * scale, float * y, int rows, int n, bool r)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    const half sc = scale[i % n];
    y[i] = r ? __half2float(__hmul(__float2half_rn(x[i]), sc)) : x[i] * __half2float(sc);
}

__global__ void round16_kernel(float * v, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = __half2float(__float2half_rn(v[i]));
}

__global__ void postscale_kernel(float * v, const half * scale, int rows, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * n) v[i] *= __half2float(scale[i % n]);
}

__global__ void swiglu_kernel(const float * g, const float * u, float * m, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) m[i] = g[i] * sigmoid(g[i]) * u[i];
}

__global__ void scale_kernel(float * v, int n, float w)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= w;
}

__global__ void add_kernel(const float * a, const float * b, float * out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void slot_sum_kernel(const float * y, float * out, int T, int k, int d)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * d) return;
    const int t = i / d, c = i % d;
    float acc = 0.f;
    for (int s = 0; s < k; ++s) acc += y[((size_t) t * k + s) * d + c];
    out[i] = acc;
}

__global__ void gate_add_kernel(const float * a, const float * b, const float * gate, float * out, int T, int d)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * d) out[i] = (a ? a[i] : 0.f) + b[i] * sigmoid(gate[i / d]);
}

// one projection of one expert on `rows` activations: out = svh * H128(W^T a), a = H128(suh * in)
void project(const Ctx & x, const formats::ExpertTable & tab, int e, const float * in, int rows, float * out)
{
    const size_t mark = x.scratch.mark();
    const bool r = x.num == ref::Numerics::LLAMA;
    const int n_in = (int) tab.in, n_out = (int) tab.out;
    float * a = x.scratch.alloc((size_t) rows * n_in);
    prescale_kernel<<<blocks((int64_t) rows * n_in), 256, 0, x.stream>>>(
        in, x.dev(tab.suh).as<half>() + (size_t) e * n_in, a, rows, n_in, r);
    ref::hadamard128(a, (int64_t) rows * n_in, x.stream);
    if (r) round16_kernel<<<blocks((int64_t) rows * n_in), 256, 0, x.stream>>>(a, rows * n_in);
    float * W = x.scratch.alloc((size_t) n_in * n_out);
    ref::trellis_dequant(x.dev(tab.trellis).as<uint16_t>() + tab.offset[e], tab.k[e], n_in, n_out, W, x.stream);
    DTensor Wt;
    Wt.data = W;
    Wt.ne = { n_in, n_out, 1, 1 };
    ref::linear(Wt, a, out, rows, x.stream);
    ref::hadamard128(out, (int64_t) rows * n_out, x.stream);
    postscale_kernel<<<blocks((int64_t) rows * n_out), 256, 0, x.stream>>>(
        out, x.dev(tab.svh).as<half>() + (size_t) e * n_out, rows, n_out);
    x.scratch.release(mark);
}

}  // namespace

Routing route(const Ctx & x, const Moe & m, const float * in, int T, float * logits)
{
    const int E = x.c.n_expert, k = x.c.n_expert_used;
    float * lg = logits ? logits : x.scratch.alloc((size_t) T * E);
    ref::linear(x.dev(m.router), in, lg, T, x.stream);
    std::vector<float> h((size_t) T * E);
    TRUSS_CUDA(cudaMemcpyAsync(h.data(), lg, h.size() * 4, cudaMemcpyDeviceToHost, x.stream));
    TRUSS_CUDA(cudaStreamSynchronize(x.stream));
    Routing r;
    r.ids.resize((size_t) T * k);
    r.weights.resize((size_t) T * k);
    std::vector<float> p(E);
    std::vector<int> idx(E);
    for (int t = 0; t < T; ++t) {
        const float * l = h.data() + (size_t) t * E;
        float mx = l[0], sum = 0.f;
        for (int e = 1; e < E; ++e) mx = std::max(mx, l[e]);
        for (int e = 0; e < E; ++e) sum += p[e] = std::exp(l[e] - mx);
        for (int e = 0; e < E; ++e) p[e] /= sum, idx[e] = e;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&] (int a, int b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
        float s = 0.f;
        for (int j = 0; j < k; ++j) s += p[idx[j]];
        s = std::max(s, 6.103515625e-5f);
        for (int j = 0; j < k; ++j) {
            r.ids[(size_t) t * k + j] = idx[j];
            r.weights[(size_t) t * k + j] = p[idx[j]] / s;
        }
    }
    return r;
}

void routed(const Ctx & x, const Moe & m, const float * in, const Routing & r, int T, float * out)
{
    const int d = x.c.d_model, f = x.c.d_ff_exp, k = x.c.n_expert_used;
    float * y = x.scratch.alloc((size_t) T * k * d);        // [T][k][d]: each choice's weighted output
    float * xs = x.scratch.alloc((size_t) T * d);
    float * g = x.scratch.alloc((size_t) T * f);
    float * u = x.scratch.alloc((size_t) T * f);
    float * mid = x.scratch.alloc((size_t) T * f);
    float * yd = x.scratch.alloc((size_t) T * d);
    for (int e = 0; e < x.c.n_expert; ++e) {
        std::vector<int> pos;                               // flat (t, j) choices of expert e
        for (int i = 0; i < T * k; ++i)
            if (r.ids[i] == e) pos.push_back(i);
        if (pos.empty()) continue;
        const int n = (int) pos.size();
        for (int i = 0; i < n; ++i)
            TRUSS_CUDA(cudaMemcpyAsync(xs + (size_t) i * d, in + (size_t) (pos[i] / k) * d, d * 4,
                                       cudaMemcpyDeviceToDevice, x.stream));
        project(x, m.gate, e, xs, n, g);
        project(x, m.up, e, xs, n, u);
        swiglu_kernel<<<blocks((int64_t) n * f), 256, 0, x.stream>>>(g, u, mid, n * f);
        project(x, m.down, e, mid, n, yd);
        for (int i = 0; i < n; ++i) {   // y[choice] = w * yd[i]
            const float w = r.weights[pos[i]];
            TRUSS_CUDA(cudaMemcpyAsync(y + (size_t) pos[i] * d, yd + (size_t) i * d, d * 4, cudaMemcpyDeviceToDevice,
                                       x.stream));
            scale_kernel<<<blocks(d), 256, 0, x.stream>>>(y + (size_t) pos[i] * d, d, w);
        }
    }
    slot_sum_kernel<<<blocks((int64_t) T * d), 256, 0, x.stream>>>(y, out, T, k, d);
    TRUSS_CUDA(cudaGetLastError());
}

void shared(const Ctx & x, const Moe & m, const float * in, int T, float * out)
{
    const int d = x.c.d_model, f = x.c.d_ff_shexp;
    float * g = x.scratch.alloc((size_t) T * f);
    float * u = x.scratch.alloc((size_t) T * f);
    float * mid = x.scratch.alloc((size_t) T * f);
    float * y = x.scratch.alloc((size_t) T * d);
    float * gate = x.scratch.alloc(T);
    ref::linear(x.dev(m.shexp_gate), in, g, T, x.stream, x.num);
    ref::linear(x.dev(m.shexp_up), in, u, T, x.stream, x.num);
    swiglu_kernel<<<blocks((int64_t) T * f), 256, 0, x.stream>>>(g, u, mid, T * f);
    ref::linear(x.dev(m.shexp_down), mid, y, T, x.stream, x.num);
    ref::linear(x.dev(m.shexp_gate_inp), in, gate, T, x.stream);
    gate_add_kernel<<<blocks((int64_t) T * d), 256, 0, x.stream>>>(nullptr, y, gate, out, T, d);
    TRUSS_CUDA(cudaGetLastError());
}

void ffn(const Ctx & x, const Moe & m, const float * in, int T, float * out)
{
    const Routing r = route(x, m, in, T, nullptr);
    float * a = x.scratch.alloc((size_t) T * x.c.d_model);
    float * b = x.scratch.alloc((size_t) T * x.c.d_model);
    routed(x, m, in, r, T, a);
    shared(x, m, in, T, b);
    add_kernel<<<blocks((int64_t) T * x.c.d_model), 256, 0, x.stream>>>(a, b, out, T * x.c.d_model);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::qwen4exp::reference

// ---------------------------------------------------------------------------------------------------------------
// GDN mixer

namespace truss::qwen4exp::reference {

namespace {

// gate[t][h] = softplus(alpha + dt_bias[h]) * a[h] (ggml softplus: x > 20 -> x); beta = sigmoid(beta)
__global__ void gdn_gates_kernel(float * alpha, float * beta, const float * dt_bias, const float * a, int T, int H)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    const int h = i % H;
    const float v = alpha[i] + dt_bias[h];
    alpha[i] = (v > 20.f ? v : log1pf(expf(v))) * a[h];
    beta[i] = sigmoid(beta[i]);
}

// columns [c0, c0 + n) of each row of src [T][ld] -> dst [T][n]
__global__ void take_cols_kernel(const float * src, float * dst, int T, int ld, int c0, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * n) dst[i] = src[(size_t) (i / n) * ld + c0 + i % n];
}

__global__ void mul_sigmoid_kernel(float * y, const float * z, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= sigmoid(z[i]);
}

}  // namespace

void gdn(const Ctx & x, const Gdn & g, const float * in, int T, float * out, const GdnTrace * trace)
{
    const Config & c = x.c;
    const int S = c.ssm_state, Hk = c.ssm_groups, Hv = c.ssm_v_heads, kd = c.key_dim(), vd = c.value_dim();
    const int C = 2 * kd + vd;
    auto buf = [&](float * t, size_t n) { return t ? t : x.scratch.alloc(n); };
    float * qkv = buf(trace ? trace->qkv : nullptr, (size_t) T * C);
    float * z = buf(trace ? trace->z : nullptr, (size_t) T * vd);
    float * gate = buf(trace ? trace->gate : nullptr, (size_t) T * Hv);
    float * beta = buf(trace ? trace->beta : nullptr, (size_t) T * Hv);
    float * conv = buf(trace ? trace->conv : nullptr, (size_t) T * C);
    float * core = buf(trace ? trace->core : nullptr, (size_t) T * vd);
    float * normed = buf(trace ? trace->normed : nullptr, (size_t) T * vd);
    float * q = x.scratch.alloc((size_t) T * kd), * k = x.scratch.alloc((size_t) T * kd);
    float * v = x.scratch.alloc((size_t) T * vd);

    ref::linear(x.dev(g.qkv), in, qkv, T, x.stream, x.num);
    ref::linear(x.dev(g.gate), in, z, T, x.stream, x.num);
    ref::linear(x.dev(g.alpha), in, gate, T, x.stream, x.num);
    ref::linear(x.dev(g.beta), in, beta, T, x.stream, x.num);
    gdn_gates_kernel<<<blocks((int64_t) T * Hv), 256, 0, x.stream>>>(gate, beta, x.dev(g.dt_bias).as<float>(),
                                                                     x.dev(g.a).as<float>(), T, Hv);
    ref::causal_conv_silu(qkv, x.dev(g.conv1d).as<float>(), nullptr, conv, T, C, c.ssm_conv, x.stream);
    // conv channels: q [key_dim], k [key_dim], v [value_dim]
    take_cols_kernel<<<blocks((int64_t) T * kd), 256, 0, x.stream>>>(conv, q, T, C, 0, kd);
    take_cols_kernel<<<blocks((int64_t) T * kd), 256, 0, x.stream>>>(conv, k, T, C, kd, kd);
    take_cols_kernel<<<blocks((int64_t) T * vd), 256, 0, x.stream>>>(conv, v, T, C, 2 * kd, vd);
    ref::l2_norm(q, q, T * Hk, S, c.rms_eps, x.stream);
    ref::l2_norm(k, k, T * Hk, S, c.rms_eps, x.stream);
    ref::gated_delta_rule(q, k, v, gate, beta, nullptr, core, T, Hk, Hv, S, x.stream);
    // gated RMSNorm over each head (gamma [head]), times sigmoid(z)
    ref::rms_norm(core, x.dev(g.norm).as<float>(), 1, normed, T * Hv, S, c.rms_eps, x.stream);
    mul_sigmoid_kernel<<<blocks((int64_t) T * vd), 256, 0, x.stream>>>(normed, z, T * vd);
    ref::linear(x.dev(g.out), normed, out, T, x.stream, x.num);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::qwen4exp::reference

// ---------------------------------------------------------------------------------------------------------------
// DSA mixer

namespace truss::qwen4exp::reference {

namespace {

// qfull [T][H][2 D] = per head [q | gate] -> q, gate [T][H][D]
__global__ void split_qgate_kernel(const float * qfull, float * q, float * gate, int T, int H, int D)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H * D) return;
    const int th = i / D, e = i % D;
    q[i] = qfull[(size_t) th * 2 * D + e];
    gate[i] = qfull[(size_t) th * 2 * D + D + e];
}

// kb[b] = mean of k[r b .. r b + r - 1] (summed in order, then scaled, as llama)
__global__ void pool_kernel(const float * k, float * kb, int nb, int r, int d)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb * d) return;
    const int b = i / d, e = i % d;
    float acc = 0.f;
    for (int j = 0; j < r; ++j) acc += k[((size_t) b * r + j) * d + e];
    kb[i] = acc * (1.f / (float) r);
}

__global__ void positions_kernel(int * pos, int n, int step)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pos[i] = i * step;
}

}  // namespace

void dsa(const Ctx & x, const Dsa & a, int ratio, const float * in, int T, float * out, const DsaTrace * trace)
{
    const Config & c = x.c;
    const int H = c.n_head, Hkv = c.n_head_kv, D = c.head_dim, Hi = c.idx_heads, Di = c.idx_head_dim;
    const int nb = T / ratio;
    auto buf = [&](float * t, size_t n) { return t ? t : x.scratch.alloc(n); };
    float * qfull = x.scratch.alloc((size_t) T * H * 2 * D);
    float * q = buf(trace ? trace->q : nullptr, (size_t) T * H * D);
    float * gate = buf(trace ? trace->gate : nullptr, (size_t) T * H * D);
    float * k = buf(trace ? trace->k : nullptr, (size_t) T * Hkv * D);
    float * v = buf(trace ? trace->v : nullptr, (size_t) T * Hkv * D);
    float * iq = buf(trace ? trace->idx_q : nullptr, (size_t) T * Hi * Di);
    float * ik_raw = x.scratch.alloc((size_t) T * Di);
    float * ik = buf(trace ? trace->idx_k : nullptr, (size_t) (nb > 0 ? nb : 1) * Di);
    float * att = buf(trace ? trace->pregate : nullptr, (size_t) T * H * D);
    float * gated = buf(trace ? trace->gated : nullptr, (size_t) T * H * D);
    int * pos = x.scratch.alloc<int>(T), * bpos = x.scratch.alloc<int>(nb > 0 ? nb : 1);
    uint8_t * sel = trace && trace->sel ? trace->sel : x.scratch.alloc<uint8_t>((size_t) T * T);
    positions_kernel<<<blocks(T), 256, 0, x.stream>>>(pos, T, 1);
    positions_kernel<<<blocks(nb), 256, 0, x.stream>>>(bpos, nb, ratio);

    // q (+ gate), k, v: per-head RMSNorm, rope on the first rope_dims
    ref::linear(x.dev(a.q), in, qfull, T, x.stream, x.num);
    split_qgate_kernel<<<blocks((int64_t) T * H * D), 256, 0, x.stream>>>(qfull, q, gate, T, H, D);
    ref::rms_norm(q, x.dev(a.q_norm).as<float>(), 1, q, T * H, D, c.rms_eps, x.stream);
    ref::linear(x.dev(a.k), in, k, T, x.stream, x.num);
    ref::rms_norm(k, x.dev(a.k_norm).as<float>(), 1, k, T * Hkv, D, c.rms_eps, x.stream);
    ref::linear(x.dev(a.v), in, v, T, x.stream, x.num);
    if (trace && trace->q_normed)
        TRUSS_CUDA(cudaMemcpyAsync(trace->q_normed, q, sizeof(float) * T * H * D, cudaMemcpyDeviceToDevice, x.stream));
    if (trace && trace->k_normed)
        TRUSS_CUDA(cudaMemcpyAsync(trace->k_normed, k, sizeof(float) * T * Hkv * D, cudaMemcpyDeviceToDevice, x.stream));
    ref::rope_neox(q, T, H, D, c.rope_dims, pos, c.rope_base, x.stream);
    ref::rope_neox(k, T, Hkv, D, c.rope_dims, pos, c.rope_base, x.stream);

    // indexer: raw keys pooled per complete block, then normed and roped at the block start; queries per head
    ref::linear(x.dev(a.idx_k), in, ik_raw, T, x.stream, x.num);
    if (nb > 0) {
        pool_kernel<<<blocks((int64_t) nb * Di), 256, 0, x.stream>>>(ik_raw, ik, nb, ratio, Di);
        ref::rms_norm(ik, x.dev(a.idx_k_norm).as<float>(), 1, ik, nb, Di, c.rms_eps, x.stream);
        ref::rope_neox(ik, nb, 1, Di, c.rope_dims, bpos, c.rope_base, x.stream);
    }
    ref::linear(x.dev(a.idx_q), in, iq, T, x.stream, x.num);
    ref::rms_norm(iq, x.dev(a.idx_q_norm).as<float>(), 1, iq, T * Hi, Di, c.rms_eps, x.stream);
    ref::rope_neox(iq, T, Hi, Di, c.rope_dims, pos, c.rope_base, x.stream);
    ref::qsa_select(iq, ik, T, Hi, Di, ratio, c.idx_top_k / ratio, sel, x.stream);

    ref::masked_attention(q, k, v, sel, att, T, H, Hkv, D, 1.f / std::sqrt((float) D), x.stream);
    TRUSS_CUDA(cudaMemcpyAsync(gated, att, sizeof(float) * T * H * D, cudaMemcpyDeviceToDevice, x.stream));
    mul_sigmoid_kernel<<<blocks((int64_t) T * H * D), 256, 0, x.stream>>>(gated, gate, T * H * D);
    ref::linear(x.dev(a.out), gated, out, T, x.stream, x.num);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::qwen4exp::reference
