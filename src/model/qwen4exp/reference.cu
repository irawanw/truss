// Math follows llama-paw src/models/qwen4exp.cpp (build_hc_mix, build_hc_combine, ...): see each block.
#include "core/cuda_check.h"
#include "kernels/reference/ref.cuh"
#include "model/qwen4exp/reference.h"

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
    ref::linear(x.dev(h.down), xn, lo, T, x.stream, x.act);
    silu_scale_kernel<<<blocks((int64_t) T * c.hc_rank), 256, 0, x.stream>>>(lo, T * c.hc_rank, 1.f / hc);
    float * gate = x.scratch.alloc((size_t) T * hc * d);
    ref::linear(x.dev(h.up), lo, gate, T, x.stream, x.act);
    hc_collapse_kernel<<<blocks((int64_t) T * d), 256, 0, x.stream>>>(xn, gate, mixed, T, d, hc);
    if (h.inject && inject) ref::linear(x.dev(h.inject), xn, inject, T, x.stream, x.act);
    TRUSS_CUDA(cudaGetLastError());
}

void hc_combine(const Ctx & x, float * res, const float * block_out, const float * inject, int T)
{
    const int d = x.c.d_model, hc = x.c.hc;
    hc_combine_kernel<<<blocks((int64_t) T * hc * d), 256, 0, x.stream>>>(res, block_out, inject, T, d, hc);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::qwen4exp::reference
