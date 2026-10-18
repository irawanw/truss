// DeepSeek-V4 hyper-connections (mHC): the residual is hc parallel copies of d_model; around each sub-block (attention,
// FFN) three coefficient sets come from one fp32 projection of the whole flattened stream (dsref/model.py Block +
// kernel_torch.hc_split_sinkhorn):
//   m      = (res_flat . fn[j]) * rsqrt(mean(res_flat^2) + norm_eps)          j < (2 + hc) * hc
//   pre    = sigmoid(m[0:hc] * scale[0] + base[0:hc]) + hc_eps                  collapse weights
//   post   = 2 sigmoid(m[hc:2hc] * scale[1] + base[hc:2hc])                    expand weights
//   comb   = softmax_j(m[2hc + i*hc + j] * scale[2] + base[...]) + hc_eps, then column-normalized, then iters - 1
//            rounds of (row-normalize, column-normalize), every sum + hc_eps (Sinkhorn: comb ~ doubly stochastic)
//   collapse  y[d] = sum_c pre[c] * res[c][d]
//   post      res'[j][d] = post[j] * x[d] + sum_i comb[i][j] * res[i][d]
// The pre a sub-block's mix produces is used by the NEXT sub-block's collapse (the model keeps two coef buffers).
// coef layout per token: [pre hc][post hc][comb hc x hc] (row i = source copy, column j = destination copy).
// Deterministic (fixed split and reduction order). The projection is split over SPLIT-float chunks of the stream so a
// single decode row spreads over many SMs; partial sums go through a workspace, summed in chunk order.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace truss::hc {

struct SkDims {
    int hc, d;
    float norm_eps, hc_eps;
    int iters;
    int n_mix() const { return (2 + hc) * hc; }
    int coef() const { return (2 + hc) * hc; }   // floats of coef per token (same as n_mix)
};

// workspace for sk_mix of up to T rows
size_t sk_mix_ws_bytes(int T, const SkDims & k);

// res [T][hc][d] fp32, fn [(2 + hc) hc][hc d] fp32, scale [3], base [(2 + hc) hc] -> coef [T][(2 + hc) hc].
// hc == 4, d % 1024 == 0 (host-checked).
void sk_mix(const float * res, const float * fn, const float * scale, const float * base, int T, const SkDims & k,
            void * ws, float * coef, cudaStream_t stream);

// y [T][d] = sum_c pre[c] res[c] (pre = coef row's first hc floats); y16 (nullable) the same rows in fp16
void sk_collapse(const float * res, const float * coef, int T, const SkDims & k, float * y, half * y16,
                 cudaStream_t stream);

// res_out [T][hc][d] = post[j] x[d] + sum_i comb[i][j] res[i][d]; res_out may be res (each thread reads its column of
// all copies before writing)
void sk_post(const float * res, const float * x, const float * coef, int T, const SkDims & k, float * res_out,
             cudaStream_t stream);

// res [T][hc][d] = emb [T][d] in every copy; coef_pre [T][coef] = one-hot pre on copy 0 (the reference's
// make_identity_pre_mix; post / comb entries zero, never read)
void sk_expand(const float * emb, int T, const SkDims & k, float * res, float * coef_pre, cudaStream_t stream);

}  // namespace truss::hc
