// DeepSeek-V4 mHC kernels (hc_sinkhorn.cuh). Memory-bound: the projection reads fn (24 x 20480 fp32 = 1.97 MB) once
// per call; at decode it is split over hc d / SPLIT blocks so the read uses the whole GPU.
#include "kernels/hc/hc_sinkhorn.cuh"

#include "core/cuda_check.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace truss::hc {
namespace {

constexpr int HC = 4, NMIX = (2 + HC) * HC, SPLIT = 1024, THREADS = 256;   // one float4 per thread per chunk

// part[t][chunk][0..NMIX) = chunk's dot products, [NMIX] = chunk's sum of squares
__global__ void __launch_bounds__(THREADS) sk_part_kernel(const float * __restrict__ res, const float * __restrict__ fn,
                                                         int hcd, int rld, float * __restrict__ part)
{
    const int chunk = blockIdx.x, t = blockIdx.y, n_chunk = gridDim.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const size_t off = (size_t) chunk * SPLIT + threadIdx.x * 4;
    const float4 x = *(const float4 *) (res + (size_t) t * rld + off);
    float acc[NMIX + 1];
#pragma unroll
    for (int j = 0; j < NMIX; ++j) {
        const float4 w = __ldg((const float4 *) (fn + (size_t) j * hcd + off));
        acc[j] = x.x * w.x + x.y * w.y + x.z * w.z + x.w * w.w;
    }
    acc[NMIX] = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
    __shared__ float red[THREADS / 32][NMIX + 1];
#pragma unroll
    for (int j = 0; j <= NMIX; ++j) {
        float v = acc[j];
        for (int o = 16; o; o /= 2) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[warp][j] = v;
    }
    __syncthreads();
    if (threadIdx.x <= NMIX) {
        float v = 0.f;
        for (int w = 0; w < THREADS / 32; ++w) v += red[w][threadIdx.x];
        part[((size_t) t * n_chunk + chunk) * (NMIX + 1) + threadIdx.x] = v;
    }
}

__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }

// one thread per token: chunk sums in order, then the split + Sinkhorn of hc_split_sinkhorn
__global__ void sk_final_kernel(const float * __restrict__ part, int n_chunk, int T, int hcd,
                                const float * __restrict__ scale, const float * __restrict__ base, float norm_eps,
                                float hc_eps, int iters, int cld, float * __restrict__ coef)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float m[NMIX + 1];
    for (int j = 0; j <= NMIX; ++j) m[j] = 0.f;
    for (int c = 0; c < n_chunk; ++c)
        for (int j = 0; j <= NMIX; ++j) m[j] += part[((size_t) t * n_chunk + c) * (NMIX + 1) + j];
    const float r = rsqrtf(m[NMIX] / hcd + norm_eps);
    float * o = coef + (size_t) t * cld;
    const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
    for (int c = 0; c < HC; ++c) {
        o[c] = sigm(m[c] * r * s0 + base[c]) + hc_eps;
        o[HC + c] = 2.f * sigm(m[HC + c] * r * s1 + base[HC + c]);
    }
    float cm[HC][HC];
    for (int i = 0; i < HC; ++i) {
        float mx = -INFINITY;
        for (int j = 0; j < HC; ++j) {
            cm[i][j] = m[2 * HC + i * HC + j] * r * s2 + base[2 * HC + i * HC + j];
            mx = fmaxf(mx, cm[i][j]);
        }
        float s = 0.f;
        for (int j = 0; j < HC; ++j) s += (cm[i][j] = expf(cm[i][j] - mx));
        for (int j = 0; j < HC; ++j) cm[i][j] = cm[i][j] / s + hc_eps;
    }
    auto col_norm = [&] {
        for (int j = 0; j < HC; ++j) {
            float s = 0.f;
            for (int i = 0; i < HC; ++i) s += cm[i][j];
            for (int i = 0; i < HC; ++i) cm[i][j] /= s + hc_eps;
        }
    };
    col_norm();
    for (int it = 1; it < iters; ++it) {
        for (int i = 0; i < HC; ++i) {
            float s = 0.f;
            for (int j = 0; j < HC; ++j) s += cm[i][j];
            for (int j = 0; j < HC; ++j) cm[i][j] /= s + hc_eps;
        }
        col_norm();
    }
    for (int i = 0; i < HC; ++i)
        for (int j = 0; j < HC; ++j) o[2 * HC + i * HC + j] = cm[i][j];
}

// one warp per token, the same arithmetic as sk_final_kernel in the same order (bit-identical, TRACKER #190): lane
// j < 25 sums its chunk column in chunk order; lanes 0-3 pre, 4-7 post, 8-23 the comb element (i, j) = lane - 8, whose
// row / column sums gather the four entries by shuffles and add them in index order like the serial loops
__global__ void __launch_bounds__(128) sk_final_warp_kernel(const float * __restrict__ part, int n_chunk, int T, int hcd,
                                                            const float * __restrict__ scale, const float * __restrict__ base,
                                                            float norm_eps, float hc_eps, int iters, int cld,
                                                            float * __restrict__ coef)
{
    const int t = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (t >= T) return;   // whole warps leave together (t is per warp)
    float m = 0.f;
    if (lane <= NMIX)
        for (int c = 0; c < n_chunk; ++c) m += part[((size_t) t * n_chunk + c) * (NMIX + 1) + lane];
    const float r = rsqrtf(__shfl_sync(0xffffffffu, m, NMIX) / hcd + norm_eps);
    float * o = coef + (size_t) t * cld;
    if (lane < HC) o[lane] = sigm(m * r * scale[0] + base[lane]) + hc_eps;
    else if (lane < 2 * HC) o[lane] = 2.f * sigm(m * r * scale[1] + base[lane]);
    const bool cb = lane >= 2 * HC && lane < NMIX;
    const int k = cb ? lane - 2 * HC : 0, i = k / HC, j = k % HC;
    float x = cb ? m * r * scale[2] + base[lane] : 0.f;
    auto row = [&](int jj) { return __shfl_sync(0xffffffffu, x, 2 * HC + i * HC + jj); };
    auto col = [&](int ii) { return __shfl_sync(0xffffffffu, x, 2 * HC + ii * HC + j); };
    float mx = -INFINITY;
#pragma unroll
    for (int jj = 0; jj < HC; ++jj) mx = fmaxf(mx, row(jj));
    x = expf(x - mx);
    float s = 0.f;
#pragma unroll
    for (int jj = 0; jj < HC; ++jj) s += row(jj);
    x = x / s + hc_eps;
    s = 0.f;
#pragma unroll
    for (int ii = 0; ii < HC; ++ii) s += col(ii);
    x /= s + hc_eps;
    for (int it = 1; it < iters; ++it) {
        s = 0.f;
#pragma unroll
        for (int jj = 0; jj < HC; ++jj) s += row(jj);
        x /= s + hc_eps;
        s = 0.f;
#pragma unroll
        for (int ii = 0; ii < HC; ++ii) s += col(ii);
        x /= s + hc_eps;
    }
    if (cb) o[lane] = x;
}

__global__ void sk_collapse_kernel(const float * __restrict__ res, const float * __restrict__ coef, int T, int d, int rld,
                                   int cld, float * __restrict__ y, half * __restrict__ y16)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x, n4 = (size_t) T * d / 4;
    if (i >= n4) return;
    const int t = (int) (i * 4 / d), c4 = (int) (i * 4 % d);
    const float * p = coef + (size_t) t * cld;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float4 r = *(const float4 *) (res + (size_t) t * rld + (size_t) c * d + c4);
        const float w = p[c];
        a.x += w * r.x; a.y += w * r.y; a.z += w * r.z; a.w += w * r.w;
    }
    *(float4 *) (y + (size_t) t * d + c4) = a;
    if (y16) {
        half2 * h = (half2 *) (y16 + (size_t) t * d + c4);
        h[0] = __floats2half2_rn(a.x, a.y);
        h[1] = __floats2half2_rn(a.z, a.w);
    }
}

__global__ void sk_post_kernel(const float * res, const float * __restrict__ x, const float * __restrict__ coef, int T,
                               int d, int rld, int cld, float * out)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x, n4 = (size_t) T * d / 4;
    if (i >= n4) return;
    const int t = (int) (i * 4 / d), c4 = (int) (i * 4 % d);
    const float * p = coef + (size_t) t * cld;
    float4 r[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) r[c] = *(const float4 *) (res + (size_t) t * rld + (size_t) c * d + c4);
    const float4 xv = *(const float4 *) (x + (size_t) t * d + c4);
#pragma unroll
    for (int j = 0; j < HC; ++j) {
        const float pj = p[HC + j];
        float4 a = make_float4(pj * xv.x, pj * xv.y, pj * xv.z, pj * xv.w);
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const float w = p[2 * HC + c * HC + j];
            a.x += w * r[c].x; a.y += w * r[c].y; a.z += w * r[c].z; a.w += w * r[c].w;
        }
        *(float4 *) (out + (size_t) t * rld + (size_t) j * d + c4) = a;
    }
}

__global__ void sk_expand_kernel(const float * __restrict__ emb, int T, int d, int rld, int cld,
                                 float * __restrict__ res, float * __restrict__ coef)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x, n4 = (size_t) T * d / 4;
    if (i < n4) {
        const int t = (int) (i * 4 / d), c4 = (int) (i * 4 % d);
        const float4 e = *(const float4 *) (emb + (size_t) t * d + c4);
#pragma unroll
        for (int c = 0; c < HC; ++c) *(float4 *) (res + (size_t) t * rld + (size_t) c * d + c4) = e;
    }
    if (i < (size_t) T * NMIX) coef[i / NMIX * cld + i % NMIX] = i % NMIX == 0 ? 1.f : 0.f;
}

void check(const SkDims & k)
{
    if (k.hc != HC || k.d % SPLIT) throw std::invalid_argument("hc::sk_*: needs hc == 4 and d % 1024 == 0");
    if (k.rld() < k.hc * k.d || k.rld() % 4 || k.cld() < NMIX) throw std::invalid_argument("hc::sk_*: row strides");
}

unsigned blocks4(int T, int d) { return (unsigned) (((size_t) T * d / 4 + 255) / 256); }

}  // namespace

size_t sk_mix_ws_bytes(int T, const SkDims & k)
{
    return sizeof(float) * (size_t) T * (k.hc * k.d / SPLIT) * (NMIX + 1);
}

void sk_mix(const float * res, const float * fn, const float * scale, const float * base, int T, const SkDims & k,
            void * ws, float * coef, cudaStream_t stream)
{
    check(k);
    if (T <= 0) return;
    if (T > 65535) throw std::invalid_argument("hc::sk_mix: T > 65535");
    const int hcd = k.hc * k.d, n_chunk = hcd / SPLIT;
    sk_part_kernel<<<dim3(n_chunk, T), THREADS, 0, stream>>>(res, fn, hcd, k.rld(), (float *) ws);
    static const bool serial = std::getenv("TRUSS_HC_FINAL_SERIAL") && std::atoi(std::getenv("TRUSS_HC_FINAL_SERIAL"));
    if (serial)   // the one-thread-per-token reference (A/B only)
        sk_final_kernel<<<(T + 127) / 128, 128, 0, stream>>>((const float *) ws, n_chunk, T, hcd, scale, base, k.norm_eps,
                                                             k.hc_eps, k.iters, k.cld(), coef);
    else
        sk_final_warp_kernel<<<(T + 3) / 4, 128, 0, stream>>>((const float *) ws, n_chunk, T, hcd, scale, base,
                                                              k.norm_eps, k.hc_eps, k.iters, k.cld(), coef);
    TRUSS_CUDA(cudaGetLastError());
}

void sk_collapse(const float * res, const float * coef, int T, const SkDims & k, float * y, half * y16,
                 cudaStream_t stream)
{
    check(k);
    if (T <= 0) return;
    sk_collapse_kernel<<<blocks4(T, k.d), 256, 0, stream>>>(res, coef, T, k.d, k.rld(), k.cld(), y, y16);
    TRUSS_CUDA(cudaGetLastError());
}

void sk_post(const float * res, const float * x, const float * coef, int T, const SkDims & k, float * res_out,
             cudaStream_t stream)
{
    check(k);
    if (T <= 0) return;
    sk_post_kernel<<<blocks4(T, k.d), 256, 0, stream>>>(res, x, coef, T, k.d, k.rld(), k.cld(), res_out);
    TRUSS_CUDA(cudaGetLastError());
}

void sk_expand(const float * emb, int T, const SkDims & k, float * res, float * coef_pre, cudaStream_t stream)
{
    check(k);
    if (T <= 0) return;
    const size_t n = std::max((size_t) T * k.d / 4, (size_t) T * NMIX);
    sk_expand_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(emb, T, k.d, k.rld(), k.cld(), res, coef_pre);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::hc
