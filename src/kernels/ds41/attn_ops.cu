// DeepSeek-V4.1 attention elementwise ops (attn_ops.cuh). All memory-bound and tiny next to the projections.
#include "kernels/ds41/attn_ops.cuh"

#include "core/cuda_check.h"

#include <cmath>
#include <stdexcept>

namespace truss::ds41 {

std::vector<float> rope_inv_freq(int rd, float base, int orig_ctx, float factor, float beta_fast, float beta_slow)
{
    // fp32 like torch: base ** (arange(0, dim, 2) / dim) on a float32 tensor, then 1 / that
    std::vector<float> f(rd / 2);
    for (int i = 0; i < rd / 2; ++i) f[i] = 1.0f / powf(base, (float) (2 * i) / (float) rd);
    if (orig_ctx > 0) {
        auto corrected = [&] (double rot) { return rd * std::log(orig_ctx / (rot * 2 * M_PI)) / (2 * std::log(base)); };
        const int low = std::max((int) std::floor(corrected(beta_fast)), 0);
        const int high = std::min((int) std::ceil(corrected(beta_slow)), rd - 1);
        const float span = std::max((float) (high - low), 1e-3f);
        for (int i = 0; i < rd / 2; ++i) {
            const float ramp = std::min(std::max(((float) i - (float) low) / span, 0.0f), 1.0f);
            const float smooth = 1.0f - ramp;
            f[i] = f[i] / factor * (1.0f - smooth) + f[i] * smooth;
        }
    }
    return f;
}

namespace {

__global__ void rope_tail_kernel(float * x, int rows, int heads, int D, int rd, int pos_base, int pos_mul,
                                 const float * __restrict__ inv_freq, float sign)
{
    const int half_rd = rd / 2;
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x, n = (size_t) rows * heads * half_rd;
    if (i >= n) return;
    const int k = (int) (i % half_rd), h = (int) (i / half_rd % heads), r = (int) (i / half_rd / heads);
    // fp32 angle exactly as torch.outer(arange(seqlen), freqs): position converted to float, one fp32 product
    const float ang = (float) (pos_base + r * pos_mul) * inv_freq[k];
    float s, c;
    sincosf(ang, &s, &c);
    s *= sign;
    float * p = x + ((size_t) r * heads + h) * D + (D - rd) + 2 * k;
    const float x0 = p[0], x1 = p[1];
    p[0] = x0 * c - x1 * s;
    p[1] = x0 * s + x1 * c;
}

template <class X>
__global__ void ring_write_kernel(X * ring, int R, const float * __restrict__ src, int W, int pos0, int T, int first)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x, n = (size_t) (T - first) * W;
    if (i >= n) return;
    const int t = first + (int) (i / W), c = (int) (i % W);
    const float v = src[(size_t) t * W + c];
    X * dst = ring + (size_t) ((pos0 + t) % R) * W + c;
    if constexpr (sizeof(X) == 2) *dst = __float2half_rn(v);
    else *dst = v;
}

// one thread per (group, dim): softmax over the ratio positions of the score, weighted sum of kv (fixed order)
__global__ void pool_kernel(const float * __restrict__ kvc, const float * __restrict__ scc,
                            const float * __restrict__ raw_kv, const float * __restrict__ raw_sc, int R, int D,
                            int ratio, int pos0, int j0, int n, float * __restrict__ out)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * D) return;
    const int g = (int) (i / D), d = (int) (i % D), j = j0 + g;
    float kv[8], sc[8], mx = -INFINITY;
    for (int r = 0; r < ratio; ++r) {
        const int p = j * ratio + r;
        const size_t at = p >= pos0 ? (size_t) (p - pos0) * D + d : (size_t) (p % R) * D + d;
        kv[r] = p >= pos0 ? kvc[at] : raw_kv[at];
        sc[r] = p >= pos0 ? scc[at] : raw_sc[at];
        mx = fmaxf(mx, sc[r]);
    }
    float den = 0.f, acc = 0.f;
    for (int r = 0; r < ratio; ++r) den += (sc[r] = expf(sc[r] - mx));
    for (int r = 0; r < ratio; ++r) acc += kv[r] * (sc[r] / den);
    out[(size_t) g * D + d] = acc;
}

__global__ void pick_groups_kernel(const float * __restrict__ y, int T, int G, int C, float * __restrict__ out)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x, n = (size_t) T * G * C;
    if (i >= n) return;
    const int t = (int) (i / ((size_t) G * C)), g = (int) (i / C % G), c = (int) (i % C);
    out[i] = y[((size_t) t * G + g) * G * C + (size_t) g * C + c];
}

__global__ void to_half_kernel(const float * __restrict__ x, size_t n, half * __restrict__ y)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}

unsigned blocks(size_t n) { return (unsigned) ((n + 255) / 256); }

}  // namespace

void rope_tail(float * x, int rows, int heads, int D, int rd, int pos_base, int pos_mul, const float * inv_freq,
               bool inverse, cudaStream_t stream)
{
    if (rd % 2 || rd > D) throw std::invalid_argument("ds41::rope_tail: rd");
    const size_t n = (size_t) rows * heads * (rd / 2);
    if (!n) return;
    rope_tail_kernel<<<blocks(n), 256, 0, stream>>>(x, rows, heads, D, rd, pos_base, pos_mul, inv_freq,
                                                     inverse ? -1.f : 1.f);
    TRUSS_CUDA(cudaGetLastError());
}

void ring_write(half * ring, int R, const float * src, int W, int pos0, int T, cudaStream_t stream)
{
    const int first = T > R ? T - R : 0;
    if (T <= 0) return;
    ring_write_kernel<half><<<blocks((size_t) (T - first) * W), 256, 0, stream>>>(ring, R, src, W, pos0, T, first);
    TRUSS_CUDA(cudaGetLastError());
}

void ring_write_f32(float * ring, int R, const float * src, int W, int pos0, int T, cudaStream_t stream)
{
    const int first = T > R ? T - R : 0;
    if (T <= 0) return;
    ring_write_kernel<float><<<blocks((size_t) (T - first) * W), 256, 0, stream>>>(ring, R, src, W, pos0, T, first);
    TRUSS_CUDA(cudaGetLastError());
}

void pool(const float * kv, const float * sc, const float * raw_kv, const float * raw_sc, int R, int D, int ratio,
          int pos0, int j0, int n, float * out, cudaStream_t stream)
{
    if (ratio < 2 || ratio > 8) throw std::invalid_argument("ds41::pool: ratio 2..8");
    if (j0 * ratio < pos0 - R) throw std::invalid_argument("ds41::pool: group starts before the raw ring");
    if (n <= 0) return;
    pool_kernel<<<blocks((size_t) n * D), 256, 0, stream>>>(kv, sc, raw_kv, raw_sc, R, D, ratio, pos0, j0, n, out);
    TRUSS_CUDA(cudaGetLastError());
}

void pick_groups(const float * y, int T, int G, int C, float * out, cudaStream_t stream)
{
    const size_t n = (size_t) T * G * C;
    if (!n) return;
    pick_groups_kernel<<<blocks(n), 256, 0, stream>>>(y, T, G, C, out);
    TRUSS_CUDA(cudaGetLastError());
}

void to_half(const float * x, size_t n, half * y, cudaStream_t stream)
{
    if (!n) return;
    to_half_kernel<<<blocks(n), 256, 0, stream>>>(x, n, y);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::ds41
