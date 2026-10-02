// One warp per normalized row. Rows are strided over lanes (element i at lane i % 32), so rotary pair (p, p + 32)
// sits in one lane when ROPE_DIMS = 64.
#include "dsa_prepare.cuh"

#include "core/cuda_check.h"
#include "dsa_prefill.cuh"

#include <stdexcept>

namespace truss::dsa {
namespace {

constexpr int THREADS = 256, WARPS = THREADS / 32;

unsigned grid(int64_t rows) { return (unsigned) ((rows + WARPS - 1) / WARPS); }

template <class Shape> __global__ void rope_table_kernel(int pos0, int T, float base, float2 * cs)
{
    constexpr int HALF = Shape::ROPE_DIMS / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * HALF) return;
    const int t = i / HALF, p = i % HALF;
    const double theta = (double) (pos0 + t) * pow((double) base, -2.0 * p / Shape::ROPE_DIMS);
    cs[i] = make_float2((float) cos(theta), (float) sin(theta));
}

// x [N] (N = 32 E) in registers, element 32 j + lane in x[j]: RMSNorm * gamma, then rope with cs (nullable)
template <int E, int ROPE_DIMS>
__device__ void norm_rope(float (&x)[E], const float * gamma, const float2 * cs, float eps)
{
    static_assert(ROPE_DIMS == 64, "rotary pair in one lane");
    constexpr int N = 32 * E;
    const int lane = threadIdx.x % 32;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < E; ++j) acc += x[j] * x[j];
    for (int o = 16; o; o /= 2) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    const float inv = rsqrtf(acc / N + eps);
#pragma unroll
    for (int j = 0; j < E; ++j) x[j] *= inv * gamma[32 * j + lane];
    if (cs) {
        const float2 r = cs[lane];
        const float a = x[0], b = x[1];
        x[0] = a * r.x - b * r.y;
        x[1] = a * r.y + b * r.x;
    }
}

// norm_rope with this lane's rotation given directly
template <int E, int ROPE_DIMS> __device__ void norm_rope_at(float (&x)[E], const float * gamma, float2 r, float eps)
{
    static_assert(ROPE_DIMS == 64, "rotary pair in one lane");
    constexpr int N = 32 * E;
    const int lane = threadIdx.x % 32;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < E; ++j) acc += x[j] * x[j];
    for (int o = 16; o; o /= 2) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    const float inv = rsqrtf(acc / N + eps);
#pragma unroll
    for (int j = 0; j < E; ++j) x[j] *= inv * gamma[32 * j + lane];
    const float a = x[0], b = x[1];
    x[0] = a * r.x - b * r.y;
    x[1] = a * r.y + b * r.x;
}

template <class Shape>
__global__ void qkv_kernel(const float * qfull, const float * k, const float * v, const float * q_norm,
                           const float * k_norm, const float2 * cs, int pos0, int T, float eps, half * q16,
                           float * gate, KvCache kv)
{
    constexpr int H = Shape::H, HKV = Shape::HKV, D = Shape::D, E = D / 32, HALF = Shape::ROPE_DIMS / 2;
    const int64_t r = (int64_t) blockIdx.x * WARPS + threadIdx.x / 32;   // (token, head) of q, then of k, then v
    const int lane = threadIdx.x % 32;
    if (r >= (int64_t) T * (H + 2 * HKV)) return;
    float x[E];
    if (r < (int64_t) T * H) {
        const int64_t t = r / H;
        const float * src = qfull + r * 2 * D;
#pragma unroll
        for (int j = 0; j < E; ++j) x[j] = src[32 * j + lane], gate[r * D + 32 * j + lane] = src[D + 32 * j + lane];
        norm_rope<E, Shape::ROPE_DIMS>(x, q_norm, cs + t * HALF, eps);
#pragma unroll
        for (int j = 0; j < E; ++j) q16[r * D + 32 * j + lane] = __float2half(x[j]);
        return;
    }
    const int64_t rk = r - (int64_t) T * H, t = rk % ((int64_t) T * HKV) / HKV;
    const bool is_v = rk >= (int64_t) T * HKV;
    const int64_t row = rk % ((int64_t) T * HKV);   // (token, kv head)
    const float * src = (is_v ? v : k) + row * D;
#pragma unroll
    for (int j = 0; j < E; ++j) x[j] = src[32 * j + lane];
    if (!is_v) norm_rope<E, Shape::ROPE_DIMS>(x, k_norm, cs + t * HALF, eps);
    const int64_t cell_row = (int64_t) pos0 * HKV + row;   // (cell, kv head)
    if (!kv.int8()) {
        half * dst = (is_v ? kv.v16 : kv.k16) + cell_row * D;
#pragma unroll
        for (int j = 0; j < E; ++j) dst[32 * j + lane] = __float2half(x[j]);
        return;
    }
    // int8: lane holds dims 32 j + lane, so group g (dims 64 g .. 64 g + 63) is j = 2 g, 2 g + 1 across the warp
    static_assert(KV_GROUP == 64 && E % 2 == 0, "groups of 64 = two dims per lane");
    int8_t * code = (is_v ? kv.vq : kv.kq) + cell_row * D;
    half * scale = (is_v ? kv.vs : kv.ks) + cell_row * (D / KV_GROUP);
#pragma unroll
    for (int g = 0; g < E / 2; ++g) {
        float m = fmaxf(fabsf(x[2 * g]), fabsf(x[2 * g + 1]));
        for (int o = 16; o; o /= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const half sh = __float2half(m / 127.f);
        const float sc = __half2float(sh), inv = sc > 0.f ? 1.f / sc : 0.f;
        if (lane == 0) scale[g] = sh;
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int j = 2 * g + u;
            code[32 * j + lane] = (int8_t) fmaxf(-127.f, fminf(127.f, rintf(x[j] * inv)));
        }
    }
}

template <class Shape>
__global__ void index_kernel(const float * idx_q, const float * idx_k, const float * q_norm, const float * k_norm,
                             const float2 * cs, float rope_base, int pos0, int T, float eps, const float * partial,
                             half * idx_q16, half * idx_k_cache)
{
    constexpr int IH = Shape::IH, ID = Shape::ID, E = ID / 32, R = Shape::RATIO, HALF = Shape::ROPE_DIMS / 2;
    const int64_t r = (int64_t) blockIdx.x * WARPS + threadIdx.x / 32;   // (token, head) of q, then completed block
    const int lane = threadIdx.x % 32, b_lo = pos0 / R, nb = (pos0 + T) / R - b_lo;
    if (r >= (int64_t) T * IH + nb) return;
    float x[E];
    if (r < (int64_t) T * IH) {
#pragma unroll
        for (int j = 0; j < E; ++j) x[j] = idx_q[r * ID + 32 * j + lane];
        norm_rope<E, Shape::ROPE_DIMS>(x, q_norm, cs + r / IH * HALF, eps);
#pragma unroll
        for (int j = 0; j < E; ++j) idx_q16[r * ID + 32 * j + lane] = __float2half(x[j]);
        return;
    }
    const int b = b_lo + (int) (r - (int64_t) T * IH);
#pragma unroll
    for (int j = 0; j < E; ++j) {
        float acc = 0.f;
        for (int e = 0; e < R; ++e) {
            const int p = R * b + e;   // cells before the chunk come from the carried partial block
            acc += p >= pos0 ? idx_k[(int64_t) (p - pos0) * ID + 32 * j + lane] : partial[e * ID + 32 * j + lane];
        }
        x[j] = acc * (1.f / R);
    }
    // the block start may precede the chunk's rope table: its own fp64 angle
    const double theta = (double) (R * b) * pow((double) rope_base, -2.0 * lane / Shape::ROPE_DIMS);
    const float2 rot = make_float2((float) cos(theta), (float) sin(theta));
    norm_rope_at<E, Shape::ROPE_DIMS>(x, k_norm, rot, eps);
#pragma unroll
    for (int j = 0; j < E; ++j) idx_k_cache[(int64_t) b * ID + 32 * j + lane] = __float2half(x[j]);
}

// partial <- raw keys of the block open after the chunk (cells R floor((pos0 + T) / R) .. pos0 + T - 1)
template <class Shape> __global__ void partial_kernel(const float * idx_k, int pos0, int T, float * partial)
{
    constexpr int ID = Shape::ID, R = Shape::RATIO;
    const int start = (pos0 + T) / R * R;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // (cell of the open block, dim)
    if (i >= (R - 1) * ID) return;
    const int p = start + i / ID;
    if (p >= pos0 && p < pos0 + T) partial[i] = idx_k[(int64_t) (p - pos0) * ID + i % ID];
}

}  // namespace

template <class Shape> void rope_table(int pos0, int T, float base, float2 * cs, cudaStream_t stream)
{
    const int n = T * Shape::ROPE_DIMS / 2;
    rope_table_kernel<Shape><<<(n + 255) / 256, 256, 0, stream>>>(pos0, T, base, cs);
    TRUSS_CUDA(cudaGetLastError());
}

template <class Shape>
void prepare_qkv(const float * qfull, const float * k, const float * v, const float * q_norm, const float * k_norm,
                 const float2 * cs, int pos0, int T, float eps, half * q16, float * gate, half * k_cache,
                 half * v_cache, cudaStream_t stream)
{
    KvCache kv;
    kv.k16 = k_cache, kv.v16 = v_cache;
    prepare_qkv<Shape>(qfull, k, v, q_norm, k_norm, cs, pos0, T, eps, q16, gate, kv, stream);
}

template <class Shape>
void prepare_qkv(const float * qfull, const float * k, const float * v, const float * q_norm, const float * k_norm,
                 const float2 * cs, int pos0, int T, float eps, half * q16, float * gate, const KvCache & kv,
                 cudaStream_t stream)
{
    static_assert(Shape::D % 32 == 0, "head dim in lane strides");
    const int64_t rows = (int64_t) T * (Shape::H + 2 * Shape::HKV);
    qkv_kernel<Shape><<<grid(rows), THREADS, 0, stream>>>(qfull, k, v, q_norm, k_norm, cs, pos0, T, eps, q16, gate, kv);
    TRUSS_CUDA(cudaGetLastError());
}

template <class Shape>
void prepare_index(const float * idx_q, const float * idx_k, const float * q_norm, const float * k_norm,
                   const float2 * cs, float rope_base, int pos0, int T, float eps, float * partial, half * idx_q16,
                   half * idx_k_cache, cudaStream_t stream)
{
    static_assert(Shape::ID % 32 == 0, "indexer head dim in lane strides");
    const int nb = (pos0 + T) / Shape::RATIO - pos0 / Shape::RATIO;
    const int64_t rows = (int64_t) T * Shape::IH + nb;
    index_kernel<Shape><<<grid(rows), THREADS, 0, stream>>>(idx_q, idx_k, q_norm, k_norm, cs, rope_base, pos0, T, eps,
                                                            partial, idx_q16, idx_k_cache);
    const int n = (Shape::RATIO - 1) * Shape::ID;
    partial_kernel<Shape><<<(n + 255) / 256, 256, 0, stream>>>(idx_k, pos0, T, partial);
    TRUSS_CUDA(cudaGetLastError());
}

template <class Shape> void carry_partial(const float * idx_k, int pos0, int T, float * partial, cudaStream_t stream)
{
    const int n = (Shape::RATIO - 1) * Shape::ID;
    partial_kernel<Shape><<<(n + 255) / 256, 256, 0, stream>>>(idx_k, pos0, T, partial);
    TRUSS_CUDA(cudaGetLastError());
}

template void carry_partial<FlashNext>(const float *, int, int, float *, cudaStream_t);
template void rope_table<FlashNext>(int, int, float, float2 *, cudaStream_t);
template void prepare_qkv<FlashNext>(const float *, const float *, const float *, const float *, const float *,
                                     const float2 *, int, int, float, half *, float *, half *, half *, cudaStream_t);
template void prepare_qkv<FlashNext>(const float *, const float *, const float *, const float *, const float *,
                                     const float2 *, int, int, float, half *, float *, const KvCache &, cudaStream_t);
template void prepare_index<FlashNext>(const float *, const float *, const float *, const float *, const float2 *,
                                       float, int, int, float, float *, half *, half *, cudaStream_t);

}  // namespace truss::dsa
