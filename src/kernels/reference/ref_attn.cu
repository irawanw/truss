#include "core/cuda_check.h"
#include "kernels/reference/ref.cuh"

#include <cuda_fp16.h>

#include <cmath>

namespace truss::ref {

namespace {

__global__ void rope_kernel(float * x, int rows, int heads, int hd, int n_rot, const int * pos, float base)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // (row, head, pair)
    const int half = n_rot / 2;
    if (i >= rows * heads * half) return;
    const int r = i / (heads * half), h = (i / half) % heads, p = i % half;
    const double theta = (double) pos[r] * pow((double) base, -2.0 * p / n_rot);
    const float c = (float) cos(theta), s = (float) sin(theta);
    float * v = x + ((size_t) r * heads + h) * hd;
    const float a = v[p], b = v[p + half];
    v[p] = a * c - b * s;
    v[p + half] = a * s + b * c;
}

// one block per query: block scores into shared memory, rank-count selection, then the cell mask
__global__ void select_kernel(const float * q, const float * kb, int T, int heads, int d, int r, int top_blocks,
                              uint8_t * sel)
{
    extern __shared__ float sc[];
    const int t = blockIdx.x;
    const int nb = (t + 1) / r;   // complete blocks visible to t
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        float acc = 0.f;
        for (int h = 0; h < heads; ++h) {
            float dot = 0.f;
            for (int i = 0; i < d; ++i) dot += q[((size_t) t * heads + h) * d + i] * kb[(size_t) b * d + i];
            acc += fmaxf(dot, 0.f);
        }
        sc[b] = acc;
    }
    __syncthreads();
    uint8_t * row = sel + (size_t) t * T;
    for (int j = threadIdx.x; j < T; j += blockDim.x) row[j] = 0;
    __syncthreads();
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        int better = 0;   // blocks ranked before b: higher score, or equal score and later
        for (int o = 0; o < nb; ++o) better += sc[o] > sc[b] || (sc[o] == sc[b] && o > b);
        if (better < top_blocks)
            for (int e = 0; e < r; ++e) row[b * r + e] = 1;
    }
    for (int j = nb * r + threadIdx.x; j <= t; j += blockDim.x) row[j] = 1;   // tail
}

// one block per (query, head); fixed-order softmax in fp64 accumulators over the selected cells
__global__ void attn_kernel(const float * q, const float * k, const float * v, const uint8_t * sel, float * out, int T,
                            int Hq, int Hkv, int d, float scale)
{
    extern __shared__ float sm[];   // [T] scores
    const int t = blockIdx.x, h = blockIdx.y, hk = h / (Hq / Hkv);
    const float * qv = q + ((size_t) t * Hq + h) * d;
    const uint8_t * row = sel + (size_t) t * T;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float dot = -INFINITY;
        if (row[j]) {
            dot = 0.f;
            for (int i = 0; i < d; ++i) dot += qv[i] * k[((size_t) j * Hkv + hk) * d + i];
            dot *= scale;
        }
        sm[j] = dot;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int j = 0; j < T; ++j) mx = fmaxf(mx, sm[j]);
    double den = 0;
    for (int j = 0; j < T; ++j) den += row[j] ? exp((double) (sm[j] - mx)) : 0.0;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        double acc = 0;
        for (int j = 0; j < T; ++j)
            if (row[j]) acc += exp((double) (sm[j] - mx)) * v[((size_t) j * Hkv + hk) * d + i];
        out[((size_t) t * Hq + h) * d + i] = (float) (acc / den);
    }
}

__device__ __forceinline__ float r16(float x) { return __half2float(__float2half(x)); }

// same layout as attn_kernel, llama-paw's flash-attention numerics (see ref.cuh)
__global__ void attn_llama_kernel(const float * q, const float * k, const float * v, const uint8_t * sel, float * out,
                                  int T, int Hq, int Hkv, int d, float scale)
{
    constexpr int TILE = 64, MMA_K = 16;
    constexpr float MAX_OFFSET = 3.0f * 0.6931f;   // FATTN_KQ_MAX_OFFSET
    extern __shared__ float sm[];                    // [T] scores
    const int t = blockIdx.x, h = blockIdx.y, hk = h / (Hq / Hkv);
    const float * qv = q + ((size_t) t * Hq + h) * d;
    const uint8_t * row = sel + (size_t) t * T;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float dot = -INFINITY;
        if (row[j]) {
            dot = 0.f;
            for (int i = 0; i < d; ++i) dot += r16(qv[i] * scale) * r16(k[((size_t) j * Hkv + hk) * d + i]);
        }
        sm[j] = dot;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float mx = -INFINITY, rowsum = 0.f, acc = 0.f;   // acc holds an fp16 value
        for (int j0 = 0; j0 < T; j0 += TILE) {
            const int j1 = min(T, j0 + TILE);
            float mx_new = mx;
            for (int j = j0; j < j1; ++j) mx_new = fmaxf(mx_new, sm[j] + MAX_OFFSET);
            if (mx_new == -INFINITY) continue;
            const float resc = mx == -INFINITY ? 0.f : expf(mx - mx_new);
            rowsum *= resc;
            acc = r16(acc * r16(resc));
            mx = mx_new;
            for (int g = j0; g < j1; g += MMA_K) {
                float part = 0.f;
                for (int j = g; j < min(j1, g + MMA_K); ++j) {
                    if (!row[j]) continue;
                    const float p = expf(sm[j] - mx);
                    rowsum += p;
                    part += r16(p) * r16(v[((size_t) j * Hkv + hk) * d + i]);
                }
                acc = r16(acc + part);
            }
        }
        out[((size_t) t * Hq + h) * d + i] = acc / rowsum;
    }
}

}  // namespace

void rope_neox(float * x, int rows, int heads, int hd, int n_rot, const int * pos, float base, cudaStream_t s)
{
    const int n = rows * heads * (n_rot / 2);
    rope_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, rows, heads, hd, n_rot, pos, base);
    TRUSS_CUDA(cudaGetLastError());
}

void qsa_select(const float * idx_q, const float * idx_k, int T, int heads, int d, int r, int top_blocks,
                uint8_t * sel, cudaStream_t s)
{
    select_kernel<<<T, 256, (T / r + 1) * sizeof(float), s>>>(idx_q, idx_k, T, heads, d, r, top_blocks, sel);
    TRUSS_CUDA(cudaGetLastError());
}

void masked_attention(const float * q, const float * k, const float * v, const uint8_t * sel, float * out, int T,
                      int Hq, int Hkv, int d, float scale, cudaStream_t s, Numerics num)
{
    if (num == Numerics::LLAMA)
        attn_llama_kernel<<<dim3(T, Hq), 256, T * sizeof(float), s>>>(q, k, v, sel, out, T, Hq, Hkv, d, scale);
    else
        attn_kernel<<<dim3(T, Hq), 256, T * sizeof(float), s>>>(q, k, v, sel, out, T, Hq, Hkv, d, scale);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::ref
