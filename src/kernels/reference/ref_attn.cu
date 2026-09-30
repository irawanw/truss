#include "core/cuda_check.h"
#include "kernels/reference/ref.cuh"

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
                      int Hq, int Hkv, int d, float scale, cudaStream_t s)
{
    attn_kernel<<<dim3(T, Hq), 256, T * sizeof(float), s>>>(q, k, v, sel, out, T, Hq, Hkv, d, scale);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::ref
