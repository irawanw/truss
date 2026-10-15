// Sigmoid router with a selection bias (HF Lfm2MoeTopKRouter, DeepSeek-V3-style "aux-loss-free" routing):
//   s = sigmoid(logit); the top k of s + bias choose the experts (ties: the lower id); the weights are the chosen s,
//   divided by (their sum + 1e-6) when renorm, times scale.
// One warp per row; ids / weights in descending selection key, as ffn::route.
#include "kernels/ffn/ffn_ops.cuh"

#include "core/cuda_check.h"

#include <cmath>
#include <stdexcept>

namespace truss::ffn {

namespace {

template <int PER_LANE>
__global__ void route_sigmoid_kernel(const float * logits, const float * bias, int T, int E, int k, bool renorm,
                                     float scale, int * ids, float * wts)
{
    const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (t >= T) return;
    const float * l = logits + (size_t) t * E;
    float s[PER_LANE], key[PER_LANE];
#pragma unroll
    for (int j = 0; j < PER_LANE; ++j) {
        const int e = 32 * j + lane;
        s[j] = e < E ? 1.f / (1.f + expf(-l[e])) : 0.f;
        key[j] = e < E ? s[j] + (bias ? bias[e] : 0.f) : -INFINITY;
    }
    int my_id = -1;
    float my_s = 0.f, sum = 0.f;
    for (int r = 0; r < k; ++r) {
        // the best (key, lowest id) of the warp: keys may be negative, so compare floats, ties by id
        float bk = -INFINITY;
        int be = 1 << 30;
#pragma unroll
        for (int j = 0; j < PER_LANE; ++j) {
            const int e = 32 * j + lane;
            if (key[j] > bk || (key[j] == bk && e < be)) bk = key[j], be = e;
        }
        for (int o = 16; o; o /= 2) {
            const float ok = __shfl_xor_sync(0xffffffffu, bk, o);
            const int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (ok > bk || (ok == bk && oe < be)) bk = ok, be = oe;
        }
        // the owner of be retires it and broadcasts its sigmoid
        float bs = 0.f;
#pragma unroll
        for (int j = 0; j < PER_LANE; ++j)
            if (32 * j + lane == be) bs = s[j], key[j] = -INFINITY;
        for (int o = 16; o; o /= 2) bs += __shfl_xor_sync(0xffffffffu, bs, o);   // one lane held it, the rest 0
        if (lane == r) my_id = be, my_s = bs;
        sum += bs;
    }
    if (lane < k) {
        ids[(size_t) t * k + lane] = my_id;
        wts[(size_t) t * k + lane] = (renorm ? my_s / (sum + 1e-6f) : my_s) * scale;
    }
}

}  // namespace

void route_sigmoid(const float * logits, const float * bias, int T, int E, int k, bool renorm, float scale, int * ids,
                   float * wts, cudaStream_t stream)
{
    if (E % 32 || E > 1024 || k > 32 || k > E) throw std::invalid_argument("ffn::route_sigmoid: unsupported E / k");
    if (T <= 0) return;
    const int warps = 8, blocks = (T + warps - 1) / warps;
    if (E <= 128) route_sigmoid_kernel<4><<<blocks, 32 * warps, 0, stream>>>(logits, bias, T, E, k, renorm, scale, ids, wts);
    else if (E <= 512) route_sigmoid_kernel<16><<<blocks, 32 * warps, 0, stream>>>(logits, bias, T, E, k, renorm, scale, ids, wts);
    else route_sigmoid_kernel<32><<<blocks, 32 * warps, 0, stream>>>(logits, bias, T, E, k, renorm, scale, ids, wts);
    TRUSS_CUDA(cudaGetLastError());
}

// The lone-router fast path (B2): this file's route_sigmoid tail folded into the f32 router projection, one launch.
// The projection is f32_gemv_multi's WPO=2 arithmetic verbatim (the same (block, warp) -> (output, wi) mapping, the
// same float4 loop, xor tree and wi combine), so logits, ids and wts are bit-identical to the two-kernel path; each
// block writes its 2 logits, fences, bumps arrive, and the last block runs the tail (route_publish's pattern: the
// word resets itself, zeroed once by the caller, graph-safe). T == 1 only (decode); every other shape keeps the old
// path, so prefill / verify / the hint batch never see this kernel.
template <int PER_LANE>
__device__ void route_tail(const float * logits, const float * bias, int E, int k, bool renorm, float scale, int * ids,
                           float * wts)
{
    const int lane = threadIdx.x % 32;
    float s[PER_LANE], key[PER_LANE];
#pragma unroll
    for (int j = 0; j < PER_LANE; ++j) {
        const int e = 32 * j + lane;
        s[j] = e < E ? 1.f / (1.f + expf(-logits[e])) : 0.f;
        key[j] = e < E ? s[j] + (bias ? bias[e] : 0.f) : -INFINITY;
    }
    int my_id = -1;
    float my_s = 0.f, sum = 0.f;
    for (int r = 0; r < k; ++r) {
        float bk = -INFINITY;
        int be = 1 << 30;
#pragma unroll
        for (int j = 0; j < PER_LANE; ++j) {
            const int e = 32 * j + lane;
            if (key[j] > bk || (key[j] == bk && e < be)) bk = key[j], be = e;
        }
        for (int o = 16; o; o /= 2) {
            const float ok = __shfl_xor_sync(0xffffffffu, bk, o);
            const int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (ok > bk || (ok == bk && oe < be)) bk = ok, be = oe;
        }
        float bs = 0.f;
#pragma unroll
        for (int j = 0; j < PER_LANE; ++j)
            if (32 * j + lane == be) bs = s[j], key[j] = -INFINITY;
        for (int o = 16; o; o /= 2) bs += __shfl_xor_sync(0xffffffffu, bs, o);
        if (lane == r) my_id = be, my_s = bs;
        sum += bs;
    }
    if (lane < k) {
        ids[lane] = my_id;
        wts[lane] = (renorm ? my_s / (sum + 1e-6f) : my_s) * scale;
    }
}

__global__ void __launch_bounds__(128)
route_gemv_kernel(const float * W, int in, const float * bias, const float * __restrict__ x, int E, int k,
                  float * logits, int * ids, float * wts, unsigned * arrive)
{
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 2 + warp / 2, wi = warp % 2;
    const float4 * w = reinterpret_cast<const float4 *>(W + (size_t) o * in);
    float acc = 0.f;
    for (int i = lane + 32 * wi; i < in / 4; i += 64) {
        const float4 a = __ldg(w + i);
        const float4 b = reinterpret_cast<const float4 *>(x)[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    __shared__ float part[4];
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (lane == 0 && wi == 0) logits[o] = part[warp] + part[warp + 1];
    __syncthreads();
    __threadfence();
    __shared__ bool last;
    if (threadIdx.x == 0) last = atomicAdd(arrive, 1u) == (unsigned) gridDim.x - 1u;
    __syncthreads();
    if (last && warp == 0) {
        if (E <= 128) route_tail<4>(logits, bias, E, k, true, 1.f, ids, wts);
        else if (E <= 512) route_tail<16>(logits, bias, E, k, true, 1.f, ids, wts);
        else route_tail<32>(logits, bias, E, k, true, 1.f, ids, wts);
        if (threadIdx.x == 0) *arrive = 0;
    }
}

void route_sigmoid_gemv(const float * W, int in, const float * bias, const float * x, int E, int k,
                        float * logits, int * ids, float * wts, unsigned * arrive, cudaStream_t stream)
{
    if (E % 32 || E > 1024 || k > 32 || k > E) throw std::invalid_argument("ffn::route_sigmoid_gemv: unsupported E / k");
    if (in < 2048 || in % 128) throw std::invalid_argument("ffn::route_sigmoid_gemv: needs the WPO=2 shape (in >= 2048)");
    route_gemv_kernel<<<(E + 1) / 2, 128, 0, stream>>>(W, in, bias, x, E, k, logits, ids, wts, arrive);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::ffn
