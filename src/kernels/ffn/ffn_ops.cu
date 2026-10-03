#include "ffn_ops.cuh"

#include "core/cuda_check.h"

#include <stdexcept>

namespace truss::ffn {
namespace {

constexpr int THREADS = 256;

__device__ __forceinline__ float sigmoid(float v) { return 1.f / (1.f + __expf(-v)); }

unsigned grid(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

// one warp per token; expert e = 32 j + lane in p[j]
template <int PER_LANE>
__global__ void route_kernel(const float * logits, int T, int E, int k, int * ids, float * wts)
{
    const int t = blockIdx.x * (THREADS / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (t >= T) return;
    const float * l = logits + (size_t) t * E;
    float p[PER_LANE];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < PER_LANE; ++j) {
        p[j] = 32 * j + lane < E ? l[32 * j + lane] : -INFINITY;
        mx = fmaxf(mx, p[j]);
    }
    for (int o = 16; o; o /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < PER_LANE; ++j) sum += p[j] = expf(p[j] - mx);   // exp(-inf) = 0 for padding
    for (int o = 16; o; o /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, o);
#pragma unroll
    for (int j = 0; j < PER_LANE; ++j) p[j] /= sum;

    float kept = 0.f, my_w = 0.f;
    int my_id = -1;
    // (probability, expert) as one 64-bit key: probabilities are >= 0 (taken: -1, never picked while k <= E), so
    // their bits order like the values; the low word ~e makes a tie go to the lower expert. One shuffle per step.
    for (int s = 0; s < k; ++s) {
        unsigned long long key = 0;
#pragma unroll
        for (int j = 0; j < PER_LANE; ++j) {
            const int e = 32 * j + lane;
            if (e < E && p[j] >= 0.f) {
                const unsigned long long kj = (unsigned long long) __float_as_uint(p[j]) << 32 | (unsigned) ~e;
                key = kj > key ? kj : key;
            }
        }
        for (int o = 16; o; o /= 2) {
            const unsigned long long ok = __shfl_xor_sync(0xffffffffu, key, o);
            key = ok > key ? ok : key;
        }
        const float best = __uint_as_float((unsigned) (key >> 32));
        const int best_e = (int) ~(unsigned) key;
        kept += best;
        if (lane == s) my_id = best_e, my_w = best;
#pragma unroll
        for (int j = 0; j < PER_LANE; ++j)
            if (32 * j + lane == best_e) p[j] = -1.f;   // taken
    }
    if (lane < k) {
        ids[(size_t) t * k + lane] = my_id;
        wts[(size_t) t * k + lane] = my_w / fmaxf(kept, 6.103515625e-5f);
    }
}

__global__ void swiglu_kernel(const float * g, const float * u, int n, half * mid16)
{
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) mid16[i] = __float2half(g[i] * sigmoid(g[i]) * u[i]);
}

__global__ void shared_add_kernel(const float * routed, const float * y, const float * gate, int T, int d, float * out)
{
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t) T * d) out[i] = routed[i] + y[i] * sigmoid(gate[i / d]);
}

__global__ void add_kernel(float * y, const float * x, int n)
{
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

__global__ void count_kernel(const int * ids, int n, float * counts)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(counts + ids[i], 1.f);
}

}  // namespace

void add(float * y, const float * x, int n, cudaStream_t stream)
{
    add_kernel<<<grid(n), THREADS, 0, stream>>>(y, x, n);
    TRUSS_CUDA(cudaGetLastError());
}

void count(const int * ids, int n, float * counts, cudaStream_t stream)
{
    count_kernel<<<grid(n), THREADS, 0, stream>>>(ids, n, counts);
    TRUSS_CUDA(cudaGetLastError());
}

void route(const float * logits, int T, int E, int k, int * ids, float * wts, cudaStream_t stream)
{
    if (E % 32 || E > 1024 || k > 32 || k > E) throw std::invalid_argument("ffn::route: unsupported E / k");
    if (E <= 512) route_kernel<16><<<grid((int64_t) T * 32), THREADS, 0, stream>>>(logits, T, E, k, ids, wts);
    else route_kernel<32><<<grid((int64_t) T * 32), THREADS, 0, stream>>>(logits, T, E, k, ids, wts);
    TRUSS_CUDA(cudaGetLastError());
}

void swiglu(const float * g, const float * u, int n, half * mid16, cudaStream_t stream)
{
    swiglu_kernel<<<grid(n), THREADS, 0, stream>>>(g, u, n, mid16);
    TRUSS_CUDA(cudaGetLastError());
}

void shared_add(const float * routed, const float * y, const float * gate, int T, int d, float * out,
                cudaStream_t stream)
{
    shared_add_kernel<<<grid((int64_t) T * d), THREADS, 0, stream>>>(routed, y, gate, T, d, out);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::ffn
