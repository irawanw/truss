#include "ffn_ops.cuh"

#include "core/cuda_check.h"

#include <cstdlib>
#include <stdexcept>

namespace truss::ffn {
namespace {

constexpr int THREADS = 256;

__device__ __forceinline__ float sigmoid(float v) { return 1.f / (1.f + __expf(-v)); }

unsigned grid(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

// one warp's top-k of one token's logits; expert e = 32 j + lane in p[j]. Lanes 0..k-1 end with the winners'
// (id, weight), in descending probability. route_kernel and the doorbell's fused route+publish share it so the two
// paths cannot drift apart: the ids and weights the driver splits on have to be the same bits either way.
template <int PER_LANE>
__device__ __forceinline__ void topk_row(const float * l, int E, int k, int & my_id, float & my_w)
{
    const int lane = threadIdx.x % 32;
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

    float kept = 0.f;
    int my_id_ = -1;
    my_w = 0.f;
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
        if (lane == s) my_id_ = best_e, my_w = best;
#pragma unroll
        for (int j = 0; j < PER_LANE; ++j)
            if (32 * j + lane == best_e) p[j] = -1.f;   // taken
    }
    my_id = my_id_;
    if (lane < k) my_w /= fmaxf(kept, 6.103515625e-5f);
}

// one warp per token
template <int PER_LANE>
__global__ void route_kernel(const float * logits, int T, int E, int k, int * ids, float * wts)
{
    const int t = blockIdx.x * (THREADS / 32) + threadIdx.x / 32;
    int my_id;
    float my_w;
    topk_row<PER_LANE>(logits + (size_t) t * E, E, k, my_id, my_w);
    const int lane = threadIdx.x % 32;
    if (lane < k) {
        ids[(size_t) t * k + lane] = my_id;
        wts[(size_t) t * k + lane] = my_w;
    }
}

// The doorbell's routing and publish in one launch (TRUSS_ROUTE_PUBLISH, Phase 2). Block 0's warps run the top-k:
// T rows of this layer's logits, then (pre-gating hint) T rows of the next layer's, and write ids and weights to
// device + mapped memory and the prediction to mapped memory. Blocks 1..copy_ctas stream the FFN input rows into
// mapped memory: publish_kernel did the same 80 KB from ONE 1024-thread CTA, which the link serves at a quarter of
// the width a handful of CTAs reach, and did it after the routing instead of next to it. Every CTA fences before it
// arrives on `arrive` (a device word, zero before the first launch and reset by the last arriver, so the count is
// per launch and the CTA count may change between rows); the last one rings the doorbell. The driver therefore
// still sees every byte it reads before the seq word, and it sees it ~2 launches earlier.
constexpr int RP_THREADS = 512;   // block 0: 16 warps = T rows of logits + T of prediction, T <= cpu::MAX_ROWS

template <int PER_LANE>
__global__ void __launch_bounds__(RP_THREADS) route_publish_kernel(
        const float * logits, const float * pred_logits, int T, int E, int k, int * ids, float * wts,
        const float4 * x, int n4, int * dst_ids, int * dst_pred, float * dst_wts, float4 * dst_x,
        volatile int * doorbell, int seq, unsigned * arrive, int copy_ctas)
{
    if (blockIdx.x == 0) {
        const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
        const bool pr = pred_logits && w >= T;   // the upper warps predict the next layer's routing
        const int t = pr ? w - T : w;
        if (t < T) {
            int my_id;
            float my_w;
            topk_row<PER_LANE>((pr ? pred_logits : logits) + (size_t) t * E, E, k, my_id, my_w);
            if (lane < k) {
                const size_t i = (size_t) t * k + lane;
                if (pr)
                    dst_pred[i] = my_id;   // the prediction's weights were never read; the ids are the point
                else {
                    ids[i] = my_id;
                    wts[i] = my_w;
                    dst_ids[i] = my_id;
                    dst_wts[i] = my_w;
                }
            }
        }
    } else {
        const int per = (n4 + copy_ctas - 1) / copy_ctas;
        const int lo = (blockIdx.x - 1) * per, hi = min(n4, lo + per);
        for (int i = lo + threadIdx.x; i < hi; i += RP_THREADS) dst_x[i] = x[i];
    }
    __threadfence_system();   // the data reaches host memory before the doorbell does
    __syncthreads();
    // doorbell == nullptr: the caller publishes the rows with the copy engine and rings behind that copy, so this
    // launch only writes and fences (route_publish_words).
    if (doorbell && threadIdx.x == 0) {
        const unsigned old = atomicAdd(arrive, 1u);
        if (old == (unsigned) copy_ctas) {   // the last of 1 + copy_ctas CTAs
            *arrive = 0;
            __threadfence_system();
            *doorbell = seq;
        }
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

// How many CTAs publish a layer's FFN rows. Measured on an idle GPU (tk-bench-router, 8 rows, 80 KB): one CTA
// pushes it at 4.1-4.4 GB/s, and that is the link's ceiling - the fused launch takes 20.0 us with one copy CTA and
// 23.1 / 25.6 / 26.8 / 28.6 / 33.2 us with 2 / 4 / 8 / 16 / 32, because every extra CTA pays a system fence that
// drains its own write queue over the same link. So: one. TRUSS_PUB_CTAS overrides it, for the measurement.
int publish_ctas(int n_x)
{
    if (n_x <= 0) return 0;
    int ctas = 1;
    if (const char * e = std::getenv("TRUSS_PUB_CTAS")) {
        const int v = std::atoi(e);
        if (v > 0) ctas = v;
    }
    return ctas;
}

void route_publish(const float * logits, const float * pred_logits, int T, int E, int k, int * ids, float * wts,
                   const float * x, int n_x, int * dst_ids, int * dst_pred, float * dst_wts, float * dst_x,
                   volatile int * doorbell, int seq, unsigned * arrive, int copy_ctas, cudaStream_t stream)
{
    if (E % 32 || E > 1024 || k > 32 || k > E) throw std::invalid_argument("ffn::route_publish: unsupported E / k");
    if (T < 1 || T > RP_THREADS / 64) throw std::invalid_argument("ffn::route_publish: rows must be 1..8");
    if (n_x < 0 || n_x % 4 || (n_x && !x)) throw std::invalid_argument("ffn::route_publish: bad FFN rows");
    if (pred_logits && !dst_pred) throw std::invalid_argument("ffn::route_publish: nowhere for the prediction");
    if (!!doorbell != !!arrive) throw std::invalid_argument("ffn::route_publish: doorbell and arrive go together");
    const int n4 = n_x / 4;
    if (n4 > 0 && copy_ctas < 1) copy_ctas = 1;
    if (n4 == 0) copy_ctas = 0;
    const unsigned blocks = (unsigned) (1 + copy_ctas);
    if (E <= 512)
        route_publish_kernel<16><<<blocks, RP_THREADS, 0, stream>>>(
                logits, pred_logits, T, E, k, ids, wts, reinterpret_cast<const float4 *>(x), n4, dst_ids, dst_pred,
                dst_wts, reinterpret_cast<float4 *>(dst_x), doorbell, seq, arrive, copy_ctas);
    else
        route_publish_kernel<32><<<blocks, RP_THREADS, 0, stream>>>(
                logits, pred_logits, T, E, k, ids, wts, reinterpret_cast<const float4 *>(x), n4, dst_ids, dst_pred,
                dst_wts, reinterpret_cast<float4 *>(dst_x), doorbell, seq, arrive, copy_ctas);
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
