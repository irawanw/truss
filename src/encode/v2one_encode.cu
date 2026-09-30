// v2one tail-biting Viterbi (v2one_encode.h). One block per tile, all 128 steps inside the kernel.
//
// Cost layout (as exllamav3): the best cost of a state n depends on its predecessor only through n >> S, so after
// each step only m(o) = min over states with low 16 - S bits o is kept: 2^(16-S) floats per step. Step t:
//   m_t(o) = min over a < 2^S of  m_{t-1}(n >> S) + err_t(n),   n = a << (16 - S) | o
// and the argmin a (the state's top S bits) is the backpointer. Costs live in shared memory for S >= 3 (<= 64 KB,
// double-buffered), in a global workspace slice for S = 2 (128 KB).
#include "encode/v2one_encode.h"

#include "core/cuda_check.h"
#include "kernels/trellis/codec_v2one.cuh"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace truss::encode {
namespace {

constexpr int STEPS = 128, NT = 512;

template <int S> constexpr bool COSTS_IN_SMEM = S >= 3;
template <int S> constexpr int OB = 16 - S;           // overlap bits
template <int S> constexpr int NO = 1 << OB<S>;       // overlap values

template <int S>
__host__ __device__ constexpr size_t bp_bytes()
{
    return (size_t) STEPS * NO<S>;
}
template <int S>
__host__ __device__ constexpr size_t cost_bytes()
{
    return 2 * (size_t) NO<S> * sizeof(float);
}

__device__ __forceinline__ float err2(uint32_t n, float x0, float x1)
{
    const uint32_t h = v2one_codebook_bits(n);
    const float d0 = x0 - __half2float(__ushort_as_half((uint16_t) (h & 0xffffu)));
    const float d1 = x1 - __half2float(__ushort_as_half((uint16_t) (h >> 16)));
    return d0 * d0 + d1 * d1;
}

// block-wide argmin of m over all overlaps (lowest index on ties); every thread gets the result
template <int S>
__device__ int argmin_overlap(const float * m, float * red_v, int * red_i)
{
    float best = INFINITY;
    int arg = 0;
    for (int o = threadIdx.x; o < NO<S>; o += NT)
        if (m[o] < best) best = m[o], arg = o;   // o increases per thread: first minimum kept
    for (int off = 16; off; off >>= 1) {
        const float bv = __shfl_down_sync(0xffffffffu, best, off);
        const int bi = __shfl_down_sync(0xffffffffu, arg, off);
        if (bv < best || (bv == best && bi < arg)) best = bv, arg = bi;
    }
    if ((threadIdx.x & 31) == 0) red_v[threadIdx.x >> 5] = best, red_i[threadIdx.x >> 5] = arg;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < NT / 32; ++w)
            if (red_v[w] < red_v[0] || (red_v[w] == red_v[0] && red_i[w] < red_i[0])) red_v[0] = red_v[w], red_i[0] = red_i[w];
    }
    __syncthreads();
    const int r = red_i[0];
    __syncthreads();
    return r;
}

template <int S>
__global__ __launch_bounds__(NT) void viterbi_kernel(const float * __restrict__ tiles, float * __restrict__ q,
                                                     uint16_t * __restrict__ states, uint8_t * __restrict__ bp_all,
                                                     float * __restrict__ cost_all)
{
    constexpr int ob = OB<S>, no = NO<S>, na = 1 << S;
    extern __shared__ float smem[];
    __shared__ float x[256];
    __shared__ float red_v[NT / 32];
    __shared__ int red_i[NT / 32];
    __shared__ uint16_t ring[STEPS];

    const int tile = blockIdx.x;
    uint8_t * bp = bp_all + (size_t) tile * bp_bytes<S>();
    float * m0 = COSTS_IN_SMEM<S> ? smem : cost_all + (size_t) tile * 2 * no;
    float * m1 = m0 + no;
    for (int i = threadIdx.x; i < 256; i += NT) x[i] = tiles[(size_t) tile * 256 + i];

    // one pass over the ring starting at step `roll`; fixed >= 0 pins the overlap before the first step
    auto pass = [&](int roll, int fixed) -> float * {
        float * cur = m0, * nxt = m1;
        for (int o = threadIdx.x; o < no; o += NT) cur[o] = (fixed < 0 || o == fixed) ? 0.f : INFINITY;
        __syncthreads();
        for (int i = 0; i < STEPS; ++i) {
            const int t = (i + roll) & (STEPS - 1);
            const float x0 = x[2 * t], x1 = x[2 * t + 1];
            for (int o = threadIdx.x; o < no; o += NT) {
                float best = INFINITY;
                int arg = 0;
#pragma unroll
                for (int a = 0; a < na; ++a) {
                    const uint32_t n = ((uint32_t) a << ob) | (uint32_t) o;
                    const float c = cur[n >> S] + err2(n, x0, x1);
                    if (c < best) best = c, arg = a;
                }
                nxt[o] = best;
                bp[(size_t) t * no + o] = (uint8_t) arg;
            }
            __syncthreads();
            float * tmp = cur;
            cur = nxt;
            nxt = tmp;
        }
        return cur;
    };
    // walk back from step `last` with overlap o for `count` steps, writing the states into ring[]; returns the
    // overlap before the earliest step walked
    auto trace = [&](int last, int o, int count) {
        for (int i = 0; i < count; ++i) {
            const int t = (last - i) & (STEPS - 1);
            const uint32_t n = ((uint32_t) bp[(size_t) t * no + o] << ob) | (uint32_t) o;
            ring[t] = (uint16_t) n;
            o = (int) (n >> S);
        }
        return o;
    };

    // pass 1: ring rotated by half, free start; its path through step 0 gives the seam overlap
    const float * m = pass(STEPS / 2, -1);
    int end = argmin_overlap<S>(m, red_v, red_i);
    __shared__ int seam;
    if (threadIdx.x == 0) {
        trace(STEPS / 2 - 1, end, STEPS / 2);   // steps 63 .. 0
        seam = ring[0] >> S;                    // overlap before step 0 = low bits of step 127's state
    }
    __syncthreads();
    // pass 2: in order, seam pinned at both ends
    const int fixed = seam;
    pass(0, fixed);
    if (threadIdx.x == 0) trace(STEPS - 1, fixed, STEPS);
    __syncthreads();
    for (int t = threadIdx.x; t < STEPS; t += NT) {
        const uint32_t n = ring[t], h = v2one_codebook_bits(n);
        states[(size_t) tile * STEPS + t] = (uint16_t) n;
        q[(size_t) tile * 256 + 2 * t] = __half2float(__ushort_as_half((uint16_t) (h & 0xffffu)));
        q[(size_t) tile * 256 + 2 * t + 1] = __half2float(__ushort_as_half((uint16_t) (h >> 16)));
    }
}

template <int S>
size_t per_tile()
{
    return bp_bytes<S>() + (COSTS_IN_SMEM<S> ? 0 : cost_bytes<S>());
}

template <int S>
void run(const float * tiles, int n, float * q, uint16_t * states, void * ws, size_t ws_bytes, cudaStream_t stream)
{
    const size_t pt = per_tile<S>();
    const int chunk = (int) std::min<size_t>(ws_bytes / pt, (size_t) n);
    if (chunk < 1) throw std::runtime_error("v2one_encode: workspace below one tile (" + std::to_string(pt) + " B)");
    const size_t smem = COSTS_IN_SMEM<S> ? cost_bytes<S>() : 0;
    TRUSS_CUDA(cudaFuncSetAttribute(viterbi_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
    uint8_t * bp = static_cast<uint8_t *>(ws);
    float * costs = reinterpret_cast<float *>(bp + (size_t) chunk * bp_bytes<S>());
    for (int t0 = 0; t0 < n; t0 += chunk) {
        const int c = std::min(chunk, n - t0);
        viterbi_kernel<S><<<c, NT, smem, stream>>>(tiles + (size_t) t0 * 256, q + (size_t) t0 * 256,
                                                   states + (size_t) t0 * STEPS, bp, costs);
        TRUSS_CUDA(cudaGetLastError());
    }
}

}  // namespace

size_t v2one_workspace_bytes_per_tile(int S)
{
    switch (S) {
        case 2: return per_tile<2>();
        case 3: return per_tile<3>();
        case 4: return per_tile<4>();
        case 5: return per_tile<5>();
    }
    throw std::runtime_error("v2one: S must be 2..5, got " + std::to_string(S));
}

void v2one_encode(int S, const float * tiles, int n, float * q, uint16_t * states, void * ws, size_t ws_bytes,
                  cudaStream_t stream)
{
    switch (S) {
        case 2: return run<2>(tiles, n, q, states, ws, ws_bytes, stream);
        case 3: return run<3>(tiles, n, q, states, ws, ws_bytes, stream);
        case 4: return run<4>(tiles, n, q, states, ws, ws_bytes, stream);
        case 5: return run<5>(tiles, n, q, states, ws, ws_bytes, stream);
    }
    throw std::runtime_error("v2one: S must be 2..5, got " + std::to_string(S));
}

}  // namespace truss::encode
