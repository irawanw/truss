// Gated delta rule (see gdn_prefill.cuh). Block = (value head, 32 state columns) = 4 warps x 8 columns; a column
// is held by 4 lanes (lane = 4 c + qd), lane qd holding rows {16 m + 4 qd + e : m < 8, e < 4} (32 registers), so a
// column's reductions take 2 shuffle levels and its k, q rows load as 8 conflict-free LDS.128 each. Per token, with
// the column s (pre-update) and decay a = exp(g):
//   s' = a s + k delta,  delta = beta (v - a s.k)
//   o  = s'.q = a (s.q) + (k.q) delta
// so s.k and s.q reduce together, and k.q (the same for every column) is computed once per token at staging.
// k, q, v, g, beta come through shared memory in chunks of TC tokens (cp.async, double buffered).
// History (TRACKER #46): warp per column with k, q from global 2.5 us/token/layer; staged k, q with 4 columns per
// warp (32 rows per lane, 5 shuffle levels) 1.83.
#include "gdn_prefill.cuh"

#include "core/cuda_check.h"

#include <stdexcept>

namespace truss::gdn {
namespace {

constexpr int S = 128;
constexpr int WARPS = 4, CW = 8, COLS = WARPS * CW;   // columns per warp, per block
constexpr int RL = S / 4;                             // rows per lane
constexpr int TC = 8;                                 // tokens per staged chunk

struct Chunk {
    float k[TC][S], q[TC][S], v[TC][COLS], g[TC], b[TC], kq[TC];
};

__device__ __forceinline__ void cp4(void * dst, const void * src)
{
    const unsigned d = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp16(void * dst, const void * src)
{
    const unsigned d = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}

__device__ __forceinline__ int row_of(int qd, int m, int e) { return 16 * m + 4 * qd + e; }

__global__ __launch_bounds__(WARPS * 32) void delta_rule_kernel(const float * __restrict__ q, const float * __restrict__ k,
                                                               const float * __restrict__ v, const float * __restrict__ g,
                                                               const float * __restrict__ beta, const float * state_in,
                                                               float * state_out, float * __restrict__ out, int T, int Hk,
                                                               int Hv)
{
    __shared__ __align__(16) Chunk ch[2];
    const int h = blockIdx.x, hk = h % Hk, lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    const int qd = lane & 3, cl = wid * CW + (lane >> 2);   // row quarter; column inside the block
    const int col = blockIdx.y * COLS + cl;
    float s[RL];
#pragma unroll
    for (int m = 0; m < 8; ++m)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            s[4 * m + e] = state_in ? state_in[((size_t) h * S + row_of(qd, m, e)) * S + col] : 0.f;
    const float scale = rsqrtf((float) S);

    // tokens [t0, t0 + n) -> ch[b]; rows past T are not loaded (never read)
    auto issue = [&](int t0, int b) {
        const int n = min(TC, T - t0);
        for (int i = threadIdx.x; i < n * (S / 4); i += WARPS * 32) {
            const int t = i / (S / 4), e = (i % (S / 4)) * 4;
            cp16(&ch[b].k[t][e], k + ((size_t) (t0 + t) * Hk + hk) * S + e);
            cp16(&ch[b].q[t][e], q + ((size_t) (t0 + t) * Hk + hk) * S + e);
        }
        for (int i = threadIdx.x; i < n * (COLS / 4); i += WARPS * 32) {
            const int t = i / (COLS / 4), e = (i % (COLS / 4)) * 4;
            cp16(&ch[b].v[t][e], v + ((size_t) (t0 + t) * Hv + h) * S + blockIdx.y * COLS + e);
        }
        for (int t = threadIdx.x; t < n; t += WARPS * 32) {
            cp4(&ch[b].g[t], g + (size_t) (t0 + t) * Hv + h);
            cp4(&ch[b].b[t], beta + (size_t) (t0 + t) * Hv + h);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    const int n_chunks = (T + TC - 1) / TC;
    if (n_chunks > 0) issue(0, 0);
    for (int ci = 0; ci < n_chunks; ++ci) {
        const int b = ci & 1;
        if (ci + 1 < n_chunks) {
            issue((ci + 1) * TC, b ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        Chunk & C = ch[b];
        const int n = min(TC, T - ci * TC);
        for (int t = wid; t < n; t += WARPS) {   // k.q per token, one warp each
            float a = 0.f;
#pragma unroll
            for (int r = 0; r < S / 32; ++r) a += C.k[t][lane + 32 * r] * C.q[t][lane + 32 * r];
#pragma unroll
            for (int m = 16; m; m >>= 1) a += __shfl_xor_sync(0xffffffffu, a, m);
            if (lane == 0) C.kq[t] = a;
        }
        __syncthreads();
        for (int tt = 0; tt < n; ++tt) {
            float kr[RL];
            float sk = 0.f, sq = 0.f;
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                const float4 k4 = *(const float4 *) &C.k[tt][row_of(qd, m, 0)];
                const float4 q4 = *(const float4 *) &C.q[tt][row_of(qd, m, 0)];
                kr[4 * m] = k4.x; kr[4 * m + 1] = k4.y; kr[4 * m + 2] = k4.z; kr[4 * m + 3] = k4.w;
                sk += s[4 * m] * k4.x + s[4 * m + 1] * k4.y + s[4 * m + 2] * k4.z + s[4 * m + 3] * k4.w;
                sq += s[4 * m] * q4.x + s[4 * m + 1] * q4.y + s[4 * m + 2] * q4.z + s[4 * m + 3] * q4.w;
            }
            sk += __shfl_xor_sync(0xffffffffu, sk, 1);
            sq += __shfl_xor_sync(0xffffffffu, sq, 1);
            sk += __shfl_xor_sync(0xffffffffu, sk, 2);
            sq += __shfl_xor_sync(0xffffffffu, sq, 2);
            const float a = expf(C.g[tt]);
            const float delta = C.b[tt] * (C.v[tt][cl] - a * sk);
#pragma unroll
            for (int r = 0; r < RL; ++r) s[r] = a * s[r] + kr[r] * delta;
            if (qd == 0) out[((size_t) (ci * TC + tt) * Hv + h) * S + col] = (a * sq + C.kq[tt] * delta) * scale;
        }
        __syncthreads();
    }
    if (state_out)
#pragma unroll
        for (int m = 0; m < 8; ++m)
#pragma unroll
            for (int e = 0; e < 4; ++e) state_out[((size_t) h * S + row_of(qd, m, e)) * S + col] = s[4 * m + e];
}

}  // namespace

void delta_rule(const float * q, const float * k, const float * v, const float * g, const float * beta,
                const float * state_in, float * state_out, float * out, int T, int Hk, int Hv, cudaStream_t stream)
{
    if (Hv % Hk) throw std::runtime_error("gdn::delta_rule: value heads must be a multiple of key heads");
    delta_rule_kernel<<<dim3(Hv, S / COLS), WARPS * 32, 0, stream>>>(q, k, v, g, beta, state_in, state_out, out, T, Hk,
                                                                     Hv);
    TRUSS_CUDA(cudaGetLastError());
}

void delta_rule(const float * q, const float * k, const float * v, const float * g, const float * beta, float * state,
                float * out, int T, int Hk, int Hv, cudaStream_t stream)
{
    delta_rule(q, k, v, g, beta, state, state, out, T, Hk, Hv, stream);
}

}  // namespace truss::gdn
