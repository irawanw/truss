#include "spec.cuh"

#include "core/cuda_check.h"
#include <cstdlib>
#include <cstring>

namespace truss::sampling {
namespace {

constexpr int THREADS = 1024, WARPS = THREADS / 32;
constexpr int BISECT = 40;   // as sample.cu

__device__ float block_sum(float v, float * sh)
{
    for (int o = 16; o; o /= 2) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v;
    __syncthreads();
    float t = 0.f;
    for (int w = 0; w < WARPS; ++w) t += sh[w];
    return t;
}

__device__ float block_max(float v, float * sh)
{
    for (int o = 16; o; o /= 2) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v;
    __syncthreads();
    float t = -INFINITY;
    for (int w = 0; w < WARPS; ++w) t = fmaxf(t, sh[w]);
    return t;
}

__device__ __forceinline__ uint64_t mix(uint64_t z)   // splitmix64 finalizer
{
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

__device__ __forceinline__ float unit(uint64_t h) { return ((float) (h >> 40) + 0.5f) * (1.f / 16777216.f); }   // (0, 1)

// salts: the draft, accept and residual draws never share a stream with sample()'s rows or each other
constexpr uint64_t SALT_DRAFT = 0xd1b54a32d192ed03ull, SALT_ACCEPT = 0xa24baed4963ee407ull,
                   SALT_RESID = 0x9fb21c651e98df25ull;

// sample.cu's cut, verbatim: y = x / T kept when y >= cut; m = max y
__device__ float cut_of(const float * x, int n, float it, const SampleParams & p, float * sh, float & m_out)
{
    float m = -INFINITY, lo = INFINITY;
    for (int j = threadIdx.x; j < n; j += THREADS) {
        const float y = x[j] * it;
        m = fmaxf(m, y), lo = fminf(lo, y);
    }
    m = block_max(m, sh);
    lo = -block_max(-lo, sh);
    float cut = lo;
    if (p.min_p > 0.f) cut = fmaxf(cut, m + logf(p.min_p));
    if (p.top_k > 0 && p.top_k < n) {
        float a = lo, b = m;
        for (int s = 0; s < BISECT; ++s) {
            const float c = 0.5f * (a + b);
            float k = 0.f;
            for (int j = threadIdx.x; j < n; j += THREADS) k += x[j] * it >= c ? 1.f : 0.f;
            (block_sum(k, sh) >= (float) p.top_k ? a : b) = c;
        }
        cut = fmaxf(cut, a);
    }
    if (p.top_p > 0.f && p.top_p < 1.f) {
        float z = 0.f;
        for (int j = threadIdx.x; j < n; j += THREADS) {
            const float y = x[j] * it;
            z += y >= cut ? __expf(y - m) : 0.f;
        }
        z = block_sum(z, sh);
        float a = cut, b = m;
        for (int s = 0; s < BISECT; ++s) {
            const float c = 0.5f * (a + b);
            float q = 0.f;
            for (int j = threadIdx.x; j < n; j += THREADS) {
                const float y = x[j] * it;
                q += y >= c ? __expf(y - m) : 0.f;
            }
            (block_sum(q, sh) >= p.top_p * z ? a : b) = c;
        }
        cut = a;
    }
    m_out = m;
    return cut;
}
// ---- min_p-first compaction (S1) ---------------------------------------------------------------------------
// min_p's cut needs one max pass, and it is the tightest of the three cuts almost always: the served row keeps a few
// hundred tokens of 152K. So find it, keep the survivors, and run top_k's and top_p's bisections on those.
//
// EXACT, not approximate, and the reason is that a bisection only needs its DECISIONS to agree, not its counts:
//  - top_k searches [lo, m] exactly as before. For a midpoint c >= the min_p cut, "how many >= c" over the survivors
//    is the same number as over the whole row. For a c below it the two counts differ, but both are >= top_k
//    whenever the survivor count S is (the full row has at least as many), so both take the same branch. When
//    S < top_k the branches can differ, and it still ends the same: the loop's invariant is count(>=a) >= top_k,
//    so a can never exceed the top_k-th value, which is below the min_p cut when S < top_k - and the caller takes
//    max(min_p cut, a) either way.
//  - top_p searches [cut, m], and cut >= the min_p cut always, so every midpoint counts survivors only anyway.
//  - Every partial sum is bit-identical because each thread keeps ITS OWN survivors in the order it met them
//    (the strided loop is ascending in j): thread t's contribution to z, to the counts and to the Gumbel argmax is
//    the same sequence of adds in the same order, so block_sum sees the same 32 numbers and z, the cut, the probs
//    and the draws come out equal to the last bit. A shared compacted array re-dealt across threads would not.
//
// A thread holds CAP survivors; a row where any thread needs more (a flat distribution: min_p kept nearly the whole
// vocabulary) sets the flag and the row runs the full path above - same results, just not the fast one.
constexpr int CAP = 4;   // survivors kept per thread: 1024 threads x CAP x 8 B = 32 KB of dynamic shared

struct Surv
{
    int j[THREADS * CAP];
    float y[THREADS * CAP];
};

// m and lo over the whole row, exactly the two passes cut_of starts with
__device__ void maxmin(const float * x, int n, float it, float * sh, float & m, float & lo)
{
    m = -INFINITY, lo = INFINITY;
    for (int j = threadIdx.x; j < n; j += THREADS) {
        const float y = x[j] * it;
        m = fmaxf(m, y), lo = fminf(lo, y);
    }
    m = block_max(m, sh);
    lo = -block_max(-lo, sh);
}

// thread t's own survivors, in j order; cnt = how many, over = it did not fit
__device__ void gather(const float * x, int n, float it, float cut_minp, Surv & s, int & cnt, bool & over)
{
    cnt = 0;
    over = false;
    for (int j = threadIdx.x; j < n; j += THREADS) {
        const float y = x[j] * it;
        if (y < cut_minp) continue;
        if (cnt == CAP) { over = true; return; }
        s.j[threadIdx.x * CAP + cnt] = j;
        s.y[threadIdx.x * CAP + cnt] = y;
        ++cnt;
    }
}

// cut_of's answer from the survivor lists. lo/m/cut_minp come from the max pass; n bounds the top_k guard.
__device__ float cut_compact(Surv & s, int cnt, int S, float lo, float m, float cut_minp, int n,
                             const SampleParams & p, float * sh)
{
    float cut = fmaxf(lo, cut_minp);
    if (p.top_k > 0 && p.top_k < n && S >= p.top_k) {   // S < top_k: the cut is cut_minp, as proven above
        float a = lo, b = m;
        for (int sq = 0; sq < BISECT; ++sq) {
            const float c = 0.5f * (a + b);
            float k = 0.f;
            for (int q = 0; q < cnt; ++q) k += s.y[threadIdx.x * CAP + q] >= c ? 1.f : 0.f;
            (block_sum(k, sh) >= (float) p.top_k ? a : b) = c;
        }
        cut = fmaxf(cut, a);
    }
    if (p.top_p > 0.f && p.top_p < 1.f) {
        float z = 0.f;
        for (int q = 0; q < cnt; ++q) {
            const float y = s.y[threadIdx.x * CAP + q];
            z += y >= cut ? __expf(y - m) : 0.f;
        }
        z = block_sum(z, sh);
        float a = cut, b = m;
        for (int sq = 0; sq < BISECT; ++sq) {
            const float c = 0.5f * (a + b);
            float q = 0.f;
            for (int k = 0; k < cnt; ++k) {
                const float y = s.y[threadIdx.x * CAP + k];
                q += y >= c ? __expf(y - m) : 0.f;
            }
            (block_sum(q, sh) >= p.top_p * z ? a : b) = c;
        }
        cut = a;
    }
    return cut;
}

// block-wide argmax of (v, i), ties to the lower index; thread 0 gets the result
__device__ int block_argmax(float v, int i, float * sv, int * si)
{
    for (int o = 16; o; o /= 2) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, i, o);
        if (ov > v || (ov == v && oi < i)) v = ov, i = oi;
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0) sv[threadIdx.x / 32] = v, si[threadIdx.x / 32] = i;
    __syncthreads();
    if (threadIdx.x == 0)
        for (int w = 1; w < WARPS; ++w)
            if (sv[w] > v || (sv[w] == v && si[w] < i)) v = sv[w], i = si[w];
    return i;
}

template <bool COMPACT>
__global__ void __launch_bounds__(THREADS) probs_kernel(float * xs, int n, SampleParams p)
{
    extern __shared__ float dyn[];   // the survivor lists, sized sizeof(Surv) when COMPACT
    __shared__ float sh[WARPS];
    float * x = xs + (size_t) blockIdx.x * n;
    const float it = 1.f / p.temperature;
    Surv * s = reinterpret_cast<Surv *>(dyn);
    float m = 0.f, cut = 0.f;
    int cnt = -1;   // >= 0: this thread's survivors are in the list, and it is the whole basis for the cut
    if (COMPACT) {
        float lo;
        maxmin(x, n, it, sh, m, lo);
        const float cut_minp = m + logf(p.min_p);
        bool over = false;
        gather(x, n, it, cut_minp, *s, cnt, over);
        if (__syncthreads_or(over))
            cnt = -1;   // a flat row (min_p kept nearly everything): the full path below, same answer
        else
            cut = cut_compact(*s, cnt, (int) block_sum((float) cnt, sh), lo, m, cut_minp, n, p, sh);
    }
    if (!COMPACT || cnt < 0) cut = cut_of(x, n, it, p, sh, m);
    float z = 0.f;
    if (cnt >= 0)
        for (int q = 0; q < cnt; ++q) {
            const float y = s->y[threadIdx.x * CAP + q];
            z += y >= cut ? __expf(y - m) : 0.f;
        }
    else
        for (int j = threadIdx.x; j < n; j += THREADS) {
            const float y = x[j] * it;
            z += y >= cut ? __expf(y - m) : 0.f;
        }
    z = block_sum(z, sh);   // every thread has read its own x[j] above; the writes below touch only those
    const float iz = 1.f / z;
    for (int j = threadIdx.x; j < n; j += THREADS) {
        const float y = x[j] * it;
        x[j] = y >= cut ? __expf(y - m) * iz : 0.f;
    }
}

template <bool COMPACT>
__global__ void __launch_bounds__(THREADS) draft_kernel(const float * x, int n, const int * map, SampleParams p,
                                                        uint64_t counter, float * q_full, int * out)
{
    extern __shared__ float dyn[];
    __shared__ float sh[WARPS], sv[WARPS];
    __shared__ int si[WARPS];
    Surv * s = reinterpret_cast<Surv *>(dyn);
    const float it = 1.f / p.temperature;
    float m = 0.f, cut = 0.f;
    int cnt = -1;
    if (COMPACT) {
        float lo;
        maxmin(x, n, it, sh, m, lo);
        const float cut_minp = m + logf(p.min_p);
        bool over = false;
        gather(x, n, it, cut_minp, *s, cnt, over);
        if (__syncthreads_or(over))
            cnt = -1;
        else
            cut = cut_compact(*s, cnt, (int) block_sum((float) cnt, sh), lo, m, cut_minp, n, p, sh);
    }
    if (!COMPACT || cnt < 0) cut = cut_of(x, n, it, p, sh, m);
    float z = 0.f;
    if (cnt >= 0)
        for (int q = 0; q < cnt; ++q) {
            const float y = s->y[threadIdx.x * CAP + q];
            z += y >= cut ? __expf(y - m) : 0.f;
        }
    else
        for (int j = threadIdx.x; j < n; j += THREADS) {
            const float y = x[j] * it;
            z += y >= cut ? __expf(y - m) : 0.f;
        }
    z = block_sum(z, sh);
    const float iz = 1.f / z;
    const uint64_t stream = mix(p.seed ^ mix(counter + SALT_DRAFT));
    float v = -INFINITY;
    int i = n;
    if (cnt >= 0)
        for (int q = 0; q < cnt; ++q) {   // own survivors, own order: the same per-thread argmax as the loop below
            const int j = s->j[threadIdx.x * CAP + q];
            const float y = s->y[threadIdx.x * CAP + q];
            if (y < cut) continue;
            q_full[map ? map[j] : j] = __expf(y - m) * iz;
            const float g = y - logf(-logf(unit(mix(stream + (uint64_t) j * 0x9e3779b97f4a7c15ull))));
            if (g > v || (g == v && j < i)) v = g, i = j;
        }
    else
        for (int j = threadIdx.x; j < n; j += THREADS) {
            const float y = x[j] * it;
            if (y < cut) continue;
            const float q = __expf(y - m) * iz;
            q_full[map ? map[j] : j] = q;
            // Gumbel-max on log q (= y - m - log z): the same argmax as on y
            const float g = y - logf(-logf(unit(mix(stream + (uint64_t) j * 0x9e3779b97f4a7c15ull))));
            if (g > v || (g == v && j < i)) v = g, i = j;
        }
    i = block_argmax(v, i, sv, si);
    if (threadIdx.x == 0) out[0] = map ? map[i] : i;
}

// one block: rows in order; thread 0 decides, all threads draw the residual / bonus
__global__ void __launch_bounds__(THREADS) accept_kernel(const float * probs, const float * q_full, const int * drafts,
                                                         int nw, int V, uint64_t seed, uint64_t counter, int * out)
{
    __shared__ float sv[WARPS];
    __shared__ int si[WARPS];
    __shared__ int rej;
    int j = 0;
    for (; j < nw; ++j) {
        if (threadIdx.x == 0) {
            const int x = drafts[j];
            const float px = probs[(size_t) j * V + x], qx = q_full[(size_t) j * V + x];
            const float u = unit(mix(mix(seed ^ mix(counter + j + SALT_ACCEPT))));
            rej = !(u * qx < px);   // accept with probability min(1, px / qx); qx > 0: x was drawn from q
        }
        __syncthreads();
        if (rej) break;
        __syncthreads();
    }
    // j == nw: every draft accepted, the bonus is a sample of p_nw; else the residual max(0, p_j - q_j)
    const float * pr = probs + (size_t) j * V;
    const float * qr = j < nw ? q_full + (size_t) j * V : nullptr;
    const uint64_t stream = mix(seed ^ mix(counter + j + SALT_RESID));
    float v = -INFINITY;
    int i = V;
    for (int t = threadIdx.x; t < V; t += THREADS) {
        const float r = qr ? pr[t] - qr[t] : pr[t];
        if (!(r > 0.f)) continue;
        const float g = logf(r) - logf(-logf(unit(mix(stream + (uint64_t) t * 0x9e3779b97f4a7c15ull))));
        if (g > v || (g == v && t < i)) v = g, i = t;
    }
    i = block_argmax(v, i, sv, si);
    if (threadIdx.x == 0) {
        if (i >= V) {   // empty residual (p_j == q_j up to rounding): fall back to p_j itself
            float bv = -INFINITY;
            for (int t = 0; t < V; ++t)
                if (pr[t] > bv) bv = pr[t], i = t;
        }
        out[0] = j, out[1] = i;
    }
}

}  // namespace

// The compact path needs a min_p cut to compact against, and TRUSS_SAMPLER=old switches it off for the A/B.
// Greedy/argmax callers never come through here.
namespace {
bool use_compact(const SampleParams & p)
{
    static const bool old = [] {
        const char * e = std::getenv("TRUSS_SAMPLER");
        return e && std::strcmp(e, "old") == 0;
    }();
    return p.min_p > 0.f && !old;
}
}  // namespace

void probs_inplace(float * x, int n, int rows, const SampleParams & p, cudaStream_t stream)
{
    if (rows <= 0) return;
    if (use_compact(p))
        probs_kernel<true><<<rows, THREADS, sizeof(Surv), stream>>>(x, n, p);
    else
        probs_kernel<false><<<rows, THREADS, 0, stream>>>(x, n, p);
    TRUSS_CUDA(cudaGetLastError());
}

void draft_sample(const float * x, int n, const int * map, int V, const SampleParams & p, uint64_t counter,
                  float * q_full, int * out, cudaStream_t stream)
{
    TRUSS_CUDA(cudaMemsetAsync(q_full, 0, sizeof(float) * V, stream));
    if (use_compact(p))
        draft_kernel<true><<<1, THREADS, sizeof(Surv), stream>>>(x, n, map, p, counter, q_full, out);
    else
        draft_kernel<false><<<1, THREADS, 0, stream>>>(x, n, map, p, counter, q_full, out);
    TRUSS_CUDA(cudaGetLastError());
}

void spec_accept(const float * probs, const float * q_full, const int * drafts, int nw, int V, uint64_t seed,
                 uint64_t counter, int * out, cudaStream_t stream)
{
    accept_kernel<<<1, THREADS, 0, stream>>>(probs, q_full, drafts, nw, V, seed, counter, out);
    TRUSS_CUDA(cudaGetLastError());
}

}  // namespace truss::sampling
