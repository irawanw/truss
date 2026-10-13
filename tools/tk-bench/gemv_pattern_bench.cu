// Why does the hc down projection sit at ~410 GB/s (the split sweep plateaus there for every S/KB)? Two candidate
// answers: the gemv's access pattern, or its arithmetic. This bench reads the engine's own down-projection bytes with
// NO mma, so the only thing that varies is the pattern:
//   strided   exactly the gemv's map (grid (tiles/S, KB), 16 warps of 512 per tile, warp p = kb*S + ks takes Q8
//             blocks p, p + P, ...; 8 bytes per thread per block, two rows g and g + 8);
//   contig    the same warps, the same number of blocks each, but a contiguous run per warp (what a regrouped k
//             split would look like - it changes the summation order, so it is only a ceiling probe here);
//   linear    the whole matrix grid-stride, 16 or 8 bytes per thread.
// Trustworthy numbers, per the lead's calibration (a pure read tops out ~850-900 GB/s, D2D memcpy ~800 counting
// read + write): every warp's sum goes to a slot unconditionally so no load can be optimised away, the host zeroes
// the slots, launches once more and checks they add up to the sum of the matrix's bytes (each byte read exactly
// once); and timing replays a captured graph of 50 launches, because a 3 MB kernel runs ~4 us and a CPU-side launch
// costs about 2 of them, which is what made an earlier version of this bench report 1450 GB/s for empty kernels.
// usage: gemv_pattern_bench [reps]
#include "core/cuda_check.h"
#include "kernels/dense/q8_gemm.cuh"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

using namespace truss;

namespace {

constexpr int IN = 10240, OUTS = 320;   // the hc down matrix; the engine's launch adds inject's 4 rows to the tiles
constexpr int SLOTS = 1 << 15, PER = 50;

__device__ __forceinline__ uint32_t warp_sum(uint32_t v)
{
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// one slot per warp: the store is unconditional and every lane's value feeds it, so the loads above are live
__device__ __forceinline__ void emit(uint32_t acc, uint32_t * out)
{
    if ((threadIdx.x & 31) == 0)
        out[(size_t) (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x / 32) + (threadIdx.x >> 5)] = warp_sum(acc);
}

// the gemv's own map: warp p = kb * S + ks of P takes blocks p, p + P, ...
template <int S>
__global__ void __launch_bounds__(512) strided_kernel(const int8_t * __restrict__ w, int in, int outs, int KB,
                                                      uint32_t * out)
{
    constexpr int WARPS = S > 4 ? S : 4, TILES = WARPS / S;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, q = lane & 3, nb = in / 32;
    const int tile = blockIdx.x * TILES + warp / S, ks = warp % S, kb = blockIdx.y, P = S * KB;
    const int o[2] = { tile * 16 + g, tile * 16 + g + 8 };
    const int8_t * w0 = o[0] < outs ? w + (size_t) o[0] * in + 8 * q : nullptr;
    const int8_t * w1 = o[1] < outs ? w + (size_t) o[1] * in + 8 * q : nullptr;
    uint32_t acc = 0;
#pragma unroll 4
    for (int b = kb * S + ks; b < nb; b += P) {
        const uint2 a = w0 ? __ldg(reinterpret_cast<const uint2 *>(w0 + b * 32)) : make_uint2(0, 0);
        const uint2 c = w1 ? __ldg(reinterpret_cast<const uint2 *>(w1 + b * 32)) : make_uint2(0, 0);
        acc += a.x + a.y + c.x + c.y;
    }
    emit(acc, out);
}

// the same warps and the same count of blocks, contiguous
template <int S>
__global__ void __launch_bounds__(512) contig_kernel(const int8_t * __restrict__ w, int in, int outs, int KB,
                                                     uint32_t * out)
{
    constexpr int WARPS = S > 4 ? S : 4, TILES = WARPS / S;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, q = lane & 3, nb = in / 32;
    const int tile = blockIdx.x * TILES + warp / S, ks = warp % S, kb = blockIdx.y, P = S * KB;
    const int o[2] = { tile * 16 + g, tile * 16 + g + 8 };
    const int8_t * w0 = o[0] < outs ? w + (size_t) o[0] * in + 8 * q : nullptr;
    const int8_t * w1 = o[1] < outs ? w + (size_t) o[1] * in + 8 * q : nullptr;
    const int p = kb * S + ks, per = nb / P, last = p * per + per + (p < nb % P ? 1 : 0);
    uint32_t acc = 0;
#pragma unroll 4
    for (int b = p * per; b < last; ++b) {
        const uint2 a = w0 ? __ldg(reinterpret_cast<const uint2 *>(w0 + b * 32)) : make_uint2(0, 0);
        const uint2 c = w1 ? __ldg(reinterpret_cast<const uint2 *>(w1 + b * 32)) : make_uint2(0, 0);
        acc += a.x + a.y + c.x + c.y;
    }
    emit(acc, out);
}

template <typename V>
__global__ void linear_kernel(const V * __restrict__ w, size_t n, uint32_t * out)
{
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const V v = __ldg(w + i);
        acc += v.x + v.y + v.z + v.w;
    }
    emit(acc, out);
}

__global__ void linear2_kernel(const uint2 * __restrict__ w, size_t n, uint32_t * out)
{
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const uint2 v = __ldg(w + i);
        acc += v.x + v.y;
    }
    emit(acc, out);
}

float elapsed(cudaEvent_t a, cudaEvent_t b)
{
    float ms = 0.f;
    TRUSS_CUDA(cudaEventElapsedTime(&ms, a, b));
    return ms * 1e3f;
}

}  // namespace

int main(int argc, char ** argv)
{
    const int reps = argc > 1 ? std::atoi(argv[1]) : 300;
    const size_t nbytes = (size_t) OUTS * IN;
    std::vector<int8_t> h(nbytes);
    for (size_t i = 0; i < nbytes; ++i) h[i] = (int8_t) (i * 31 + 7);
    // every byte is read exactly once per launch, so all variants must produce this sum
    const uint64_t want = std::accumulate(h.begin(), h.end(), (uint64_t) 0,
                                          [](uint64_t a, int8_t b) { return a + (uint32_t) (uint8_t) b; });
    const int copies = (int) std::max<size_t>(2, (160u << 20) / nbytes + 1);
    std::vector<int8_t *> W(copies);
    for (int c = 0; c < copies; ++c) {
        TRUSS_CUDA(cudaMalloc(&W[c], nbytes));
        TRUSS_CUDA(cudaMemcpy(W[c], h.data(), nbytes, cudaMemcpyHostToDevice));
    }
    int8_t * dst;
    TRUSS_CUDA(cudaMalloc(&dst, nbytes));
    uint32_t * out;
    TRUSS_CUDA(cudaMalloc(&out, sizeof(uint32_t) * SLOTS));
    cudaStream_t s;
    TRUSS_CUDA(cudaStreamCreate(&s));
    cudaEvent_t a, b;
    TRUSS_CUDA(cudaEventCreate(&a));
    TRUSS_CUDA(cudaEventCreate(&b));
    std::vector<uint32_t> slot(SLOTS);
    std::printf("down projection %zu B, %d copies (%.0f MB, L2 is 6 MB); the engine's split is S 16 KB 4\n"
                "expected byte sum %llu\n",
                nbytes, copies, copies * nbytes / 1e6, (unsigned long long) want);

    // timed with a captured graph: PER launches replayed as one submission, so the CPU's launch cost is not measured
    auto run = [&](const char * name, const char * cfg, auto launch, float mult = 1.f) {
        cudaGraph_t g;
        TRUSS_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < PER; ++i) launch(s, W[i % copies]);
        TRUSS_CUDA(cudaStreamEndCapture(s, &g));
        cudaGraphExec_t e;
        TRUSS_CUDA(cudaGraphInstantiate(&e, g, 0));
        TRUSS_CUDA(cudaGraphDestroy(g));
        const int iters = reps / PER;
        TRUSS_CUDA(cudaGraphLaunch(e, s));
        TRUSS_CUDA(cudaStreamSynchronize(s));   // warm
        TRUSS_CUDA(cudaEventRecord(a, s));
        for (int i = 0; i < iters; ++i) TRUSS_CUDA(cudaGraphLaunch(e, s));
        TRUSS_CUDA(cudaEventRecord(b, s));
        TRUSS_CUDA(cudaStreamSynchronize(s));
        const float us = elapsed(a, b) / (iters * PER);
        cudaGraphExecDestroy(e);
        TRUSS_CUDA(cudaMemsetAsync(out, 0, sizeof(uint32_t) * SLOTS, s));
        launch(s, W[0]);
        TRUSS_CUDA(cudaStreamSynchronize(s));
        TRUSS_CUDA(cudaMemcpy(slot.data(), out, sizeof(uint32_t) * SLOTS, cudaMemcpyDeviceToHost));
        const uint64_t got = std::accumulate(slot.begin(), slot.end(), (uint64_t) 0);
        std::printf("  %-20s %-16s %6.2f us  %5.0f GB/s  %s\n", name, cfg, us, nbytes * mult / us / 1e3,
                    got == want ? "all bytes read once" : "MISSING BYTES");
    };
    // the engine's launch for down+inject: 21 tile columns, S 16 (one tile per 16-warp CTA), KB 4
    run("strided S16 KB4", "21x4 CTAs x512", [&](cudaStream_t st, int8_t * w) {
        strided_kernel<16><<<dim3(21, 4), 512, 0, st>>>(w, IN, OUTS, 4, out);
    });
    run("contig S16 KB4", "21x4 CTAs x512", [&](cudaStream_t st, int8_t * w) {
        contig_kernel<16><<<dim3(21, 4), 512, 0, st>>>(w, IN, OUTS, 4, out);
    });
    run("strided S16 KB8", "21x8 CTAs x512", [&](cudaStream_t st, int8_t * w) {
        strided_kernel<16><<<dim3(21, 8), 512, 0, st>>>(w, IN, OUTS, 8, out);
    });
    run("contig S16 KB8", "21x8 CTAs x512", [&](cudaStream_t st, int8_t * w) {
        contig_kernel<16><<<dim3(21, 8), 512, 0, st>>>(w, IN, OUTS, 8, out);
    });
    run("contig S4 KB8", "21x8 CTAs x128", [&](cudaStream_t st, int8_t * w) {
        contig_kernel<4><<<dim3(21, 8), 128, 0, st>>>(w, IN, OUTS, 8, out);
    });
    run("linear uint4", "1024 CTAs x256", [&](cudaStream_t st, int8_t * w) {
        linear_kernel<uint4><<<1024, 256, 0, st>>>(reinterpret_cast<const uint4 *>(w), nbytes / 16, out);
    });
    run("linear uint2", "1024 CTAs x256", [&](cudaStream_t st, int8_t * w) {
        linear2_kernel<<<1024, 256, 0, st>>>(reinterpret_cast<const uint2 *>(w), nbytes / 8, out);
    });
    run("memcpy D2D", "3.3 MB r+w", [&](cudaStream_t st, int8_t * w) {
        TRUSS_CUDA(cudaMemcpyAsync(dst, w, nbytes, cudaMemcpyDeviceToDevice, st));
    }, 2.f);
    TRUSS_CUDA(cudaGetLastError());
    return 0;
}
