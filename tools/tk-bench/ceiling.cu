// CP3 ceilings on this GPU, measured apart from any real kernel:
//  A) decode ceiling: the per-tile work of the TRUSS GEMV (shuffles, extraction, mul1 decode, mma, fp32 fold) with
//     trellis words from registers, so memory cannot limit it. Reported as weights/s and as the GB/s that rate
//     would stream at K bits per weight.
//  B) stream ceiling: read rate of candidate access patterns over a 1 GiB buffer.
// usage: tk-bench-ceiling (codec: mul1)
#include "kernels/trellis/codec_mul1.cuh"
#include "../codec-lab/proto_v2pair.cuh"

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

using namespace truss;

enum Mode { FULL = 0, NO_MMA = 1, NO_DECODE = 2, MMA_ONLY = 3 };

// one warp = WNT tile columns; iters k slices; words vary per slice so nothing is hoisted
template <class Codec, int WNT, int mode>
__global__ __launch_bounds__(256) void decode_bench(int iters, float * out)
{
    constexpr int LOADS = WNT / Codec::TILES_PER_VEC;
    const int lane = threadIdx.x % 32;
    const Codec codec(lane);
    uint32_t bw[LOADS];
    #pragma unroll
    for (int l = 0; l < LOADS; ++l) bw[l] = (blockIdx.x * 7919u + threadIdx.x * 104729u + l * 31u) | 1u;
    const half2 act_lo = __floats2half2_rn(0.01f * lane, 0.02f), act_hi = __floats2half2_rn(0.03f, -0.01f * lane);
    const half2 hzero = __half2half2(__ushort_as_half(0));
    FragCh ch[WNT];
    float2 acc[WNT][2];
    #pragma unroll
    for (int t = 0; t < WNT; ++t) { ch[t][0] = ch[t][1] = hzero; acc[t][0] = acc[t][1] = make_float2(0.f, 0.f); }
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int l = 0; l < LOADS; ++l) bw[l] += 0x9e3779b9u;
        #pragma unroll
        for (int t = 0; t < WNT; ++t) {
            FragB f0, f1;
            const uint32_t w = bw[t / Codec::TILES_PER_VEC];
            if constexpr (mode == MMA_ONLY)
                f0[0] = f0[1] = f1[0] = f1[1] = __halves2half2(__ushort_as_half((uint16_t) w), __ushort_as_half((uint16_t) (w >> 16)));
            else
                codec.tile(w, t % Codec::TILES_PER_VEC, f0, f1);
            if constexpr (mode == NO_MMA)
                ch[t][0] = __hadd2(ch[t][0], __hadd2(__hadd2(f0[0], f0[1]), __hadd2(f1[0], f1[1])));
            else
                mma_w(f0, f1, act_lo, act_hi, ch[t]);
        }
        if (i & 1) {
            #pragma unroll
            for (int t = 0; t < WNT; ++t)
                #pragma unroll
                for (int f = 0; f < 2; ++f) {
                    acc[t][f].x += __low2float(ch[t][f]); acc[t][f].y += __high2float(ch[t][f]); ch[t][f] = hzero;
                }
        }
    }
    float s = 0.f;
    #pragma unroll
    for (int t = 0; t < WNT; ++t) s += acc[t][0].x + acc[t][0].y + acc[t][1].x + acc[t][1].y + __low2float(ch[t][0]);
    if (s == 1234.5f) out[0] = s;   // keeps the work alive
}

// per_sm_cap > 0 limits resident blocks per SM (dynamic shared memory as ballast) to study occupancy.
// Reports the median of 5 timed runs (clocks on this box move 10-15% between runs).
template <class Codec, int bits, int WNT, int mode>
static void run_decode(const char * name, int sms, float * d_out, int per_sm_cap = 0)
{
    auto kern = decode_bench<Codec, WNT, mode>;
    size_t ballast = 0;
    if (per_sm_cap > 0) {
        CK(cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        ballast = 100 * 1024 / (per_sm_cap + 1) + 1024;
        CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) ballast));
    }
    int per_sm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, 256, ballast));
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, kern));
    const int blocks = sms * per_sm * 8, iters = 2000;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    kern<<<blocks, 256, ballast>>>(10, d_out);
    std::vector<float> t;
    for (int r = 0; r < 5; ++r) {
        CK(cudaEventRecord(e0));
        kern<<<blocks, 256, ballast>>>(iters, d_out);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    const double w = (double) blocks * 8 * iters * WNT * 256;
    const double wps = w / (t[2] * 1e-3);
    printf("decode %-13s K%d WNT%-2d %-9s regs %3d blk/SM %d | %6.2f Tw/s = %5.0f GB/s at K%d (%3.0f%% of 936) spread %.0f%%\n",
           Codec::NAME, bits, WNT, name, fa.numRegs, per_sm, wps / 1e12, wps * bits / 8 / 1e9, bits,
           100.0 * wps * bits / 8 / 936e9, 100.0 * (t[4] - t[0]) / t[2]);
}

// ---------------------------------------------------------------------------------------------------------
// streams. sum into a register so the loads are real.

__global__ void stream_flat(const uint4 * __restrict__ p, size_t n, uint32_t * out)
{
    uint32_t s = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = __ldcs(p + i);
        s ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (s == 0x12345678u) out[0] = s;
}

// each warp streams its own contiguous chunk of `chunk` bytes via cp.async into a STAGES-deep smem ring of 512 B
// stages (one 16 B cp.async per lane per stage), then reads it back from smem.
template <int STAGES>
__global__ __launch_bounds__(256) void stream_warp_cpasync(const char * __restrict__ p, size_t n_chunks, size_t chunk,
                                                           uint32_t * out)
{
    __shared__ __align__(16) char ring[8][STAGES][512];
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    uint32_t s = 0;
    const size_t gw = blockIdx.x * 8 + w, nw = (size_t) gridDim.x * 8;
    const int nst = (int) (chunk / 512);
    for (size_t c = gw; c < n_chunks; c += nw) {
        const char * src = p + c * chunk + lane * 16;
        auto issue = [&] (int st) {
            const uint32_t dst = (uint32_t) __cvta_generic_to_shared(&ring[w][st % STAGES][lane * 16]);
            if (st < nst) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src + (size_t) st * 512));
            asm volatile("cp.async.commit_group;\n");
        };
        #pragma unroll
        for (int st = 0; st < STAGES - 1; ++st) issue(st);
        for (int st = 0; st < nst; ++st) {
            issue(st + STAGES - 1);
            asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 1));
            __syncwarp();
            const uint4 v = *(const uint4 *) &ring[w][st % STAGES][lane * 16];
            s ^= v.x ^ v.y ^ v.z ^ v.w;
            __syncwarp();
        }
    }
    if (s == 0x12345678u) out[0] = s;
}

// today's pattern: each warp reads `run` contiguous bytes, then jumps `stride` bytes (k-slice-major tiles)
__global__ void stream_strided(const uint32_t * __restrict__ p, size_t n_words, int run_words, int stride_words,
                               int runs_per_warp, uint32_t * out)
{
    const int lane = threadIdx.x % 32;
    const size_t gw = (blockIdx.x * (size_t) blockDim.x + threadIdx.x) / 32;
    const int groups = stride_words / run_words;
    const size_t base = (gw / groups) * (size_t) stride_words * runs_per_warp + (gw % groups) * run_words;
    uint32_t s = 0;
    if (base + (size_t) stride_words * runs_per_warp > n_words) return;
    for (int r = 0; r < runs_per_warp; ++r)
        for (int j = lane; j < run_words; j += 32) s ^= __ldcs(p + base + (size_t) r * stride_words + j);
    if (s == 0x12345678u) out[0] = s;
}

int main()
{
    int dev = 0, sms;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, dev));
    int clk = 0;
    CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev));
    printf("%s, %d SMs, max clock %.0f MHz\n", pr.name, sms, clk / 1e3);
    float * d_out;
    CK(cudaMalloc(&d_out, 64));

    run_decode<Mul1<2>, 2, 8, FULL>("full", sms, d_out);
    run_decode<Mul1<2>, 2, 8, NO_MMA>("no-mma", sms, d_out);
    run_decode<Mul1<2>, 2, 8, MMA_ONLY>("mma-only", sms, d_out);
    run_decode<Mul1<2>, 2, 2, FULL>("full", sms, d_out);
    run_decode<Mul1<2>, 2, 4, FULL>("full", sms, d_out);
    run_decode<Mul1<2>, 2, 8, FULL>("full 2/SM", sms, d_out, 2);
    run_decode<Mul1<3>, 3, 8, FULL>("full 2/SM", sms, d_out, 2);
    run_decode<Mul1<3>, 3, 4, FULL>("full", sms, d_out);
    run_decode<Mul1<4>, 4, 4, FULL>("full", sms, d_out);
    run_decode<Mul1<3>, 3, 8, FULL>("full", sms, d_out);
    run_decode<Mul1<3>, 3, 8, NO_MMA>("no-mma", sms, d_out);
    run_decode<Mul1<4>, 4, 8, FULL>("full", sms, d_out);
    run_decode<Mul1<4>, 4, 8, NO_MMA>("no-mma", sms, d_out);
    run_decode<lab::V2Pair<2>, 2, 8, FULL>("full", sms, d_out);
    run_decode<lab::V2Pair<2>, 2, 8, NO_MMA>("no-mma", sms, d_out);
    run_decode<lab::V2Pair<3>, 3, 8, FULL>("full", sms, d_out);
    run_decode<lab::V2Pair<4>, 4, 8, FULL>("full", sms, d_out);

    const size_t bytes = (size_t) 1 << 30;
    char * buf;
    CK(cudaMalloc(&buf, bytes));
    CK(cudaMemset(buf, 1, bytes));
    uint32_t * d_u;
    CK(cudaMalloc(&d_u, 64));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto timeit = [&] (const char * name, auto launch) {
        launch();
        CK(cudaEventRecord(e0));
        for (int r = 0; r < 5; ++r) launch();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        ms /= 5;
        printf("stream %-34s %6.0f GB/s (%3.0f%% of 936)\n", name, bytes / (ms * 1e-3) / 1e9, 100.0 * bytes / (ms * 1e-3) / 936e9);
    };
    timeit("flat uint4 ldcs", [&] { stream_flat<<<sms * 8, 256>>>((const uint4 *) buf, bytes / 16, d_u); });
    for (size_t chunk : { (size_t) 2560, (size_t) 10240, (size_t) 40960 }) {
        char nm[64];
        snprintf(nm, sizeof nm, "warp cp.async 4st chunk %zu", chunk);
        timeit(nm, [&] { stream_warp_cpasync<4><<<sms * 3, 256>>>(buf, bytes / chunk, chunk, d_u); });
        snprintf(nm, sizeof nm, "warp cp.async 8st chunk %zu", chunk);
        timeit(nm, [&] { stream_warp_cpasync<8><<<sms * 3, 256>>>(buf, bytes / chunk, chunk, d_u); });
    }
    // today's gate/up K2 pattern: 512 B runs, 2560 B stride (40 tiles x 64 B per k slice), 20 slices per warp
    timeit("strided 512B/2560B (today K2 g/u)", [&] {
        const int runs = 20, groups = 5;
        const size_t warps = bytes / 4 / (640 * runs) * groups;
        stream_strided<<<(unsigned) (warps * 32 / 256), 256>>>((const uint32_t *) buf, bytes / 4, 128, 640, runs, d_u);
    });
    printf("done\n");
    return 0;
}
