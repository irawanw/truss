// DSA prefill attention timing on REAL selections (lead 10-07): blocks / n_blocks dumped from the engine
// (TRUSS_DSA_DUMP, header pos0 T TOP, then n[T], blocks[T][TOP]); random fp16-exact q, gate and an int8 KV cache.
// Run with TRUSS_DSA_MQ=0 and =1; each run writes its output so the two can be compared.
// usage: dsa_mq_bench <dump> <out.f16> [reps]
#include "core/cuda_check.h"
#include "kernels/dsa/dsa_prefill.cuh"

#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace truss;
using Shape = dsa::FlashNext;

int main(int argc, char ** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage: dsa_mq_bench <dump> <out> [reps]\n"); return 2; }
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;
    FILE * f = std::fopen(argv[1], "rb");
    int hdr[3];
    if (!f || std::fread(hdr, 4, 3, f) != 3) { std::fprintf(stderr, "bad dump\n"); return 2; }
    const int pos0 = hdr[0], T = hdr[1], TOP = hdr[2];
    std::vector<int> hn(T), hb((size_t) T * TOP);
    if (std::fread(hn.data(), 4, T, f) != (size_t) T || std::fread(hb.data(), 4, hb.size(), f) != hb.size()) return 2;
    std::fclose(f);
    const int H = Shape::H, HKV = Shape::HKV, D = Shape::D, n_ctx = pos0 + T;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto up = [](const auto & v) {
        using V = typename std::decay_t<decltype(v)>::value_type;
        V * d; TRUSS_CUDA(cudaMalloc(&d, v.size() * sizeof(V)));
        TRUSS_CUDA(cudaMemcpy(d, v.data(), v.size() * sizeof(V), cudaMemcpyHostToDevice));
        return d;
    };
    std::vector<half> hq((size_t) T * H * D);
    for (auto & x : hq) x = __float2half(nd(rng));
    std::vector<float> hg((size_t) T * H * D);
    for (auto & x : hg) x = nd(rng);
    const size_t nkv = (size_t) n_ctx * HKV * D;
    std::vector<int8_t> hk(nkv), hv(nkv);
    for (size_t i = 0; i < nkv; ++i) hk[i] = (int8_t) (rng() % 255 - 127), hv[i] = (int8_t) (rng() % 255 - 127);
    std::vector<half> hs(nkv / dsa::KV_GROUP);
    for (auto & x : hs) x = __float2half(0.01f + 0.01f * (rng() % 100) / 100.f);
    half * q = up(hq);
    float * gate = up(hg);
    dsa::KvCache kc;
    kc.kq = up(hk), kc.vq = up(hv), kc.ks = up(hs), kc.vs = up(hs);
    int * blocks = up(hb), * n_blocks = up(hn);
    half * out;
    TRUSS_CUDA(cudaMalloc(&out, sizeof(half) * T * H * D));
    dsa::attention<Shape>(q, gate, kc, blocks, n_blocks, pos0, T, out, nullptr, 0, nullptr);
    TRUSS_CUDA(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) dsa::attention<Shape>(q, gate, kc, blocks, n_blocks, pos0, T, out, nullptr, 0, nullptr);
    cudaEventRecord(e1);
    TRUSS_CUDA(cudaEventSynchronize(e1));
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    const char * mq = std::getenv("TRUSS_DSA_MQ");
    std::printf("pos0 %d T %d: attention %.2f ms/layer (TRUSS_DSA_MQ=%s)\n", pos0, T, ms / reps, mq ? mq : "0");
    std::vector<half> ho((size_t) T * H * D);
    TRUSS_CUDA(cudaMemcpy(ho.data(), out, ho.size() * 2, cudaMemcpyDeviceToHost));
    FILE * o = std::fopen(argv[2], "wb");
    std::fwrite(ho.data(), 2, ho.size(), o);
    std::fclose(o);
    return 0;
}
