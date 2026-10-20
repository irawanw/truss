// trellis_dense_bench: the trellis-dense window GEMV on real PAW-27B-X3.1 matrices, the same 9 shape groups as
// tqLLM's tests/x3_window_bench.cu (so the numbers can be compared with tqLLM's X3_STATUS "Phase T1/T2" table).
// Per group: the us per call at n = 1, 4, 8, 16 (graph-timed, best of 5) and the GB/s of the trellis bytes read; the
// 16-wide check: one 12-column call against 8 + 4.
//
// usage: trellis_dense_bench <PAW-27B-X3.1.gguf> [group ...]   (a group is "name+name", the matrices of one matvec)
#include "blocks/linear.h"
#include "engine/gpu.h"
#include "formats/gguf.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace truss;

namespace {

// Graph-timed: the launch overhead of three kernels per matvec would dominate the measurement.
float graph_us(const std::function<void()> & body, cudaStream_t s, int iters = 50)
{
    cudaGraph_t g;
    cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < iters; ++i) body();
    TRUSS_CUDA(cudaStreamEndCapture(s, &g));
    // the legacy 3-arg instantiate segfaults on this box (TRACKER #169): instantiate with flags
    TRUSS_CUDA(cudaGraphInstantiateWithFlags(&ge, g, 0));
    cudaGraphLaunch(ge, s);
    cudaStreamSynchronize(s);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    float best = 1e30f;
    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(a, s);
        cudaGraphLaunch(ge, s);
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        best = std::min(best, ms * 1000.f / iters);
    }
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return best;
}

}  // namespace

int main(int argc, char ** argv)
{
    if (argc < 2) {
        printf("usage: %s <gguf> [group ...]\n", argv[0]);
        return 1;
    }
    std::vector<std::string> groups;
    for (int i = 2; i < argc; ++i) groups.push_back(argv[i]);
    if (groups.empty())
        groups = { "blk.0.ffn_gate+blk.0.ffn_up", "blk.0.ffn_gate",  "blk.0.ffn_down", "blk.0.attn_qkv",
                    "blk.0.attn_gate",  "blk.0.ssm_out",     "blk.3.attn_q",  "blk.3.attn_k+blk.3.attn_v",
                    "blk.3.attn_output" };

    std::unique_ptr<gguf::File> f = gguf::File::open(argv[1]);
    engine::Gpu g(0);
    cudaStream_t s = g.s;

    printf("%-34s %8s %10s %10s %10s %10s %8s\n", "group", "rate", "n=1 us", "n=4 us", "n=8 us", "n=16 us", "GB/s");
    double sum1 = 0;
    for (const std::string & grp : groups) {
        // the matrices of the group
        std::vector<std::string> names;
        size_t p = 0;
        while (p <= grp.size()) {
            const size_t q = grp.find('+', p);
            names.push_back(grp.substr(p, q == std::string::npos ? std::string::npos : q - p));
            if (q == std::string::npos) break;
            p = q + 1;
        }
        blocks::Linear lin(g, blocks::Activations::FP16);
        std::vector<const gguf::Tensor *> ws;
        int in = 0, out = 0, k = 0;
        size_t bytes = 0;
        for (const std::string & n : names) {
            const gguf::Tensor * w = f->find(n + ".m3_trellis");
            if (!w || !w->trellis) {
                printf("%-34s SKIP (no group)\n", n.c_str());
                ws.clear();
                break;
            }
            ws.push_back(w);
            in = (int) w->trellis->in;
            out += (int) w->trellis->out;
            k = w->trellis->k;
            bytes += (size_t) w->trellis->trellis->bytes;
        }
        if (ws.empty()) continue;
        for (const gguf::Tensor * w : ws) lin.add(w);
        lin.upload(nullptr, {});

        // the input and the outputs
        half * dx = nullptr;
        TRUSS_CUDA(cudaMalloc(&dx, (size_t) 16 * in * sizeof(half)));
        TRUSS_CUDA(cudaMemset(dx, 0, (size_t) 16 * in * sizeof(half)));
        std::vector<float *> ys;
        for (const gguf::Tensor * w : ws) {
            float * y = nullptr;
            TRUSS_CUDA(cudaMalloc(&y, (size_t) 16 * w->trellis->out * sizeof(float)));
            ys.push_back(y);
        }
        std::vector<blocks::Linear::Proj> ps;
        for (size_t i = 0; i < ws.size(); ++i) ps.push_back({ ws[i], ys[i] });

        float us[4] = { 0, 0, 0, 0 };
        const int ns[4] = { 1, 4, 8, 16 };
        for (int i = 0; i < 4; ++i) {
            us[i] = graph_us([&] { lin.lin_multi(ps, lin.quant(dx, ns[i], in), ns[i]); }, s);
        }
        // the 16-wide check: one 12-column call against 8 + 4
        const float u12 = graph_us([&] { lin.lin_multi(ps, lin.quant(dx, 12, in), 12); }, s);
        const float u8 = graph_us([&] { lin.lin_multi(ps, lin.quant(dx, 8, in), 8); }, s);
        const float u4 = graph_us([&] { lin.lin_multi(ps, lin.quant(dx, 4, in), 4); }, s);
        sum1 += us[0];
        printf("%-34s %8d %10.1f %10.1f %10.1f %10.1f %8.0f   12col %.0f vs 8+4 %.0f (%.2fx)\n", grp.c_str(), k, us[0],
               us[1], us[2], us[3], bytes / (us[0] * 1e3), u12, u8 + u4, (u8 + u4) / u12);
        cudaFree(dx);
        for (float * y : ys) cudaFree(y);
    }
    printf("sum over the groups, n=1: %.1f us\n", sum1);
    return 0;
}
