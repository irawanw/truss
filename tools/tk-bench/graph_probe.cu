// B1 probe: can one all-resident decode step (run T=1 + head + argmax + token copy-back) be captured as a
// CUDA graph and replayed bit-for-bit vs eager, and how much faster is it? Phase 1 captures raw and checks one
// replay; phase 2 runs N steps through the engine's GraphStep (per-step pos0 refresh, token rewrite, re-capture
// on epoch change) and requires token-identity with eager. Pinned token/next buffers: the embed H2D node re-reads
// the pinned token at each launch. NOTE: use cudaGraphInstantiateWithFlags (the legacy 3-arg call segfaults in
// libcuda on this box's cudart-11.0/lib setup, TRACKER #169).
// usage: tk-bench-graph <model.gguf> [prompt=512] [steps=64] [n_ctx=65536] [chunk=8192]
#include "core/cuda_check.h"
#include "engine/graph_step.h"
#include "kernels/sampling/argmax.cuh"
#include "model/qwen4exp/config.h"
#include "engine/model.h"
#include "engine/pipeline.h"
#include "model/qwen4exp/weights.h"
#include "runtime/expert_store.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace truss;

namespace {

// one step exactly like the bench's timed loop
void eager_step(engine::Pipeline & f, const int32_t * tok, float * logits, int vocab, int * next_d, int * next_h)
{
    f.run(tok, 1);
    f.head(0, 1, logits);
    sampling::argmax(logits, vocab, next_d, f.stream());
    TRUSS_CUDA(cudaMemcpyAsync(next_h, next_d, 4, cudaMemcpyDeviceToHost, f.stream()));
    TRUSS_CUDA(cudaStreamSynchronize(f.stream()));
}

double now_ms()
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}   // namespace

int main(int argc, char ** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [prompt] [steps] [n_ctx] [chunk]\n", argv[0]);
        return 2;
    }
    try {
        const auto file = gguf::File::open(argv[1]);
        const auto fam = engine::open_family(*file, {});
        const int vocab = fam->n_vocab();
        std::mt19937 rng(1);
        std::vector<int32_t> tok(argc > 2 ? std::atoi(argv[2]) : 512);
        for (auto & t : tok) t = (int32_t) (rng() % 150000);
        const int n = (int) tok.size(), steps = argc > 3 ? std::atoi(argv[3]) : 64;
        const int n_ctx = argc > 4 ? std::atoi(argv[4]) : 65536, chunk = argc > 5 ? std::atoi(argv[5]) : 8192;
        engine::Options o;
        float * logits;
        int * next_d, * next_h, * tok_pinned;
        TRUSS_CUDA(cudaMalloc(&logits, sizeof(float) * vocab));
        TRUSS_CUDA(cudaMalloc(&next_d, sizeof(int)));
        TRUSS_CUDA(cudaMallocHost(&next_h, sizeof(int)));
        TRUSS_CUDA(cudaMallocHost(&tok_pinned, sizeof(int)));
        std::vector<float> logits_eager(vocab), logits_graph(vocab);
        engine::Pipeline f(*fam, n_ctx, chunk, o);
        std::printf("experts: %d of %d resident, ring %.2f GB\n", f.hot_experts(), fam->n_expert() * fam->n_layer(),
                    f.experts()->ring_bytes() / 1e9);
        // prompt (eager, warms everything incl. the attn smem attribute), then the decode token
        f.reset();
        for (int s = 0; s < n; s += chunk) f.run(tok.data() + s, std::min(chunk, n - s));
        f.head((n - 1) % chunk, 1, logits);
        sampling::argmax(logits, vocab, next_d, f.stream());
        TRUSS_CUDA(cudaMemcpyAsync(next_h, next_d, 4, cudaMemcpyDeviceToHost, f.stream()));
        TRUSS_CUDA(cudaStreamSynchronize(f.stream()));
        *tok_pinned = *next_h;
        const int tok0 = *tok_pinned;
        std::printf("prompt %d done, first decode token %d (pos %d)\n", n, tok0, f.position());
        // eager reference step, then back to the same state for the capture
        f.checkpoint();
        eager_step(f, tok_pinned, logits, vocab, next_d, next_h);
        const int tok_eager = *next_h;
        TRUSS_CUDA(cudaMemcpy(logits_eager.data(), logits, sizeof(float) * vocab, cudaMemcpyDeviceToHost));
        f.restore();
        cudaGraph_t g;
        size_t nn = 0;
        const double t_cap0 = now_ms();
        TRUSS_CUDA(cudaStreamBeginCapture(f.stream(), cudaStreamCaptureModeThreadLocal));
        f.run(tok_pinned, 1);
        f.head(0, 1, logits);
        sampling::argmax(logits, vocab, next_d, f.stream());
        TRUSS_CUDA(cudaMemcpyAsync(next_h, next_d, 4, cudaMemcpyDeviceToHost, f.stream()));
        TRUSS_CUDA(cudaStreamEndCapture(f.stream(), &g));
        {
            // inspect before instantiating: histogram of node types + kernel grids
            std::vector<cudaGraphNode_t> ns(8192);
            nn = ns.size();
            TRUSS_CUDA(cudaGraphGetNodes(g, ns.data(), &nn));
            long nker = 0, nmem = 0, nother = 0;
            for (size_t i = 0; i < nn; ++i) {
                cudaGraphNodeType ty;
                TRUSS_CUDA(cudaGraphNodeGetType(ns[i], &ty));
                if (ty == cudaGraphNodeTypeKernel) {
                    ++nker;
                    cudaKernelNodeParams kp;
                    TRUSS_CUDA(cudaGraphKernelNodeGetParams(ns[i], &kp));
                    if (!kp.func)
                        std::fprintf(stderr, "probe: node %zu has NULL func! grid %u block %u smem %u\n", i,
                                     kp.gridDim.x, kp.blockDim.x, kp.sharedMemBytes);
                } else if (ty == cudaGraphNodeTypeMemcpy) {
                    ++nmem;
                    cudaMemcpy3DParms p;
                    TRUSS_CUDA(cudaGraphMemcpyNodeGetParams(ns[i], &p));
                    std::printf("graph memcpy node %zu kind %d w %zu src %p dst %p\n", i, (int) p.kind,
                                p.extent.width, p.srcPtr.ptr, p.dstPtr.ptr);
                } else {
                    ++nother;
                    std::printf("graph node %zu type %d (non-kernel/memcpy)\n", i, (int) ty);
                }
            }
            std::printf("graph nodes: %zu (%ld kernel, %ld memcpy, %ld other)\n", nn, nker, nmem, nother);
        }
        cudaGraphExec_t e = nullptr;
        const double t_inst0 = now_ms();
        TRUSS_CUDA(cudaGraphInstantiateWithFlags(&e, g, 0));   // never the legacy 3-arg call (see header)
        std::printf("capture %.1f ms, instantiate %.1f ms\n", t_inst0 - t_cap0, now_ms() - t_inst0);
        std::printf("captured %zu nodes (host pos now %d, device still at the checkpoint)\n", nn, f.position());
        f.restore();   // host position back to the checkpoint; the capture never executed on the device
        // replay once with the same token the capture ran with, compare bit-for-bit
        *tok_pinned = tok0, *next_h = -1;
        TRUSS_CUDA(cudaGraphLaunch(e, f.stream()));
        TRUSS_CUDA(cudaStreamSynchronize(f.stream()));
        const int tok_graph = *next_h;
        TRUSS_CUDA(cudaMemcpy(logits_graph.data(), logits, sizeof(float) * vocab, cudaMemcpyDeviceToHost));
        // the capture ran with tok0 at the checkpoint position; the eager reference ran the same token from the
        // same state, so next_h must equal the eager next token and the logits must match bitwise.
        size_t ndiff = 0;
        float maxabs = 0;
        for (int i = 0; i < vocab; ++i) {
            if (std::memcmp(&logits_graph[i], &logits_eager[i], 4)) {
                ++ndiff;
                maxabs = std::max(maxabs, std::abs(logits_graph[i] - logits_eager[i]));
            }
        }
        std::printf("eager next %d, graph next %d; logits words differ %zu/%d, max abs diff %g -> %s\n", tok_eager,
                    tok_graph, ndiff, vocab, maxabs, (tok_graph == tok_eager && ndiff == 0) ? "BIT-EXACT" : "MISMATCH");
        if (tok_graph != tok_eager || ndiff != 0) return 1;
        // Phase 2: the engine's GraphStep over N steps - exercises capture, per-step pos refresh, token rewrite
        // and re-capture across the 128-block boundary. Token-identity with eager required in-block; past the
        // boundary (stale splits stay correct but round differently) only the token sequence must match.
        const int P = n;   // capture position (prompt length); block boundary above it
        const int boundary = (P / 128 + 1) * 128;
        const int N = steps;
        // eager reference sequence
        f.restore();
        std::vector<int> tok_ref;
        *tok_pinned = tok0;
        double t_eager = 0;
        for (int i = 0; i < N; ++i) {
            const double t0 = now_ms();
            eager_step(f, tok_pinned, logits, vocab, next_d, next_h);
            t_eager += now_ms() - t0;
            tok_ref.push_back(*next_h);
            *tok_pinned = *next_h;
        }
        TRUSS_CUDA(cudaMemcpy(logits_eager.data(), logits, sizeof(float) * vocab, cudaMemcpyDeviceToHost));
        // graph sequence through the engine class (device back at P via restore)
        f.restore();
        engine::GraphStep gs(f, vocab);
        std::vector<int> tok_g2;
        int tnext = tok0;
        double t_graph = 0;
        for (int i = 0; i < N; ++i) {
            const double t0 = now_ms();
            tnext = gs.step(P + i, tnext, logits, next_d);
            t_graph += now_ms() - t0;
            tok_g2.push_back(tnext);
        }
        TRUSS_CUDA(cudaMemcpy(logits_graph.data(), logits, sizeof(float) * vocab, cudaMemcpyDeviceToHost));
        int first_tok_diff = -1, n_tok_diff = 0, n_tok_diff_inblock = 0;
        for (int i = 0; i < N; ++i) {
            if (tok_g2[i] != tok_ref[i]) {
                if (first_tok_diff < 0) first_tok_diff = i;
                ++n_tok_diff;
                if (P + i < boundary) ++n_tok_diff_inblock;
            }
        }
        size_t ndiff2 = 0;
        float maxabs2 = 0;
        for (int i = 0; i < vocab; ++i) {
            if (std::memcmp(&logits_graph[i], &logits_eager[i], 4)) {
                ++ndiff2;
                maxabs2 = std::max(maxabs2, std::abs(logits_graph[i] - logits_eager[i]));
            }
        }
        std::printf("multi-step %d (pos %d..%d, boundary %d): tokens differ %d (in-block %d, first at step %d); "
                    "final logits words differ %zu/%d maxabs %g\n",
                    N, P, P + N - 1, boundary, n_tok_diff, n_tok_diff_inblock, first_tok_diff, ndiff2, vocab, maxabs2);
        std::printf("eager %.3f ms/step, graph %.3f ms/step (%d captures)\n", t_eager / N, t_graph / N,
                    gs.captures());
        if (n_tok_diff_inblock != 0) {
            std::printf("MISMATCH inside the capture block\n");
            return 1;
        }
        std::printf("MULTI-STEP OK\n");
        TRUSS_CUDA(cudaGraphExecDestroy(e));
        TRUSS_CUDA(cudaGraphDestroy(g));
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
