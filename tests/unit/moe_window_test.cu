// truss::moe::window: correctness vs llama-paw's unfused PAW_X3 chain (mm_id gate/up -> swiglu -> mm_id down ->
// moe_reduce) on the same random trellis weights, and timing inside a CUDA graph.
// Shape: Flash-Next layer (2560 x 640, 512 experts, top-10). Env: TRUSS_KFIX (0 = mixed 2..4), TRUSS_REPS.
// usage: moe_window_test [n_rows ...]
#include "kernels/moe/moe_window.cuh"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using Shape = truss::moe::FlashNext;

static int64_t env_int(const char * n, int64_t d) { const char * e = getenv(n); return e ? atoll(e) : d; }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static int run_case(ggml_backend_t backend, int64_t n_tokens, std::mt19937 & rng, void * ws)
{
    const int64_t n_embd = Shape::D_MODEL, n_ff = Shape::D_FF, n_expert = 512, n_used = Shape::TOPK;
    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead() * 3, NULL, true };
    ggml_context * ctx = ggml_init(ip);

    struct proj_host { int64_t in, out; std::vector<int32_t> meta; int64_t words; };
    proj_host ph[3];
    ggml_tensor * proj[12];
    const int64_t kfix = env_int("TRUSS_KFIX", 0);
    for (int p = 0; p < 3; ++p) {
        ph[p].in = p == 2 ? n_ff : n_embd;
        ph[p].out = p == 2 ? n_embd : n_ff;
        ph[p].meta.resize(2 * n_expert);
        const int64_t ntiles = (ph[p].in / 16) * (ph[p].out / 16);
        int64_t words = 0;
        for (int64_t e = 0; e < n_expert; ++e) {
            ph[p].meta[2 * e] = kfix ? (int32_t) kfix : 2 + (int32_t) ((e + p) % 3);
            ph[p].meta[2 * e + 1] = (int32_t) words;
            words += 16 * ph[p].meta[2 * e] * ntiles;
        }
        ph[p].words = words;
        proj[4 * p + 0] = ggml_new_tensor_1d(ctx, GGML_TYPE_I16, words);
        proj[4 * p + 1] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_expert);
        proj[4 * p + 2] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ph[p].in, n_expert);
        proj[4 * p + 3] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ph[p].out, n_expert);
    }
    ggml_tensor * t_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * t_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * t_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, n_tokens);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * x3d = ggml_reshape_3d(ctx, t_x, n_embd, 1, n_tokens);
    ggml_tensor * gate = ggml_paw_x3_mm_id(ctx, proj[0], proj[1], proj[2], proj[3], t_ids, x3d);
    ggml_tensor * up = ggml_paw_x3_mm_id(ctx, proj[4], proj[5], proj[6], proj[7], t_ids, x3d);
    ggml_tensor * par = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * down = ggml_paw_x3_mm_id(ctx, proj[8], proj[9], proj[10], proj[11], t_ids, par);
    ggml_tensor * ref = ggml_paw_moe_reduce(ctx, down, t_w);
    ggml_build_forward_expand(gf, ref);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::uniform_real_distribution<float> mag(0.25f, 1.0f);
    for (int p = 0; p < 3; ++p) {
        std::vector<int16_t> tw(ph[p].words);
        for (auto & w : tw) w = (int16_t) (rng() & 0xffff);
        ggml_backend_tensor_set(proj[4 * p + 0], tw.data(), 0, ggml_nbytes(proj[4 * p + 0]));
        ggml_backend_tensor_set(proj[4 * p + 1], ph[p].meta.data(), 0, ggml_nbytes(proj[4 * p + 1]));
        for (int s = 2; s < 4; ++s) {
            ggml_tensor * t_s = proj[4 * p + s];
            std::vector<ggml_fp16_t> v(ggml_nelements(t_s));
            for (auto & h : v) h = ggml_fp32_to_fp16(((rng() & 1) ? 0.05f : -0.05f) * mag(rng));
            ggml_backend_tensor_set(t_s, v.data(), 0, ggml_nbytes(t_s));
        }
    }
    // realistic routing: distinct experts per row, overlap between rows like a verify window (~35 unique at 4 rows)
    std::vector<int32_t> ids(n_used * n_tokens);
    std::vector<float> w(n_used * n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        std::vector<int32_t> pick;
        while ((int64_t) pick.size() < n_used) {
            const int32_t e = (t > 0 && (rng() % 8) == 0) ? ids[(t - 1) * n_used + rng() % n_used] : (int32_t) (rng() % n_expert);
            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
        }
        float sum = 0.0f;
        for (int64_t s = 0; s < n_used; ++s) { ids[t * n_used + s] = pick[s]; w[t * n_used + s] = mag(rng); sum += w[t * n_used + s]; }
        for (int64_t s = 0; s < n_used; ++s) w[t * n_used + s] /= sum;
    }
    ggml_backend_tensor_set(t_ids, ids.data(), 0, ggml_nbytes(t_ids));
    ggml_backend_tensor_set(t_w, w.data(), 0, ggml_nbytes(t_w));
    {
        std::vector<float> v(ggml_nelements(t_x));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto & f : v) f = nd(rng);
        ggml_backend_tensor_set(t_x, v.data(), 0, ggml_nbytes(t_x));
    }
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    truss::moe::Weights W;
    W.n_expert = (int) n_expert;
    for (int p = 0; p < 3; ++p)
        W.proj[p] = { (const uint16_t *) proj[4 * p]->data, (const int32_t *) proj[4 * p + 1]->data,
                      (const half *) proj[4 * p + 2]->data, (const half *) proj[4 * p + 3]->data };
    float * d_out;
    CK(cudaMalloc(&d_out, sizeof(float) * n_tokens * n_embd));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    // NaN-fill the workspace and output so any output the kernels skip fails the check (stale data once passed)
    CK(cudaMemset(ws, 0xff, truss::moe::workspace_bytes<Shape>()));
    CK(cudaMemset(d_out, 0xff, sizeof(float) * n_tokens * n_embd));
    truss::moe::workspace_init<Shape>(ws, st);
    truss::moe::window<Shape>(W, (const float *) t_x->data, (const int *) t_ids->data, (const float *) t_w->data,
                      (int) n_tokens, d_out, ws, st);
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());

    std::vector<float> b(ggml_nelements(ref));
    ggml_backend_tensor_get(ref, b.data(), 0, ggml_nbytes(ref));
    double worst_rel = 0.0, worst_cos = 1.0;
    int64_t nonfinite = 0;
    auto check = [&] {
        std::vector<float> a(n_tokens * n_embd);
        CK(cudaMemcpy(a.data(), d_out, sizeof(float) * a.size(), cudaMemcpyDeviceToHost));
        worst_rel = 0.0; worst_cos = 1.0; nonfinite = 0;
        for (int64_t t = 0; t < n_tokens; ++t) {
            double dd = 0, rr = 0, aa = 0, ar = 0;
            for (int64_t i = 0; i < n_embd; ++i) {
                const double av = a[t * n_embd + i], bv = b[t * n_embd + i];
                nonfinite += !std::isfinite(av);
                dd += (av - bv) * (av - bv); rr += bv * bv; aa += av * av; ar += av * bv;
            }
            worst_rel = std::max(worst_rel, std::sqrt(dd / std::max(rr, 1e-30)));
            worst_cos = std::min(worst_cos, ar / std::sqrt(std::max(aa * rr, 1e-30)));
        }
        return !nonfinite && worst_rel <= 0.02 && worst_cos >= 0.9998;
    };
    const bool ok_first = check();

    // bytes the window must read: each distinct routed expert once, all three projections, at its rate
    std::vector<char> seen(n_expert, 0);
    int64_t uniq = 0, tbytes = 0;
    for (int64_t i = 0; i < n_used * n_tokens; ++i) {
        const int32_t e = ids[i];
        if (seen[e]) continue;
        seen[e] = 1; ++uniq;
        for (int p = 0; p < 3; ++p) tbytes += ph[p].in * ph[p].out * ph[p].meta[2 * e] / 8;
    }
    const int reps = (int) env_int("TRUSS_REPS", 300);
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    truss::moe::window<Shape>(W, (const float *) t_x->data, (const int *) t_ids->data, (const float *) t_w->data,
                      (int) n_tokens, d_out, ws, st);
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaGraphLaunch(ge, st));
    CK(cudaEventRecord(e0, st));
    for (int r = 0; r < reps; ++r) CK(cudaGraphLaunch(ge, st));
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    ms /= reps;
    // the counters reset themselves: the output after all replays must still match
    CK(cudaMemset(d_out, 0xff, sizeof(float) * n_tokens * n_embd));
    CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    const bool ok_replay = check();
    const bool ok = ok_first && ok_replay;
    printf("rows=%-2lld K=%s uniq=%-3lld %.1f MB | rel_rms %.5f cos %.8f %s | %.1f us/layer, %.0f GB/s (%.0f%% of 936)\n",
           (long long) n_tokens, kfix ? std::to_string(kfix).c_str() : "mix2-4", (long long) uniq, tbytes / 1e6,
           worst_rel, worst_cos, ok ? "PASS" : ok_first ? "FAIL(replay)" : "FAIL", ms * 1e3, tbytes / (ms * 1e-3) / 1e9,
           100.0 * tbytes / (ms * 1e-3) / 936e9);

    cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(st); cudaFree(d_out);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv)
{
    std::vector<int64_t> rows;
    for (int i = 1; i < argc; ++i) rows.push_back(atoll(argv[i]));
    if (rows.empty()) rows = { 1, 2, 4, 5, 8 };
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    void * ws;
    CK(cudaMalloc(&ws, truss::moe::workspace_bytes<Shape>()));
    std::mt19937 rng(4321);
    int fails = 0;
    for (int64_t n : rows) fails += run_case(backend, n, rng, ws);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
