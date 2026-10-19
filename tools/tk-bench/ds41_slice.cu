// Cheap DeepSeek-V4.1 speed probe: layers [0, N) as one engine::Stage on one GPU (only those layers' experts load,
// no head), a prompt of T real tokens (prefill), then S decode steps (T = 1 chunks, the next prompt tokens as input:
// teacher-forced, so routing follows real text). Per-layer ms for both, plus the section split (TRUSS_PROFILE_SECTIONS
// is set here): 0 Engram + mHC mix / collapse + norm, 1 attention, 2 mHC post + FFN mix / collapse + norm + router,
// 3..5 the MoE block's own sections. Seconds to run instead of a whole-model load on two GPUs; whole-model
// estimates scale the per-layer numbers (layers 0-1 are window-only, so N >= 4 to include compressed + indexer layers).
// TRUSS_BENCH_PROFRANGE=prefill|decode: cudaProfilerStart/Stop around that timed part (nsys --capture-range=cudaProfilerApi).
// TRUSS_EXPERT_GB caps the resident experts to mimic the served residency (the rest go through the ring / CPU tier).
// usage: TRUSS_EXPERT_OVERLAY=<experts>,<engram L1>,<engram L14> tk-bench-ds41-slice <dense.gguf> <ids.i32> <N>
//          [T=2048] [S=64] [chunk=T] [usage.f32|-]
#include "core/cuda_check.h"
#include "engine/model.h"
#include "engine/options.h"
#include "engine/stage.h"
#include "runtime/expert_store.h"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace truss;

static double now_ms()
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <dense.gguf> <ids.i32> <n_layers> [T=2048] [steps=64] [chunk=T] [usage|-]\n", argv[0]);
        return 2;
    }
    try {
        setenv("TRUSS_PROFILE_SECTIONS", "1", 0);
        const int N = std::atoi(argv[3]), T = argc > 4 ? std::atoi(argv[4]) : 2048, S = argc > 5 ? std::atoi(argv[5]) : 64;
        const int chunk = argc > 6 ? std::atoi(argv[6]) : T;
        std::ifstream in(argv[2], std::ios::binary | std::ios::ate);
        const size_t have = (size_t) in.tellg() / 4;
        if (have < (size_t) T + S) throw std::runtime_error("ids file has " + std::to_string(have) + " tokens, need T + steps");
        std::vector<int32_t> ids(T + S);
        in.seekg(0);
        in.read((char *) ids.data(), ids.size() * 4);
        const auto file = gguf::File::open(argv[1]);
        if (const char * ov = std::getenv("TRUSS_EXPERT_OVERLAY")) file->overlay_list(ov);
        const auto fam = engine::open_family(*file, {});
        engine::Options o;
        o.act = blocks::Activations::FP16;
        o.prefill_rows = chunk;
        engine::apply_env(o);
        if (argc > 7 && std::string(argv[7]) != "-")
            o.expert_usage = runtime::ExpertStore::load_usage(argv[7], fam->n_layer(), fam->n_expert());
        const double t_load = now_ms();
        engine::Stage st(*fam, T + S, chunk, o, engine::StageSpec{ 0, N, -1, 1.0, nullptr, nullptr });
        file->release_pages();
        std::printf("slice layers 0-%d on device %d: load %.1f s\n", N - 1, st.device(), (now_ms() - t_load) / 1e3);
        const char * name[6] = { "engram+hc+norm", "attention", "hc post+ffn mix+router", "moe a", "moe b", "moe c" };
        auto sections = [&](const char * what, double per) {
            double s[6];
            st.prof().ms(s, true);
            std::printf("  %s sections, ms per layer per token:", what);
            for (int i = 0; i < 6; ++i) std::printf(" %s %.3f |", name[i], s[i] / N / per);
            std::printf("\n");
        };
        const char * pr = std::getenv("TRUSS_BENCH_PROFRANGE");
        const std::string range = pr ? pr : "";
        st.select(0);
        for (int pass = 0; pass < 2; ++pass) {   // warm-up, then timed
            st.reset();
            double s0[6];
            st.prof().ms(s0, true);
            TRUSS_CUDA(cudaDeviceSynchronize());
            const double p0 = now_ms();
            if (pass && range == "prefill") cudaProfilerStart();
            for (int p = 0; p < T; p += chunk) st.run_chunk(ids.data() + p, p, std::min(chunk, T - p), nullptr, false);
            TRUSS_CUDA(cudaDeviceSynchronize());
            if (pass && range == "prefill") cudaProfilerStop();
            const double pms = now_ms() - p0;
            std::vector<double> ms;
            double d0[6];
            st.prof().ms(d0, false);
            if (pass) {
                std::printf("prefill %d tokens (chunk %d): %.1f ms = %.3f ms per layer = %.0f tok/s for this slice, "
                            "%.0f tok/s at 40 such layers (est.)\n", T, chunk, pms, pms / N, 1e3 * T / pms,
                            1e3 * T / (pms / N * 40));
                sections("prefill", T);
            }
            st.prof().ms(d0, true);
            if (pass && range == "decode") cudaProfilerStart();
            for (int s = 0; s < S; ++s) {
                const double t0 = now_ms();
                st.run_chunk(ids.data() + T + s, T + s, 1, nullptr, false);
                TRUSS_CUDA(cudaStreamSynchronize(st.gpu().s));
                ms.push_back(now_ms() - t0);
            }
            if (pass && range == "decode") cudaProfilerStop();
            if (!pass) continue;
            std::vector<double> v = ms;
            std::sort(v.begin(), v.end());
            double sum = 0;
            for (double x : ms) sum += x;
            std::printf("decode %d steps: mean %.3f ms (%.3f per layer), median %.3f, min %.3f, max %.3f; "
                        "x40 layers (est.) %.1f ms = %.1f tok/s before head and hand-off\n", S, sum / S, sum / S / N,
                        v[S / 2], v[0], v[S - 1], sum / S / N * 40, 1e3 / (sum / S / N * 40));
            sections("decode", S);
            const runtime::ExpertStore::Stats & es = st.moe() ? st.moe()->store().stats() : runtime::ExpertStore::Stats {};
            std::printf("experts: %d of %d resident in the slice; store totals since load: %ld asked, %ld fetched "
                        "(%.1f MB), %ld hinted (%.1f MB), admitted %.1f MB\n", st.moe() ? st.moe()->n_hot() : 0,
                        fam->n_expert() * N, (long) es.experts_asked, (long) es.misses, es.bytes / 1e6,
                        (long) es.hinted, es.hint_bytes / 1e6, es.admit_bytes / 1e6);
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
