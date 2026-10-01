// Speculative (MTP) greedy decode vs plain greedy decode on the full model: the same prompt, N generated tokens each
// way; the token sequences must be identical (verify windows, rollback and the MTP cache must not change the
// output), and the speculative run's speed and tokens per verify pass are printed. Exit code 1 on a mismatch.
// usage: tk-bench-spec <model.gguf> <mtp.gguf> @prompt.i32 [tokens=128] [drafts=3] [usage file|-] [n_ctx=65536]
//                      [chunk=8192] [draft vocab ids file|-] [cpu tier dir|-] [draft min p=0]
#include "core/cuda_check.h"
#include "kernels/sampling/argmax.cuh"
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/forward.h"
#include "model/qwen4exp/weights.h"
#include "runtime/expert_store.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace truss;
namespace q = truss::qwen4exp;

namespace {

std::vector<int32_t> read_ids(const char * arg)
{
    if (arg[0] != '@') throw std::runtime_error(std::string("prompt must be @file.i32: ") + arg);
    FILE * f = std::fopen(arg + 1, "rb");
    if (!f) throw std::runtime_error(std::string("cannot open ") + (arg + 1));
    std::vector<int32_t> v;
    int32_t t;
    while (std::fread(&t, 4, 1, f) == 1) v.push_back(t);
    std::fclose(f);
    return v;
}

struct Greedy {   // argmax of logit rows on the device, copied back
    int vocab;
    float * logits;
    int * d_ids;
    int * h_ids;
    explicit Greedy(int v) : vocab(v)
    {
        TRUSS_CUDA(cudaMalloc(&logits, sizeof(float) * v * 9));
        TRUSS_CUDA(cudaMalloc(&d_ids, sizeof(int) * 9));
        TRUSS_CUDA(cudaMallocHost(&h_ids, sizeof(int) * 9));
    }
    void rows(q::Forward & f, int first, int n, int32_t * out)
    {
        f.head(first, n, logits);
        for (int r = 0; r < n; ++r) sampling::argmax(logits + (size_t) r * vocab, vocab, d_ids + r, f.stream());
        TRUSS_CUDA(cudaMemcpyAsync(h_ids, d_ids, sizeof(int) * n, cudaMemcpyDeviceToHost, f.stream()));
        TRUSS_CUDA(cudaStreamSynchronize(f.stream()));
        std::copy(h_ids, h_ids + n, out);
    }
};

void prompt(q::Forward & f, const std::vector<int32_t> & tok, int chunk)
{
    for (size_t s = 0; s < tok.size(); s += chunk) f.run(tok.data() + s, (int) std::min<size_t>(chunk, tok.size() - s));
}

}  // namespace

int main(int argc, char ** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <mtp.gguf> @prompt.i32 [tokens] [drafts] [usage|-] [n_ctx] [chunk]\n",
                     argv[0]);
        return 2;
    }
    try {
        const auto file = gguf::File::open(argv[1]);
        const auto mfile = gguf::File::open(argv[2]);
        const q::Config c = q::Config::from_gguf(*file);
        const q::Weights w = q::bind(*file, c);
        const q::Mtp mtp = q::bind_mtp(*mfile, c);
        const std::vector<int32_t> tok = read_ids(argv[3]);
        const int N = argc > 4 ? std::atoi(argv[4]) : 128, nd = argc > 5 ? std::atoi(argv[5]) : 3;
        const int n_ctx = argc > 7 ? std::atoi(argv[7]) : 65536, chunk = argc > 8 ? std::atoi(argv[8]) : 8192;
        const std::string draft_vocab = argc > 9 && std::string(argv[9]) != "-" ? argv[9] : "";
        const float min_p = argc > 11 ? std::atof(argv[11]) : 0.f;
        const std::string cpu_dir = argc > 10 && std::string(argv[10]) != "-" ? argv[10] : "";
        q::Forward::Options o;
        o.mtp = &mtp;
        o.spec_rows = nd + 1;
        // the prompt path's VRAM is sized by the prompt, not by max_chunk: a chunk cap above the prompt's real
        // length holds cached experts out of the ring's spare region for the whole run (TRACKER #70)
        const int step = (int) std::min<size_t>(chunk, tok.size());
        o.prefill_rows = step;
        if (!draft_vocab.empty()) {
            FILE * fv = std::fopen(draft_vocab.c_str(), "rb");
            if (!fv) throw std::runtime_error("cannot open " + draft_vocab);
            int32_t t;
            while (std::fread(&t, 4, 1, fv) == 1) o.draft_vocab.push_back(t);
            std::fclose(fv);
        }
        o.draft_min_p = min_p;
        o.doorbell = !(argc > 12 && std::string(argv[12]) == "sync");
        o.cpu_dir = cpu_dir;
        if (const char * e = std::getenv("TRUSS_CPU_SHARE")) o.cpu_share = (float) std::atof(e);   // sweeps
        if (const char * e = std::getenv("TRUSS_CPU_THREADS")) o.cpu_threads = std::atoi(e);
        if (const char * e = std::getenv("TRUSS_CPU_DYNAMIC")) o.cpu_dynamic = std::atoi(e) != 0;   // Strata split
        if (const char * e = std::getenv("TRUSS_CPU_TRELLIS")) o.cpu_trellis = std::atoi(e) != 0;   // CPU from the pack
        if (const char * e = std::getenv("TRUSS_PCIE_GBPS")) o.pcie_gbps = (float) std::atof(e);
        if (const char * e = std::getenv("TRUSS_PCIE_FRAC")) o.pcie_frac = (float) std::atof(e);   // fixed share
        if (const char * e = std::getenv("TRUSS_RING_GB")) o.ring_bytes_override = (size_t) std::atof(e) * (1ull << 30);
        if (const char * e = std::getenv("TRUSS_HINT_K")) o.hint_k = std::atoi(e);   // pre-gated prefetch width
        if (argc > 6 && std::string(argv[6]) != "-")
            o.expert_usage = runtime::ExpertStore::load_usage(argv[6], c.n_layer, c.n_expert);   // MTP: Forward adds
                                                                                               // the mean layer
        // Drop the shards the moment upload() is done: they peak near 47 GB resident, and load_cpu_tier()'s vector
        // would otherwise allocate on top of that and the kernel would OOM us (TRACKER #72).
        o.after_upload = [&] { file->release_pages(); mfile->release_pages(); };
        Greedy g(c.n_vocab);   // before the Forward: its expert budget takes the memory left
        q::Forward f(c, w, n_ctx, chunk, o);
        file->release_pages();
        mfile->release_pages();
        std::printf("experts: %d resident, ring %.2f GB; drafts %d\n", f.hot_experts(), f.experts().ring_bytes() / 1e9, nd);

        // plain greedy
        std::vector<int32_t> ref;
        prompt(f, tok, step);
        int32_t t;
        g.rows(f, (int) (tok.size() - 1) % step, 1, &t);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) {
            ref.push_back(t);
            f.run(&t, 1);
            g.rows(f, 0, 1, &t);
        }
        const double plain = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        // speculative
        f.reset();
        f.cpu_stats(true);   // CPU tier time below covers the spec phase only
        f.cpu_shape_reset();   // CPU shape below covers the spec phase only
        double sect[6];      // layer sections, reset after the prompt below so they cover the spec phase only
        prompt(f, tok, step);
        g.rows(f, (int) (tok.size() - 1) % step, 1, &t);
        f.section_ms(sect, true);
        { double m4[4], d3[3]; long dn; f.section_moe_ms(m4, true); f.driver_ms(d3, dn, true); }
        std::vector<int32_t> got;
        const runtime::ExpertStore::Stats st0 = f.experts().stats();
        int passes = 0;
        long drafted = 0, accepted = 0;
        // device-side time per section (events bracket device work; host waits excluded): splits GPU compute from
        // driver/CPU/PCIe stalls (TRACKER #61: pass-time breakdown without nsys, which needs root here).
        cudaEvent_t e0, e1;
        TRUSS_CUDA(cudaEventCreate(&e0));
        TRUSS_CUDA(cudaEventCreate(&e1));
        float ms_draft = 0.f, ms_verify = 0.f, ms_head = 0.f;
        t0 = std::chrono::steady_clock::now();
        while ((int) got.size() < N) {
            int32_t win[9], best[9];
            win[0] = t;
            TRUSS_CUDA(cudaEventRecord(e0, f.stream()));
            const int nw = f.draft(t, nd, win + 1);
            TRUSS_CUDA(cudaEventRecord(e1, f.stream()));
            TRUSS_CUDA(cudaEventSynchronize(e1));
            { float m = 0; TRUSS_CUDA(cudaEventElapsedTime(&m, e0, e1)); ms_draft += m; }
            TRUSS_CUDA(cudaEventRecord(e0, f.stream()));
            f.verify(win, nw + 1);
            TRUSS_CUDA(cudaEventRecord(e1, f.stream()));
            TRUSS_CUDA(cudaEventSynchronize(e1));
            { float m = 0; TRUSS_CUDA(cudaEventElapsedTime(&m, e0, e1)); ms_verify += m; }
            TRUSS_CUDA(cudaEventRecord(e0, f.stream()));
            g.rows(f, 0, nw + 1, best);
            TRUSS_CUDA(cudaEventRecord(e1, f.stream()));
            TRUSS_CUDA(cudaEventSynchronize(e1));
            { float m = 0; TRUSS_CUDA(cudaEventElapsedTime(&m, e0, e1)); ms_head += m; }
            int j = 0;
            while (j < nw && best[j] == win[j + 1]) ++j;
            f.accept(j + 1);
            for (int i = 0; i <= j; ++i) got.push_back(win[i]);
            t = best[j];
            ++passes, drafted += nw, accepted += j;
        }
        const double spec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        got.resize(N);
        int same = 0;
        while (same < N && got[same] == ref[same]) ++same;
        std::printf("plain: %d tokens in %.2f s = %.1f tok/s\n", N, plain, N / plain);
        std::printf("spec : %d tokens in %.2f s = %.1f tok/s, %d passes (%.2f tokens/pass), drafts accepted %ld of %ld\n",
                    (int) got.size(), spec, N / spec, passes, (double) N / passes, accepted, drafted);
        const runtime::ExpertStore::Stats & st = f.experts().stats();
        std::printf("       per pass: %.1f cold experts routed, %.1f fetched on demand (%.1f MB), %.1f prefetched (%.1f MB)\n",
                    (double) (st.experts_asked - st0.experts_asked) / passes, (double) (st.misses - st0.misses) / passes,
                    (st.bytes - st0.bytes) / 1e6 / passes, (double) (st.hinted - st0.hinted) / passes,
                    (st.hint_bytes - st0.hint_bytes) / 1e6 / passes);
        const auto cs = f.cpu_stats();
        if (cs.second) {
            long long ph[4];
            f.cpu_phase_us(ph);
            std::printf("       CPU tier: %.2f ms/pass in wait over %ld waits (items sum %.2f ms/pass)\n",
                        cs.first / 1e3 / passes, cs.second, f.cpu_item_us() / 1e3 / passes);
            std::printf("       CPU phases ms/pass: gate/up %.2f + silu %.2f + h-quant %.2f + down %.2f\n",
                        ph[0] / 1e3 / passes, ph[1] / 1e3 / passes, ph[2] / 1e3 / passes, ph[3] / 1e3 / passes);
            long long sh[4];
            f.cpu_shape(sh);
            std::printf("       CPU calls/pass %.1f: %.1f distinct experts, %.1f slots, R = %.2f rows/expert\n",
                        sh[0] / (double) passes, sh[1] / (double) passes, sh[2] / (double) passes,
                        sh[2] / (double) std::max(1LL, sh[1]));
        }
        {
            long dc = 0, dp = 0;
            double ce = 0, cc = 0;
            f.dyn_stats(dc, dp, ce, cc);
            if (dc + dp)
                std::printf("       dynamic split (whole run): %ld misses to the CPU, %ld over PCIe; CPU call %.3f + %.3f ms/expert\n",
                            dc, dp, cc, ce);
        }
        std::printf("       device ms/pass: draft %.2f + verify %.2f + head %.2f = %.2f (rest: host stalls/gaps)\n",
                    ms_draft / passes, ms_verify / passes, ms_head / passes,
                    (ms_draft + ms_verify + ms_head) / passes);
        f.section_ms(sect);
        if (sect[0] > 0)
            std::printf("       layer device ms/pass: ple+hc_mix %.2f + mixer %.2f + hc_mix %.2f + routed MoE %.2f "
                        "+ shared %.2f + cpu join/combine %.2f = %.2f\n",
                        sect[0] / passes, sect[1] / passes, sect[2] / passes, sect[3] / passes, sect[4] / passes,
                        sect[5] / passes, (sect[0] + sect[1] + sect[2] + sect[3] + sect[4] + sect[5]) / passes);
        double m4[4];
        f.section_moe_ms(m4);
        if (m4[0] + m4[1] + m4[2] + m4[3] > 0)
            std::printf("       routed MoE ms/pass: router+publish %.2f + wait for the host plan %.2f + wait for copies %.2f "
                        "+ expert kernel %.2f\n", m4[0] / passes, m4[1] / passes, m4[2] / passes, m4[3] / passes);
        double d3[3];
        long dn = 0;
        f.driver_ms(d3, dn);
        if (dn)
            std::printf("       driver ms/pass: split %.2f + CPU start %.2f + copies and plan %.2f (%.1f layers/pass)\n",
                        d3[0] / passes, d3[1] / passes, d3[2] / passes, (double) dn / passes);
        TRUSS_CUDA(cudaEventDestroy(e0));
        TRUSS_CUDA(cudaEventDestroy(e1));
        std::printf("tokens identical to plain greedy: %d of %d  %s\n", same, N, same == N ? "PASS" : "FAIL");
        return same == N ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
