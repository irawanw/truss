// Speculative (MTP) greedy decode vs plain greedy decode on the full model: the same prompt, N generated tokens each
// way; the token sequences must be identical (verify windows, rollback and the MTP cache must not change the
// output), and the speculative run's speed and tokens per verify pass are printed. Exit code 1 on a mismatch.
// Several prompts (@a.i32,@b.i32,...): each runs plain + spec in turn in one process (reset between), one line per
// prompt, and the breakdown below sums the spec phases of all of them (TRACKER #84: a change of numerics changes the
// greedy text of one prompt, and with it the tokens per pass and the experts per pass; a set of prompts averages
// that out). TRUSS_BENCH_PLAIN=0 skips the plain runs (no identity check, faster sweeps).
// usage: tk-bench-spec <model.gguf> <mtp.gguf> @prompt.i32[,@prompt2.i32...] [tokens=128] [drafts=3] [usage file|-]
//                      [n_ctx=65536] [chunk=8192] [draft vocab ids file|-] [cpu tier dir|-] [draft min p=0]
#include "core/cuda_check.h"
#include <cuda_profiler_api.h>
#include <sys/prctl.h>
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


// TRUSS_STACK_SIGNAL=1: SIGUSR1 makes every thread print its backtrace (raw addresses + /proc/self/maps base of
// the executable) to stderr: a hang probe where ptrace / cuda-gdb cannot attach (TRACKER #115)
#include <dirent.h>
#include <execinfo.h>
#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>
namespace {
void dump_self(int)
{
    void * f[48];
    const int n = backtrace(f, 48);
    char h[64];
    const int k = std::snprintf(h, sizeof h, "== tid %ld\n", (long) syscall(SYS_gettid));
    (void) !write(2, h, (size_t) k);
    backtrace_symbols_fd(f, n, 2);
}
void dump_all(int sig)
{
    const long self = syscall(SYS_gettid);
    if (DIR * d = opendir("/proc/self/task")) {
        while (dirent * e = readdir(d)) {
            const long t = std::atol(e->d_name);
            if (t > 0 && t != self) syscall(SYS_tgkill, getpid(), t, SIGUSR2), usleep(20000);
        }
        closedir(d);
    }
    dump_self(sig);
}
}  // namespace

int main(int argc, char ** argv)
{
    // TRUSS_PTRACE_ANY=1: any process of this user may attach (cuda-gdb -p) though ptrace_scope is 1 (hang hunting,
    // TRACKER #115)
    if (const char * e = std::getenv("TRUSS_PTRACE_ANY"); e && std::atoi(e)) prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);
    if (const char * e = std::getenv("TRUSS_STACK_SIGNAL"); e && std::atoi(e)) {
        void * pre[1];
        backtrace(pre, 1);   // loads libgcc's unwinder now, not inside a handler
        signal(SIGUSR1, dump_all);
        signal(SIGUSR2, dump_self);
    }
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <mtp.gguf> @prompt.i32 [tokens] [drafts] [usage|-] [n_ctx] [chunk]\n",
                     argv[0]);
        return 2;
    }
    try {
        const auto file = gguf::File::open(argv[1]);
        if (const char * ov = std::getenv("TRUSS_EXPERT_OVERLAY")) {   // X3.1 experts over the served GGUF (#118)
            file->overlay(ov);
            std::fprintf(stderr, "expert overlay %s: %zu tensors replaced\n", ov, file->overlaid());
        }
        const auto mfile = gguf::File::open(argv[2]);
        const q::Config c = q::Config::from_gguf(*file);
        const q::Weights w = q::bind(*file, c);
        const q::Mtp mtp = q::bind_mtp(*mfile, c);
        std::vector<std::vector<int32_t>> prompts;
        {
            const std::string list = argv[3];
            for (size_t a = 0; a <= list.size();) {
                const size_t b = std::min(list.find(',', a), list.size());
                prompts.push_back(read_ids(list.substr(a, b - a).c_str()));
                a = b + 1;
            }
        }
        size_t longest = 0;
        for (const auto & p : prompts) longest = std::max(longest, p.size());
        const bool do_plain = !(std::getenv("TRUSS_BENCH_PLAIN") && std::atoi(std::getenv("TRUSS_BENCH_PLAIN")) == 0);
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
        const int step = (int) std::min<size_t>(chunk, longest);
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
        o.ple_file = file.get();   // PLE rows by O_DIRECT reads (TRUSS_PLE_DIRECT=0: the mapping)
        q::apply_env(o);   // TRUSS_CPU_*, TRUSS_PCIE_*, TRUSS_ADAPT_*, TRUSS_HINT_K, TRUSS_RING_GB, TRUSS_PREFILL_ROWS
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
        std::printf("experts: %d resident, ring %.2f GB (a prompt chunk takes %.2f GB of it); drafts %d\n", f.hot_experts(),
                    f.experts().ring_bytes() / 1e9, f.experts().stream_area() / 1e9, nd);

        // per prompt: plain greedy (unless TRUSS_BENCH_PLAIN=0), then speculative; every counter below is read and
        // reset around the spec phase only, and summed over the prompts
        struct Sum {
            double spec_s = 0, plain_s = 0, ms_draft = 0, ms_verify = 0, ms_head = 0, sect[6] = {}, m4[4] = {}, d3[3] = {};
            double ple_ms = 0, cpu_wait_us = 0, pre_s = 0, pre_ple = 0;
            long pre_tok = 0;
            long passes = 0, tokens = 0, drafted = 0, accepted = 0, dn = 0, ple_calls = 0, ple_rows = 0, cpu_waits = 0;
            long long item_us = 0, ph[4] = {}, sh[4] = {};
            long asked = 0, misses = 0, hinted = 0;
            double bytes = 0, hint_bytes = 0, adm_bytes = 0;
            long adm_n = 0;
            int identical = 0, compared = 0;
        } S;
        cudaEvent_t e0, e1;
        TRUSS_CUDA(cudaEventCreate(&e0));
        TRUSS_CUDA(cudaEventCreate(&e1));
        for (size_t pi = 0; pi < prompts.size(); ++pi) {
            const std::vector<int32_t> & tok = prompts[pi];
            std::vector<int32_t> ref;
            int32_t t;
            double plain = 0;
            if (do_plain) {
                f.reset();
                prompt(f, tok, step);
                g.rows(f, (int) (tok.size() - 1) % step, 1, &t);
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < N; ++i) {
                    ref.push_back(t);
                    f.run(&t, 1);
                    g.rows(f, 0, 1, &t);
                }
                plain = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }

            // speculative (the prompt read is timed: prefill tok/s)
            f.reset();
            double ple0;
            long pc0, pr0;
            f.ple_host_ms(ple0, pc0, pr0, true);
            const auto tp = std::chrono::steady_clock::now();
            prompt(f, tok, step);
            g.rows(f, (int) (tok.size() - 1) % step, 1, &t);
            const double pre = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count();
            double pre_ple;   // the prompt's PLE host gather (synchronous table reads), part of `pre`
            f.ple_host_ms(pre_ple, pc0, pr0, true);
            f.cpu_stats(true);
            f.cpu_shape_reset();
            double sect[6], m4[4], d3[3], pm;
            long dn, pc, pr;
            f.section_ms(sect, true);
            f.section_moe_ms(m4, true);
            f.driver_ms(d3, dn, true);
            f.ple_host_ms(pm, pc, pr, true);
            const long long item0 = f.cpu_item_us();
            long long ph0[4];
            f.cpu_phase_us(ph0);
            std::vector<int32_t> got;
            const runtime::ExpertStore::Stats st0 = f.experts().stats();
            int passes = 0;
            long drafted = 0, accepted = 0;
            // device-side time per section (events bracket device work; host waits excluded): splits GPU compute from
            // driver/CPU/PCIe stalls (TRACKER #61: pass-time breakdown without nsys, which needs root here).
            float ms_draft = 0.f, ms_verify = 0.f, ms_head = 0.f;
            // TRUSS_BENCH_PROFRANGE=1: the decode loops are the only cudaProfilerApi range (nsys --capture-range)
            const bool prange = std::getenv("TRUSS_BENCH_PROFRANGE") && std::atoi(std::getenv("TRUSS_BENCH_PROFRANGE"));
            if (prange) TRUSS_CUDA(cudaProfilerStart());
            const auto t0 = std::chrono::steady_clock::now();
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
            if (prange) TRUSS_CUDA(cudaProfilerStop());
            got.resize(N);
            int same = 0;
            if (do_plain)
                while (same < N && got[same] == ref[same]) ++same;
            std::printf("prompt %zu (%zu tokens): prefill %.0f tok/s (%.2f s, PLE host gather %.2f s), ", pi, tok.size(),
                        tok.size() / pre, pre, pre_ple / 1e3);
            if (do_plain) std::printf("plain %.1f tok/s, ", N / plain);
            std::printf("spec %.1f tok/s, %.2f tokens/pass", N / spec, (double) N / passes);
            if (do_plain) std::printf(", identical %d of %d", same, N);
            std::printf("\n");
            // accumulate
            S.pre_s += pre, S.pre_tok += (long) tok.size(), S.pre_ple += pre_ple / 1e3;
            if (const char * dp = std::getenv("TRUSS_BENCH_DUMP")) {   // the spec tokens, appended per prompt
                FILE * fd = std::fopen(dp, pi ? "ab" : "wb");
                if (fd) std::fwrite(got.data(), 4, std::min<size_t>(got.size(), N), fd), std::fclose(fd);
            }
            S.spec_s += spec, S.plain_s += plain, S.passes += passes, S.tokens += N, S.drafted += drafted;
            S.accepted += accepted, S.ms_draft += ms_draft, S.ms_verify += ms_verify, S.ms_head += ms_head;
            if (do_plain) S.identical += same, S.compared += N;
            f.section_ms(sect), f.section_moe_ms(m4), f.driver_ms(d3, dn), f.ple_host_ms(pm, pc, pr);
            for (int k = 0; k < 6; ++k) S.sect[k] += sect[k];
            for (int k = 0; k < 4; ++k) S.m4[k] += m4[k];
            for (int k = 0; k < 3; ++k) S.d3[k] += d3[k];
            S.dn += dn, S.ple_ms += pm, S.ple_calls += pc, S.ple_rows += pr;
            const auto cs = f.cpu_stats();
            S.cpu_wait_us += cs.first, S.cpu_waits += cs.second;
            S.item_us += f.cpu_item_us() - item0;
            long long ph[4], sh[4];
            f.cpu_phase_us(ph);
            f.cpu_shape(sh);
            for (int k = 0; k < 4; ++k) S.ph[k] += ph[k] - ph0[k], S.sh[k] += sh[k];
            const runtime::ExpertStore::Stats & st = f.experts().stats();
            S.asked += st.experts_asked - st0.experts_asked, S.misses += st.misses - st0.misses;
            S.hinted += st.hinted - st0.hinted;
            S.bytes += st.bytes - st0.bytes, S.hint_bytes += st.hint_bytes - st0.hint_bytes;
            S.adm_bytes += (double) (st.admit_bytes - st0.admit_bytes), S.adm_n += st.admitted_claims - st0.admitted_claims;
        }
        TRUSS_CUDA(cudaEventDestroy(e0));
        TRUSS_CUDA(cudaEventDestroy(e1));

        const double passes = (double) S.passes;
        if (do_plain) std::printf("plain: %ld tokens in %.2f s = %.1f tok/s\n", S.tokens, S.plain_s, S.tokens / S.plain_s);
        std::printf("prefill: %ld prompt tokens in %.2f s = %.0f tok/s (PLE host gather %.2f s)\n", S.pre_tok, S.pre_s,
                    S.pre_tok / S.pre_s, S.pre_ple);
        std::printf("spec : %ld tokens in %.2f s = %.1f tok/s, %ld passes (%.2f tokens/pass), drafts accepted %ld of %ld\n",
                    S.tokens, S.spec_s, S.tokens / S.spec_s, S.passes, S.tokens / passes, S.accepted, S.drafted);
        std::printf("       per pass: %.1f cold experts routed, %.1f fetched on demand (%.1f MB), %.1f prefetched (%.1f MB)\n",
                    S.asked / passes, S.misses / passes, S.bytes / 1e6 / passes, S.hinted / passes,
                    S.hint_bytes / 1e6 / passes);
        if (f.kv_lent()) std::printf("       KV lent to the expert ring at the end: %.2f GB\n", f.kv_lent() / 1e9);
        {
            long long sp[3];
            f.split_stats(sp);
            if (sp[1] + sp[2])
                std::printf("       E7 split prompts: %.2f GB copied, cold experts %lld to the CPU, %lld to the GPU\n", sp[0] / 1e9,
                            sp[1], sp[2]);
        }
        if (S.adm_n || S.adm_bytes)
            std::printf("       admission on the idle link: %.1f experts/pass issued (%.1f MB/pass), %ld landed in all\n",
                        S.adm_n / passes, S.adm_bytes / 1e6 / passes, f.experts().admit_landed());
        if (S.cpu_waits) {
            std::printf("       CPU tier: %.2f ms/pass in wait over %ld waits (items sum %.2f ms/pass)\n",
                        S.cpu_wait_us / 1e3 / passes, S.cpu_waits, S.item_us / 1e3 / passes);
            std::printf("       CPU phases ms/pass: gate/up %.2f + silu %.2f + h-quant %.2f + down %.2f\n",
                        S.ph[0] / 1e3 / passes, S.ph[1] / 1e3 / passes, S.ph[2] / 1e3 / passes, S.ph[3] / 1e3 / passes);
            std::printf("       CPU calls/pass %.1f: %.1f distinct experts, %.1f slots, R = %.2f rows/expert\n",
                        S.sh[0] / passes, S.sh[1] / passes, S.sh[2] / passes, S.sh[2] / (double) std::max(1LL, S.sh[1]));
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
                    S.ms_draft / passes, S.ms_verify / passes, S.ms_head / passes,
                    (S.ms_draft + S.ms_verify + S.ms_head) / passes);
        if (S.sect[0] > 0)
            std::printf("       layer device ms/pass: ple+hc_mix %.2f + mixer %.2f + hc_mix %.2f + routed MoE %.2f "
                        "+ shared %.2f + cpu join/combine %.2f = %.2f\n",
                        S.sect[0] / passes, S.sect[1] / passes, S.sect[2] / passes, S.sect[3] / passes,
                        S.sect[4] / passes, S.sect[5] / passes,
                        (S.sect[0] + S.sect[1] + S.sect[2] + S.sect[3] + S.sect[4] + S.sect[5]) / passes);
        if (S.m4[0] + S.m4[1] + S.m4[2] + S.m4[3] > 0)
            std::printf("       routed MoE ms/pass: router+publish %.2f + wait for the host plan %.2f + wait for copies %.2f "
                        "+ expert kernel %.2f\n", S.m4[0] / passes, S.m4[1] / passes, S.m4[2] / passes, S.m4[3] / passes);
        if (f.adapt_admitted()) std::printf("       adaptive tier: %ld experts admitted\n", f.adapt_admitted());
        if (S.ple_calls)
            std::printf("       PLE host gather ms/pass: %.2f (%.1f calls, %.0f table rows per pass)\n", S.ple_ms / passes,
                        S.ple_calls / passes, S.ple_rows / passes);
        if (S.dn)
            std::printf("       driver ms/pass: split %.2f + CPU start %.2f + copies and plan %.2f (%.1f layers/pass)\n",
                        S.d3[0] / passes, S.d3[1] / passes, S.d3[2] / passes, S.dn / passes);
        if (!do_plain) {
            std::printf("tokens identical to plain greedy: not checked (TRUSS_BENCH_PLAIN=0)\n");
            return 0;
        }
        std::printf("tokens identical to plain greedy: %d of %d  %s\n", S.identical, S.compared,
                    S.identical == S.compared ? "PASS" : "FAIL");
        return S.identical == S.compared ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
