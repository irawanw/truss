// Order 9 step 1: reproduce the IN-SITU CPU-tier call shape with the REAL pool + kernels (libtruss_cpu.a),
// CPU-only, no engine change. Shape from row 1007_064447: 3.68 experts/call (74% K3.5 / 26% K2.5),
// R=1.17, T=3, 22 workers pinned like TRUSS_CPU_PIN=1, caller pinned to core 0 (driver), experts from an
// ~8.5 GB DRAM/TLB-cold arena (THP off like the pack). Reports wall/last_call, item/gate/down/hq per call
// /23, 1x1 + empty + 1-thread + 11-thread controls, per-thread CPU-ms, SMT-sibling busy%, MHz, TLB variant.
#include "cpu/expert_q4.h"
#include "cpu/expert_trellis.h"
#include "formats/trellis_k.h"
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sched.h>
#include <execinfo.h>
#include <signal.h>
static void segv(int sig) { void * bt[32]; int n = backtrace(bt, 32); backtrace_symbols_fd(bt, n, 2); signal(sig, SIG_DFL); }
using namespace truss::cpu;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

static std::vector<int> physical_cores()   // mirror of expert_q4.cc:160
{
    std::vector<int> out, sibs;
    std::vector<std::pair<long, long>> seen;
    cpu_set_t set; CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return out;
    auto topo = [](int c, const char * what) {
        char path[96]; snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", c, what);
        long v = -1; FILE * f = fopen(path, "r");
        if (f) { if (fscanf(f, "%ld", &v) != 1) v = -1; fclose(f); }
        return v;
    };
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &set)) continue;
        auto key = std::make_pair(topo(c, "physical_package_id"), topo(c, "core_id"));
        if (std::find(seen.begin(), seen.end(), key) != seen.end()) { sibs.push_back(c); continue; }
        seen.push_back(key); out.push_back(c);
    }
    for (int c : sibs) out.push_back(c);
    return out;
}

struct TaskStat { unsigned long ut = 0, st = 0; };
static std::vector<std::pair<std::string, TaskStat>> read_tasks()
{
    std::vector<std::pair<std::string, TaskStat>> v;
    for (auto & e : std::filesystem::directory_iterator("/proc/self/task")) {
        std::ifstream f(e.path().string() + "/stat");
        std::string line;
        if (!std::getline(f, line)) continue;
        auto rp = line.rfind(')');
        if (rp == std::string::npos) continue;
        std::istringstream ss(line.substr(rp + 2));
        std::string tok; TaskStat s; int idx = 0;
        while (ss >> tok) { ++idx; if (idx == 12) s.ut = std::stoul(tok); if (idx == 13) s.st = std::stoul(tok); }
        v.push_back({e.path().filename().string(), s});
    }
    std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    return v;
}

struct ProcStat { unsigned long busy[64] = {}, idle[64] = {}; };
static ProcStat read_proc_stat()
{
    ProcStat p;
    std::ifstream f("/proc/stat");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("cpu", 0) != 0 || line[3] == ' ') continue;
        std::istringstream ss(line.substr(3));
        std::string id;
        if (!(ss >> id)) continue;
        unsigned long v[10] = {};
        for (int i = 0; i < 10 && (ss >> v[i]); ++i);
        const int c = std::stoi(id);
        if (c < 64) { p.busy[c] = v[0] + v[1] + v[2] + v[3] + v[6] + v[7]; p.idle[c] = v[4] + v[5]; }
    }
    return p;
}

static std::vector<double> read_mhz()
{
    std::vector<double> m(64, 0.0);
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    int proc = -1;
    while (std::getline(f, line)) {
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        if (line.rfind("processor", 0) == 0) proc = std::stoi(line.substr(colon + 1));
        else if (line.rfind("cpu MHz", 0) == 0 && proc >= 0 && proc < 64) m[proc] = std::stod(line.substr(colon + 1));
    }
    return m;
}

int main(int argc, char ** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    signal(SIGSEGV, segv); signal(SIGBUS, segv);
    const int CALLS = argc > 1 ? atoi(argv[1]) : 2000;
    const int NT = argc > 2 ? atoi(argv[2]) : 22;
    const int NEXP = 3900;
    const double FRAC35 = 0.74;
    setenv("TRUSS_CPU_PIN", "1", 1);
    std::mt19937_64 rng(42);

    const size_t tileB35 = 112, tileB25 = 80, TILES = 6400;
    const size_t RAW = 3 * TILES * tileB35 + 4 * 2560 + 4 * 640;
    const size_t STRIDE = (RAW + 4095) & ~(size_t) 4095;
    printf("alloc %.2f GB ...\n", STRIDE * (double) NEXP / 1e9);
    char * arena = (char *) aligned_alloc(4096, STRIDE * NEXP);
    if (!arena) { printf("arena FAILED\n"); return 1; }
    {   // touch every 4 KB page
        volatile uint64_t * w = (uint64_t *) arena; const size_t nw = STRIDE * NEXP / 8;
        for (size_t i = 0; i < nw; i += 512) w[i] = 1;
    }
    printf("arena touched\n");
    std::vector<uint16_t> sgn(2560);
    for (int i = 0; i < 2560; ++i) sgn[i] = (uint16_t)((i & 1) ? 0x3C00 : 0xBC00);
    std::vector<TrellisExpert> exps(NEXP);
    for (int i = 0; i < NEXP; ++i) {
        char * p = arena + (size_t) i * STRIDE;
        const int kind = (i % 100) < (int)(FRAC35 * 100) ? 35 : 25;
        const size_t tb = kind == 35 ? tileB35 : tileB25;
        uint16_t * sc = (uint16_t *)(p + 3 * TILES * tileB35);   // gate svh | up svh | down suh | down svh
        exps[i] = { TrellisMat{(uint32_t *) p, kind, 2560, 640, sgn.data(), sc},
                    TrellisMat{(uint32_t *)(p + TILES * tb), kind, 2560, 640, sgn.data(), sc + 640},
                    TrellisMat{(uint32_t *)(p + 2 * TILES * tb), kind, 640, 2560, sc + 1280, sc + 1920} };
    }
    const double bytes35 = 3.0 * TILES * tileB35, bytes25 = 3.0 * TILES * tileB25;

    std::vector<std::vector<Slot>> calls(CALLS);
    for (int c = 0; c < CALLS; ++c) {
        const int G = (c % 100) < 68 ? 4 : 3;
        for (int g = 0; g < G; ++g) {
            const int e = (int)(rng() % NEXP);
            calls[c].push_back({0, nullptr, 0.1f, &exps[e]});
            if ((c * 7 + g) % 100 < 17) calls[c].push_back({1, nullptr, 0.1f, &exps[e]});
        }
    }
    std::vector<std::vector<Slot>> seq_calls(CALLS);
    for (int c = 0; c < CALLS; ++c) seq_calls[c] = calls[c % 25];

    std::vector<float> x(3 * 2560), y(3 * 2560, 0.f);
    for (auto & v : x) v = ((rng() % 2000) - 1000) / 500.f;

    auto cores = physical_cores();   // BEFORE pinning the caller, else affinity hides the other cores
    printf("pins (workers): ");
    for (int i = 0; i < NT && (size_t)(i + 1) < cores.size(); ++i) printf("%d ", cores[i + 1]);
    printf("(n=%zu)\n", cores.size());

    auto pool = std::make_unique<ExpertPool>(NT);
    { cpu_set_t s; CPU_ZERO(&s); CPU_SET(0, &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s); }
    printf("pool up; warmup\n");
    for (int c = 0; c < 50; ++c) pool->run(x.data(), 3, calls[c], y.data());
    printf("warmup done\n");
    pool->reset_stats();
    auto tasks0 = read_tasks();
    ProcStat ps0 = read_proc_stat();

    std::vector<double> wall(CALLS), lastms(CALLS);
    auto t_run0 = clk::now();
    for (int c = 0; c < CALLS; ++c) {
        if (c % 250 == 0) fprintf(stderr, "run %d\n", c);
        auto a = clk::now();
        pool->start(x.data(), 3, calls[c], y.data());
        pool->wait();
        wall[c] = ms(a, clk::now());
        lastms[c] = pool->last_call_ms();
    }
    double run_s = ms(t_run0, clk::now()) / 1000.0;
    long long item_us = pool->item_us(), gate_us = pool->gate_us(), down_us = pool->down_us(), hq_us = pool->hq_us();
    long long shp[4]; pool->shape(shp);
    auto tasks1 = read_tasks();
    ProcStat ps1 = read_proc_stat();
    std::vector<double> mhz_now = read_mhz();
    pool.reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    auto sib_of = [&](int c) {
        char path[96]; snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        std::ifstream f(path); std::string l;
        if (!std::getline(f, l)) return std::vector<int>{};
        std::vector<int> r; std::istringstream ss(l); std::string tok;
        while (std::getline(ss, tok, ',')) if (tok.find('-') == std::string::npos) { int v = std::stoi(tok); if (v != c) r.push_back(v); }
        return r;
    };
    std::vector<int> sibs;
    for (int i = 0; i < NT && (size_t)(i + 1) < cores.size(); ++i) for (int s : sib_of(cores[i + 1])) sibs.push_back(s);
    double sib_busy = 0, sib_tot = 0, mhz_us = 0;
    for (int s : sibs) { sib_busy += ps1.busy[s] - ps0.busy[s]; sib_tot += (ps1.busy[s] - ps0.busy[s]) + (ps1.idle[s] - ps0.idle[s]); }
    int mhz_n = 0;
    for (int i = 0; i < NT && (size_t)(i + 1) < cores.size(); ++i) { mhz_us += mhz_now[cores[i + 1]]; ++mhz_n; }
    sib_busy = sib_tot ? 100.0 * sib_busy / sib_tot : 0;

    std::vector<double> tid_ms;
    if (tasks0.size() == tasks1.size())
        for (size_t i = 0; i < tasks1.size(); ++i)
            tid_ms.push_back((tasks1[i].second.ut + tasks1[i].second.st - tasks0[i].second.ut - tasks0[i].second.st) * 10.0);
    std::sort(tid_ms.begin(), tid_ms.end(), std::greater<double>());
    double wsum = 0, lsum = 0;
    for (int c = 0; c < CALLS; ++c) { wsum += wall[c]; lsum += lastms[c]; }
    std::sort(wall.begin(), wall.end());

    printf("== IN-SITU SHAPE (%d calls, %d workers, %.1f s) ==\n", CALLS, NT, run_s);
    printf("wall mean %.3f p50 %.3f p95 %.3f ms | last_call mean %.3f\n", wsum / CALLS, wall[CALLS / 2], wall[(int)(CALLS * 0.95)], lsum / CALLS);
    printf("item %.3f gate %.3f down %.3f hq %.3f ms/call (all-thread sums) | /%d = %.3f ms parallel compute\n",
           item_us / 1e3 / CALLS, gate_us / 1e3 / CALLS, down_us / 1e3 / CALLS, hq_us / 1e3 / CALLS, NT + 1, item_us / 1e3 / CALLS / (NT + 1));
    printf("shape/call: groups %.2f slots %.2f rows %.2f\n", (double) shp[1] / shp[0], (double) shp[2] / shp[0], (double) shp[3] / shp[0]);
    double bavg = FRAC35 * bytes35 + (1 - FRAC35) * bytes25;
    printf("bytes/expert avg %.3f MB; GB/s(bytes / parallel-compute) %.2f aggregate, %.3f/thread\n",
           bavg / 1e6, bavg * (shp[1] / (double) shp[0]) / (item_us / 1e6 / CALLS), bavg * (shp[1] / (double) shp[0]) / (item_us / 1e6 / CALLS) / (NT + 1));
    printf("SMT siblings busy%% during run: %.1f | our pins MHz avg: %.0f\n", sib_busy, mhz_n ? mhz_us / mhz_n : 0.0);
    printf("thread CPU-ms over run (incl spin) top8: ");
    for (size_t i = 0; i < std::min<size_t>(tid_ms.size(), 8); ++i) printf("%.0f ", tid_ms[i]);
    printf("... min %.0f (n=%zu)\n", tid_ms.empty() ? -1 : tid_ms.back(), tid_ms.size());

    auto widen = [] { cpu_set_t s; CPU_ZERO(&s); for (int c = 0; c < 48; ++c) CPU_SET(c, &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s); };
    // paced variants: next start at prev start + gap (like serving's 0.675 ms layer period); workers spin hot
    auto paced = [&](const char * name, double gap_ms, int n) {
        widen();
        auto p = std::make_unique<ExpertPool>(NT);
        { cpu_set_t s; CPU_ZERO(&s); CPU_SET(0, &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s); }
        for (int c = 0; c < 50; ++c) p->run(x.data(), 3, calls[c], y.data());
        std::vector<double> w(n), lm(n);
        for (int c = 0; c < n; ++c) {
            auto target = clk::now() - std::chrono::duration<double, std::milli>(0);   // no-op anchor
            auto a = clk::now();
            p->start(x.data(), 3, calls[c], y.data());
            p->wait();
            w[c] = ms(a, clk::now());
            lm[c] = p->last_call_ms();
            double used = ms(a, clk::now());
            double sleep_ms = gap_ms - used;
            if (sleep_ms > 0.05) std::this_thread::sleep_for(std::chrono::microseconds((int)(sleep_ms * 1000)));
        }
        double s2 = 0, l2 = 0; for (int c = 0; c < n; ++c) { s2 += w[c]; l2 += lm[c]; }
        std::sort(w.begin(), w.end());
        printf("%-26s wall mean %.3f ms (p50 %.3f p95 %.3f) last_call %.3f\n", name, s2 / n, w[n / 2], w[(int)(n * 0.95)], l2 / n);
    };
    paced("paced gap 0.675 ms", 0.675, 2000);
    paced("paced gap 1.5 ms", 1.5, 2000);
    auto control = [&](const char * name, int threads, const std::vector<std::vector<Slot>> & cs, int T, int n) {
        widen();
        auto p = std::make_unique<ExpertPool>(threads);
        { cpu_set_t s; CPU_ZERO(&s); CPU_SET(0, &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s); }
        for (int c = 0; c < 50 && c < n; ++c) p->run(x.data(), T, cs[c % cs.size()], y.data());
        std::vector<double> w(n);
        for (int c = 0; c < n; ++c) {
            auto a = clk::now();
            p->start(x.data(), T, cs[c % cs.size()], y.data());
            p->wait();
            w[c] = ms(a, clk::now());
        }
        double s = 0; for (double v : w) s += v;
        std::sort(w.begin(), w.end());
        printf("%-26s wall mean %.3f ms (p95 %.3f)\n", name, s / n, w[(int)(n * 0.95)]);
    };
    std::vector<std::vector<Slot>> one(200, std::vector<Slot>{ {0, nullptr, 0.1f, &exps[7]} });
    control("1x1 (G=1,T=1)", NT, one, 1, 500);
    control("empty (G=0)", NT, std::vector<std::vector<Slot>>(200), 1, 500);
    control("in-situ shape 1 thread", 1, calls, 3, 500);
    control("in-situ shape 11 threads", 11, calls, 3, 1000);
    control("TLB-hot (25 experts)", NT, seq_calls, 3, 2000);
    printf("done\n");
    return 0;
}
