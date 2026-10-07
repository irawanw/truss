// Lead Order-2 (2026-10-06): CPU-tier standard-quant feasibility microbench. One routed expert (gate, up 640x2560,
// down 2560x640; the served qwen4exp shape) measured per weight format, single thread and on a pool of pinned
// workers mirroring ExpertPool (TRUSS_CPU_PIN=1: worker i on physical core i+1, skipping core 0), cycling more
// experts than the L3 holds so every expert streams from DRAM, as CPU-tier misses do (the tk-bench-cpu-q4 house
// style). Formats: the served pack's trellis K2.5 and K3.5 through the engine's own kernels (truss_cpu) and the
// ggml AVX2 vec_dot kernels (llama-paw's built libggml-cpu, the AVX2 target build) for Q3_K / Q4_K / Q8_0 on the
// same shapes. K-quants block by 256: gate/up in=2560 is 10 blocks, but down in=640 is 2.5 blocks, so down is
// zero-padded to 768 (the padding's bytes are counted; Q8_0 blocks by 32 and needs none). R = rows per expert:
// decode groups rows per expert (slots/expert 1.203), so R=1 and R=2 are timed and blended 0.8/0.2 downstream.
// CPU-only: no CUDA call is made; the ggml dots are called directly (they are direct AVX2 builds, no ggml_init,
// so the backend registry — and the GPU — are never touched).
// usage: tk-bench-cpu-quant [threads=22] [experts=64] [seconds=1.0] [rounds=4]
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <pthread.h>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <vector>

#define GGML_COMMON_DECL_C  // ggml-common.h is a multi-include header: the C decl pass emits the block structs
#include "ggml-common.h"    // block_q3_K, block_q4_K, block_q8_K, block_q8_0 (llama-paw's own sizes)

#include "cpu/expert_trellis.h"

// llama-paw's exported AVX2 kernels (declarations per ggml/src/ggml-cpu/quants.h + ggml/include/ggml.h)
extern "C" {
size_t ggml_quantize_chunk(int32_t type, const float * src, void * dst, int64_t start, int64_t nrows, int64_t n_per_row,
                           const float * im);
void quantize_row_q8_K(const float * x, void * y, int64_t k);
void quantize_row_q8_0(const float * x, void * y, int64_t k);
void ggml_vec_dot_q3_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
}

namespace {

constexpr int D_MODEL = 2560, D_FF = 640, DOWN_PAD = 768;  // gate/up in=2560, down in=640 padded to 3x256
constexpr int32_t T_Q3_K = 11, T_Q4_K = 12, T_Q8_0 = 8;    // ggml/include/ggml.h enum values
constexpr int N_TRELLIS = 2;                                // K2.5, K3.5
constexpr double MIX_K25 = 0.512;                           // served pack: 37728 K25 of 73728 projections
constexpr int COLD_EXPERTS = 21296;                         // the host-RAM budget's expert count

volatile float sink = 0.f;  // keeps every dot's output live

uint64_t rng_s = 0x9E3779B97F4A7C15ull;
inline uint64_t xs64()
{
    rng_s ^= rng_s << 13;
    rng_s ^= rng_s >> 7;
    rng_s ^= rng_s << 17;
    return rng_s;
}
inline float frand() { return (float)((int64_t)(xs64() >> 11) % 2000001 - 1000000) / 1000000.f; }  // [-1,1]

// one logical CPU per physical core (the first SMT sibling), ascending: the engine's ExpertPool scheme
std::vector<int> physical_cores()
{
    std::vector<int> out, firsts;
    for (int cpu = 0; cpu < 512; ++cpu)
    {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
        if (!f) continue;
        std::string s;
        std::getline(f, s);
        const int first = atoi(s.c_str());
        if (std::find(firsts.begin(), firsts.end(), first) == firsts.end())
        {
            firsts.push_back(first);
            if (first == cpu) out.push_back(cpu);
        }
    }
    return out;
}

void pin_to(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::string loadavg()
{
    std::ifstream f("/proc/loadavg");
    std::string s;
    std::getline(f, s);
    return s.substr(0, s.find_first_of(" \n"));
}

// mlocked anonymous arena: pinned, hugepage-free DRAM, the engine's pack-byte analogue
struct Arena
{
    uint8_t * p = nullptr;
    size_t cap = 0;
    void make(size_t bytes)
    {
        cap = bytes;
        p = (uint8_t *)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (p == MAP_FAILED) { std::fprintf(stderr, "mmap %zu MB failed\n", bytes >> 20); std::exit(1); }
        if (mlock(p, bytes)) { std::fprintf(stderr, "mlock %zu MB failed (ulimit -l)\n", bytes >> 20); std::exit(1); }
        std::memset(p, 1, bytes);  // fault every page in before timing
    }
};

// trellis tile words (u32): 8*K -> K2.5 = 20, K3.5 = 28; tiles per projection = (in/16)*(out/16) = 6400 both ways
size_t tr_words(int K) { return K == 25 ? 20 : 28; }
size_t tr_tiles() { return (size_t)(D_MODEL / 16) * (D_FF / 16); }

// per-expert layout: part 0 gate (in 2560, out 640), 1 up (same), 2 down (in 640, out 2560).
// trellis part = tiles | suh | svh; ggml part = rows of blocks (down padded to 768 inputs).
struct ExpertSet
{
    int fmt = 0;  // 0 trellis25, 1 trellis35, 2 q3_k, 3 q4_k, 4 q8_0
    int K = 25;
    Arena a;
    size_t per = 0;  // weight bytes per expert: the number the RAM budget multiplies
    int n = 0;

    size_t bs() const { return fmt == 2 ? sizeof(block_q3_K) : fmt == 3 ? sizeof(block_q4_K) : sizeof(block_q8_0); }
    size_t nb_in() const { return fmt == 4 ? D_MODEL / QK8_0 : D_MODEL / QK_K; }      // blocks per gate/up row
    size_t nb_dn() const { return fmt == 4 ? D_FF / QK8_0 : DOWN_PAD / QK_K; }        // blocks per down row
    size_t tr_part() const { return tr_tiles() * tr_words(K) * 4 + (size_t)(D_MODEL + D_FF) * 2; }  // tiles+suh+svh
    size_t offs(int part) const
    {
        if (fmt < N_TRELLIS) return part * tr_part();
        const size_t g = (size_t)D_FF * nb_in() * bs();
        return part == 2 ? 2 * g : part * g;
    }
    const uint32_t * tiles(int e, int part) const { return reinterpret_cast<const uint32_t *>(a.p + per * e + offs(part)); }
    const uint16_t * suh(int e, int part) const { return reinterpret_cast<const uint16_t *>(a.p + per * e + offs(part) + tr_tiles() * tr_words(K) * 4); }
    const uint16_t * svh(int e, int part) const { return suh(e, part) + (part == 2 ? D_FF : D_MODEL); }
    const uint8_t * wrows(int e, int part) const { return a.p + per * e + offs(part); }
};

size_t expert_bytes(int fmt, int K)
{
    if (fmt < N_TRELLIS) return 3 * (tr_tiles() * tr_words(K) * 4 + (size_t)(D_MODEL + D_FF) * 2);
    const size_t bs = fmt == 2 ? sizeof(block_q3_K) : fmt == 3 ? sizeof(block_q4_K) : sizeof(block_q8_0);
    const size_t nb_in = fmt == 4 ? D_MODEL / QK8_0 : D_MODEL / QK_K, nb_dn = fmt == 4 ? D_FF / QK8_0 : DOWN_PAD / QK_K;
    return 2 * (size_t)D_FF * nb_in * bs + (size_t)D_MODEL * nb_dn * bs;
}

void build_trellis(ExpertSet & X)
{
    for (int e = 0; e < X.n; ++e)
        for (int part = 0; part < 3; ++part)
        {
            uint8_t * base = X.a.p + X.per * e + X.offs(part);
            const size_t tw = tr_tiles() * tr_words(X.K);
            uint32_t * t = (uint32_t *)base;
            for (size_t i = 0; i < tw; ++i) t[i] = (uint32_t)xs64();
            const int n_in = part == 2 ? D_FF : D_MODEL, n_out = part == 2 ? D_MODEL : D_FF;
            uint16_t * suh = (uint16_t *)(base + tw * 4), * svh = suh + n_in;
            for (int i = 0; i < n_in; ++i) suh[i] = _cvtss_sh(0.5f + (frand() + 1.f) * 0.25f, 0x08);
            for (int i = 0; i < n_out; ++i) svh[i] = _cvtss_sh((frand() < 0 ? -1.f : 1.f) * (0.5f + (frand() + 1.f) * 0.25f), 0x08);
        }
}

void build_ggml(ExpertSet & X, int32_t type, float * src)
{
    for (int e = 0; e < X.n; ++e)
        for (int part = 0; part < 3; ++part)
        {
            const int rows = part == 2 ? D_MODEL : D_FF;
            const int nin = part == 2 ? (type == T_Q8_0 ? D_FF : DOWN_PAD) : D_MODEL;
            if (part == 2 && type != T_Q8_0)
                for (int o = 0; o < rows; ++o)
                    for (int i = 0; i < nin; ++i) src[(size_t)o * nin + i] = i < D_FF ? frand() : 0.f;
            else
                for (size_t i = 0; i < (size_t)rows * nin; ++i) src[i] = frand();
            ggml_quantize_chunk(type, src, X.a.p + X.per * e + X.offs(part), 0, rows, nin, nullptr);
        }
}

const float * XROWS[2];  // the R activation rows (shared, read-only): one contiguous [2][D_MODEL]

// the pool's own h = silu(gate) * up (scalar expf, same math as ExpertPool::item_trellis)
inline void silu_mul(const float * g, const float * u, float * h, int n)
{
    for (int j = 0; j < n; ++j) h[j] = g[j] / (1.f + std::exp(-g[j])) * u[j];
}

struct Scratch
{
    std::vector<float> P, c, g, u, h, y, hpad;
    std::vector<uint8_t> xq, hq;  // activation rows: q8_K (10 or 3 blocks) or q8_0 (80 or 20), R rows
    int R = 0;
    void grow(int R_)
    {
        R = R_;
        P.resize((size_t)truss::cpu::trellis_prep_floats(D_MODEL, R));
        c.resize((size_t)R * D_MODEL);
        g.resize((size_t)R * D_FF); u.resize((size_t)R * D_FF); h.resize((size_t)R * D_FF); y.resize((size_t)R * D_MODEL);
        hpad.resize((size_t)R * DOWN_PAD);
        xq.resize((size_t)R * (D_MODEL / QK_K) * sizeof(block_q8_K));
        hq.resize((size_t)R * (DOWN_PAD / QK_K) * sizeof(block_q8_K));
    }
};

void run_expert(const ExpertSet & X, int e, int R, Scratch & S)
{
    if (X.fmt < N_TRELLIS)
    {
        using namespace truss::cpu;
        TrellisMat W[3] = { { X.tiles(e, 0), X.K, D_MODEL, D_FF, X.suh(e, 0), X.svh(e, 0) },
                            { X.tiles(e, 1), X.K, D_MODEL, D_FF, X.suh(e, 1), X.svh(e, 1) },
                            { X.tiles(e, 2), X.K, D_FF, D_MODEL, X.suh(e, 2), X.svh(e, 2) } };
        for (int part = 0; part < 2; ++part)
        {
            trellis_prep(W[part], XROWS[0], D_MODEL, R, S.P.data());
            trellis_gemv_raw(W[part], S.P.data(), R, 0, D_FF, S.c.data(), D_FF);
            trellis_out(W[part], S.c.data(), R, D_FF, 0, D_FF, part == 0 ? S.g.data() : S.u.data(), D_FF);
        }
        for (int r = 0; r < R; ++r) silu_mul(S.g.data() + r * D_FF, S.u.data() + r * D_FF, S.h.data() + r * D_FF, D_FF);
        trellis_prep(W[2], S.h.data(), D_FF, R, S.P.data());
        trellis_gemv_raw(W[2], S.P.data(), R, 0, D_MODEL, S.c.data(), D_MODEL);
        trellis_out(W[2], S.c.data(), R, D_MODEL, 0, D_MODEL, S.y.data(), D_MODEL);
        sink += S.y[0];
        return;
    }
    const size_t bs = X.bs(), nb_in = X.nb_in(), nb_dn = X.nb_dn();
    const int n_dn = X.fmt == 4 ? D_FF : DOWN_PAD;  // dot length on down's input
    const size_t ab = X.fmt == 4 ? sizeof(block_q8_0) : sizeof(block_q8_K);
    for (int part = 0; part < 2; ++part)
    {
        const uint8_t * w = X.wrows(e, part);
        float * dst = part == 0 ? S.g.data() : S.u.data();
        for (int r = 0; r < R; ++r)
            for (int o = 0; o < D_FF; ++o)
            {
                const void * xr = S.xq.data() + r * nb_in * ab;
                const uint8_t * wr = w + (size_t)o * nb_in * bs;
                if (X.fmt == 4) ggml_vec_dot_q8_0_q8_0(D_MODEL, dst + r * D_FF + o, 0, wr, 0, xr, 0, 1);
                else if (X.fmt == 3) ggml_vec_dot_q4_K_q8_K(D_MODEL, dst + r * D_FF + o, 0, wr, 0, xr, 0, 1);
                else ggml_vec_dot_q3_K_q8_K(D_MODEL, dst + r * D_FF + o, 0, wr, 0, xr, 0, 1);
            }
    }
    for (int r = 0; r < R; ++r)
    {
        silu_mul(S.g.data() + r * D_FF, S.u.data() + r * D_FF, S.h.data() + r * D_FF, D_FF);
        if (X.fmt == 4)
            quantize_row_q8_0(S.h.data() + r * D_FF, S.hq.data() + r * nb_dn * ab, D_FF);
        else
        {
            float * hp = S.hpad.data() + r * DOWN_PAD;
            std::memcpy(hp, S.h.data() + r * D_FF, D_FF * sizeof(float));
            std::fill(hp + D_FF, hp + DOWN_PAD, 0.f);
            quantize_row_q8_K(hp, S.hq.data() + r * nb_dn * ab, DOWN_PAD);
        }
    }
    const uint8_t * wd = X.wrows(e, 2);
    for (int r = 0; r < R; ++r)
        for (int o = 0; o < D_MODEL; ++o)
        {
            const void * hr = S.hq.data() + r * nb_dn * ab;
            const uint8_t * wr = wd + (size_t)o * nb_dn * bs;
            if (X.fmt == 4) ggml_vec_dot_q8_0_q8_0(n_dn, S.y.data() + r * D_MODEL + o, 0, wr, 0, hr, 0, 1);
            else if (X.fmt == 3) ggml_vec_dot_q4_K_q8_K(n_dn, S.y.data() + r * D_MODEL + o, 0, wr, 0, hr, 0, 1);
            else ggml_vec_dot_q3_K_q8_K(n_dn, S.y.data() + r * D_MODEL + o, 0, wr, 0, hr, 0, 1);
        }
    sink += S.y[0];
}

double st_run(const ExpertSet & X, int R, double seconds, Scratch & S)
{
    const double t0 = now_s();
    long count = 0;
    while (now_s() - t0 < seconds) { run_expert(X, (int)(count % X.n), R, S); ++count; }
    return (now_s() - t0) * 1000.0 / count;
}

double pool_run(const ExpertSet & X, int R, double seconds, int nthreads, const std::vector<int> & cores, Scratch * per)
{
    std::atomic<bool> stop{ false };
    std::atomic<long> ticket{ 0 }, done{ 0 };
    std::vector<std::thread> th;
    const double t0 = now_s();
    for (int i = 0; i < nthreads; ++i)
        th.emplace_back(
            [&, i]
            {
                if (i + 1 < (int)cores.size()) pin_to(cores[i + 1]);  // the pool's scheme: skip core 0 (the driver's)
                for (;;)
                {
                    const long k = ticket.fetch_add(8, std::memory_order_relaxed);
                    if (stop.load(std::memory_order_relaxed)) return;
                    for (int j = 0; j < 8; ++j) run_expert(X, (int)((k + j) % X.n), R, per[i]);
                    done.fetch_add(8, std::memory_order_relaxed);
                }
            });
    while (now_s() - t0 < seconds) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    stop.store(true);
    for (auto & t : th) t.join();
    return (now_s() - t0) * 1000.0 / (double)done.load();
}

double dram_ref(int nthreads, double seconds, const std::vector<int> & cores, size_t total_bytes)
{
    uint8_t * buf = (uint8_t *)mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (buf == MAP_FAILED || mlock(buf, total_bytes)) { std::fprintf(stderr, "dram arena failed\n"); return -1; }
    std::memset(buf, 3, total_bytes);
    std::atomic<bool> stop{ false };
    std::atomic<long> passes{ 0 };
    std::atomic<uint64_t> keep{ 0 };
    std::vector<std::thread> th;
    const size_t slice = total_bytes / nthreads;
    const double t0 = now_s();
    for (int i = 0; i < nthreads; ++i)
        th.emplace_back(
            [&, i]
            {
                if (i + 1 < (int)cores.size()) pin_to(cores[i + 1]);
                const uint64_t * s = reinterpret_cast<const uint64_t *>(buf + i * slice);
                const size_t n = slice / 64;
                uint64_t local = 0;
                while (!stop.load(std::memory_order_relaxed))
                {
                    for (size_t k = 0; k < n * 8; k += 8) local += s[k];  // one touch per 64B line: a full pass
                    passes.fetch_add(1, std::memory_order_relaxed);
                }
                keep.fetch_add(local);
            });
    while (now_s() - t0 < seconds) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    stop.store(true);
    for (auto & t : th) t.join();
    const double gbps = (double)passes.load() * slice / (now_s() - t0) / 1e9;
    munmap(buf, total_bytes);
    return gbps;
}

struct Result
{
    double st_ms = 0, pool_ms = 0;
};

}  // namespace

int main(int argc, char ** argv)
{
    const int nthreads = argc > 1 ? atoi(argv[1]) : 22;
    const int nexperts = argc > 2 ? atoi(argv[2]) : 64;
    const double seconds = argc > 3 ? atof(argv[3]) : 1.0;
    const int rounds = argc > 4 ? atoi(argv[4]) : 4;

    const std::vector<int> cores = physical_cores();
    std::fprintf(stderr, "cpu_quant: %zu physical cores, %d workers (skip core 0), %d experts, %.1fs, %d rounds\n",
                 cores.size(), nthreads, nexperts, seconds, rounds);
    pin_to(cores.empty() ? 0 : cores[0]);  // the driver thread: core 0, as the engine's

    static float xrows[2][D_MODEL];
    for (int r = 0; r < 2; ++r)
        for (int i = 0; i < D_MODEL; ++i) xrows[r][i] = frand();
    XROWS[0] = xrows[0]; XROWS[1] = xrows[1];

    std::fprintf(stderr, "sizes: block_q3_K=%zu block_q4_K=%zu block_q8_K=%zu block_q8_0=%zu\n", sizeof(block_q3_K),
                 sizeof(block_q4_K), sizeof(block_q8_K), sizeof(block_q8_0));

    ExpertSet XS[5];
    const char * NAMES[5] = { "trellis_k25", "trellis_k35", "q3_k_pad768", "q4_k_pad768", "q8_0" };
    const int32_t TYPES[5] = { 0, 0, T_Q3_K, T_Q4_K, T_Q8_0 };
    for (int f = 0; f < 5; ++f)
    {
        XS[f].fmt = f;
        XS[f].K = f == 0 ? 25 : 35;
        XS[f].n = nexperts;
        XS[f].per = expert_bytes(f, XS[f].K);
        XS[f].a.make((size_t)XS[f].per * nexperts);
        if (f < N_TRELLIS) build_trellis(XS[f]);
        else
        {
            float * src = new float[(size_t)D_MODEL * DOWN_PAD];
            build_ggml(XS[f], TYPES[f], src);
            delete[] src;
        }
        std::fprintf(stderr, "%s: %.4f MB/expert -> %.1f GB for %d cold experts\n", NAMES[f], XS[f].per / 1e6,
                     XS[f].per * (double)COLD_EXPERTS / 1e9, COLD_EXPERTS);
    }

    Scratch * pool1 = new Scratch[nthreads], * pool2 = new Scratch[nthreads];
    for (int i = 0; i < nthreads; ++i) { pool1[i].grow(1); pool2[i].grow(2); }
    Scratch S1, S2;
    S1.grow(1); S2.grow(2);
    Scratch * SCR[2] = { &S1, &S2 };

    std::fprintf(stderr, "dram ref (2 GB, %d pinned threads)...\n", nthreads);
    const double dram = dram_ref(nthreads, 0.8, cores, (size_t)2 << 30);
    std::fprintf(stderr, "RESULT | fmt=dram_ref pool_gbps=%.1f load=%s\n", dram, loadavg().c_str());

    Result best[5][2];
    for (int round = 0; round < rounds; ++round)
        for (int slot = 0; slot < 5; ++slot)
        {
            const int f = (round + slot) % 5;  // rotation: drift hits every format equally
            for (int R = 1; R <= 2; ++R)
            {
                if (f >= N_TRELLIS)  // shared activation rows (outside timing): x -> q8_K or q8_0
                {
                    const size_t ab = f == 4 ? sizeof(block_q8_0) : sizeof(block_q8_K);
                    const size_t nb = f == 4 ? D_MODEL / QK8_0 : D_MODEL / QK_K;
                    std::vector<uint8_t> xq((size_t)R * nb * ab);
                    for (int r = 0; r < R; ++r)
                        if (f == 4) quantize_row_q8_0(XROWS[r], xq.data() + r * nb * ab, D_MODEL);
                        else quantize_row_q8_K(XROWS[r], xq.data() + r * nb * ab, D_MODEL);
                    SCR[R - 1]->xq = xq;
                    Scratch * p = R == 1 ? pool1 : pool2;
                    for (int i = 0; i < nthreads; ++i) p[i].xq = xq;
                }
                const double st = st_run(XS[f], R, seconds, *SCR[R - 1]);
                const double pm = pool_run(XS[f], R, seconds, nthreads, cores, R == 1 ? pool1 : pool2);
                const double gbps = XS[f].per / pm / 1e6;  // B/ms -> GB/s
                Result & b = best[f][R - 1];
                if (b.st_ms == 0 || st < b.st_ms) b.st_ms = st;
                if (b.pool_ms == 0 || pm < b.pool_ms) b.pool_ms = pm;
                std::fprintf(stderr, "RESULT | fmt=%s R=%d st_ms=%.4f pool_ms=%.5f pool_gbps=%.1f bytes=%zu ram_gb=%.1f load=%s\n",
                             NAMES[f], R, st, pm, gbps, XS[f].per, XS[f].per * (double)COLD_EXPERTS / 1e9, loadavg().c_str());
            }
        }

    std::fprintf(stderr, "\n# best (min over rounds); R1.2 = 0.8*R1 + 0.2*R2 (slots/expert 1.203)\n");
    double mix[5];
    for (int f = 0; f < 5; ++f)
    {
        mix[f] = 0.8 * best[f][0].pool_ms + 0.2 * best[f][1].pool_ms;
        std::fprintf(stderr, "BEST | fmt=%-13s st1=%.4f st2=%.4f pool1=%.5f pool2=%.5f pool_R12=%.5f gbps=%.1f ram_gb=%.1f\n",
                     NAMES[f], best[f][0].st_ms, best[f][1].st_ms, best[f][0].pool_ms, best[f][1].pool_ms, mix[f],
                     XS[f].per / mix[f] / 1e6, XS[f].per * (double)COLD_EXPERTS / 1e9);
    }
    const double tm = MIX_K25 * mix[0] + (1.0 - MIX_K25) * mix[1];
    const double tb = MIX_K25 * XS[0].per + (1.0 - MIX_K25) * XS[1].per;
    std::fprintf(stderr, "BEST | fmt=trellis_mix   pool_R12=%.5f gbps=%.1f bytes=%.0f ram_gb=%.1f (served pack today)\n",
                 tm, tb / tm / 1e6, tb, tb * (double)COLD_EXPERTS / 1e9);
    std::fprintf(stderr, "sink=%f\n", (double)sink);
    return 0;
}
