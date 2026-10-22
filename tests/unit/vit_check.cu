// Vision tower check: runs Vit on the Phase 0 fixtures (HF transformers taps) and prints cosine / max-abs-diff per tap.
//   vit_check <flashnext-mmproj-f16.gguf> <fixture dir> [--time]
// Fixtures: pixel_values.npy, grid_thw.npy, patch_embed/block0_out/block13_out/block26_out/vit_out.npy (f32, from
// ML_projects/flashnext/20261022_vision/scripts/p0_fixtures.py). Exit 0 when vit_out cosine >= 0.999 and block0 >= 0.9999.
#include "model/qwen4exp/vision/vit.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace truss;

static std::vector<float> npy(const std::string & p, std::vector<long> * shape = nullptr)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
    char m[10]; f.read(m, 10);
    unsigned hl = (unsigned char) m[8] | ((unsigned char) m[9] << 8);
    std::string h(hl, ' '); f.read(&h[0], hl);
    if (h.find("<f4") == std::string::npos && h.find("<i8") == std::string::npos) { fprintf(stderr, "%s: dtype not f4/i8: %s\n", p.c_str(), h.c_str()); exit(2); }
    std::vector<long> sh; size_t a = h.find("shape"), b = h.find('(', a), c = h.find(')', b);
    for (size_t i = b + 1; i < c;) { while (i < c && (h[i] == ' ' || h[i] == ',')) i++; if (i >= c) break; sh.push_back(strtol(h.c_str() + i, nullptr, 10)); while (i < c && h[i] != ',') i++; }
    long n = 1; for (long s : sh) n *= s;
    std::vector<float> v(n);
    if (h.find("<i8") != std::string::npos) { std::vector<long> t(n); f.read((char *) t.data(), n * 8); for (long i = 0; i < n; i++) v[i] = (float) t[i]; }
    else f.read((char *) v.data(), n * 4);
    if (shape) *shape = sh;
    return v;
}

static bool cmp(const char * name, const std::vector<float> & ref, const std::vector<float> & got, double need)
{
    if (ref.size() != got.size()) { printf("%-12s SIZE MISMATCH ref %zu got %zu\n", name, ref.size(), got.size()); return false; }
    double dot = 0, na = 0, nb = 0, mx = 0, amax = 0;
    for (size_t i = 0; i < ref.size(); i++) { dot += (double) ref[i] * got[i]; na += (double) ref[i] * ref[i]; nb += (double) got[i] * got[i]; mx = std::fmax(mx, std::fabs(ref[i] - got[i])); amax = std::fmax(amax, std::fabs(ref[i])); }
    const double cosv = dot / std::sqrt(na * nb + 1e-30);
    printf("%-12s cos %.6f  max|diff| %.4g  (ref absmax %.4g, rel %.2e)  %s\n", name, cosv, mx, amax, mx / amax, cosv >= need ? "ok" : "FAIL");
    return cosv >= need;
}

int main(int argc, char ** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: vit_check <mmproj.gguf> <fixture dir> [--time]\n"); return 2; }
    const std::string dir = argv[2];
    try {
        auto f = gguf::File::open(argv[1]);
        qwen4exp::Vit vit(*f, 0);
        auto pv = npy(dir + "/pixel_values.npy"); auto g = npy(dir + "/grid_thw.npy");
        const int gh = (int) g[1], gw = (int) g[2];
        std::map<std::string, std::vector<float>> taps;
        vit.forward(pv.data(), gh, gw, &taps);
        bool ok = true;
        ok &= cmp("patch+pos", npy(dir + "/patch_pos.npy"), taps["patch_embed"], 0.9999);
        ok &= cmp("block0_out", npy(dir + "/block0_out.npy"), taps["block0_out"], 0.9999);
        ok &= cmp("block13_out", npy(dir + "/block13_out.npy"), taps["block13_out"], 0.999);
        ok &= cmp("block26_out", npy(dir + "/block26_out.npy"), taps["block26_out"], 0.999);
        ok &= cmp("vit_out", npy(dir + "/vit_out.npy"), taps["vit_out"], 0.999);
        if (argc > 3 && !strcmp(argv[3], "--time")) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 5; i++) vit.forward(pv.data(), gh, gw);
            printf("forward %dx%d patches: %.1f ms\n", gh, gw, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5);
        }
        printf(ok ? "VIT CHECK PASS\n" : "VIT CHECK FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception & e) { fprintf(stderr, "error: %s\n", e.what()); return 2; }
}
