// Vision tower check: runs Vit on the Phase 0 fixtures (HF transformers taps) and prints cosine / max-abs-diff per tap.
//   vit_check <flashnext-mmproj-f16.gguf> <fixture dir> [--time]
// Fixtures: pixel_values.npy, grid_thw.npy, patch_embed/block0_out/block13_out/block26_out/vit_out.npy (f32, from
// ML_projects/flashnext/20261022_vision/scripts/p0_fixtures.py). Exit 0 when vit_out cosine >= 0.999 and block0 >= 0.9999.
#include "model/qwen4exp/vision/vit.h"
#include "npy_check.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace truss;

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
