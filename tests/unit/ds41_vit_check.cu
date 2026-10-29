// DS41 vision tower check: runs deepseek41::Vit on the fixtures of DeepSeek's reference (vision.py / image_processor.py,
// f32 math on the bf16 weights) and prints cosine / max-abs-diff per tap.
//   ds41_vit_check <ds41-mmproj-f16.gguf> <fixture dir> [--time]
// Fixtures (ML_projects/dsv41flash/20261025_vision/scripts/v1_fixtures.py): pixel_values, grid [n_vit_h, n_vit_w,
// n_llm_h, n_llm_w], patch_embed, block0_out, block15_out, block31_out, vit_out, rows, rows_bf16 (.npy).
// Exit 0 when every tap passes: block0 >= 0.9999, later taps >= 0.999, aligner rows at least as close to the f32
// reference as DeepSeek's own bf16 run is (rows_bf16), and the span layout (START, rows + NEWLINE per line, END) holds.
#include "model/deepseek41/vision/vit.h"
#include "npy_check.h"

#include <chrono>
#include <cstring>

using namespace truss;

static double cosine(const std::vector<float> & a, const std::vector<float> & b)
{
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) { dot += (double) a[i] * b[i]; na += (double) a[i] * a[i]; nb += (double) b[i] * b[i]; }
    return dot / std::sqrt(na * nb + 1e-30);
}

int main(int argc, char ** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: ds41_vit_check <ds41-mmproj.gguf> <fixture dir> [--time]\n"); return 2; }
    const std::string dir = argv[2];
    try {
        auto f = gguf::File::open(argv[1]);
        deepseek41::Vit vit(*f, 0);
        auto pv = npy(dir + "/pixel_values.npy"); auto g = npy(dir + "/grid.npy");
        const int nh = (int) g[0], nw = (int) g[1], lh = (int) g[2], lw = (int) g[3], W = vit.out_hidden();
        std::map<std::string, std::vector<float>> taps;
        const float * span_dev = vit.forward(pv.data(), nh, nw, &taps);
        const int span = vit.span_rows(nh, nw);
        std::vector<float> sp((size_t) span * W);
        cudaMemcpy(sp.data(), span_dev, sp.size() * 4, cudaMemcpyDeviceToHost);
        bool ok = span == lh * (lw + 1) + 2;
        ok &= cmp("patch_embed", npy(dir + "/patch_embed.npy"), taps["patch_embed"], 0.9999);
        ok &= cmp("block0_out", npy(dir + "/block0_out.npy"), taps["block0_out"], 0.9999);
        ok &= cmp("block15_out", npy(dir + "/block15_out.npy"), taps["block15_out"], 0.999);
        ok &= cmp("block31_out", npy(dir + "/block31_out.npy"), taps["block31_out"], 0.999);
        ok &= cmp("vit_out", npy(dir + "/vit_out.npy"), taps["vit_out"], 0.999);
        const auto ref = npy(dir + "/rows.npy");
        const double floor = cosine(npy(dir + "/rows_bf16.npy"), ref);
        ok &= cmp("rows", ref, taps["rows"], floor);
        printf("rows floor (reference bf16 vs f32) %.6f\n", floor);
        // span layout: START, line i = rows [i*lw, (i+1)*lw) then NEWLINE, END
        auto host = [&](const char * n) { const auto & t = f->at(n); return std::vector<float>((const float *) t.data, (const float *) t.data + W); };
        const auto st = host("span.start"), nl = host("span.newline"), en = host("span.end");
        auto row_is = [&](int r, const float * want) { return !memcmp(sp.data() + (size_t) r * W, want, (size_t) W * 4); };
        bool lay = row_is(0, st.data()) && row_is(span - 1, en.data());
        for (int i = 0; i < lh && lay; i++) {
            for (int j = 0; j < lw && lay; j++) lay = row_is(1 + i * (lw + 1) + j, taps["rows"].data() + ((size_t) i * lw + j) * W);
            lay = lay && row_is(1 + i * (lw + 1) + lw, nl.data());
        }
        printf("span %d rows (%dx%d tokens): layout %s\n", span, lh, lw, lay ? "ok" : "FAIL");
        ok &= lay;
        if (argc > 3 && !strcmp(argv[3], "--time")) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 3; i++) vit.forward(pv.data(), nh, nw);
            printf("forward %dx%d patches: %.1f ms\n", nh, nw, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 3);
        }
        printf(ok ? "DS41 VIT CHECK PASS\n" : "DS41 VIT CHECK FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception & e) { fprintf(stderr, "error: %s\n", e.what()); return 2; }
}
