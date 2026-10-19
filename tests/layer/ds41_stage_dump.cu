// Runs layers [0, N) of deepseek41 on one GPU for one prompt and lets the model's TRUSS_DS41_DUMP hooks write each
// layer's residual / attention / MoE tensors (src/model/deepseek41/model.cu dump()); tests/layer/ds41_stage_cmp.py
// compares them with the PoC's Python captures. Only the N layers' experts load (one stage, no head).
// usage: TRUSS_DS41_DUMP=<dir> TRUSS_EXPERT_OVERLAY=<experts>[,<engram L1>,...] ds41_stage_dump <dense.gguf>
//          <ids.i32> <seq> <ctx> <N>
#include "core/cuda_check.h"
#include "engine/model.h"
#include "engine/options.h"
#include "engine/stage.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace truss;

int main(int argc, char ** argv)
{
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s <dense.gguf> <ids.i32> <seq> <ctx> <n_layers>\n", argv[0]);
        return 2;
    }
    try {
        const int seq = std::atoi(argv[3]), ctx = std::atoi(argv[4]), N = std::atoi(argv[5]);
        std::vector<int32_t> ids(ctx);
        std::ifstream in(argv[2], std::ios::binary);
        in.seekg((std::streamoff) seq * ctx * 4);
        in.read((char *) ids.data(), ctx * 4);
        const auto file = gguf::File::open(argv[1]);
        if (const char * ov = std::getenv("TRUSS_EXPERT_OVERLAY")) file->overlay_list(ov);
        const auto fam = engine::open_family(*file, {});
        engine::Options o;
        o.act = blocks::Activations::FP16;
        o.prefill_rows = ctx;
        engine::apply_env(o);
        engine::Stage st(*fam, ctx, ctx, o, engine::StageSpec{ 0, N, -1, 1.0, nullptr, nullptr });
        st.select(0);
        st.reset();
        st.run_chunk(ids.data(), 0, ctx, nullptr, false);
        TRUSS_CUDA(cudaDeviceSynchronize());
        std::printf("layers 0-%d done for seq %d (%d tokens)\n", N - 1, seq, ctx);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
