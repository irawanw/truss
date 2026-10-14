// Engine golden test: one fixed script of C API calls, every result written to a file.
//
// The C API (include/truss/truss.h) is the engine's model-agnostic surface, so this test does not change when the
// engine underneath is rebuilt: run it on the old build (golden) and on the new one, and `cmp` the two files. Every
// engine path the server uses is in the script - prompt chunks that stream whole expert layers and chunks small
// enough to fetch, single-token decode, checkpoint and restore, greedy and sampled speculative rounds, device
// sampling with penalties, and (TRUSS_SLOTS >= 2) a second sequence interleaved with the first. The environment picks
// the engine options (TRUSS_ACT, TRUSS_CPU_TRELLIS, TRUSS_EXPERT_GB, TRUSS_LAYER_SPLIT, TRUSS_KV_INT8, ...);
// tests/golden/run.sh holds the configurations.
//
// usage: engine_golden <model.gguf> <tokens.i32> <out.bin> [mtp.gguf|-] [drafts] [draft_min_p]
// The output holds, in order: every logits vector and token the script produced (no timing, no addresses).
#include "truss/truss.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Out {
    FILE * f;
    void tokens(const int32_t * t, int n) { std::fwrite(t, 4, (size_t) n, f); }
    void floats(const std::vector<float> & v) { std::fwrite(v.data(), 4, v.size(), f); }
};

void ok(int rc, truss_model *, const char * what)
{
    if (rc < 0) throw std::runtime_error(std::string(what) + ": " + truss_last_error());
}

int argmax(const std::vector<float> & v)
{
    int b = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[b]) b = (int) i;
    return b;
}

// prompt, decode, checkpoint / restore, speculative rounds and sampling on one slot
void script(truss_model * m, int slot, const std::vector<int32_t> & tok, int off, Out & o)
{
    const int V = truss_n_vocab(m);
    std::vector<float> lg(V);
    ok(truss_reset_s(m, slot), m, "reset");
    // 200 prompt tokens in 64-token chunks (prompt path), then 20 more in one call (fetch-sized chunk)
    ok(truss_eval_step_s(m, slot, tok.data() + off, 200, 64, lg.data()), m, "prompt");
    o.floats(lg);
    ok(truss_eval_s(m, slot, tok.data() + off + 200, 20, lg.data()), m, "short chunk");
    o.floats(lg);
    int32_t next = argmax(lg);
    for (int i = 0; i < 12; ++i) {   // greedy decode
        int32_t t = next;
        ok(truss_eval_argmax_s(m, slot, &t, 1, &next), m, "decode");
        o.tokens(&next, 1);
    }
    // checkpoint, 8 tokens, restore, the same 8 again: both logits go out (they must match each other too)
    ok(truss_checkpoint_s(m, slot), m, "checkpoint");
    ok(truss_eval_s(m, slot, tok.data() + off + 220, 8, lg.data()), m, "after checkpoint");
    o.floats(lg);
    const int p = truss_restore_s(m, slot);
    ok(p, m, "restore");
    o.tokens(&p, 1);
    ok(truss_eval_s(m, slot, tok.data() + off + 220, 8, lg.data()), m, "after restore");
    o.floats(lg);
    next = argmax(lg);
    if (truss_drafts(m) > 0) {
        int32_t em[16], nn;
        int ne;
        for (int i = 0; i < 10; ++i) {   // greedy speculative rounds
            ok(truss_spec_step_s(m, slot, next, em, &ne, &nn), m, "spec step");
            o.tokens(&ne, 1), o.tokens(em, ne), o.tokens(&nn, 1);
            next = nn;
        }
    }
    truss_sampling sp{};
    sp.temperature = 0.8f, sp.top_p = 0.95f, sp.top_k = 40, sp.min_p = 0.05f, sp.seed = 1234 + slot;
    sp.repetition_penalty = 1.1f, sp.presence_penalty = 0.3f, sp.penalty_last_n = 64;
    sp.history = tok.data() + off, sp.n_history = 228;
    for (int i = 0; i < 6; ++i) {   // device sampling with penalties
        int32_t t = next;
        ok(truss_eval_sample_s(m, slot, &t, 1, &sp, &next), m, "sample");
        o.tokens(&next, 1);
    }
    if (truss_drafts(m) > 0) {
        int32_t em[16], nn;
        int ne;
        for (int i = 0; i < 6; ++i) {   // sampled speculative rounds
            ok(truss_spec_step_sampled_s(m, slot, next, &sp, em, &ne, &nn), m, "sampled spec step");
            o.tokens(&ne, 1), o.tokens(em, ne), o.tokens(&nn, 1);
            next = nn;
        }
    }
    const int pos = truss_position_s(m, slot);
    o.tokens(&pos, 1);
}

}  // namespace

int main(int argc, char ** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <tokens.i32> <out.bin> [mtp.gguf|-] [drafts] [draft_min_p]\n", argv[0]);
        return 2;
    }
    std::vector<int32_t> tok;
    if (FILE * f = std::fopen(argv[2], "rb")) {
        int32_t t;
        while (std::fread(&t, 4, 1, f) == 1) tok.push_back(t);
        std::fclose(f);
    }
    if (tok.size() < 256) {
        std::fprintf(stderr, "need >= 256 tokens in %s\n", argv[2]);
        return 2;
    }
    truss_params p = truss_default_params();
    p.n_ctx = 8192, p.max_chunk = 256;
    if (argc > 4 && std::strcmp(argv[4], "-") != 0) p.mtp = argv[4];
    if (argc > 5) p.drafts = std::atoi(argv[5]);
    if (argc > 6) p.draft_min_p = (float) std::atof(argv[6]);
    truss_model * m = truss_open_params(argv[1], &p);
    if (!m) {
        std::fprintf(stderr, "open: %s\n", truss_last_error());
        return 1;
    }
    FILE * f = std::fopen(argv[3], "wb");
    if (!f) return 1;
    Out o{ f };
    try {
        script(m, 0, tok, 0, o);
        if (truss_slots(m) >= 2) {   // a second sequence, then the first one again on top of its state
            script(m, 1, tok, 16, o);
            int32_t t = 1000, next;
            ok(truss_eval_argmax_s(m, 0, &t, 1, &next), m, "slot 0 after slot 1");
            o.tokens(&next, 1);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "engine_golden: %s\n", e.what());
        std::fclose(f);
        truss_close(m);
        return 1;
    }
    std::fclose(f);
    truss_close(m);
    std::printf("engine_golden: wrote %s\n", argv[3]);
    return 0;
}
