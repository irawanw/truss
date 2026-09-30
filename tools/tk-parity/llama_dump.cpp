// Reference activations from llama-paw for layer-by-layer parity (CP2): runs one prompt through llama-paw as a
// single ubatch with every row an output row, and saves the graph tensors whose names match, as raw little-endian
// f32 (i32 for index tensors) plus index.tsv (name, dtype, ggml shape, file). The attention and FFN
// hyper-connection mixes are both "hc_mixed-<il>": they are saved as "#1" and "#2" in call order; any other
// repeated name gets "#2", "#3", ... from its second occurrence.
//
// usage: tk-parity-llama-dump <model.gguf> <out_dir> "<prompt>" | @<ids.i32> [base names, comma separated]
//   The whole model runs on the GPU (llama-paw's X3 MoE op is CUDA-only), so the full Flash-Next does not fit one
//   3090: dump a layer slice (tools/tk-parity/slice_gguf.py).
#include "ggml-backend.h"
#include "ggml-cpu.h"   // ggml_fp16_to_fp32_row
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Dump {
    std::filesystem::path dir;
    std::set<std::string> bases;           // empty = everything
    std::map<std::string, int> seen;
    FILE * index = nullptr;
    long n = 0;
    std::vector<uint8_t> buf;
};

std::string base_name(const char * name)
{
    const char * dash = std::strrchr(name, '-');
    return dash ? std::string(name, dash) : std::string(name);
}

bool on_tensor(ggml_tensor * t, bool ask, void * ud)
{
    auto & d = *static_cast<Dump *>(ud);
    const bool want = d.bases.empty() || d.bases.count(base_name(t->name));
    if (ask) return want;
    if (!want) return true;
    if (!ggml_is_contiguous(t) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_I32)) {
        std::fprintf(stderr, "skip %s (%s, %s)\n", t->name, ggml_type_name(t->type),
                     ggml_is_contiguous(t) ? "contiguous" : "strided");
        return true;
    }
    std::string name = t->name;
    if (const int k = d.seen[name]++; k > 0 || name.rfind("hc_mixed", 0) == 0) name += "#" + std::to_string(k + 1);
    d.buf.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, d.buf.data(), 0, d.buf.size());
    std::vector<float> f32;
    const void * out = d.buf.data();
    size_t out_bytes = d.buf.size();
    if (t->type == GGML_TYPE_F16) {
        f32.resize(ggml_nelements(t));
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) d.buf.data(), f32.data(), (int64_t) f32.size());
        out = f32.data();
        out_bytes = f32.size() * 4;
    }
    std::string file = name;
    for (char & c : file) if (c == '#' || c == '/') c = '_';
    file += t->type == GGML_TYPE_I32 ? ".i32" : ".f32";
    FILE * fp = std::fopen((d.dir / file).c_str(), "wb");
    if (!fp || std::fwrite(out, 1, out_bytes, fp) != out_bytes) {
        std::fprintf(stderr, "write failed: %s\n", (d.dir / file).c_str());
        std::exit(1);
    }
    std::fclose(fp);
    std::fprintf(d.index, "%s\t%s\t%lld,%lld,%lld,%lld\t%s\n", name.c_str(), t->type == GGML_TYPE_I32 ? "i32" : "f32",
                 (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], file.c_str());
    ++d.n;
    return true;
}

}  // namespace

int main(int argc, char ** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <out_dir> \"<prompt>\" [base names, comma separated]\n", argv[0]);
        return 2;
    }
    Dump d;
    d.dir = argv[2];
    std::filesystem::create_directories(d.dir);
    if (argc > 4) {
        std::stringstream ss(argv[4]);
        for (std::string s; std::getline(ss, s, ',');) d.bases.insert(s);
    }
    d.index = std::fopen((d.dir / "index.tsv").c_str(), "w");

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const std::string prompt = argv[3];
    std::vector<llama_token> toks;
    if (prompt.size() > 1 && prompt[0] == '@') {   // raw int32 token ids from a file
        std::FILE * f = std::fopen(prompt.c_str() + 1, "rb");
        if (!f) return 1;
        for (int32_t t; std::fread(&t, 4, 1, f) == 1;) toks.push_back(t);
        std::fclose(f);
    } else {
        toks.resize(prompt.size() + 8);
        const int m = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), toks.data(), (int32_t) toks.size(),
                                     false, true);
        if (m <= 0) return 1;
        toks.resize(m);
    }
    const int n = (int) toks.size();
    if (n <= 0) return 1;

    llama_context_params cp = llama_context_default_params();
    // the whole prompt is one ubatch (the dump is per ubatch): context and batch sized to it, at least 512
    const uint32_t n_ctx = (uint32_t) std::max(512, (n + 255) / 256 * 256);
    cp.n_ctx = n_ctx;
    cp.n_batch = cp.n_ubatch = n_ctx;
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &d;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = toks[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) {
        std::fprintf(stderr, "llama_decode failed\n");
        return 1;
    }
    FILE * tf = std::fopen((d.dir / "tokens.i32").c_str(), "wb");
    std::fwrite(toks.data(), 4, toks.size(), tf);
    std::fclose(tf);
    std::fclose(d.index);
    std::printf("%d tokens, %ld tensors -> %s\n", n, d.n, d.dir.c_str());
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
