// Catalog of a model GGUF as TRUSS reads it: bytes per tensor class (the budget sheet's rows, CP0 §e), a check of
// every trellis expert table, and, for a known architecture, the model binding (every tensor bound, every shape
// right).
// usage: tk-pack-inspect <model.gguf | any shard>          summary
//        tk-pack-inspect <model.gguf> list                 one line per tensor (tests/unit/gguf_reader_check.py)
#include "formats/gguf.h"
#include "formats/trellis_table.h"
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/weights.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using truss::gguf::File;
using truss::gguf::Tensor;

namespace {

bool has(const std::string & s, const char * part) { return s.find(part) != std::string::npos; }

bool ends_with(const std::string & s, const std::string & sfx)
{
    return s.size() >= sfx.size() && s.compare(s.size() - sfx.size(), sfx.size(), sfx) == 0;
}

// Budget-sheet class of a tensor, by name (qwen4exp naming). Order matters: first match wins.
const char * tensor_class(const std::string & n)
{
    if (has(n, "_exps.")) return "routed experts";
    if (has(n, "_shexp")) return "shared expert";
    if (has(n, "ffn_gate_inp")) return "router";
    if (has(n, "hc_")) return "hyper-connections";
    if (has(n, "indexer.")) return "DSA indexer";
    if (has(n, "ple") || has(n, "per_layer_token_embd")) return "PLE";
    if (n.rfind("token_embd", 0) == 0) return "token_embd";
    if (n.rfind("output.", 0) == 0) return "lm_head";
    if (has(n, "nextn") || has(n, "mtp")) return "MTP";
    if (has(n, "attn_") || has(n, "ssm_")) return "mixer";
    return "other";
}

std::string shape_str(const Tensor & t)
{
    std::string s;
    for (size_t i = 0; i < t.shape.size(); ++i) s += (i ? "," : "") + std::to_string(t.shape[i]);
    return s;
}

void list(const File & f)
{
    for (const Tensor & t : f.tensors())
        std::printf("%s\t%s\t%s\t%d\t%llu\t%llu\n", t.name.c_str(), truss::gguf::type_info(t.type)->name,
                    shape_str(t).c_str(), t.shard, (unsigned long long) t.file_offset, (unsigned long long) t.bytes);
}

void classes(const File & f)
{
    struct Row { uint64_t bytes = 0; long tensors = 0; std::map<std::string, uint64_t> by_type; };
    std::map<std::string, Row> rows;
    std::vector<std::string> others;
    uint64_t total = 0;
    for (const Tensor & t : f.tensors()) {
        const char * cls = tensor_class(t.name);
        Row & r = rows[cls];
        r.bytes += t.bytes;
        ++r.tensors;
        r.by_type[truss::gguf::type_info(t.type)->name] += t.bytes;
        total += t.bytes;
        if (std::strcmp(cls, "other") == 0 && others.size() < 12) others.push_back(t.name);
    }
    std::printf("\n%-18s %8s %10s %9s  by type (GiB)\n", "class", "tensors", "GiB", "GB");
    for (auto & [cls, r] : rows) {
        std::printf("%-18s %8ld %10.3f %9.3f ", cls.c_str(), r.tensors, r.bytes / 1073741824.0, r.bytes / 1e9);
        for (auto & [ty, b] : r.by_type) std::printf(" %s %.3f", ty.c_str(), b / 1073741824.0);
        std::printf("\n");
    }
    std::printf("%-18s %8zu %10.3f %9.3f\n", "total", f.tensors().size(), total / 1073741824.0, total / 1e9);
    if (!others.empty()) {
        std::printf("\nunclassified (first %zu):", others.size());
        for (auto & n : others) std::printf(" %s", n.c_str());
        std::printf("\n");
    }
}

void expert_tables(const File & f)
{
    long projections = 0;
    uint64_t words = 0, scales = 0, meta = 0;
    std::map<int, long> k_hist;
    const std::string sfx = ".m3_trellis";
    for (const Tensor & t : f.tensors()) {
        if (!ends_with(t.name, sfx)) continue;
        const auto e = truss::formats::read_expert_table(f, t.name.substr(0, t.name.size() - sfx.size()));
        projections += e.n_expert;
        words += e.trellis->bytes;
        scales += e.suh->bytes + e.svh->bytes;
        meta += e.meta->bytes;
        for (int k : e.k) ++k_hist[k];
    }
    if (!projections) return;
    std::printf("\ntrellis expert tables: %ld expert-projections, all consistent\n", projections);
    std::printf("  words %.3f GiB, suh+svh %.3f GiB, meta %.1f KiB; %.4f MB per expert (3 projections)\n  K:",
                words / 1073741824.0, scales / 1073741824.0, meta / 1024.0,
                (double) (words + scales + meta) / (projections / 3.0) / 1e6);
    for (auto & [k, n] : k_hist) std::printf(" K%d %ld", k, n);
    std::printf("\n");
}

void bind_model(const File & f)
{
    const std::string arch = f.get_string("general.architecture");
    if (arch != "qwen4exp") {
        std::printf("\nno TRUSS model binding for architecture %s\n", arch.c_str());
        return;
    }
    const auto c = truss::qwen4exp::Config::from_gguf(f);
    const auto w = truss::qwen4exp::bind(f, c);
    int gdn = 0, dsa = 0;
    for (const auto & l : w.layers) (l.mixer == truss::qwen4exp::Mixer::GDN ? gdn : dsa)++;
    std::printf("\nqwen4exp binding: all %zu tensors bound; %d layers (%d GDN, %d DSA), PLE at", f.tensors().size(),
                c.n_layer, gdn, dsa);
    for (auto l : c.ple_layers) std::printf(" %lld", (long long) l);
    std::printf("; d_model %d, vocab %d, experts %d top-%d, hc %d x rank %d\n", c.d_model, c.n_vocab, c.n_expert,
                c.n_expert_used, c.hc, c.hc_rank);
}

}  // namespace

int main(int argc, char ** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [list]\n", argv[0]);
        return 2;
    }
    try {
        const auto f = File::open(argv[1]);
        if (argc > 2 && std::strcmp(argv[2], "list") == 0) {
            list(*f);
            return 0;
        }
        std::printf("%s: %zu shards, %zu tensors, %zu metadata keys, arch %s\n", argv[1], f->shard_paths().size(),
                    f->tensors().size(), f->metadata().size(), f->get_string("general.architecture").c_str());
        classes(*f);
        expert_tables(*f);
        bind_model(*f);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
