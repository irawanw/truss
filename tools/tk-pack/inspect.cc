// Catalog of a model GGUF as TRUSS reads it: bytes per tensor class (the budget sheet's rows, CP0 §e) and a
// consistency check of every trellis expert table (m3_meta K and word offsets against m3_trellis, m3_suh, m3_svh).
// usage: tk-pack-inspect <model.gguf | any shard>          summary
//        tk-pack-inspect <model.gguf> list                 one line per tensor (tests/unit/gguf_reader_check.py)
#include "formats/gguf.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using truss::gguf::File;
using truss::gguf::Tensor;
using truss::gguf::Type;

namespace {

bool has(const std::string & s, const char * part) { return s.find(part) != std::string::npos; }

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

struct ExpertCheck {
    int layers = 0, experts = 0, errors = 0;
    uint64_t trellis_bytes = 0, scale_bytes = 0, meta_bytes = 0;
    std::map<int, long> k_hist;            // K -> expert-projections
};

// One routed projection "<prefix>.m3_*": meta [n_expert][2] = (K, uint16 word offset); expert e holds
// in*out*K/16 words (16*K uint16 per 16x16 tile); suh [n_expert][in], svh [n_expert][out].
void check_trellis(const File & f, const std::string & prefix, ExpertCheck & ec)
{
    const Tensor & tr = f.at(prefix + ".m3_trellis");
    const Tensor & meta = f.at(prefix + ".m3_meta");
    const Tensor & suh = f.at(prefix + ".m3_suh");
    const Tensor & svh = f.at(prefix + ".m3_svh");
    auto bad = [&] (const std::string & why) {
        if (ec.errors++ < 20) std::printf("  BAD %s: %s\n", prefix.c_str(), why.c_str());
    };
    if (tr.type != Type::I16 || meta.type != Type::I32 || suh.type != Type::F16 || svh.type != Type::F16)
        return bad("unexpected tensor types");
    const int64_t n_expert = meta.shape.size() == 2 && meta.shape[0] == 2 ? meta.shape[1] : -1;
    if (n_expert <= 0 || suh.shape.size() != 2 || svh.shape.size() != 2 || suh.shape[1] != n_expert ||
        svh.shape[1] != n_expert)
        return bad("meta/suh/svh shapes disagree");
    const int64_t in = suh.shape[0], out = svh.shape[0];
    if (in % 16 || out % 16) return bad("in/out not multiples of 16");
    const auto * m = reinterpret_cast<const int32_t *>(meta.data);
    int64_t next = 0;
    for (int64_t e = 0; e < n_expert; ++e) {
        const int k = m[2 * e];
        const int64_t off = m[2 * e + 1];
        if (k < 1 || k > 8) return bad("expert " + std::to_string(e) + " K=" + std::to_string(k));
        if (off != next) return bad("expert " + std::to_string(e) + " offset " + std::to_string(off) + ", expected " +
                                    std::to_string(next));
        next += in * out * k / 16;
        ++ec.k_hist[k];
    }
    if (next != tr.elements())
        return bad("experts need " + std::to_string(next) + " words, tensor has " + std::to_string(tr.elements()));
    ec.experts += (int) n_expert;
    ec.trellis_bytes += tr.bytes;
    ec.scale_bytes += suh.bytes + svh.bytes;
    ec.meta_bytes += meta.bytes;
}

std::string shape_str(const Tensor & t)
{
    std::string s;
    for (size_t i = 0; i < t.shape.size(); ++i) s += (i ? "," : "") + std::to_string(t.shape[i]);
    return s;
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
            for (const Tensor & t : f->tensors())
                std::printf("%s\t%s\t%s\t%d\t%llu\t%llu\n", t.name.c_str(), truss::gguf::type_info(t.type)->name,
                            shape_str(t).c_str(), t.shard, (unsigned long long) t.file_offset,
                            (unsigned long long) t.bytes);
            return 0;
        }
        std::printf("%s: %zu shards, %zu tensors, %zu metadata keys, arch %s\n", argv[1], f->shard_paths().size(),
                    f->tensors().size(), f->metadata().size(), f->get_string("general.architecture").c_str());

        struct Row { uint64_t bytes = 0; long tensors = 0; std::map<std::string, uint64_t> by_type; };
        std::map<std::string, Row> rows;
        std::vector<std::string> others;
        uint64_t total = 0;
        ExpertCheck ec;
        for (const Tensor & t : f->tensors()) {
            const char * cls = tensor_class(t.name);
            Row & r = rows[cls];
            r.bytes += t.bytes;
            ++r.tensors;
            r.by_type[truss::gguf::type_info(t.type)->name] += t.bytes;
            total += t.bytes;
            if (std::strcmp(cls, "other") == 0 && others.size() < 12) others.push_back(t.name);
            const std::string sfx = ".m3_trellis";
            if (t.name.size() > sfx.size() && t.name.compare(t.name.size() - sfx.size(), sfx.size(), sfx) == 0)
                check_trellis(*f, t.name.substr(0, t.name.size() - sfx.size()), ec);
        }
        std::printf("\n%-18s %8s %10s %9s  by type (GiB)\n", "class", "tensors", "GiB", "GB");
        for (auto & [cls, r] : rows) {
            std::printf("%-18s %8ld %10.3f %9.3f ", cls.c_str(), r.tensors, r.bytes / 1073741824.0, r.bytes / 1e9);
            for (auto & [ty, b] : r.by_type) std::printf(" %s %.3f", ty.c_str(), b / 1073741824.0);
            std::printf("\n");
        }
        std::printf("%-18s %8zu %10.3f %9.3f\n", "total", f->tensors().size(), total / 1073741824.0, total / 1e9);
        if (!others.empty()) {
            std::printf("\nunclassified (first %zu):", others.size());
            for (auto & n : others) std::printf(" %s", n.c_str());
            std::printf("\n");
        }

        const int64_t layers = f->get_int(f->get_string("general.architecture") + ".block_count");
        std::printf("\ntrellis expert tables: %d expert-projections checked, %d errors\n", ec.experts, ec.errors);
        if (ec.experts) {
            const double per_expert = (double) (ec.trellis_bytes + ec.scale_bytes + ec.meta_bytes) / (ec.experts / 3.0);
            std::printf("  words %.3f GiB, suh+svh %.3f GiB, meta %.1f KiB; %.4f MB per expert (3 projections, %lld "
                        "layers)\n  K:", ec.trellis_bytes / 1073741824.0, ec.scale_bytes / 1073741824.0,
                        ec.meta_bytes / 1024.0, per_expert / 1e6, (long long) layers);
            for (auto & [k, n] : ec.k_hist) std::printf(" K%d %ld", k, n);
            std::printf("\n");
        }
        return ec.errors ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
