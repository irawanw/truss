// Offline dump of the routed-expert byte table (host only, no CUDA): expert_bytes() exactly as ExpertStore::plan
// costs it - gate|up|down trellis tiles, words * 2 - for every layer of the main model and the MTP block.
// usage: dump_expert_bytes <model.gguf> <mtp.gguf> <n_layer> <n_expert> <out.i32>
#include "formats/gguf.h"
#include "formats/trellis_table.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace truss;

static formats::ExpertTable table(const gguf::File & f, const std::string & p, const char * what)
{
    return formats::read_expert_table(f, p + what);
}

static void one(const gguf::File & f, const std::string & p, int n_expert, std::vector<int32_t> & out)
{
    const formats::ExpertTable g = table(f, p, "ffn_gate_exps");
    const formats::ExpertTable u = table(f, p, "ffn_up_exps");
    const formats::ExpertTable d = table(f, p, "ffn_down_exps");
    out.resize((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) {
        const size_t b = (size_t) g.words(e) * 2 + (size_t) u.words(e) * 2 + (size_t) d.words(e) * 2;
        out[e] = (int32_t) b;
        if (b > INT32_MAX) std::abort();
    }
}

int main(int argc, char ** argv)
{
    if (argc < 6) { std::fprintf(stderr, "usage: %s <model> <mtp> <n_layer> <n_expert> <out>\n", argv[0]); return 2; }
    const int L = std::atoi(argv[3]), E = std::atoi(argv[4]);
    auto f = gguf::File::open(argv[1]), m = gguf::File::open(argv[2]);
    std::vector<int32_t> all;
    all.reserve((size_t) (L + 1) * E);
    for (int l = 0; l <= L; ++l) {   // the MTP block is blk.<L>. in its own file
        std::vector<int32_t> row;
        one(*(l < L ? f : m), "blk." + std::to_string(l) + ".", E, row);
        all.insert(all.end(), row.begin(), row.end());
        if (l == 0 || l == L) {
            long long sum = 0;
            for (int32_t b : row) sum += b;
            std::fprintf(stderr, "layer %d: %.3f GiB, min %.2f MB, max %.2f MB\n", l, sum / 1073741824.0,
                         *std::min_element(row.begin(), row.end()) / 1e6,
                         *std::max_element(row.begin(), row.end()) / 1e6);
        }
    }
    FILE * o = std::fopen(argv[5], "wb");
    if (!o) { std::fprintf(stderr, "cannot write %s\n", argv[5]); return 1; }
    std::fwrite(all.data(), 4, all.size(), o);
    std::fclose(o);
    long long sum = 0;
    for (int32_t b : all) sum += b;
    std::fprintf(stderr, "%zu values, %.2f GiB of routed experts\n", all.size(), sum / 1073741824.0);
    return 0;
}
