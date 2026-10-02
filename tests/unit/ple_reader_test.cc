// PleReader (O_DIRECT row reads + row cache) against the mapping's ple_gather: bit-identical fp16 rows for random
// rows, rows straddling a 4 KiB page, repeats inside a call (cache misses read twice) and across calls (cache hits),
// and with a cache smaller than the working set (evictions). Also the time per row, cold and warm.
// usage: ple_reader_test <model gguf (shard 1)>
#include "formats/gguf.h"
#include "model/qwen4exp/ple_reader.h"

#include <cuda_fp16.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace truss;

int main(int argc, char ** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: ple_reader_test <model gguf>\n");
        return 2;
    }
    const auto file = gguf::File::open(argv[1]);
    const gguf::Tensor * table = file->find("per_layer_token_embd.q8"), * scale = file->find("per_layer_token_embd.scale");
    if (!table || !scale) {
        std::fprintf(stderr, "no PLE table in %s\n", argv[1]);
        return 2;
    }
    const int rb = (int) table->shape[0];
    const int64_t n_rows = table->shape[1];
    auto mapped = [&](const std::vector<int32_t> & rows) {   // ple_gather's arithmetic: int8 x fp16 scale -> fp16
        std::vector<half> out(rows.size() * rb);
        const auto * t = reinterpret_cast<const int8_t *>(table->data);
        const auto * s = reinterpret_cast<const half *>(scale->data);
        for (size_t r = 0; r < rows.size(); ++r)
            for (int j = 0; j < rb; ++j)
                out[r * rb + j] = __float2half(t[(int64_t) rows[r] * rb + j] * __half2float(s[rows[r]]));
        return out;
    };
    std::mt19937_64 rng(7);
    std::uniform_int_distribution<int64_t> any(0, n_rows - 1);
    int fails = 0;
    for (size_t cache : { (size_t) 1 << 20, (size_t) 64 }) {
        qwen4exp::PleReader rd(*file, *table, *scale, rb, 16, cache);
        for (int call = 0; call < 6; ++call) {
            std::vector<int32_t> rows;
            const int n = call % 2 ? 16 * 3 : 16 * 4096;   // a decode window / a prompt chunk
            for (int i = 0; i < n; ++i) rows.push_back((int32_t) any(rng));
            for (int i = 0; i < 64; ++i) {   // rows that straddle a page: start within rb bytes of a page end
                const int64_t page = any(rng) % (table->bytes / 4096 - 1) + 1;
                const int64_t r = ((int64_t) table->file_offset / 4096 * 4096 + page * 4096 - (int64_t) table->file_offset) / rb;
                if (r >= 0 && r < n_rows) rows.push_back((int32_t) r);
            }
            for (int i = 0; i < 32; ++i) rows.push_back(rows[i]);                       // repeats in this call
            if (call) for (int i = 0; i < 32; ++i) rows.push_back((int32_t) (i * 7919)); // repeats across calls
            std::vector<half> got(rows.size() * rb);
            const auto t0 = std::chrono::steady_clock::now();
            rd.collect(rd.issue(rows.data(), rows.size()), got.data());
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            const auto want = mapped(rows);
            const bool same = std::memcmp(got.data(), want.data(), got.size() * sizeof(half)) == 0;
            fails += !same;
            std::printf("cache %7zu call %d: %6zu rows in %8.0f us (%.2f us/row) %s\n", cache, call, rows.size(), us,
                        us / rows.size(), same ? "identical" : "DIFFERENT");
        }
        const auto & st = rd.stats();
        std::printf("cache %7zu: %llu rows, %llu cache hits, %llu pages read, %.1f ms waiting\n", cache,
                    (unsigned long long) st.rows, (unsigned long long) st.cache_hits, (unsigned long long) st.pages,
                    st.wait_ms);
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
