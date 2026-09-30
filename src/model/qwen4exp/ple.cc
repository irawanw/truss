#include "model/qwen4exp/ple.h"

#include <cuda_fp16.h>

#include <stdexcept>
#include <vector>

namespace truss::qwen4exp {

void ple_rows(const Config & c, const int32_t * tokens, int T, int32_t * rows)
{
    const int n_gram = c.ple_ngram, heads = c.ple_heads();
    std::vector<int64_t> win(n_gram);
    for (int i = 0; i < T; ++i) {
        win[0] = tokens[i];
        bool cut = false;
        for (int s = 1; s < n_gram; ++s) {
            const int64_t t = cut || i - s < 0 ? -1 : tokens[i - s];
            cut = cut || t < 0 || t == c.ple_eos;
            win[s] = cut ? c.ple_eos : t;
        }
        for (int n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) win[0] * (uint64_t) c.ple_multipliers[0];
            for (int j = 1; j < n; ++j) mixed ^= (uint64_t) win[j] * (uint64_t) c.ple_multipliers[j];
            for (int g = 0; g < c.ple_heads_per_ngram; ++g) {
                const int h = (n - 2) * c.ple_heads_per_ngram + g;
                rows[(size_t) i * heads + h] =
                    (int32_t) (mixed % (uint64_t) c.ple_head_vocab[h] + (uint64_t) c.ple_head_offsets[h]);
            }
        }
    }
}

void ple_gather(const Config & c, const Weights & w, const int32_t * rows, int T, float * emb)
{
    if (!w.ple_table) throw std::runtime_error("ple_gather: the model has no PLE table");
    const int hd = c.ple_head_dim, heads = c.ple_heads();
    const auto * table = reinterpret_cast<const int8_t *>(w.ple_table->data);
    const auto * scale = reinterpret_cast<const half *>(w.ple_scale->data);
    for (size_t r = 0; r < (size_t) T * heads; ++r) {
        const int64_t row = rows[r];
        const float s = __half2float(scale[row]);
        for (int i = 0; i < hd; ++i) emb[r * hd + i] = table[row * hd + i] * s;
    }
}

}  // namespace truss::qwen4exp
