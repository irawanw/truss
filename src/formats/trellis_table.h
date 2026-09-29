// Per-expert rate and word offset of a routed trellis projection stored as PAW X3 tensors
// "<prefix>.m3_{trellis,meta,suh,svh}" (llama-paw paw-x3): the experts' tiles are concatenated in one I16 tensor,
// meta [n_expert][2] = (K, uint16 word offset), suh [n_expert][in], svh [n_expert][out].
#pragma once
#include "formats/gguf.h"

#include <string>
#include <vector>

namespace truss::formats {

struct ExpertTable {
    int n_expert = 0;
    int64_t in = 0, out = 0;               // weights: out x in per expert
    std::vector<int> k;                    // bits per weight, integer 1..8
    std::vector<int64_t> offset;           // uint16 words from the start of the trellis tensor
    const gguf::Tensor *trellis = nullptr, *meta = nullptr, *suh = nullptr, *svh = nullptr;

    int64_t words(int e) const { return in * out * k[e] / 16; }   // 16*K uint16 per 16x16 tile
};

// Reads and checks the table: types, shapes, offsets contiguous and in order, total = the trellis tensor.
// Throws std::runtime_error naming the projection and the first inconsistency.
ExpertTable read_expert_table(const gguf::File & f, const std::string & prefix);

}  // namespace truss::formats
