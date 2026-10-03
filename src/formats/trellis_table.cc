#include "formats/trellis_table.h"

#include <stdexcept>

namespace truss::formats {

ExpertTable read_expert_table(const gguf::File & f, const std::string & prefix)
{
    auto fail = [&] (const std::string & why) { throw std::runtime_error("trellis table " + prefix + ": " + why); };
    ExpertTable t;
    t.trellis = &f.at(prefix + ".m3_trellis");
    t.meta = &f.at(prefix + ".m3_meta");
    t.suh = &f.at(prefix + ".m3_suh");
    t.svh = &f.at(prefix + ".m3_svh");
    using gguf::Type;
    if (t.trellis->type != Type::I16 || t.meta->type != Type::I32 || t.suh->type != Type::F16 ||
        t.svh->type != Type::F16)
        fail("unexpected tensor types");
    if (t.meta->shape.size() != 2 || t.meta->shape[0] != 2) fail("meta is not [n_expert][2]");
    t.n_expert = (int) t.meta->shape[1];
    if (t.suh->shape.size() != 2 || t.svh->shape.size() != 2 || t.suh->shape[1] != t.n_expert ||
        t.svh->shape[1] != t.n_expert)
        fail("suh/svh shapes disagree with meta");
    t.in = t.suh->shape[0];
    t.out = t.svh->shape[0];
    if (t.in % 16 || t.out % 16) fail("in/out not multiples of 16");

    const auto * m = reinterpret_cast<const int32_t *>(t.meta->data);
    t.k.resize(t.n_expert);
    t.offset.resize(t.n_expert);
    int64_t next = 0;
    for (int e = 0; e < t.n_expert; ++e) {
        t.k[e] = m[2 * e];
        t.offset[e] = m[2 * e + 1];
        if (!k_valid(t.k[e])) fail("expert " + std::to_string(e) + " K=" + std::to_string(t.k[e]));
        if (t.offset[e] != next)
            fail("expert " + std::to_string(e) + " at word " + std::to_string(t.offset[e]) + ", expected " +
                 std::to_string(next));
        next += t.words(e);
    }
    if (next != t.trellis->elements())
        fail("experts hold " + std::to_string(next) + " words, tensor has " + std::to_string(t.trellis->elements()));
    return t;
}

}  // namespace truss::formats
