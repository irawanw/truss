// general.architecture -> model family: the one place that lists every family TRUSS serves (docs/ENGINE.md).
// Adding a family: its folder under src/model/ and one line here.
#include "engine/model.h"
#include "model/deepseek41/model.h"
#include "model/lfm2moe/model.h"
#include "model/qwen4exp/model.h"

#include <stdexcept>

namespace truss::engine {

std::unique_ptr<Family> open_family(const gguf::File & f, const DrafterFiles & d)
{
    const std::string arch = f.get_string("general.architecture");
    if (arch == "qwen4exp") return std::make_unique<qwen4exp::Family>(f, d);
    if (arch == "lfm2moe") return std::make_unique<lfm2moe::Family>(f, d);
    if (arch == "deepseek41") return std::make_unique<deepseek41::Family>(f, d);
    throw std::runtime_error("unsupported architecture " + arch);
}

}  // namespace truss::engine
