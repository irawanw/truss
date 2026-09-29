#include "core/device_tensors.h"

#include "core/cuda_check.h"

#include <stdexcept>
#include <unordered_set>

namespace truss {

namespace {
constexpr size_t ALIGN = 256;
size_t align_up(size_t n) { return (n + ALIGN - 1) / ALIGN * ALIGN; }
}  // namespace

DeviceTensors::DeviceTensors(const std::vector<const gguf::Tensor *> & tensors)
{
    std::vector<const gguf::Tensor *> unique;
    std::unordered_set<const gguf::Tensor *> seen;
    for (const auto * t : tensors)
        if (t && seen.insert(t).second) {
            unique.push_back(t);
            bytes_ += align_up(t->bytes);
        }
    TRUSS_CUDA(cudaMalloc(&base_, bytes_ ? bytes_ : ALIGN));
    auto * p = static_cast<char *>(base_);
    for (const auto * t : unique) {
        TRUSS_CUDA(cudaMemcpy(p, t->data, t->bytes, cudaMemcpyHostToDevice));
        DTensor d;
        d.data = p;
        d.type = t->type;
        for (size_t i = 0; i < t->shape.size(); ++i) d.ne[i] = t->shape[i];
        map_.emplace(t, d);
        p += align_up(t->bytes);
    }
}

DeviceTensors::~DeviceTensors() { cudaFree(base_); }

const DTensor & DeviceTensors::operator()(const gguf::Tensor * t) const
{
    const auto it = map_.find(t);
    if (it == map_.end()) throw std::runtime_error("DeviceTensors: " + (t ? t->name : std::string("null")) +
                                                   " was not uploaded");
    return it->second;
}

}  // namespace truss
