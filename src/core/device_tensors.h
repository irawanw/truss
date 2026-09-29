// Model tensors on the device: one allocation holds every uploaded tensor (256-byte aligned), and a map from the
// host catalog entry (gguf::Tensor) to its device copy. Ops look their weights up here with the host binding
// (model/<family>/weights.h), so the device side needs no second copy of the model structure.
#pragma once
#include "formats/gguf.h"

#include <cuda_runtime.h>

#include <array>
#include <unordered_map>
#include <vector>

namespace truss {

struct DTensor {
    const void * data = nullptr;           // device
    gguf::Type type = gguf::Type::F32;
    std::array<int64_t, 4> ne = { 1, 1, 1, 1 };   // ggml order

    int64_t elements() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
    template <class T> const T * as() const { return static_cast<const T *>(data); }
};

class DeviceTensors {
public:
    // Uploads `tensors` (duplicates allowed, uploaded once) with synchronous copies from the file mapping.
    explicit DeviceTensors(const std::vector<const gguf::Tensor *> & tensors);
    ~DeviceTensors();
    DeviceTensors(const DeviceTensors &) = delete;
    DeviceTensors & operator=(const DeviceTensors &) = delete;

    const DTensor & operator()(const gguf::Tensor * t) const;   // throws if `t` was not uploaded
    size_t bytes() const { return bytes_; }

private:
    void * base_ = nullptr;
    size_t bytes_ = 0;
    std::unordered_map<const gguf::Tensor *, DTensor> map_;
};

}  // namespace truss
