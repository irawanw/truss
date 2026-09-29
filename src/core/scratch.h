// Bump allocator over one device buffer for temporaries inside an op sequence. reset() frees everything at once;
// nothing is freed individually. Exhaustion throws (sizes are known up front; no silent growth).
#pragma once
#include "core/cuda_check.h"

#include <cstddef>
#include <stdexcept>

namespace truss {

class Scratch {
public:
    explicit Scratch(size_t bytes) : cap_(bytes) { TRUSS_CUDA(cudaMalloc(&base_, bytes)); }
    ~Scratch() { cudaFree(base_); }
    Scratch(const Scratch &) = delete;
    Scratch & operator=(const Scratch &) = delete;

    template <class T = float> T * alloc(size_t count)
    {
        const size_t bytes = (count * sizeof(T) + 255) / 256 * 256;
        if (used_ + bytes > cap_) throw std::runtime_error("Scratch: out of space");
        T * p = reinterpret_cast<T *>(static_cast<char *>(base_) + used_);
        used_ += bytes;
        return p;
    }
    void reset() { used_ = 0; }
    // scoped reuse inside a loop: everything allocated after mark() is released by release(mark). Safe on one
    // stream: later kernels that reuse the space run after the earlier ones.
    size_t mark() const { return used_; }
    void release(size_t m) { used_ = m; }

private:
    void * base_ = nullptr;
    size_t cap_, used_ = 0;
};

}  // namespace truss
