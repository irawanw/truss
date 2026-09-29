// CUDA error check for host code: throws with the call site, so tools and tests fail loudly with a location.
#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define TRUSS_CUDA(call)                                                                                            \
    do {                                                                                                            \
        const cudaError_t truss_err_ = (call);                                                                      \
        if (truss_err_ != cudaSuccess)                                                                              \
            throw std::runtime_error(std::string("CUDA ") + cudaGetErrorString(truss_err_) + " at " + __FILE__ +   \
                                     ":" + std::to_string(__LINE__) + ": " #call);                                  \
    } while (0)
