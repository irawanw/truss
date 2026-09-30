// cuBLAS status check for host code (the cuBLAS twin of cuda_check.h): throws with the call.
#pragma once
#include <cublas_v2.h>

#include <stdexcept>
#include <string>

#define TRUSS_CUBLAS(call)                                                                                          \
    do {                                                                                                            \
        const cublasStatus_t truss_st_ = (call);                                                                    \
        if (truss_st_ != CUBLAS_STATUS_SUCCESS)                                                                     \
            throw std::runtime_error("cuBLAS error " + std::to_string((int) truss_st_) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + ": " #call);                                        \
    } while (0)
