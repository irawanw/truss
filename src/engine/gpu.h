// One stage's device context: its device, stream, cuBLAS handle, current scratch and owned allocations. Every block
// of the stage enqueues on this stream and allocates its temporaries from `sc` (reset per layer by the stage), so a
// chunk is one ordered sequence of launches that a CUDA graph could capture.
#pragma once
#include "core/cublas_check.h"
#include "core/cuda_check.h"
#include "core/scratch.h"
#include "engine/util.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace truss::engine {

struct Gpu {
    int device = -1;                 // -1: the device current at construction
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    Scratch * sc = nullptr;          // the scratch of the buffers in use (small for decode, big for prompt chunks)
    std::vector<void *> owned;       // cudaFree at destruction

    explicit Gpu(int dev) : device(dev)
    {
        if (device >= 0) TRUSS_CUDA(cudaSetDevice(device));   // the caller's guard restores its device
        TRUSS_CUDA(cudaStreamCreate(&s));
        TRUSS_CUBLAS(cublasCreate(&blas));
    }
    ~Gpu()
    {
        if (device >= 0) cudaSetDevice(device);
        for (void * p : owned) cudaFree(p);
        if (blas) cublasDestroy(blas);
        if (s) cudaStreamDestroy(s);
    }
    Gpu(const Gpu &) = delete;
    Gpu & operator=(const Gpu &) = delete;

    template <class X> X * alloc(size_t n)   // zeroed, freed with the stage
    {
        X * p = dmalloc<X>(n);
        owned.push_back(p);
        return p;
    }
    void free(void * p)                      // an alloc()ed buffer given back early
    {
        for (size_t i = 0; i < owned.size(); ++i)
            if (owned[i] == p) {
                owned.erase(owned.begin() + (long) i);
                break;
            }
        TRUSS_CUDA(cudaFree(p));
    }
    void copy(float * dst, const float * src, size_t n)
    {
        TRUSS_CUDA(cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, s));
    }
    void sync() { TRUSS_CUDA(cudaStreamSynchronize(s)); }
    static size_t align(size_t x) { return (x + 255) / 256 * 256; }
};

}  // namespace truss::engine
