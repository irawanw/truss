// Small engine helpers every engine and block file uses: error checks with context, bounded waits, the fault
// injector for the robustness tests, device allocation and a scoped current device.
//
// They were file-local helpers of the old qwen4exp::Forward; they live here so that the engine, the blocks and every
// model family share one definition (docs/ENGINE.md).
#pragma once
#include "core/cuda_check.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace truss::engine {

// throw std::runtime_error("truss: " + what) unless ok
inline void require(bool ok, const std::string & what)
{
    if (!ok) throw std::runtime_error("truss: " + what);
}

// TRUSS_LOAD_TIMES=1: seconds since the library was loaded, at each startup step (stderr)
void load_mark(const char * what);

// TRUSS_SPIN_TIMEOUT_S (default 60): how long a hand-off wait may go without progress before it is an error. Every
// wait in the hand-off chain runs with a stage lock held, so one that can never be granted has to fail rather than
// hang (review H1: a stage held forever wedges every slot, and the driver's global lock with it).
double spin_timeout_s();

// TRUSS_ROBUST=0 switches the error-path recovery off: the negative control for scripts/r_fault_test.py, which must
// then fail the way review C1/C2 describe. The fault injector and the bounded spins stay on.
bool robust_on();

// TRUSS_FAULT=<site>:<n>[@<slot>][,...]: throw on the n-th pass of a named point (scripts/r_fault_test.py). A site
// counts the passes of one slot (@slot: only that slot; without it, any); n = + (or 1) fires the first time it is
// reached after the spec was set. Read on every call: the test arms and clears it between phases.
void inject_fault(const char * site, int slot);

template <class X> X * dmalloc(size_t n, bool zero = true)
{
    X * p;
    TRUSS_CUDA(cudaMalloc(&p, n * sizeof(X)));
    if (zero) TRUSS_CUDA(cudaMemset(p, 0, n * sizeof(X)));
    return p;
}

// the device current for one scope (a split's stages live on different devices); dev < 0: leave it
struct DeviceGuard {
    int prev = -1;
    explicit DeviceGuard(int dev)
    {
        TRUSS_CUDA(cudaGetDevice(&prev));
        if (dev >= 0 && dev != prev) TRUSS_CUDA(cudaSetDevice(dev));
        else prev = -1;
    }
    ~DeviceGuard()
    {
        if (prev >= 0) cudaSetDevice(prev);
    }
    DeviceGuard(const DeviceGuard &) = delete;
    DeviceGuard & operator=(const DeviceGuard &) = delete;
};

}  // namespace truss::engine
