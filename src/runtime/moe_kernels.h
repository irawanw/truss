// The routed-expert kernels of one expert shape, behind function pointers.
//
// moe::window / moe::prefill are templates on a shape struct (d_model, d_ff, top-k) so each shape gets its own tuned
// tiling (CODE.md: a new shape = a struct + a Tune<> + an explicit instantiation). The runtime and the MoE block run
// any shape through this table instead; a model family picks it with MoeKernels::of<Shape>() and nothing above the
// kernels names a shape.
#pragma once
#include "kernels/moe/moe_prefill.cuh"
#include "kernels/moe/moe_window.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace truss::runtime {

struct MoeKernels {
    int d_model = 0, d_ff = 0, topk = 0;
    size_t (*window_ws)() = nullptr;
    void (*window_init)(void * ws, cudaStream_t) = nullptr;
    void (*window)(const moe::Weights &, const float * x, const int * ids, const float * wts, int n_rows, float * out,
                   void * ws, cudaStream_t) = nullptr;
    size_t (*prefill_ws)(int max_tokens) = nullptr;
    void (*prefill)(const moe::Weights &, const float * x, const int * ids, const float * wts, int n_tokens,
                    float * out, void * ws, int max_tokens, cudaStream_t) = nullptr;

    template <class Shape> static MoeKernels of()
    {
        MoeKernels k;
        k.d_model = Shape::D_MODEL, k.d_ff = Shape::D_FF, k.topk = Shape::TOPK;
        k.window_ws = &moe::workspace_bytes<Shape>;
        k.window_init = &moe::workspace_init<Shape>;
        k.window = [](const moe::Weights & W, const float * x, const int * ids, const float * wts, int n, float * out,
                      void * ws, cudaStream_t s) { moe::window<Shape>(W, x, ids, wts, n, out, ws, s); };
        k.prefill_ws = &moe::prefill_workspace_bytes<Shape>;
        k.prefill = &moe::prefill<Shape>;
        return k;
    }
};

}  // namespace truss::runtime
