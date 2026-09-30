// qwen4exp prompt prefill on one GPU: the fast ops chained layer by layer over chunks of a sequence.
//
// Per chunk (positions pos .. pos + T - 1): token embedding -> per layer [PLE] -> hc mix -> GDN or DSA mixer -> hc
// combine -> hc mix -> routed experts (trellis, moe::prefill) + gated shared expert -> hc combine. Dense projections
// run on Q8_0 weights with fp16 activations (dense::q8_gemm_a16), the router and shared-expert gate in fp32
// (cuBLAS SGEMM, so routing keeps full precision). State carried across chunks: DSA K/V and indexer-block caches,
// GDN recurrent state and conv rows, PLE conv history, the PLE n-gram window.
//
// v1 keeps every weight on the device (the 8-layer parity slice fits; the full model needs the cold-expert
// stream, next). The math is qwen4exp::reference's; tests/layer/qwen4exp_prefill checks it layer by layer.
#pragma once
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/weights.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace truss::qwen4exp {

class Prefill {
public:
    // n_ctx: the longest sequence; max_chunk: the most tokens per run() call (a multiple of the DSA block ratio)
    // c and w must outlive the Prefill (w references the file mapping the weights are uploaded from)
    Prefill(const Config & c, const Weights & w, int n_ctx, int max_chunk);
    ~Prefill();
    Prefill(const Prefill &) = delete;
    Prefill & operator=(const Prefill &) = delete;

    // called after each layer with the device residual [T][hc][d_model] of the current chunk
    using LayerHook = std::function<void(int layer, const float * res, int T)>;

    // The next T tokens of the sequence. T <= max_chunk; T % DSA ratio == 0 except on the last chunk.
    void run(const int32_t * tokens, int T, const LayerHook & hook = nullptr);

    int position() const { return pos_; }   // tokens consumed so far
    cudaStream_t stream() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_;
    int pos_ = 0;
};

}  // namespace truss::qwen4exp
