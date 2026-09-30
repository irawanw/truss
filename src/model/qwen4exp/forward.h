// qwen4exp prompt prefill on one GPU: the fast ops chained layer by layer over chunks of a sequence.
//
// Per chunk (positions pos .. pos + T - 1): token embedding -> per layer [PLE] -> hc mix -> GDN or DSA mixer -> hc
// combine -> hc mix -> routed experts (trellis, moe::prefill) + gated shared expert -> hc combine. Dense projections
// run on Q8_0 weights with fp16 activations (dense::q8_gemm_a16), the router and shared-expert gate in fp32
// (cuBLAS SGEMM, so routing keeps full precision). State carried across chunks: DSA K/V and indexer-block caches,
// GDN recurrent state and conv rows, PLE conv history, the PLE n-gram window.
//
// Routed experts live in a runtime::ExpertStore: as many as fit stay resident, the rest stream over PCIe per layer,
// overlapped with the previous layer's compute. The math is qwen4exp::reference's; tests/layer/qwen4exp_forward
// checks it layer by layer.
#pragma once
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/weights.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace truss::qwen4exp {

// Activations into the dense Q8_0 GEMMs. Q8_1 (default): llama-paw's rounding on the int8 GEMM; full-model KL to
// llama-paw's Q8 logits 0.0131 = its own run-to-run floor, and the lower perplexity of the two on chat text (TRACKER
// #53). FP16: closer to exact math per matmul (TRACKER #33) but +3% perplexity on chat-format text vs Q8_1; kept until
// an unquantized anchor decides which is closer to the real model.
enum class Activations { FP16, Q8_1 };

class Forward {
public:
    // n_ctx: the longest sequence; max_chunk: the most tokens per run() call (a multiple of the DSA block ratio)
    // c and w must outlive the Forward (w references the file mapping the weights are uploaded from)
    // expert_budget: device bytes for routed experts (0: all memory left after everything else, minus a margin)
    Forward(const Config & c, const Weights & w, int n_ctx, int max_chunk, size_t expert_budget = 0,
            Activations act = Activations::Q8_1);
    ~Forward();
    Forward(const Forward &) = delete;
    Forward & operator=(const Forward &) = delete;

    // called after each layer with the device residual [T][hc][d_model] of the current chunk
    using LayerHook = std::function<void(int layer, const float * res, int T)>;

    // The next T tokens of the sequence (any T <= max_chunk: a prompt chunk, or one decoded token).
    void run(const int32_t * tokens, int T, const LayerHook & hook = nullptr);

    // logits [n][n_vocab] (device, fp32) of rows first .. first + n - 1 of the last chunk: the head hc mix, then
    // the output projection
    void head(int first, int n, float * logits);

    // start a new sequence (zero recurrent state, conv histories, the PLE window; caches are overwritten by position)
    void reset();

    int position() const { return pos_; }   // tokens consumed so far
    int hot_experts() const;                // resident experts per layer
    size_t cold_bytes() const;              // streamed per chunk
    cudaStream_t stream() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_;
    int pos_ = 0;
};

}  // namespace truss::qwen4exp
