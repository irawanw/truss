// Every dense projection of a stage, whatever the weight's format. Blocks call lin*() with a tensor; the tensor's
// type picks the kernel, so neither a block nor a model family ever names a weight format.
//
// Formats today: Q8_0 (repacked at upload into dense::Q8Matrix; dense::q8_gemv / gemm, int8 or fp16 activations)
// and F32 (router-like projections, cuBLAS SGEMM, or a row-invariant gemv at decode sizes so routing does not depend
// on the window size). Trellis-coded dense matrices (m3_trellis / m3_suh / m3_svh, PAW-27B-X3.1) are the next format:
// add them as a case in upload() and lin(), nothing else changes.
//
// Activations::Q8_1 quantizes each fp16 input once per call site (Act) and every projection reading it shares that
// (TRACKER #60); FP16 runs the a16 kernels. Moved unchanged from the old qwen4exp::Forward (bit-identical).
#pragma once
#include "blocks/weights.h"
#include "core/device_tensors.h"
#include "engine/gpu.h"
#include "formats/gguf.h"
#include "kernels/dense/q8_gemm.cuh"

#include <cuda_fp16.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace truss::blocks {


// Activations into the dense Q8_0 GEMMs. Q8_1: llama-paw's rounding on the int8 GEMM (full-model KL to llama-paw's
// Q8 logits = its own run-to-run floor, TRACKER #53). FP16: closer to exact math per matmul (TRACKER #33), the
// served default (KL 0.040 vs 0.065).
enum class Activations { FP16, Q8_1 };

class Linear {
public:
    Linear(engine::Gpu & g, Activations act) : g_(g), act_(act) {}
    ~Linear();

    // Upload: add() every tensor this stage reads (nullptr ignored, duplicates fine), then upload() once.
    // host_rows: a Q8_0 table read one row per token (an untied token embedding) kept in pinned host memory and
    // gathered over PCIe, leaving its VRAM to the expert cache (TRACKER #60); nullptr: none. no_w16: matrices the
    // a16 GEMM scratch is not sized for (the embedding and the output head, which tiles itself to fit it).
    void add(T t);
    void upload(T host_rows, const std::vector<T> & no_w16);

    // An fp16 GEMM input and, with Q8_1 activations, its quantized form, in the caller's scratch scope
    struct Act {
        const half * x;
        int8_t * q = nullptr;
        half * d = nullptr;
    };
    Act quant(const half * x, int rows, int in);

    void lin(T t, const half * x, int rows, float * y);    // y [rows][out] = W x
    void lin(T t, const Act & a, int rows, float * y);
    void lin32(T t, const float * x, int rows, float * y);   // fp32 weights and activations
    // several projections of the same input: one launch at decode rows (dense::q8_gemv_multi, TRACKER #83), each
    // result bit-identical to its own lin() at those rows
    using Proj = std::pair<T, float *>;
    void lin_multi(const std::vector<Proj> & ps, const Act & a, int rows);
    // lin_multi with the first matrix's output written as silu(row * scale) in fp16 by the gemv itself (TRUSS_HC_SILU)
    void lin_multi_silu(const std::vector<Proj> & ps, half * y16, float scale, const Act & a, int rows);
    void lin32_multi(const std::vector<Proj> & ps, const float * x, int rows);

    const DTensor & d(T t) const { return (*dev_)(t); }
    const float * f32(T t) const { return d(t).as<float>(); }
    const dense::Q8Matrix & q8(T t) const { return q8_.at(t); }
    bool has_q8(T t) const { return q8_.count(t) != 0; }
    Activations act() const { return act_; }
    half * w16() const { return w16_; }        // q8_gemm_a16's dequantized-weight scratch
    size_t w16_elems() const { return w16_elems_; }

private:
    engine::Gpu & g_;
    const Activations act_;
    std::vector<T> plain_, q8_list_;
    std::unique_ptr<DeviceTensors> dev_;                 // every non-Q8_0 tensor, as stored
    std::unordered_map<T, dense::Q8Matrix> q8_;          // Q8_0 matrices, repacked
    half * w16_ = nullptr;
    size_t w16_elems_ = 0;
    int8_t * host_q_ = nullptr;                          // pinned: the host-resident table (upload's host_rows)
    half * host_d_ = nullptr;
};

}  // namespace truss::blocks
