// Flash-Next vision tower (Qwen3-VL style ViT, 27 blocks) running on one GPU from flashnext-mmproj-f16.gguf.
// See docs/reference/model-qwen4exp.md "Vision". Gated: nothing here runs unless a request carries an image.
//
//   Vit vit(*gguf::File::open("flashnext-mmproj-f16.gguf"));
//   const float * rows = vit.forward(pixel_values, gh, gw);   // [gh*gw/4][out_hidden] f32 on the device
//
// pixel_values: host f32 [gh*gw][3*2*16*16] as produced by server/vision_pre.py (merge-window-grouped patch order).
// gh, gw: patch grid (image size / 16), both even. Output rows are in text-token order (raster over the merged grid).
// Numerics: residual stream f32 (block outputs reach |x| ~ 1e4), GEMM inputs f16 with f32 accumulate, LayerNorm /
// softmax / rotary in f32. Attention is full (non-causal) over one image's tokens, up to MAX_TOKENS patches.
#pragma once
#include "formats/gguf.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <map>
#include <string>
#include <vector>

namespace truss::qwen4exp {

class Vit {
public:
    static constexpr int MAX_TOKENS = 8192;          // patches per image (1344 x 1344 -> 7056)
    explicit Vit(const gguf::File & f, int device = 0);
    ~Vit();
    Vit(const Vit &) = delete;
    Vit & operator=(const Vit &) = delete;

    // Returns a device pointer to [gh*gw/4][out_hidden()] f32, valid until the next forward(). If taps is not null it
    // receives host copies of "patch_embed" (after pos-embed add), "block0_out", "block13_out", "block26_out", "vit_out".
    const float * forward(const float * pixel_values, int gh, int gw, std::map<std::string, std::vector<float>> * taps = nullptr);
    int out_hidden() const { return out_; }
    int hidden() const { return d_; }

private:
    struct Block { const void *qkv_w, *proj_w, *fc1_w, *fc2_w; const float *qkv_b, *proj_b, *fc1_b, *fc2_b, *n1_w, *n1_b, *n2_w, *n2_b; };
    void * dev(const gguf::Tensor & t, bool f32);
    int dev_, depth_, d_, heads_, mlp_, out_, patch_, tpatch_, side_, merge_;
    cublasHandle_t blas_;
    std::vector<void *> owned_;
    std::vector<Block> blocks_;
    const void * patch_w_ = nullptr, * pos_ = nullptr, * m_fc1_w_ = nullptr, * m_fc2_w_ = nullptr;
    const float * patch_b_ = nullptr, * m_fc1_b_ = nullptr, * m_fc2_b_ = nullptr, * m_n_w_ = nullptr, * m_n_b_ = nullptr;
    // per-call buffers, grown on demand
    int cap_ = 0;
    float * x_ = nullptr, * y32_ = nullptr, * q_ = nullptr, * k_ = nullptr, * v_ = nullptr, * out_rows_ = nullptr, * cos_ = nullptr, * sin_ = nullptr, * posw_ = nullptr;
    int * posi_ = nullptr;
    void * a16_ = nullptr, * px16_ = nullptr;
    void grow(int n);
};

}  // namespace truss::qwen4exp
