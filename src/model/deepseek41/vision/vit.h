// DeepSeek-V4.1 vision tower (ViT 32 blocks + 3x3 aligner) on one GPU from ds41-mmproj-f16.gguf
// (tools/tk-pack/ds41_mmproj.py). See docs/reference/model-deepseek41.md "Vision". Gated: nothing here runs unless a
// request carries an image.
//
//   Vit vit(*gguf::File::open("ds41-mmproj-f16.gguf"), device);
//   const float * span = vit.forward(pixel_values, n_vit_h, n_vit_w);   // [span_rows(h, w)][out_hidden()] f32, device
//
// pixel_values: host f32 [n_vit_h * n_vit_w][3*14*14], patches row-major over the patch grid, each patch (c, ph, pw),
// values (x/255 - 0.5)/0.5 as DeepSeek's image_processor.load_image makes them (server/vision_ds41.py).
// Output: the whole image span of the prompt, START + (n_llm_w aligner rows + NEWLINE) x n_llm_h + END, where
// n_llm_* = ceil(n_vit_* / 3); the learned START/NEWLINE/END vectors come from the gguf (span.*).
// Numerics: residual stream f32, GEMM inputs f16 with f32 accumulate, RMSNorm / softmax / rotary / SwiGLU in f32.
// Attention: full (non-causal) over the image's patches, as cuBLAS batched GEMMs over query chunks (scores f32,
// probabilities f16), ~100 MB of score workspace at any size instead of [heads][N][N].
// VRAM: weights 0.97 GB + workspace ~0.5 GB at the largest image (9,216 patches; a span over max_image_tokens = 1024
// is refused). reserve = allocate that workspace at open, so a model loaded afterwards on the same card sizes its
// expert budget around it (the server opens the tower first).
#pragma once
#include "formats/gguf.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <map>
#include <string>
#include <vector>

namespace truss::deepseek41 {

class Vit {
public:
    explicit Vit(const gguf::File & f, int device = 0, bool reserve = false);
    ~Vit();
    Vit(const Vit &) = delete;
    Vit & operator=(const Vit &) = delete;

    // Device pointer to [span_rows(h, w)][out_hidden()] f32, valid until the next forward() / release(). If taps is not
    // null it receives host copies of "patch_embed", "block0_out", "block15_out", "block31_out", "vit_out" (after the
    // final RMSNorm) and "rows" (aligner output, reading order).
    const float * forward(const float * pixel_values, int n_vit_h, int n_vit_w,
                          std::map<std::string, std::vector<float>> * taps = nullptr);
    int span_rows(int n_vit_h, int n_vit_w) const;   // n_llm_h * (n_llm_w + 1) + 2
    int out_hidden() const { return out_; }
    int max_patches() const { return max_patches_; }
    int max_tokens() const { return max_tokens_; }
    void release();                                  // frees the per-call workspace (weights stay)

private:
    struct Block { const void *qkv_w, *wo_w, *w1_w, *w2_w; const float *qkv_b, *wo_b, *n1_w, *n2_w; };
    void * dev(const gguf::Tensor & t, bool f32);
    void grow(int n);
    int dev_, depth_, d_, heads_, inter_, out_, patch_, ratio_, max_tokens_, max_patches_;
    float theta_;
    cublasHandle_t blas_;
    std::vector<void *> owned_;
    std::vector<Block> blocks_;
    const void * patch_w_ = nullptr, * a1_w_ = nullptr, * a2_w_ = nullptr;
    const float * patch_b_ = nullptr, * norm_w_ = nullptr, * a1_b_ = nullptr, * a2_b_ = nullptr;
    const float * span_start_ = nullptr, * span_nl_ = nullptr, * span_end_ = nullptr;
    // per-call workspace, grown on demand (cap_ = patches it fits)
    int cap_ = 0, qchunk_ = 0;
    float * x_ = nullptr, * y32_ = nullptr, * s32_ = nullptr, * rows_ = nullptr, * span_ = nullptr, * cos_ = nullptr, * sin_ = nullptr, * px_ = nullptr;
    void * q16_ = nullptr, * k16_ = nullptr, * v16_ = nullptr, * a16_ = nullptr, * p16_ = nullptr, * px16_ = nullptr;
};

}  // namespace truss::deepseek41
