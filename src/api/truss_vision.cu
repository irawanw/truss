// C API for the vision tower (include/truss/truss.h): open the mmproj on one device, run it on host pixel_values and
// hand back host f32 rows for truss_eval_img_s. Host round trips: 2 MB for a 448x448 image, and the caller keeps
// control of which device the ViT used (the front stage may live on another one).
#include "api/api_internal.h"

#include "model/qwen4exp/vision/vit.h"

#include <memory>

struct truss_vision {
    std::unique_ptr<truss::gguf::File> file;
    std::unique_ptr<truss::qwen4exp::Vit> vit;
    int device = 0;
};

extern "C" {

truss_vision * truss_vision_open(const char * mmproj_path, int device)
{
    return truss::api::guarded(
        [&]() -> truss_vision * {
            if (!mmproj_path) throw std::invalid_argument("truss_vision_open: no path");
            auto v = std::make_unique<truss_vision>();
            v->device = device;
            TRUSS_CUDA(cudaSetDevice(device));
            v->file = truss::gguf::File::open(mmproj_path);
            v->vit = std::make_unique<truss::qwen4exp::Vit>(*v->file, device);
            return v.release();
        },
        nullptr);
}

void truss_vision_close(truss_vision * v) { delete v; }

int truss_vision_width(const truss_vision * v)
{
    return v && v->vit ? v->vit->out_hidden() : -1;
}

int truss_vision_forward(truss_vision * v, const float * pixel_values, int gh, int gw, float * out_rows)
{
    return truss::api::guarded(
        [&] {
            if (!v || !pixel_values || !out_rows) throw std::invalid_argument("truss_vision_forward: null argument");
            TRUSS_CUDA(cudaSetDevice(v->device));
            const float * rows = v->vit->forward(pixel_values, gh, gw);
            const int n = gh * gw / 4, w = v->vit->out_hidden();
            TRUSS_CUDA(cudaMemcpy(out_rows, rows, sizeof(float) * (size_t) n * w, cudaMemcpyDeviceToHost));
            return n;
        },
        -1);
}

}  // extern "C"
