// C API for the vision tower (include/truss/truss.h): open the mmproj on one device, run it on host pixel_values and
// hand back host f32 rows for truss_eval_img_s. Host round trips: 2 MB for a 448x448 image, and the caller keeps
// control of which device the ViT used (the front stage may live on another one). The mmproj's general.architecture
// picks the tower: qwen4exp-mmproj (Flash-Next) or deepseek41-mmproj (DS41: the rows are the whole image span).
#include "api/api_internal.h"

#include "model/deepseek41/vision/vit.h"
#include "model/qwen4exp/vision/vit.h"

#include <memory>

namespace {

// The ViT's device must not leak into the caller: the server calls this from a thread pool whose thread may run the
// text model's next call, and that model's streams (and its gpu.h guard, "the device current at construction") live
// on another device. A leaked cudaSetDevice(1) then fails the model with "invalid resource handle".
struct DevRestore {
    int prev = -1;
    DevRestore() { cudaGetDevice(&prev); }
    ~DevRestore() { if (prev >= 0) cudaSetDevice(prev); }
};

}  // namespace

struct truss_vision {
    std::unique_ptr<truss::gguf::File> file;
    std::unique_ptr<truss::qwen4exp::Vit> vit;
    std::unique_ptr<truss::deepseek41::Vit> ds;   // deepseek41-mmproj: workspace reserved at open (vit.h "VRAM")
    int device = 0;
};

extern "C" {

truss_vision * truss_vision_open(const char * mmproj_path, int device)
{
    return truss::api::guarded(
        [&]() -> truss_vision * {
            if (!mmproj_path) throw std::invalid_argument("truss_vision_open: no path");
            auto v = std::make_unique<truss_vision>();
            const DevRestore restore;
            v->device = device;
            TRUSS_CUDA(cudaSetDevice(device));
            v->file = truss::gguf::File::open(mmproj_path);
            if (v->file->get_string("general.architecture") == "deepseek41-mmproj")
                v->ds = std::make_unique<truss::deepseek41::Vit>(*v->file, device, true);
            else
                v->vit = std::make_unique<truss::qwen4exp::Vit>(*v->file, device);
            return v.release();
        },
        nullptr);
}

void truss_vision_close(truss_vision * v) { delete v; }

int truss_vision_width(const truss_vision * v)
{
    return !v ? -1 : v->ds ? v->ds->out_hidden() : v->vit ? v->vit->out_hidden() : -1;
}

int truss_vision_rows(const truss_vision * v, int gh, int gw)
{
    return !v || gh <= 0 || gw <= 0 ? -1 : v->ds ? v->ds->span_rows(gh, gw) : gh * gw / 4;
}

int truss_vision_forward(truss_vision * v, const float * pixel_values, int gh, int gw, float * out_rows)
{
    return truss::api::guarded(
        [&] {
            if (!v || !pixel_values || !out_rows) throw std::invalid_argument("truss_vision_forward: null argument");
            const DevRestore restore;
            TRUSS_CUDA(cudaSetDevice(v->device));
            const float * rows = v->ds ? v->ds->forward(pixel_values, gh, gw) : v->vit->forward(pixel_values, gh, gw);
            const int n = truss_vision_rows(v, gh, gw), w = truss_vision_width(v);
            TRUSS_CUDA(cudaMemcpy(out_rows, rows, sizeof(float) * (size_t) n * w, cudaMemcpyDeviceToHost));
            return n;
        },
        -1);
}

}  // extern "C"
