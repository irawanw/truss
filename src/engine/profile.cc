#include "engine/profile.h"

#include "core/cuda_check.h"
#include "engine/util.h"

#include <cstdlib>

namespace truss::engine {

Profile::Profile(int n_layers, int device) : device_(device)
{
    if (!std::getenv("TRUSS_PROFILE_SECTIONS") && !std::getenv("TRUSS_SECT_CSV")) return;
    on_ = true;
    ev_.resize(n_layers);
    rec_.assign(n_layers, 0);
    bell_.assign(n_layers, 0);
    host_.resize(n_layers);
    for (auto & a : ev_)
        for (cudaEvent_t & e : a) TRUSS_CUDA(cudaEventCreate(&e));
    if (const char * p = std::getenv("TRUSS_SECT_CSV")) {   // one line per layer per pass (the pass breakdown)
        char path[1024];
        // One FILE per stage: two stages appending to one path interleave their half-written lines.
        std::snprintf(path, sizeof path, "%s.d%d", p, device);
        csv_ = std::fopen(path, "ab");
        if (csv_)
            std::fprintf(csv_, "# pass rows layer route_publish wait_plan wait_copies window cpu_join helper_join "
                               "tail ple_attn_mix mixer hc_mix moe_total shared combine host_split host_pool_start "
                               "host_plan host_hint_admit host_publish pool_span pool_wake pool_compute "
                               "pool_longest ncpu\n");
    }
}

Profile::~Profile()
{
    for (auto & a : ev_)
        for (cudaEvent_t e : a)
            if (e) cudaEventDestroy(e);
    if (csv_) std::fclose(csv_);
}

void Profile::record(int layer, int i, cudaStream_t s)
{
    if (!on_) return;
    TRUSS_CUDA(cudaEventRecord(ev_[layer][i], s));
    rec_[layer] = 1;
}

void Profile::flush(int T, int /*caller_cpu*/)
{
    if (!on_) return;
    ++pass_;
    // The events were recorded on this stage's device; the thread reading them runs both stages, so its current
    // device is whichever one ran last - reading another device's event is "invalid resource handle".
    DeviceGuard g(device_);
    for (size_t l = 0; l < ev_.size(); ++l) {
        if (!rec_[l]) continue;
        TRUSS_CUDA(cudaEventSynchronize(ev_[l][6]));   // the last one recorded for the layer, stream order
        float sec[6];
        for (int i = 1; i <= 6; ++i) {
            float ms = 0.f;
            TRUSS_CUDA(cudaEventElapsedTime(&ms, ev_[l][i - 1], ev_[l][i]));
            ms_[i - 1] += sec[i - 1] = ms;
        }
        if (bell_[l]) {
            const int at[5] = { 3, 7, 9, 8, 4 };
            for (int i = 0; i < 4; ++i) {
                float ms = 0.f;
                TRUSS_CUDA(cudaEventElapsedTime(&ms, ev_[l][at[i]], ev_[l][at[i + 1]]));
                moe_[i] += ms;
            }
            if (csv_) {   // the doorbell FFN to its last boundary, plus the whole layer chain around it
                const int b[8] = { 3, 7, 9, 8, 10, 11, 12, 6 };
                std::fprintf(csv_, "%ld %d %zu", pass_, T, l);
                for (int i = 0; i < 7; ++i) {
                    float ms = 0.f;
                    TRUSS_CUDA(cudaEventElapsedTime(&ms, ev_[l][b[i]], ev_[l][b[i + 1]]));
                    std::fprintf(csv_, " %.3f", ms);
                }
                for (int i = 0; i < 6; ++i) std::fprintf(csv_, " %.3f", sec[i]);
                // the GPU waits for the plan and the copies, so the driver had served this seq when its last event
                // completed: no race with the writer
                const Host & h = host_[l];
                for (int i = 0; i < 5; ++i) std::fprintf(csv_, " %.1f", h.t[i]);
                std::fprintf(csv_, " %.1f %.1f %.1f %.1f %d\n", h.pool, h.wake, h.item, h.longest, h.ncpu);
            }
        }
        rec_[l] = 0, bell_[l] = 0;
    }
}

void Profile::ms(double out[6], bool reset)
{
    for (int i = 0; i < 6; ++i) out[i] = ms_[i];
    if (reset) for (double & v : ms_) v = 0;
}

void Profile::moe_ms(double out[4], bool reset)
{
    for (int i = 0; i < 4; ++i) out[i] = moe_[i];
    if (reset) for (double & v : moe_) v = 0;
}

}  // namespace truss::engine
