// Layer-chain section profile of one stage (TRUSS_PROFILE_SECTIONS=1, TRUSS_SECT_CSV=<path>).
//
// Thirteen events per layer, reused every chunk: they complete in stream order, so elapsed(e[i-1], e[i]) is one
// section's device time including any host stall inside it. Sections 0..6 bracket the layer (the model's layer()
// records 0, 1, 2, 6; the MoE block 3, 4, 5); 7..12 split the doorbell MoE path (router + publish, wait for the
// host's plan, wait for the copies, expert kernel, CPU join, peer join; TRACKER #77). flush() reads the events back
// once per chunk (one sync) and sums them; with TRUSS_SECT_CSV one line per layer per pass with the driver's host
// timeline (Host, written by runtime::MoeExec's driver before it raises cpu_done).
#pragma once
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace truss::engine {

class Profile {
public:
    struct Host {   // one layer's host chain, us after the driver saw the doorbell
        double t[5] = {};   // split done, pool started, plan written, hint + admit done, pool wait returned
        double pool = 0, wake = 0, item = 0, longest = 0;
        int ncpu = 0;
    };

    Profile(int n_layers, int device);   // n_layers: model layers + 1 (a drafter's layer); off unless the env says
    ~Profile();
    bool on() const { return on_; }
    void record(int layer, int i, cudaStream_t s);
    void mark_bell(int layer) { if (on_) bell_[layer] = 1; }
    Host & host(int layer) { return host_[layer]; }
    void flush(int T, int caller_cpu);   // after a chunk
    void ms(double out[6], bool reset);      // sections 0..5 summed
    void moe_ms(double out[4], bool reset);  // doorbell MoE split summed

private:
    bool on_ = false;
    int device_ = -1;
    std::vector<std::array<cudaEvent_t, 13>> ev_;
    std::vector<uint8_t> rec_, bell_;
    std::vector<Host> host_;
    double ms_[6] = {}, moe_[4] = {};
    FILE * csv_ = nullptr;
    long pass_ = 0;
};

}  // namespace truss::engine
