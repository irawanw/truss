// CPU tier for routed experts: 4-bit ("q4s") copies of experts computed on the host, in parallel with the GPU.
//
// Why: decode misses (experts neither resident nor cached in the ring) cost PCIe bytes (x8, 13.5 GB/s); on this box
// the CPU reads RAM at ~50 GB/s even beside the renters (tools/tk-bench/cpu_q4.cc, TRACKER #59), so computing the
// rarest experts here adds miss bandwidth. The q4s copies come from the BF16 originals (flashnext_truss_cpu_q4.py):
// weight NMSE ~0.009, about the pack's K4 trellis and far below the K1/K2 the pack gives rare experts.
//
// Format q4s, per matrix [rows][cols] (out x in), blocks of 32 along the input: 16 bytes of nibbles (element i of the
// block in the low nibble of byte i for i < 16, the high nibble of byte i - 16 otherwise; value (nibble - 8) * d) and
// one fp16 scale d per block. An expert = gate q, gate d, up q, up d (640 x 2560), down q, down d (2560 x 640).
//
// Numerics: activations are quantized per 32-block to int8 (d = amax / 127, as the GPU's Q8_1 path), dot products
// exact in int32, scaled in fp32; the hidden h = silu(gate) * up is quantized the same way before down. Every row is
// computed independently and a row's experts are summed in slot order, so results do not depend on how many rows a
// call has (a verify window equals single steps).
#pragma once
#include "cpu/expert_trellis.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace truss::cpu {

struct Q4Matrix {
    const uint8_t * q;      // [rows][cols / 2]
    const uint16_t * d;     // [rows][cols / 32] fp16 bits
    const float * df = nullptr;   // optional preconverted fp32 scales (same count; gemv uses these when set)
    int rows, cols;
};

struct Q4Expert {
    Q4Matrix gate, up, down;
};

constexpr int D_MODEL = 2560, D_FF = 640, BLOCK = 32, MAX_ROWS = 8;
constexpr size_t EXPERT_BYTES = 3 * ((size_t) D_MODEL * D_FF / 2 + (size_t) D_MODEL * D_FF / BLOCK * 2);

// the matrices of one expert stored contiguously in q4s order at p
Q4Expert expert_view(const uint8_t * p);

constexpr size_t SCALES_PER_EXPERT =
    (D_FF * (D_MODEL / BLOCK) + D_FF * (D_MODEL / BLOCK) + D_MODEL * (D_FF / BLOCK));
// gate, up, down scales as fp32 into out [SCALES_PER_EXPERT] (one _cvtsh_ss per block, once at load)
void preconvert_scales(const Q4Expert & e, float * out);

// One layer's CPU work: rows x [T][D_MODEL] (fp32) and the slots routed to CPU experts. An expert is either a q4s
// copy (e) or the pack's own trellis bytes (t, expert_trellis.h); one call uses one kind.
struct Slot {
    int row;                // 0 .. T - 1
    const Q4Expert * e;
    float w;                // routing weight
    const TrellisExpert * t = nullptr;
};

class ExpertPool {
public:
    explicit ExpertPool(int threads);
    ~ExpertPool();
    ExpertPool(const ExpertPool &) = delete;
    ExpertPool & operator=(const ExpertPool &) = delete;

    // y [T][D_MODEL] = sum over slots of w * expert(x[row]) (rows without slots: 0). start() returns at once; wait()
    // blocks until y is written. x, slots and y must stay valid until wait() returns. T <= MAX_ROWS.
    void start(const float * x, int T, const std::vector<Slot> & slots, float * y);
    void wait();
    void run(const float * x, int T, const std::vector<Slot> & slots, float * y) { start(x, T, slots, y), wait(); }
    int threads() const { return (int) workers_.size(); }
    // benchmark stats: total microseconds inside wait() and its calls (wait = pool latency + work the caller
    // did not itself compute; TRACKER #61 measures the CPU tier against DRAM peak through these).
    void reset_stats();
    long long wait_us() const { return wait_us_; }
    long waits() const { return waits_; }
    long long item_us() const { return item_us_; }   // sum of item() compute time (all threads, caller included)
    // phase breakdown of item_us_ (same clock): gate/up gemv, silu, h-quantize, down gemv
    long long gate_us() const { return gate_us_; }
    long long silu_us() const { return silu_us_; }
    long long hq_us() const { return hq_us_; }
    long long down_us() const { return down_us_; }
    // call shape since the last reset: calls, distinct experts (groups), slots (expert x row pairs), and the sum of
    // rows each expert served (the gemv's R: weights are loaded once, so R costs ALU, not bytes)
    void shape(long long out[4]) const
    {
        out[0] = calls_, out[1] = groups_sum_, out[2] = slots_sum_, out[3] = rows_sum_;
    }

private:
    struct Group {                          // one expert and the rows routed to it
        const Q4Expert * e;
        const TrellisExpert * t;
        std::vector<int> rows;
        size_t p_gu = 0, p_d = 0;           // trellis: offsets of its prepared gate/up and down activations in tp_
    };
    void worker();
    void item(int i, int phase);
    bool grab(int & i, int & phase);         // lock-free: take the next item of the current phase
    void retire(int phase);                  // one item finished: phase flip or call done
    void run_items();                        // grab + run until the current phase has no items left
    void quantize_h();   // quantize every group's h rows into hq_/hd_ (once, by the last phase-0 worker)
    void prep_down();    // trellis: every group's h rows prepared for down (once, at the gate/up -> down flip)
    void item_trellis(int i, int phase);
    void finish();

    std::vector<std::thread> workers_;
    // The workers take items without the mutex and spin briefly before blocking: a layer's call is a burst of
    // ~1 ms, and condvar wake latency was 43% of it (cpu_expert_test: 903 us wall, 515 us of parallel compute).
    // Only the final "busy_ == false" and the idle wait use the mutex.
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::atomic<bool> stop_{ false }, busy_{ false };
    // q4s: phase 0 gate/up, 1 down. trellis: 0 prep gate/up activations, 1 gate/up, 2 down.
    // ticket_ = phase << TICKET_SHIFT | next item, one atomic: a worker that read the phase and then took an index
    // separately could take index 0 of the next phase after a flip, run the wrong item and retire it against the old
    // phase, leaving a pending count that never reaches 0 (the call hung; TRACKER #73).
    static constexpr int TICKET_SHIFT = 20;
    std::atomic<int> ticket_{ 0 }, pending_[3] = { 0, 0, 0 };
    bool open() const;                       // a call is running and its current phase has items left
    int phase_items_[3] = {};
    int n_phases_ = 2;
    bool trellis_ = false;
    int spin_ = 0;                           // pause iterations before blocking (TRUSS_CPU_SPIN)

    // the current call
    const float * x_ = nullptr;
    float * y_ = nullptr;
    int T_ = 0;
    std::vector<Slot> slots_;
    std::vector<Group> groups_;
    std::vector<int8_t> xq_;                // [T][D_MODEL]
    std::vector<float> xd_;                 // [T][D_MODEL / 32]
    std::vector<float> h_;                  // [group][row][D_FF]
    std::vector<int8_t> hq_;                // [group][row][D_FF], quantized once at the phase 0->1 flip
    std::vector<float> hd_;                 // [group][row][D_FF / 32] its scales
    std::vector<float> out_;                // [group][row][D_MODEL]
    std::vector<float> tp_;                 // trellis: per group [gate P][up P][down P] (trellis_prep layouts)
    std::atomic<long long> wait_us_{ 0 };   // benchmark stats (see wait_us())
    std::atomic<long> waits_{ 0 };
    std::atomic<long long> item_us_{ 0 };
    std::atomic<long long> gate_us_{ 0 }, silu_us_{ 0 }, hq_us_{ 0 }, down_us_{ 0 };
    std::atomic<long long> calls_{ 0 }, groups_sum_{ 0 }, slots_sum_{ 0 }, rows_sum_{ 0 };
};

}  // namespace truss::cpu
