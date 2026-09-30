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
    int rows, cols;
};

struct Q4Expert {
    Q4Matrix gate, up, down;
};

constexpr int D_MODEL = 2560, D_FF = 640, BLOCK = 32, MAX_ROWS = 8;
constexpr size_t EXPERT_BYTES = 3 * ((size_t) D_MODEL * D_FF / 2 + (size_t) D_MODEL * D_FF / BLOCK * 2);

// the matrices of one expert stored contiguously in q4s order at p
Q4Expert expert_view(const uint8_t * p);

// One layer's CPU work: rows x [T][D_MODEL] (fp32) and the slots routed to CPU experts.
struct Slot {
    int row;                // 0 .. T - 1
    const Q4Expert * e;
    float w;                // routing weight
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

private:
    struct Group {                          // one expert and the rows routed to it
        const Q4Expert * e;
        std::vector<int> rows;
    };
    void worker();
    void item(int i);
    void finish();

    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    long generation_ = 0;
    bool stop_ = false;
    int phase_items_[2] = {};
    std::vector<int> pending_;              // [phase] items not finished
    int phase_ = 0, next_item_ = 0, running_ = 0;
    bool busy_ = false;

    // the current call
    const float * x_ = nullptr;
    float * y_ = nullptr;
    int T_ = 0;
    std::vector<Slot> slots_;
    std::vector<Group> groups_;
    std::vector<int8_t> xq_;                // [T][D_MODEL]
    std::vector<float> xd_;                 // [T][D_MODEL / 32]
    std::vector<float> h_;                  // [group][row][D_FF]
    std::vector<float> out_;                // [group][row][D_MODEL]
};

}  // namespace truss::cpu
