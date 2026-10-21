// One stage: the layers [lb, le) of a model on one device, with everything a chunk needs there - the device context,
// projections, routed experts, buffers, per-slot sequence bookkeeping and the hand-off to the next stage of a layer
// split. Model-agnostic: the layer math is the StageModel's (engine/model.h). engine::Pipeline drives the stages.
//
// Buffers come in two sizes. Chunks of up to FETCH_ROWS rows (decode steps, verify windows) use small resident ones;
// longer prompt chunks use big ones carved from the expert ring's spare region (stream mode), so the VRAM a prompt
// needs holds cached experts during decode (TRACKER #57). A dense model (no routed experts) allocates both.
#pragma once
#include "blocks/linear.h"
#include "engine/block.h"
#include "engine/gpu.h"
#include "engine/model.h"
#include "engine/options.h"
#include "engine/profile.h"
#include "engine/round_trace.h"
#include "engine/stage_lock.h"
#include "runtime/moe_exec.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace truss::engine {

struct StageSpec {
    int lb = 0, le = 0;
    int device = -1;                           // -1: the current device
    double budget_frac = 1.0;                  // share of the free VRAM its experts take (stages sharing a device)
    std::shared_ptr<cpu::ExpertPool> pool;     // the CPU pool of an earlier stage, shared (one set of workers)
    std::shared_ptr<std::mutex> pool_mu;
};

class Stage {
public:
    Stage(const Family & fam, int n_ctx, int max_chunk, const Options & o, const StageSpec & sg);
    ~Stage();

    // services for the model's blocks
    Gpu & gpu() { return g_; }
    blocks::Linear & lin() { return *lin_; }
    runtime::MoeExec * moe() { return moe_.get(); }
    Profile & prof() { return *prof_; }
    const Options & opts() const { return o_; }
    const Family & family() const { return fam_; }
    StageModel & model() { return *model_; }
    int lb() const { return lb_; }
    int le() const { return le_; }
    bool first() const { return first_; }
    bool last() const { return last_; }
    int n_ctx() const { return n_ctx_; }
    int seq_ctx() const { return seq_ctx_; }
    int spec_rows() const { return spec_rows_; }
    int device() const { return g_.device; }
    // the buffers of the chunk in flight: residual [rows][res_width], a model's extra row buffers, MoE workspace
    float * res() const { return cur_.res; }
    float * extra(int i) const { return cur_.extra[i]; }
    void * moe_ws() const { return cur_.moe_ws; }
    int moe_rows() const { return cur_.moe_rows; }
    void use_small() { use(small_); }
    float * small_res() const { return small_.res; }       // a drafter parks a slot's deferred rows from here
    int max_rows() const { return std::max(max_chunk_, prefill_rows_); }   // the largest chunk any buffer holds
    void kv_reserve(int upto, bool shrink);    // E6

    // sequence work (Pipeline calls these with the stage held and the slot selected)
    void select(int slot);
    int slot() const { return slot_; }
    void settle();                             // the slot's deferred accept on this (front) stage
    void run_chunk(const int32_t * tokens, int pos0, int T, const LayerHook & hook, bool tentative,
                   const int32_t * next_tokens = nullptr, int next_T = 0);
    void head(int first, int n, float * logits);
    void accept(int n);
    void checkpoint(int pos);
    int restore();
    void reset();
    bool window_open() const { return slots_[slot_].open; }
    void set_defer(int slot, int n) { slots_[slot].defer_n = n; }
    cudaEvent_t slot_done(int slot) const { return slots_[slot].done; }
    int draft(int32_t next, int pos, int n, int32_t * out, const Drafter::Sampling & s);
    // the drafter's per-draft confidence p_i of the last draft call (null: none)
    const float * draft_confidence(int * n) const;

    // layer split hand-off (Pipeline): the residual rows after le - 1 go to the next stage through pinned host memory
    Stage * prev = nullptr, * next = nullptr;
    std::atomic<long> handed{ 0 }, taken{ 0 };
    std::atomic<unsigned long long> failures{ 0 };   // counted up when a forward on this stage throws, never cleared
    void drain_handoffs(long up_to);
    template <class P> void spin_until(const char * what, const Stage & other, P ready) const;

    // concurrent slots: the stage lock (Held<Stage>), and TRUSS_ROUND_TRACE's per-stage accounting
    StageLock lk;
    int trace_stage = -1;
    rt::StageState rt_st;
    void acquire(long ticket);
    void release();
    cudaStream_t stream() const { return g_.s; }
    int n_hot() const { return moe_ ? moe_->n_hot() : 0; }

private:
    struct Buffers {
        Scratch * scratch = nullptr;
        void * moe_ws = nullptr;
        int moe_rows = 0;
        float * res = nullptr;
        std::vector<float *> extra;
    };
    struct Checkpoint {
        bool valid = false;
        int pos = 0;
        std::vector<int32_t> host;               // the blocks' host-side state
        char * pinned = nullptr;                 // the blocks' device buffers
        size_t bytes = 0;
    };
    struct SlotState {
        bool open = false;                       // a verify window is open
        Window vw;
        std::vector<int32_t> vw_tokens;
        int last_T = 0;                          // rows of res from the last run
        int defer_n = 0;                         // rows to accept before this slot's next work here (front stages)
        cudaEvent_t done = nullptr;              // recorded behind its verify rows (front stages)
        Checkpoint ck;
    };
    void use(const Buffers & b);
    void alloc_buffers(int small_rows);
    size_t big_bytes() const;
    void commit_tokens(const int32_t * tokens, int n);

    const Family & fam_;
    const Options o_;
    Gpu g_;
    const int lb_, le_;
    const bool first_, last_;
    const int n_ctx_, max_chunk_, prefill_rows_, spec_rows_;
    int moe_rows_ = 0;                       // the MoE prefill's workspace rows (a piece, not the chunk)
    int seq_ctx_ = 0;
    std::unique_ptr<StageModel> model_;
    std::unique_ptr<blocks::Linear> lin_;
    std::unique_ptr<Profile> prof_;
    std::unique_ptr<runtime::MoeExec> moe_;
    std::unique_ptr<Scratch> small_scratch_, big_scratch_;
    Buffers small_, big_, cur_;
    int n_extra_ = 0;
    std::vector<Stateful *> states_;
    std::vector<SlotState> slots_;
    int slot_ = 0;
    float * out_host_ = nullptr;
    cudaEvent_t out_ready_ = nullptr, in_taken_ = nullptr;
    int rt_prev_stage_ = -1;
    int64_t rt_held_t0_ = 0;
};

template <class P> void Stage::spin_until(const char * what, const Stage & other, P ready) const
{
    const auto base = other.failures.load(std::memory_order_acquire);
    const auto t0 = std::chrono::steady_clock::now();
    while (!ready()) {
        require(other.failures.load(std::memory_order_acquire) == base,
                std::string(what) + ": the other layer-split stage's forward failed");
        require(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < spin_timeout_s(),
                std::string(what) + ": no progress for " + std::to_string((int) spin_timeout_s()) +
                    " s (TRUSS_SPIN_TIMEOUT_S)");
        std::this_thread::yield();
    }
}

}  // namespace truss::engine
