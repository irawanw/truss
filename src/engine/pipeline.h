// engine::Pipeline: a loaded model's sequences on one or two devices. The C API, the server and every bench drive
// the engine through this class; it is the same for every model family (docs/ENGINE.md).
//
// One stage covers all layers, or a layer split has a front stage (layers [0, S) on another device) and the last
// stage (the rest, the head and the drafter). Every call names a sequence slot (the calls without one act on the
// selected slot). Each stage is held only while one slot's work is enqueued on it, so with a split two slots' work
// runs on the two GPUs at once; a stage's verify window for a slot stays open until that slot's next call there
// (defer), and work that takes a hand-off from the previous stage is granted in hand-off order (StageLock).
#pragma once
#include "engine/model.h"
#include "engine/options.h"
#include "engine/stage.h"

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace truss::engine {

class Pipeline {
public:
    // fam must outlive the Pipeline. n_ctx: the longest sequence (shared by the slots); max_chunk: tokens per pass
    Pipeline(const Family & fam, int n_ctx, int max_chunk, const Options & o = {});
    ~Pipeline();

    // ---- the selected slot
    void run(const int32_t * tokens, int T, const LayerHook & hook = nullptr, const int32_t * next = nullptr,
             int next_T = 0);
    void run_prompt(const int32_t * tokens, int n, int step, const std::function<void(int)> & done = nullptr);
    void head(int first, int n, float * logits);   // logits [n][n_vocab] of rows first.. of the last run/verify
    void verify(const int32_t * tokens, int T);     // a tentative window (accept() commits its first n rows)
    void accept(int n);
    int draft(int32_t next, int n, int32_t * out);
    void set_draft_sampling(const void * params, uint64_t counter, float * q);
    void reset();
    void checkpoint();
    int restore();
    void select(int k);
    int slot() const { return active_; }
    int position() const { return seq_pos_[active_]; }
    // B1: re-align the host cursor with the device state (graph mode runs no host code per step, so the cursor
    // goes stale; the next capture must bake the device's position). Only between steps, never in a verify window.
    void set_position(int pos) { seq_pos_[active_] = pos; }

    // ---- slot calls (different slots may run on different threads at once)
    // prompt(): tokens [0, n) appended to `slot` in chunks of `step`; with logits, the last row's logits go there and
    // after() runs, both while the last stage is still held for this slot
    void prompt(int slot, const int32_t * tokens, int n, int step, float * logits, const std::function<void()> & after,
                const std::function<void(int)> & done = nullptr);
    // round(): up to nd drafts into win[1..] (win[0] = next), the verify window, its logits [nw + 1][n_vocab], then
    // pick(nw) - with the last stage held - returns how many rows to keep (1 .. nw + 1). Returns that count.
    int round(int slot, int32_t next, int nd, int32_t * win, float * logits, const Drafter::Sampling & ds,
              const std::function<int(int)> & pick);
    void reset(int slot);
    void checkpoint(int slot);
    int restore(int slot);
    int position(int slot) const { return seq_pos_.at(slot); }
    int slots() const { return (int) seq_pos_.size(); }
    int seq_ctx() const { return seq_ctx_; }
    int spec_rows() const { return m_->spec_rows(); }

    // ---- diagnostics (sums over the stages)
    cudaStream_t stream() const { return m_->stream(); }
    int hot_experts() const;
    size_t cold_bytes() const;
    void wait_pinned();
    void lock_ms(double out[4], bool reset = false);   // [front held, front wait, last held, last wait] ms
    rt::StageState * rt_last_stage() { return &m_->rt_st; }
    void section_ms(double out[6], bool reset = false) const;
    void section_moe_ms(double out[4], bool reset = false) const;
    void driver_ms(double out[3], long & n, bool reset = false) const;
    std::pair<long long, long> cpu_stats(bool reset = false);
    long long cpu_item_us() const;
    void cpu_phase_us(long long out[4]) const;
    void cpu_shape(long long out[4]) const;
    void cpu_shape_reset();
    void dyn_stats(long & cpu, long & pcie, double & ms_expert, double & ms_call) const;
    void split_stats(long long out[3]) const;
    size_t kv_lent() const;
    const runtime::ExpertStore * experts() const;   // the last stage's (a drafter's layer is its last store layer)
    // B1: graph-mode requirements and the model's position-dependent kernels
    bool split() const { return !front_.empty(); }
    bool all_resident() const { return m_->moe() && m_->moe()->all_resident(); }
    StageModel::GraphInfo graph_info() const { return m_->model().graph_info(); }
    void profile_routes(bool on);
    std::vector<float> route_counts() const;        // [layer][expert], the drafter's layer last
    static int fetch_rows();
    // host-resident per-token tables (StageModel::table_stats), summed over the stages since the last reset
    void table_stats(double & ms, long & calls, long & rows, bool reset = false, double * wait_ms = nullptr,
                     double * phase_ms = nullptr);

private:
    template <class F> void each_stage(F && f);
    template <class F> void each_stage_locked(F && f) const;
    const Family & fam_;
    std::vector<std::unique_ptr<Stage>> front_;    // a layer split's stages before the last (destroyed after m_)
    std::unique_ptr<Stage> m_;                     // the last stage (the whole model without a split)
    std::vector<int> seq_pos_{ 0 };
    std::unique_ptr<std::atomic<long long>[]> round_ns_;   // each slot's last round() end (steady clock ns)
    std::atomic<int> active_{ 0 };   // read by other threads' calls between select() and their work (review H2)
    int seq_ctx_ = 0;
    Drafter::Sampling ds_{};
};

template <class F> void Pipeline::each_stage(F && f)   // every stage in layer order, its device current
{
    for (auto & p : front_) {
        DeviceGuard g(p->device());
        f(*p);
    }
    f(*m_);
}

// Every stage in layer order, held while f runs on it: for the calls that touch a buffer a chunk's kernels may still
// be writing (review H3). Stages are taken front first, like every other path, so taking them cannot cycle.
template <class F> void Pipeline::each_stage_locked(F && f) const
{
    for (auto & p : front_) {
        DeviceGuard g(p->device());
        Held<Stage> h(*p, 0);
        f(*p);
    }
    Held<Stage> h(*m_, 0);
    f(*m_);
}

}  // namespace truss::engine
