// engine::Pipeline: construction (one stage, or a layer split), the selected-slot calls and the diagnostics. The
// concurrent slot calls prompt() and round() are in pipeline_prompt.cu / pipeline_round.cu.
#include "engine/pipeline.h"

#include "engine/limits.h"

namespace truss::engine {

Pipeline::Pipeline(const Family & fam, int n_ctx, int max_chunk, const Options & oo) : fam_(fam)
{
    Options o = oo;
    if (o.slots > 1 && o.kv_lend) {
        std::fprintf(stderr, "TRUSS_SLOTS=%d: TRUSS_KV_LEND off (it lends one sequence's unused K/V tail)\n", o.slots);
        o.kv_lend = false;
    }
    const int S = o.layer_split, L = fam.n_layer();
    if (S <= 0 || S >= L) {
        m_ = std::make_unique<Stage>(fam, n_ctx, max_chunk, o, StageSpec{ 0, L, -1, 1.0, nullptr, nullptr });
    } else {
        int cur = 0;
        TRUSS_CUDA(cudaGetDevice(&cur));
        const int d0 = o.layer_split_device >= 0 ? o.layer_split_device : cur;
        StageSpec a{ 0, S, d0, d0 == cur ? (double) o.layer_split_frac : 1.0, nullptr, std::make_shared<std::mutex>() };
        {
            DeviceGuard g(d0);
            front_.push_back(std::make_unique<Stage>(fam, n_ctx, max_chunk, o, a));
        }
        Stage & f = *front_.back();
        StageSpec b{ S, L, cur, 1.0, f.moe() ? f.moe()->shared_pool() : nullptr, a.pool_mu };
        m_ = std::make_unique<Stage>(fam, n_ctx, max_chunk, o, b);
        m_->prev = &f, f.next = m_.get();
        std::fprintf(stderr, "layer split: layers 0-%d on device %d (%d experts resident), layers %d-%d, head and "
                             "drafter on device %d (%d resident)\n",
                     S - 1, d0, f.n_hot(), S, L - 1, cur, m_->n_hot());
    }
    // TRUSS_ROUND_TRACE: which stage is which (0 the front one, 1 the last) and the device each runs on
    for (auto & p : front_) p->trace_stage = 0;
    m_->trace_stage = 1;
    int dev = m_->device();
    if (dev < 0) cudaGetDevice(&dev);
    rt::set_stage_device(1, dev);
    if (!front_.empty()) rt::set_stage_device(0, front_[0]->device());
    seq_pos_.assign(std::max(1, o.slots), 0);
    rope_delta_.assign(seq_pos_.size(), 0);
    rope_delta_ck_.assign(seq_pos_.size(), 0);
    round_ns_ = std::make_unique<std::atomic<long long>[]>(seq_pos_.size());
    for (size_t i = 0; i < seq_pos_.size(); ++i) round_ns_[i] = 0;
    seq_ctx_ = m_->seq_ctx();
    if (o.slots > 1) std::fprintf(stderr, "sequence slots: %d x %d positions\n", o.slots, seq_ctx_);
}

Pipeline::~Pipeline()
{
    int d = 0;
    cudaGetDevice(&d);
    rt::close();   // TRUSS_ROUND_TRACE: flush and join the sampler BEFORE any stage dies (its events)
    m_.reset();    // the last stage first: its driver may share the first stage's CPU pool
    front_.clear();
    cudaSetDevice(d);
}

int Pipeline::fetch_rows() { return FETCH_ROWS; }

void Pipeline::select(int k)
{
    require(k >= 0 && k < (int) seq_pos_.size(), "select(" + std::to_string(k) + "): " +
                                                     std::to_string(seq_pos_.size()) + " slots");
    active_ = k;
}

void Pipeline::run(const int32_t * tokens, int T, const LayerHook & hook, const int32_t * next, int next_T)
{
    if (next) require(next == tokens + T && next_T > 0, "lookahead must follow the chunk");
    const int slot = active_, pos = seq_pos_[slot];
    long ticket = 0;
    each_stage([&](Stage & m) {
        Held<Stage> h(m, ticket);
        m.select(slot);
        m.settle();
        m.run_chunk(tokens, pos, T, hook, false, next, next_T, ChunkExtra{ nullptr, 0, 0, 0, nullptr, 0, 0,
                                                                           rope_delta_[slot] });
        ticket = m.handed.load();
    });
    seq_pos_[slot] += T;
}

void Pipeline::run_prompt(const int32_t * tokens, int n, int step, const std::function<void(int)> & done)
{
    prompt(active_, tokens, n, step, nullptr, nullptr, done);
}

void Pipeline::head(int first, int n, float * logits)
{
    const int slot = active_;   // one snapshot (H2)
    Held<Stage> h(*m_, 0);
    m_->select(slot);
    m_->head(first, n, logits);
}

void Pipeline::verify(const int32_t * tokens, int T)
{
    require(m_->spec_rows() && T <= m_->spec_rows(),
            "verify() of " + std::to_string(T) + " rows (spec_rows " + std::to_string(m_->spec_rows()) + ")");
    const int slot = active_, pos = seq_pos_[slot];
    long ticket = 0;
    each_stage([&](Stage & m) {
        Held<Stage> h(m, ticket);
        m.select(slot);
        m.settle();
        m.run_chunk(tokens, pos, T, nullptr, true, nullptr, 0,
                    ChunkExtra{ nullptr, 0, 0, 0, nullptr, 0, 0, rope_delta_[slot] });
        ticket = m.handed.load();
    });
}

// the last stage now (its rollback and drafter commit are queued, no host wait); the earlier stages' rollbacks run
// with this slot's next call there (defer), off the draft's critical path
void Pipeline::accept(int n)
{
    const int slot = active_;
    {
        Held<Stage> h(*m_, 0);
        m_->select(slot);
        m_->accept(n);
    }
    for (auto & p : front_) p->set_defer(slot, n);
    seq_pos_[slot] += n;
}

int Pipeline::draft(int32_t next, int n, int32_t * out)
{
    const int slot = active_;
    Held<Stage> h(*m_, 0);
    m_->select(slot);
    return m_->draft(next, seq_pos_[slot], n, out, ds_);
}

void Pipeline::set_draft_sampling(const void * params, uint64_t counter, float * q) { ds_ = { params, counter, q }; }

void Pipeline::reset(int slot)
{
    if (m_->moe()) m_->moe()->sequence_reset();
    each_stage([slot](Stage & m) {
        Held<Stage> h(m, 0);
        m.select(slot);
        m.reset();
    });
    seq_pos_.at(slot) = 0;
    rope_delta_.at(slot) = 0;
}
void Pipeline::reset() { reset(active_); }

void Pipeline::checkpoint(int slot)
{
    const int pos = seq_pos_.at(slot);
    rope_delta_ck_.at(slot) = rope_delta_.at(slot);
    each_stage([&](Stage & m) {
        Held<Stage> h(m, 0);
        m.select(slot);
        m.settle();
        m.checkpoint(pos);
    });
}
void Pipeline::checkpoint() { checkpoint(active_); }

int Pipeline::restore(int slot)
{
    int p = -1;
    each_stage([&](Stage & m) {
        Held<Stage> h(m, 0);
        m.select(slot);
        m.settle();
        p = m.restore();
    });
    if (p >= 0) seq_pos_.at(slot) = p, rope_delta_.at(slot) = rope_delta_ck_.at(slot);
    return p;
}
int Pipeline::restore() { return restore(active_); }

}  // namespace truss::engine
