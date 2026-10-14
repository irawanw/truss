// engine::Stage: the sequence calls - one chunk, the head, verify-window accept, checkpoint / restore / reset, slot
// selection, the stage lock and the layer-split hand-off. The blocks' per-slot states do the model-specific part
// (engine/block.h); everything here is the same for every model.
#include "engine/stage.h"

#include "engine/limits.h"

namespace truss::engine {

void Stage::select(int k)
{
    require(k >= 0 && k < (int) slots_.size(), "select(" + std::to_string(k) + "): no such slot");
    if (k == slot_) return;
    require(!(last_ && slots_[slot_].open), "select() inside the last stage's verify window");
    if (Drafter * d = model_->drafter()) {
        d->park(*this, slot_);
        d->unpark(*this, k);
    }
    slot_ = k;
}

void Stage::settle()
{
    SlotState & s = slots_[slot_];
    if (!s.defer_n) return;
    const int n = s.defer_n;
    s.defer_n = 0;
    accept(n);
}

void Stage::commit_tokens(const int32_t * tokens, int n)
{
    for (Stateful * s : states_) s->commit(*this, slot_, tokens, n);
}

// One chunk of the model at positions pos0 .. pos0 + T - 1 on this stage. tentative: a verify window (accept()
// commits it). next: the following chunk's tokens, for blocks that prefetch per-token tables.
void Stage::run_chunk(const int32_t * tokens, int pos0, int T, const LayerHook & hook, bool tentative,
                      const int32_t * next_tokens, int next_T)
{
    SlotState & ss = slots_[slot_];
    require(T > 0 && T <= max_chunk_, "chunk of " + std::to_string(T) + " tokens (max " + std::to_string(max_chunk_) + ")");
    require(pos0 + T <= seq_ctx_, "sequence longer than n_ctx" + std::string(slots_.size() > 1 ? " / slots" : ""));
    require(!ss.open, "run()/verify() before accept() of the previous verify()");
    if (Drafter * d = model_->drafter()) d->flush(*this, slot_);   // it reads res, which this chunk rewrites
    if (moe_) moe_->check_driver();
    kv_reserve(pos0 + T, false);
    const bool stream = !fetch_mode(T);
    if (moe_) moe_->after_chunk_aux(stream);
    if (stream)
        require(T <= prefill_rows_, "prompt chunk of " + std::to_string(T) + " tokens over prefill_rows (" +
                                        std::to_string(prefill_rows_) + "); raise Options::prefill_rows");
    if (stream && moe_) moe_->drain_jobs();   // before begin_stream: a queued decode FFN must fetch in ring mode
    if (!stream) {
        use(small_);
        if (moe_) moe_->begin_ring(), moe_->adapt_tick();
    } else {   // the ring becomes the stream slots and this chunk's buffers
        if (moe_) moe_->begin_stream(g_.s);
        use(big_);
    }
    const bool split = moe_ && moe_->split_chunk(stream, T);   // E7: per-layer copies after the routing, no prefetch
    if (moe_) moe_->set_split(split);
    g_.sc->reset();
    const Chunk c{ tokens, pos0, T, slot_, tentative, stream, next_tokens, next_T };
    const size_t row_bytes = sizeof(float) * (size_t) T * fam_.res_width();
    if (first_) {
        model_->embed(*this, c, cur_.res);
    } else {   // the previous stage's residual rows (its out_host, behind its out_ready)
        spin_until("run_chunk: the previous stage's hand-off", *prev,
                   [&] { return prev->handed.load(std::memory_order_acquire) > taken.load(); });
        TRUSS_CUDA(cudaStreamWaitEvent(g_.s, prev->out_ready_, 0));
        TRUSS_CUDA(cudaMemcpyAsync(cur_.res, prev->out_host_, row_bytes, cudaMemcpyHostToDevice, g_.s));
        TRUSS_CUDA(cudaEventRecord(in_taken_, g_.s));
        taken.fetch_add(1, std::memory_order_release);
    }
    if (stream && !split && moe_)
        for (int l = 0; l < std::min(2, moe_->n_store()); ++l) moe_->prefetch(l);
    model_->layers(*this, c, cur_.res, hook);
    if (!last_) {   // hand the residual rows on, once the next stage has taken the previous ones
        if (next) {
            spin_until("run_chunk: the next stage taking the hand-off", *next,
                       [&] { return next->taken.load(std::memory_order_acquire) >= handed.load(); });
            TRUSS_CUDA(cudaStreamWaitEvent(g_.s, next->in_taken_, 0));
        }
        TRUSS_CUDA(cudaMemcpyAsync(out_host_, cur_.res, row_bytes, cudaMemcpyDeviceToHost, g_.s));
        TRUSS_CUDA(cudaEventRecord(out_ready_, g_.s));
        handed.fetch_add(1, std::memory_order_release);
    }
    prof_->flush(T, -1);
    ss.last_T = T;
    if (tentative) {
        ss.open = true;
        ss.vw_tokens.assign(tokens, tokens + T);
        ss.vw = { pos0, T, ss.vw_tokens.data() };
        return;
    }
    commit_tokens(tokens, T);
    if (Drafter * d = model_->drafter()) d->on_commit(*this, c);
}

void Stage::head(int first, int n, float * logits)
{
    require(first >= 0 && n > 0 && first + n <= slots_[slot_].last_T, "head rows outside the last chunk");
    if (moe_) moe_->check_driver();
    g_.sc->reset();   // run()'s temporaries are dead; the residual is not in the scratch
    model_->head(*this, cur_.res + (size_t) first * fam_.res_width(), n, logits, false);
}

// Commit the first n rows of the open verify window: every recurrent state as if only those rows had run. n == 0 is
// the full rollback (a round that died before its pick(), review C2): the window ran tentative, so the state before
// it is still in place and each snapshot only goes back where it came from.
void Stage::accept(int n)
{
    SlotState & ss = slots_[slot_];
    require(ss.open, "accept() without verify()");
    require(n >= 0 && n <= ss.vw.T, "accept(" + std::to_string(n) + ") of a " + std::to_string(ss.vw.T) + "-row window");
    ss.open = false;
    for (Stateful * s : states_) s->accept(*this, slot_, n, ss.vw);
    commit_tokens(ss.vw.tokens, n);
    ss.last_T = n;
    // nothing committed (n == 0): nothing new is parked for the drafter
    if (Drafter * d = model_->drafter()) d->on_accept(*this, slot_, n, ss.vw);
}

// Sequence checkpoint: every recurrent piece of the committed sequence copied to pinned host memory (no VRAM taken
// from the experts). Position-indexed caches need no copy: rows past the checkpoint are rewritten before any query
// reads them, as after a verify window.
void Stage::checkpoint(int pos)
{
    SlotState & ss = slots_[slot_];
    require(!ss.open, "checkpoint() inside a verify window");
    if (Drafter * d = model_->drafter()) d->flush(*this, slot_);
    std::vector<std::pair<float *, size_t>> bufs;
    for (Stateful * s : states_) s->ck_buffers(slot_, bufs);
    Checkpoint & ck = ss.ck;
    if (!ck.pinned) {
        for (const auto & b : bufs) ck.bytes += b.second * sizeof(float);
        TRUSS_CUDA(cudaMallocHost(&ck.pinned, ck.bytes));
    }
    size_t at = 0;
    for (const auto & [d, n] : bufs) {
        TRUSS_CUDA(cudaMemcpyAsync(ck.pinned + at, d, n * sizeof(float), cudaMemcpyDeviceToHost, g_.s));
        at += n * sizeof(float);
    }
    g_.sync();
    ck.host.clear();
    for (Stateful * s : states_) s->ck_save(slot_, ck.host);
    ck.valid = true, ck.pos = pos;
}

int Stage::restore()
{
    SlotState & ss = slots_[slot_];
    require(!ss.open, "restore() inside a verify window");
    Checkpoint & ck = ss.ck;
    if (!ck.valid) return -1;
    std::vector<std::pair<float *, size_t>> bufs;
    for (Stateful * s : states_) s->ck_buffers(slot_, bufs);
    size_t at = 0;
    for (const auto & [d, n] : bufs) {
        TRUSS_CUDA(cudaMemcpyAsync(d, ck.pinned + at, n * sizeof(float), cudaMemcpyHostToDevice, g_.s));
        at += n * sizeof(float);
    }
    g_.sync();
    size_t h = 0;
    for (Stateful * s : states_) s->ck_load(slot_, ck.host, h);
    ss.last_T = 0;
    kv_reserve(ck.pos, true);
    return ck.pos;
}

void Stage::reset()
{
    for (Stateful * s : states_) s->reset(*this, slot_);
    SlotState & ss = slots_[slot_];
    ss.last_T = 0, ss.open = false, ss.defer_n = 0;
    kv_reserve(0, true);
}

int Stage::draft(int32_t nxt, int pos, int n, int32_t * out, const Drafter::Sampling & s)
{
    Drafter * d = model_->drafter();
    require(d != nullptr, "draft() needs a drafter");
    require(!slots_[slot_].open, "draft() before accept()");
    return d->draft(*this, slot_, nxt, pos, n, out, s);
}

// Take hand-offs up to `up_to` that the previous stage wrote and this stage never read (its request threw between
// the write and the read): the stage lock grants a later ticket only once taken + 1 reaches it, so an untaken
// hand-off wedges every slot behind it. Only up to a hand-off the calling request produced itself (C1).
void Stage::drain_handoffs(long up_to)
{
    if (!robust_on()) return;   // C1 fix off
    long n = 0;
    while (taken.load(std::memory_order_relaxed) < up_to) {
        TRUSS_CUDA(cudaEventRecord(in_taken_, g_.s));
        taken.fetch_add(1, std::memory_order_release);
        ++n;
    }
    if (n) std::fprintf(stderr, "layer split: stage %d took %ld abandoned hand-off(s)\n", g_.device, n);
}

void Stage::acquire(long ticket)
{
    const bool tr = rt::on();
    const int64_t t0 = tr ? rt::now() : 0;
    rt_prev_stage_ = rt::ctx().stage;
    lk.lock(ticket, [this] { return taken.load(std::memory_order_acquire) + 1; });
    rt::ctx().stage = trace_stage;
    if (tr) {
        rt::push({ t0, rt::now() - t0, trace_stage == 0 ? rt::kLockFront : rt::kLockLast, rt::ctx().slot, trace_stage,
                   rt::kKindHost, rt::ctx().round });
        rt_held_t0_ = rt::now();
        rt::stage_begin(rt_st, g_.s, g_.device, trace_stage);   // the device takes this section over here
    }
}

void Stage::release()
{
    if (rt::on()) {
        rt::stage_end(rt_st);   // before the lock drops: what is queued now is this section's work
        rt::push({ rt_held_t0_, rt::now() - rt_held_t0_, trace_stage == 0 ? rt::kHeldFront : rt::kHeldLast,
                   rt::ctx().slot, trace_stage, rt::kKindHost, rt::ctx().round });
    }
    rt::ctx().stage = rt_prev_stage_;   // acquire stores this whether or not the trace is on; both sides match
    lk.unlock();
}

}  // namespace truss::engine
