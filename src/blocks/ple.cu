#include "blocks/ple.h"

#include "engine/stage.h"
#include "kernels/ple/ple_prefill.cuh"
#include "kernels/spec/rollback.cuh"

#include <chrono>

namespace truss::blocks {

PleTable::PleTable(const PleDims & d, const gguf::Tensor * table, const gguf::Tensor * scale, const gguf::File * file,
                   int max_rows)
    : d_(d), table_(table), scale_(scale)
{
    if (file && table) {   // PLE rows by O_DIRECT reads (TRACKER #88)
        reader_ = std::make_unique<PleReader>(*file, *table, *scale, d_.head_dim);
        stage_n_ = (size_t) max_rows * d_.heads() * d_.head_dim;
        TRUSS_CUDA(cudaMallocHost(&stage_, stage_n_ * sizeof(half)));
        TRUSS_CUDA(cudaEventCreateWithFlags(&copied_, cudaEventDisableTiming));
    }
}

PleTable::~PleTable()
{
    if (stage_) cudaFreeHost(stage_);
    if (copied_) cudaEventDestroy(copied_);
}

void PleTable::gather(engine::Gpu & g, int slot, const int32_t * tokens, int T, int pos0, half * e16)
{
    // n-gram window across chunks: hash [tail | chunk] and keep the chunk's rows
    const auto th0 = std::chrono::steady_clock::now();
    const std::vector<int32_t> & tail = tail_[slot];
    const int H = d_.heads(), E = H * d_.head_dim, n_prev = (int) tail.size();
    std::vector<int32_t> seq(tail);
    seq.insert(seq.end(), tokens, tokens + T);
    std::vector<int32_t> rows(seq.size() * H);
    ple_rows(d_, seq.data(), (int) seq.size(), rows.data());
    std::vector<half> emb16;
    const half * host16;
    if (reader_) {   // O_DIRECT reads into the pinned stage (the previous copy out of it has been read)
        engine::require((size_t) T * E <= stage_n_, "PLE stage too small for the chunk");
        int tk;
        if (tk_valid_ && tk_pos0_ == pos0 && tk_slot_ == slot) {   // already in flight
            tk = tk_;
            tk_valid_ = false;
        } else {   // a chunk's rows are used once: only decode tickets populate the cache
            tk_valid_ = false;
            tk = reader_->issue(rows.data() + (size_t) n_prev * H, (size_t) T * H, T <= 1);
        }
        TRUSS_CUDA(cudaEventSynchronize(copied_));
        reader_->collect(tk, stage_);
        host16 = stage_;
    } else {
        std::vector<float> emb((size_t) T * E);
        ple_gather(d_, table_, scale_, rows.data() + (size_t) n_prev * H, T, emb.data());
        emb16.resize(emb.size());
        for (size_t i = 0; i < emb.size(); ++i) emb16[i] = __float2half(emb[i]);
        host16 = emb16.data();
    }
    host_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - th0).count();
    ++calls_, rows_n_ += (long) T * H;
    TRUSS_CUDA(cudaMemcpyAsync(e16, host16, (size_t) T * E * 2, cudaMemcpyHostToDevice, g.s));
    if (reader_) TRUSS_CUDA(cudaEventRecord(copied_, g.s));
    if (reader_ && next_ && next_T_ > 0) {   // read the next chunk's rows while this chunk computes
        const size_t keep = std::min<size_t>(seq.size(), (size_t) d_.ngram - 1);
        std::vector<int32_t> nseq(seq.end() - keep, seq.end());
        nseq.insert(nseq.end(), next_, next_ + next_T_);
        std::vector<int32_t> nrows(nseq.size() * H);
        ple_rows(d_, nseq.data(), (int) nseq.size(), nrows.data());
        tk_ = reader_->issue(nrows.data() + keep * H, (size_t) next_T_ * H, false);
        tk_valid_ = true, tk_pos0_ = pos0 + T, tk_slot_ = slot;
    }
    next_ = nullptr, next_T_ = 0;
    if (!reader_) TRUSS_CUDA(cudaStreamSynchronize(g.s));   // emb16 is a host temporary (the stage is not)
}

void PleTable::stats(double & ms, long & calls, long & rows, bool reset, double * wait_ms, double * phase_ms)
{
    ms += host_ms_, calls += calls_, rows += rows_n_;
    if (wait_ms && reader_) *wait_ms += reader_->take_wait_ms();
    if (phase_ms && reader_) {
        double ph[3] = { 0, 0, 0 };
        reader_->take_phase_ms(ph);
        for (int i = 0; i < 3; ++i) phase_ms[i] += ph[i];
    }
    if (reset) host_ms_ = 0, calls_ = 0, rows_n_ = 0;
}

void PleTable::alloc_slots(engine::Stage &, int n_slots, int, int) { tail_.assign(n_slots, {}); }

void PleTable::reset(engine::Stage &, int slot)
{
    tail_[slot].clear();
    next_ = nullptr, next_T_ = 0, tk_valid_ = false;   // a dangling ticket is replaced by the next issue
}

void PleTable::commit(engine::Stage &, int slot, const int32_t * tokens, int n)   // the window's predecessors
{
    std::vector<int32_t> seq(tail_[slot]);
    seq.insert(seq.end(), tokens, tokens + n);
    const size_t keep = std::min<size_t>(seq.size(), d_.ngram - 1);
    tail_[slot].assign(seq.end() - keep, seq.end());
}

void PleTable::ck_save(int slot, std::vector<int32_t> & host)
{
    host.push_back((int32_t) tail_[slot].size());
    host.insert(host.end(), tail_[slot].begin(), tail_[slot].end());
}

void PleTable::ck_load(int slot, const std::vector<int32_t> & host, size_t & at)
{
    const int n = host[at++];
    tail_[slot].assign(host.begin() + (long) at, host.begin() + (long) at + n);
    at += n;
}

// ---- one PLE layer

void Ple::add_tensors(Linear & lin, const PleWeights & w)
{
    for (T t : { w.key, w.value, w.norm_key, w.norm_query, w.norm_conv, w.conv1d }) lin.add(t);
}

size_t Ple::scratch_per_row(const PleDims & d)
{
    return (size_t) d.heads() * d.head_dim * 2 + (size_t) d.hc_dim() * 8 + (size_t) d.d_model * 4 + (size_t) d.hc * 4;
}

size_t Ple::hist_floats() const
{
    const PleDims & d = t_.dims();
    return (size_t) (d.conv - 1) * d.ngram * d.hc_dim();
}

void Ple::alloc_slots(engine::Stage & st, int n_slots, int, int spec_rows)
{
    s_.resize(n_slots);
    for (Slot & s : s_) {
        s.hist = st.gpu().alloc<float>(hist_floats());
        if (spec_rows)
            s.hist_snap = st.gpu().alloc<float>(hist_floats()),
            s.normed_rows = st.gpu().alloc<float>((size_t) spec_rows * t_.dims().hc_dim());
    }
}

void Ple::forward(engine::Gpu & g, int slot, const int32_t * tokens, int T, int pos0, float * res, bool tentative)
{
    const PleDims & d = t_.dims();
    Slot & L = s_[slot];
    Scratch & sc = *g.sc;
    const size_t m = sc.mark();
    const int E = d.heads() * d.head_dim;
    half * e16 = sc.alloc<half>((size_t) T * E);
    float * key = sc.alloc((size_t) T * d.hc_dim()), * value = sc.alloc((size_t) T * d.d_model);
    float * gate = sc.alloc((size_t) T * d.hc), * normed = sc.alloc((size_t) T * d.hc_dim());
    t_.gather(g, slot, tokens, T, pos0, e16);
    const Linear::Act ea = lin_.quant(e16, T, E);
    lin_.lin_multi({ { w_.key, key }, { w_.value, value } }, ea, T);
    ple::gate(key, res, lin_.f32(w_.norm_key), lin_.f32(w_.norm_query), T, d.hc, d.d_model, d.eps, gate, g.s);
    if (tentative) g.copy(L.hist_snap, L.hist, hist_floats());
    ple::apply(value, gate, lin_.f32(w_.norm_conv), lin_.d(w_.conv1d).as<half>(), L.hist, T, d.hc, d.d_model, d.conv,
               d.ngram, d.eps, normed, res, g.s);
    if (tentative) g.copy(L.normed_rows, normed, (size_t) T * d.hc_dim());
    sc.release(m);
}

void Ple::reset(engine::Stage & st, int slot)
{
    TRUSS_CUDA(cudaMemsetAsync(s_[slot].hist, 0, sizeof(float) * hist_floats(), st.gpu().s));
}

void Ple::accept(engine::Stage & st, int slot, int n, const engine::Window & w)
{
    if (n == w.T) return;
    Slot & L = s_[slot];
    const PleDims & d = t_.dims();
    if (n > 0) spec::tail_rows(L.hist_snap, (d.conv - 1) * d.ngram, L.normed_rows, n, d.hc_dim(), L.hist, st.gpu().s);
    else st.gpu().copy(L.hist, L.hist_snap, hist_floats());   // none of the window's rows count
}

void Ple::ck_buffers(int slot, std::vector<std::pair<float *, size_t>> & out) { out.push_back({ s_[slot].hist, hist_floats() }); }

}  // namespace truss::blocks
