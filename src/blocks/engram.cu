#include "blocks/engram.h"

#include "core/cuda_check.h"
#include "engine/stage.h"
#include "kernels/ds41/attn_ops.cuh"
#include "kernels/ds41/engram.cuh"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace truss::blocks {

Engram::Engram(Linear & lin, const EngramWeights & w, const EngramDims & d, const gguf::Tensor & table, int threads)
    : lin_(lin), w_(w), d_(d), table_(table)
{
    if (d.head_dim != ds41::ENGRAM_DIM || table.shape.size() != 2 || table.shape[0] != ds41::ENGRAM_Q4_ROW)
        throw std::runtime_error("Engram: table is not int4 rows of 256 (" + table.name + ")");
    for (int i = 0; i < threads; ++i) pool_.emplace_back([this] { worker(); });
}

Engram::~Engram()
{
    {
        std::lock_guard<std::mutex> l(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto & t : pool_) t.join();
    if (pinned_) cudaFreeHost(pinned_);
}

size_t Engram::scratch_per_row(const EngramDims & d)
{
    const size_t in = (size_t) d.buckets() * d.head_dim;
    return in * 2 + (size_t) (d.hc + 1) * d.d_model * 4 + (size_t) d.buckets() * ds41::ENGRAM_Q4_ROW + 1024;
}

void Engram::build(engine::Gpu & g)
{
    const int n = d_.hc * d_.d_model;
    qk_ = g.alloc<float>(n);
    ds41::mul(lin_.f32(w_.q), lin_.f32(w_.k), n, qk_, g.s);
}

void Engram::alloc_slots(engine::Stage &, int n_slots, int, int) { alloc(n_slots); }
void Engram::alloc(int n_slots) { hist_.assign(n_slots, {}); }
void Engram::reset(engine::Stage &, int slot) { hist_[slot].clear(); }

void Engram::commit(engine::Stage &, int slot, const int32_t * tokens, int n) { commit_tokens(slot, tokens, n); }

void Engram::commit_tokens(int slot, const int32_t * tokens, int n)
{
    auto & h = hist_[slot];
    const int keep = d_.max_ngram - 1;
    for (int i = std::max(0, n - keep); i < n; ++i) h.push_back(tokens[i]);
    if ((int) h.size() > keep) h.erase(h.begin(), h.end() - keep);
}

void Engram::ck_save(int slot, std::vector<int32_t> & host)
{
    host.push_back((int32_t) hist_[slot].size());
    host.insert(host.end(), hist_[slot].begin(), hist_[slot].end());
}

void Engram::ck_load(int slot, const std::vector<int32_t> & host, size_t & at)
{
    const int n = host.at(at++);
    hist_[slot].assign(host.begin() + at, host.begin() + at + n);
    at += n;
}

void Engram::hash(int slot, const int32_t * tokens, int pos0, int T, int32_t * ids) const
{
    const auto & h = hist_[slot];
    const int N = d_.max_ngram, B = d_.buckets();
    auto mapped = [&] (int32_t tok) -> int64_t {
        return tok >= 0 && tok < d_.n_vocab ? d_.token_map[tok] : d_.pad_id;
    };
    for (int t = 0; t < T; ++t) {
        const int pos = pos0 + t;
        uint64_t rolling = 0;
        bool blocked = false;   // the sequence start or an image token is behind: pad from here on (the reference's DEAD)
        for (int s = 0; s < N; ++s) {
            if (pos < s) blocked = true;   // before the sequence start
            int32_t raw = 0;
            if (!blocked) {
                if (t - s >= 0) raw = tokens[t - s];
                else {
                    const int back = s - t;      // 1 = the newest committed token
                    if (back > (int) h.size()) throw std::runtime_error("Engram: history shorter than the n-gram");
                    raw = h[h.size() - back];
                }
                blocked = d_.dead_token >= 0 && raw == d_.dead_token;
            }
            const int64_t tok = blocked ? d_.pad_id : mapped(raw);
            const uint64_t p = (uint64_t) tok * (uint64_t) d_.multipliers[s];
            rolling = s == 0 ? p : rolling ^ p;
            if (s == 0) continue;
            for (int hd = 0; hd < d_.heads; ++hd) {
                const int b = (s - 1) * d_.heads + hd;
                ids[(size_t) t * B + b] = (int32_t) (rolling % (uint64_t) d_.primes[b] + (uint64_t) d_.offsets[b]);
            }
        }
    }
}

void Engram::worker()
{
    uint64_t seen = 0;
    for (;;) {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [&] { return stop_ || (gen_ != seen && next_ < n_jobs_); });
        if (stop_) return;
        const size_t j = next_++;
        if (next_ >= n_jobs_) seen = gen_;
        l.unlock();
        job_(j);
        l.lock();
        if (++done_ == n_jobs_) done_cv_.notify_all();
    }
}

void Engram::read_rows(const int32_t * ids, size_t n, uint8_t * dst)
{
    constexpr size_t PER_JOB = 8;
    const size_t RB = ds41::ENGRAM_Q4_ROW, rows = (size_t) table_.shape[1];
    for (size_t i = 0; i < n; ++i)   // checked here: the pool's jobs must not throw
        if (ids[i] < 0 || (size_t) ids[i] >= rows) throw std::runtime_error("Engram: row id out of range");
    const std::byte * base = table_.data;
    {
        std::lock_guard<std::mutex> l(mu_);
        job_ = [&, base] (size_t j) {
            for (size_t i = j * PER_JOB; i < std::min(n, (j + 1) * PER_JOB); ++i) {
                const std::byte * p = base + (size_t) ids[i] * RB;
                if (!gguf::read_mapped(p, RB, dst + i * RB)) std::memcpy(dst + i * RB, p, RB);
            }
        };
        n_jobs_ = (n + PER_JOB - 1) / PER_JOB, next_ = 0, done_ = 0, ++gen_;
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> l(mu_);
    done_cv_.wait(l, [&] { return done_ == n_jobs_; });
}

void Engram::forward(engine::Gpu & g, int slot, const int32_t * tokens, int pos0, int T, float * res, int res_ld)
{
    const int B = d_.buckets(), D = d_.d_model, hc = d_.hc;
    std::vector<int32_t> ids((size_t) T * B);
    hash(slot, tokens, pos0, T, ids.data());
    // image rows get no lookup: keep the other rows' ids (live, in order) and gate only those rows
    std::vector<int32_t> live;
    if (d_.dead_token >= 0)
        for (int t = 0; t < T; ++t)
            if (tokens[t] != d_.dead_token) live.push_back(t);
    const bool some_dead = d_.dead_token >= 0 && (int) live.size() < T;
    if (some_dead) {
        if (live.empty()) return;
        for (size_t i = 0; i < live.size(); ++i)
            std::memmove(ids.data() + i * B, ids.data() + (size_t) live[i] * B, sizeof(int32_t) * B);
        T = (int) live.size();
    }
    const size_t n = (size_t) T * B, RB = ds41::ENGRAM_Q4_ROW;
    if (n > pinned_rows_) {
        // the previous call's copy has finished (each call syncs before returning the rows' buffer below)
        if (pinned_) TRUSS_CUDA(cudaFreeHost(pinned_));
        TRUSS_CUDA(cudaMallocHost(&pinned_, n * RB));
        pinned_rows_ = n;
    }
    TRUSS_CUDA(cudaStreamSynchronize(g.s));   // the pinned rows of the previous call are no longer in flight
    read_rows(ids.data(), n, pinned_);
    Scratch & sc = *g.sc;
    const size_t m = sc.mark();
    uint8_t * drows = sc.alloc<uint8_t>(n * RB);
    half * x16 = sc.alloc<half>(n * d_.head_dim);
    float * kv = sc.alloc((size_t) T * (hc + 1) * D);
    int * rows = some_dead ? sc.alloc<int>(T) : nullptr;
    TRUSS_CUDA(cudaMemcpyAsync(drows, pinned_, n * RB, cudaMemcpyHostToDevice, g.s));
    if (rows) TRUSS_CUDA(cudaMemcpyAsync(rows, live.data(), sizeof(int) * T, cudaMemcpyHostToDevice, g.s));
    ds41::engram_dequant_q4(drows, (int) n, x16, g.s);
    lin_.lin(w_.wkv, x16, T, kv);
    ds41::engram_gate(res, res_ld, kv, qk_, T, hc, D, d_.eps, g.s, rows);
    if (rows) TRUSS_CUDA(cudaStreamSynchronize(g.s));   // live leaves scope with the copy maybe in flight
    sc.release(m);
}

}  // namespace truss::blocks
