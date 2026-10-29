// The C API's internal state and helpers, shared by its implementation files: truss_c.cu (open, eval, slots,
// checkpoints) and truss_sampling.cu (device sampling, speculative steps).
#pragma once
#include "truss/truss.h"

#include "core/cuda_check.h"
#include "formats/gguf.h"
#include "kernels/sampling/argmax.cuh"
#include "kernels/sampling/penalty.cuh"
#include "kernels/sampling/sample.cuh"
#include "kernels/sampling/spec.cuh"
#include "engine/model.h"
#include "engine/options.h"
#include "engine/pipeline.h"
#include "engine/round_trace.h"
#include "runtime/expert_store.h"

#include <atomic>
#include <functional>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace truss;
namespace rt = truss::engine::rt;

struct truss_model {
    std::unique_ptr<gguf::File> file;
    std::unique_ptr<engine::Family> family;
    int drafts = 0;
    int drafts_max = 0;                      // the drafts the buffers were sized for (truss_set_option may lower drafts)
    int * spec_next = nullptr, * spec_host = nullptr;   // device / pinned argmax of each verify row
    bool spec_sample = false;                // TRUSS_SPEC_SAMPLE=1: sampled drafts + min(1, p/q) acceptance
    std::vector<float *> q_full;             // per slot, device [drafts][n_vocab]: the sampled drafts' q
    int * spec_drafts = nullptr;             // device [drafts]: the window's guesses (used under the last stage's hold)
    int n_vocab = 0;
    float * logits = nullptr;                // device [n_vocab]
    int * next = nullptr;                    // device, argmax
    int n_ctx = 0, max_chunk = 0;
    // Per slot (TRUSS_SLOTS): the random stream of the next sampled row, and the running eval's progress (tokens run /
    // tokens asked, truss_progress). The device buffers above and below are shared: they are only used while the
    // engine holds its last stage for one slot (Forward::prompt's after(), Forward::round's pick()).
    std::vector<uint64_t> counter;
    std::unique_ptr<std::atomic<int>[]> prog_done, prog_total;
    // truss_select(): the slot of the calls without one. Another thread reads it (the legacy wrappers of a server
    // running slots) while truss_select writes it (review H2).
    std::atomic<int> cur{ 0 };
    int * pen_cnt = nullptr;                 // device [drafts + 1][n_vocab] penalty counts, zero between calls
    int * pen_hist = nullptr, * pen_host = nullptr;   // device / pinned [drafts + 1][TRUSS_PENALTY_CAP] row histories
    int * word_cls = nullptr;                // device [n_vocab] word class (truss_set_word_classes), host copy below
    std::vector<int> word_cls_host;
    std::vector<std::vector<int>> filler_seqs;   // truss_set_filler_seqs
    std::unique_ptr<engine::Pipeline> fwd;         // last: its expert budget takes the memory left

    ~truss_model()
    {
        fwd.reset();
        cudaFree(logits);
        cudaFree(next);
        cudaFree(spec_next);
        for (float * q : q_full) cudaFree(q);
        cudaFree(spec_drafts);
        if (spec_host) cudaFreeHost(spec_host);
        cudaFree(pen_cnt);
        cudaFree(pen_hist);
        if (pen_host) cudaFreeHost(pen_host);
        cudaFree(word_cls);
    }
};

namespace truss::api {


// A verify window is `next` plus at most DRAFTS_MAX drafts, so the window buffers on the stack in the spec steps
// have room for any configuration truss_open_params() accepts (the engine's own cap is moe::MAX_ROWS rows).
constexpr int DRAFTS_MAX = 7;
constexpr int WINDOW_ROWS = DRAFTS_MAX + 1;

extern thread_local std::string last_error;   // defined in truss_c.cu

template <class F> auto guarded(F && f, decltype(f()) fail)
{
    try {
        return f();
    } catch (const std::exception & e) {
        last_error = e.what();
        return fail;
    }
}

inline void check_slot(const truss_model & m, int slot)
{
    if (slot < 0 || slot >= (int) m.counter.size())
        throw std::invalid_argument("truss: slot " + std::to_string(slot) + " of " + std::to_string(m.counter.size()));
}

// append tokens in chunks; the last chunk's last row feeds the head (m.logits), then after() runs while the engine
// still holds the last stage for this slot (it reads m.logits)
inline void eval(truss_model & m, int slot, const int32_t * tokens, int n, const std::function<void()> & after,
                 int step_max = 0, const engine::ImageIn * img = nullptr, int n_img = 1)
{
    check_slot(m, slot);
    if (n <= 0) throw std::invalid_argument("truss_eval: no tokens");
    if (m.fwd->position(slot) + n > m.n_ctx)
        throw std::length_error("truss_eval: sequence of " + std::to_string(m.fwd->position(slot) + n) +
                                " tokens exceeds n_ctx " + std::to_string(m.n_ctx));
    // A short prompt runs as decode-sized chunks (fetch path: only the experts its rows route to move), not as a
    // prompt chunk that streams every cold expert layer (~2.3 s at 256K whatever the length; TRACKER #89).
    const int fetch_max = std::getenv("TRUSS_FETCH_PROMPT") ? std::atoi(std::getenv("TRUSS_FETCH_PROMPT")) : 0;
    int step = n <= fetch_max ? std::min(m.max_chunk, engine::Pipeline::fetch_rows()) : m.max_chunk;
    if (step_max > 0) step = std::min(step, std::max(4, step_max / 4 * 4));   // a smaller chunk (truss_eval_step_s)
    m.prog_done[slot] = 0, m.prog_total[slot] = n;
    // in chunks with the PLE lookahead (the next chunk's trigram rows are read while this chunk computes); a layer
    // split runs its stages a chunk apart (Pipeline::prompt)
    m.fwd->prompt(slot, tokens, n, step, m.logits, after, [&](int done) { m.prog_done[slot] = done; }, img, n_img);
}

inline void sync(truss_model & m) { TRUSS_CUDA(cudaStreamSynchronize(m.fwd->stream())); }

// truss_eval_imgs_*: the 1-D images of a call (truss_c.cu)
std::vector<engine::ImageIn> images_in(const float * embeds, int cols, int n_img, const int * rows0, const int * lens);

}  // namespace truss::api
