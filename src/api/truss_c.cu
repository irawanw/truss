// The C API (include/truss/truss.h) over qwen4exp::Forward. The only architecture today is qwen4exp; a second one
// becomes a dispatch on general.architecture here, behind the same functions.
#include "truss/truss.h"

#include "core/cuda_check.h"
#include "formats/gguf.h"
#include "kernels/sampling/argmax.cuh"
#include "model/qwen4exp/config.h"
#include "model/qwen4exp/forward.h"
#include "model/qwen4exp/weights.h"
#include "runtime/expert_store.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <string>

using namespace truss;
namespace q = truss::qwen4exp;

struct truss_model {
    std::unique_ptr<gguf::File> file;
    q::Config config;
    q::Weights weights;
    float * logits = nullptr;                // device [n_vocab]
    int * next = nullptr;                    // device, argmax
    int n_ctx = 0, max_chunk = 0;
    std::unique_ptr<q::Forward> fwd;         // last: its expert budget takes the memory left

    ~truss_model()
    {
        fwd.reset();
        cudaFree(logits);
        cudaFree(next);
    }
};

namespace {

thread_local std::string last_error;

template <class F> auto guarded(F && f, decltype(f()) fail)
{
    try {
        return f();
    } catch (const std::exception & e) {
        last_error = e.what();
        return fail;
    }
}

// append tokens in chunks; the last chunk's last row feeds the head
void eval(truss_model & m, const int32_t * tokens, int n)
{
    if (n <= 0) throw std::invalid_argument("truss_eval: no tokens");
    if (m.fwd->position() + n > m.n_ctx)
        throw std::length_error("truss_eval: sequence of " + std::to_string(m.fwd->position() + n) +
                                " tokens exceeds n_ctx " + std::to_string(m.n_ctx));
    int last = 0;
    for (int s = 0; s < n; s += m.max_chunk) {
        last = std::min(m.max_chunk, n - s);
        m.fwd->run(tokens + s, last);
    }
    m.fwd->head(last - 1, 1, m.logits);
}

}  // namespace

extern "C" {

truss_model * truss_open(const char * gguf_path, int n_ctx, int max_chunk, const char * expert_usage)
{
    return guarded(
        [&]() -> truss_model * {
            auto m = std::make_unique<truss_model>();
            m->file = gguf::File::open(gguf_path);
            if (m->file->get_string("general.architecture") != "qwen4exp")
                throw std::runtime_error("unsupported architecture " + m->file->get_string("general.architecture"));
            m->config = q::Config::from_gguf(*m->file);
            m->weights = q::bind(*m->file, m->config);
            m->n_ctx = n_ctx;
            m->max_chunk = max_chunk / 4 * 4;
            if (m->max_chunk <= 0 || n_ctx <= 0) throw std::invalid_argument("n_ctx and max_chunk must be positive");
            TRUSS_CUDA(cudaMalloc(&m->logits, sizeof(float) * m->config.n_vocab));
            TRUSS_CUDA(cudaMalloc(&m->next, sizeof(int)));
            q::Forward::Options o;
            if (expert_usage && *expert_usage)
                o.expert_usage = runtime::ExpertStore::load_usage(expert_usage, m->config.n_layer, m->config.n_expert);
            m->fwd = std::make_unique<q::Forward>(m->config, m->weights, n_ctx, m->max_chunk, o);
            return m.release();
        },
        nullptr);
}

void truss_close(truss_model * m) { delete m; }

const char * truss_last_error(void) { return last_error.c_str(); }

int truss_n_vocab(const truss_model * m) { return m->config.n_vocab; }

int truss_n_ctx(const truss_model * m) { return m->n_ctx; }

int truss_position(const truss_model * m) { return m->fwd->position(); }

const char * truss_meta_string(const truss_model * m, const char * key)
{
    const gguf::Value * v = m->file->get(key);
    const auto * s = v ? std::get_if<std::string>(v) : nullptr;
    return s ? s->c_str() : nullptr;
}

int64_t truss_meta_int(const truss_model * m, const char * key, int64_t def)
{
    const gguf::Value * v = m->file->get(key);
    const auto * i = v ? std::get_if<int64_t>(v) : nullptr;
    return i ? *i : def;
}

int truss_reset(truss_model * m)
{
    return guarded(
        [&] {
            m->fwd->reset();
            return 0;
        },
        -1);
}

int truss_eval(truss_model * m, const int32_t * tokens, int n, float * logits)
{
    return guarded(
        [&] {
            eval(*m, tokens, n);
            if (logits) TRUSS_CUDA(cudaMemcpyAsync(logits, m->logits, sizeof(float) * m->config.n_vocab,
                                                   cudaMemcpyDeviceToHost, m->fwd->stream()));
            TRUSS_CUDA(cudaStreamSynchronize(m->fwd->stream()));
            return 0;
        },
        -1);
}

int truss_eval_argmax(truss_model * m, const int32_t * tokens, int n, int32_t * next)
{
    return guarded(
        [&] {
            eval(*m, tokens, n);
            sampling::argmax(m->logits, m->config.n_vocab, m->next, m->fwd->stream());
            TRUSS_CUDA(cudaMemcpyAsync(next, m->next, sizeof(int), cudaMemcpyDeviceToHost, m->fwd->stream()));
            TRUSS_CUDA(cudaStreamSynchronize(m->fwd->stream()));
            return 0;
        },
        -1);
}

}  // extern "C"
