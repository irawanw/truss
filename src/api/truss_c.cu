// The C API (include/truss/truss.h) over engine::Pipeline: open, eval, slots, checkpoints. Model-agnostic: the
// family comes from the file's general.architecture (src/model/registry.cc). Sampling and spec steps:
// truss_sampling.cu.
#include "api/api_internal.h"

namespace truss::api {
thread_local std::string last_error;
}
using namespace truss::api;

extern "C" {


truss_params truss_default_params(void)
{
    truss_params p{};
    p.n_ctx = 65536;
    p.max_chunk = 8192;
    p.drafts = 3;
    p.cpu_share = 0.5f;
    p.cpu_threads = 12;
    p.draft_min_p = 0.5f;
    return p;
}

truss_model * truss_open_params(const char * gguf_path, const truss_params * p)
{
    return guarded(
        [&]() -> truss_model * {
            auto m = std::make_unique<truss_model>();
            m->file = gguf::File::open(gguf_path);
            // X3.1 experts over the served GGUF (#118); a comma list adds more files (deepseek41: experts, Engram tables)
            if (const char * ov = std::getenv("TRUSS_EXPERT_OVERLAY")) m->file->overlay_list(ov);
            engine::DrafterFiles df;
            if (p->mtp && *p->mtp) {
                if (p->drafts < 1 || p->drafts > DRAFTS_MAX)
                    throw std::invalid_argument("drafts must be 1.." + std::to_string(DRAFTS_MAX));
                df.path = p->mtp;
                df.min_p = p->draft_min_p;
                if (p->draft_vocab && *p->draft_vocab) {
                    FILE * f = std::fopen(p->draft_vocab, "rb");
                    if (!f) throw std::runtime_error(std::string("cannot open ") + p->draft_vocab);
                    int32_t t;
                    while (std::fread(&t, 4, 1, f) == 1) df.vocab.push_back(t);
                    std::fclose(f);
                }
            }
            m->family = engine::open_family(*m->file, df);
            m->n_vocab = m->family->n_vocab();
            m->n_ctx = p->n_ctx;
            m->max_chunk = p->max_chunk / 4 * 4;
            if (m->max_chunk <= 0 || m->n_ctx <= 0) throw std::invalid_argument("n_ctx and max_chunk must be positive");
            engine::Options o;
            if (p->expert_usage && *p->expert_usage)
                o.expert_usage = runtime::ExpertStore::load_usage(p->expert_usage, m->family->n_layer(), m->family->n_expert());
            if (m->family->has_drafter()) {
                m->drafts = m->drafts_max = p->drafts;
                o.spec_rows = p->drafts + 1;
            }
            if (p->cpu_dir && *p->cpu_dir) {
                o.moe.cpu_dir = p->cpu_dir;
                o.moe.cpu_share = p->cpu_share;
                o.moe.cpu_threads = p->cpu_threads;
            }
            o.act = blocks::Activations::FP16;   // served default: 16-bit dense activations (KL 0.040 vs 0.065); TRUSS_ACT=q8 for the old 8-bit path
            engine::apply_env(o);   // the bench's decode-tier knobs (TRUSS_CPU_TRELLIS, TRUSS_PCIE_FRAC, TRUSS_HINT_K, ...)
            // drop the shards' file pages once the weights are on the device (a ~47 GB resident peak otherwise, TRACKER #72)
            o.after_upload = [&] {
                m->file->release_pages();
            };
            // device buffers before the engine: its expert budget takes the memory left
            TRUSS_CUDA(cudaMalloc(&m->logits, sizeof(float) * m->n_vocab * (m->drafts + 1)));
            TRUSS_CUDA(cudaMalloc(&m->next, sizeof(int)));
            TRUSS_CUDA(cudaMalloc(&m->spec_next, sizeof(int) * (m->drafts + 1)));
            TRUSS_CUDA(cudaMallocHost(&m->spec_host, sizeof(int) * (m->drafts + 1)));
            if (const char * e = std::getenv("TRUSS_SPEC_SAMPLE")) m->spec_sample = std::atoi(e) != 0 && m->drafts > 0;
            const int n_slots = std::max(1, std::getenv("TRUSS_SLOTS") ? std::atoi(std::getenv("TRUSS_SLOTS")) : 1);
            if (m->spec_sample) {   // 3 MB a slot at 3 drafts
                m->q_full.assign(n_slots, nullptr);
                for (float *& qf : m->q_full) TRUSS_CUDA(cudaMalloc(&qf, sizeof(float) * m->n_vocab * m->drafts));
                TRUSS_CUDA(cudaMalloc(&m->spec_drafts, sizeof(int) * m->drafts));
            }
            const size_t rows = m->drafts + 1;   // penalties: 4 MB of counts at 3 drafts
            TRUSS_CUDA(cudaMalloc(&m->pen_cnt, sizeof(int) * rows * m->n_vocab));
            TRUSS_CUDA(cudaMemset(m->pen_cnt, 0, sizeof(int) * rows * m->n_vocab));
            TRUSS_CUDA(cudaMalloc(&m->pen_hist, sizeof(int) * rows * TRUSS_PENALTY_CAP));
            TRUSS_CUDA(cudaMallocHost(&m->pen_host, sizeof(int) * rows * TRUSS_PENALTY_CAP));
            m->fwd = std::make_unique<engine::Pipeline>(*m->family, m->n_ctx, m->max_chunk, o);
            m->n_ctx = m->fwd->seq_ctx();   // TRUSS_SLOTS: every limit below is per sequence
            const int ns = m->fwd->slots();
            if (m->spec_sample && ns != n_slots) throw std::logic_error("TRUSS_SLOTS: slot count mismatch");
            m->counter.assign(ns, 0);
            m->prog_done.reset(new std::atomic<int>[ns]);
            m->prog_total.reset(new std::atomic<int>[ns]);
            for (int i = 0; i < ns; ++i) m->prog_done[i] = 0, m->prog_total[i] = 0;
            m->file->release_pages();
            return m.release();
        },
        nullptr);
}

truss_model * truss_open(const char * gguf_path, int n_ctx, int max_chunk, const char * expert_usage)
{
    truss_params p = truss_default_params();
    p.n_ctx = n_ctx, p.max_chunk = max_chunk, p.expert_usage = expert_usage;
    return truss_open_params(gguf_path, &p);
}

void truss_close(truss_model * m) { delete m; }

const char * truss_last_error(void) { return last_error.c_str(); }

int truss_n_vocab(const truss_model * m) { return m->n_vocab; }

int truss_n_ctx(const truss_model * m) { return m->n_ctx; }

int truss_position(const truss_model * m) { return m->fwd->position(m->cur); }

int truss_position_s(const truss_model * m, int slot) { return m->fwd->position(slot); }

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

int truss_reset_s(truss_model * m, int slot)
{
    return guarded(
        [&] {
            check_slot(*m, slot);
            m->fwd->reset(slot);
            // A fresh sequence starts a fresh random stream. Every sampled row draws from (seed, counter), so without
            // this a run's samples depend on how many sampled rows earlier runs consumed.
            m->counter[slot] = 0;
            return 0;
        },
        -1);
}

int truss_reset(truss_model * m) { return truss_reset_s(m, m->cur); }

void truss_progress_s(const truss_model * m, int slot, int * done, int * total)
{
    if (slot < 0 || slot >= (int) m->counter.size()) {
        *done = *total = 0;
        return;
    }
    *done = m->prog_done[slot].load(std::memory_order_relaxed), *total = m->prog_total[slot].load(std::memory_order_relaxed);
}

void truss_progress(const truss_model * m, int * done, int * total) { truss_progress_s(m, m->cur, done, total); }

int truss_eval_s(truss_model * m, int slot, const int32_t * tokens, int n, float * logits)
{
    return guarded(
        [&] {
            eval(*m, slot, tokens, n, [&] {
                if (logits) TRUSS_CUDA(cudaMemcpyAsync(logits, m->logits, sizeof(float) * m->n_vocab,
                                                       cudaMemcpyDeviceToHost, m->fwd->stream()));
                sync(*m);
            });
            return 0;
        },
        -1);
}

int truss_eval(truss_model * m, const int32_t * tokens, int n, float * logits) { return truss_eval_s(m, m->cur, tokens, n, logits); }

namespace {
engine::ImageIn image_in(const float * embeds, int n_img, int cols, int img_row0, int nh, int nw)
{
    if (n_img <= 0 || nh <= 0 || nw <= 0 || nh * nw != n_img || cols <= 0)
        throw std::invalid_argument("truss_eval_img: bad image grid/width");
    return engine::ImageIn{ embeds, img_row0, n_img, cols, nh, nw };
}
}  // namespace

int truss_eval_img_s(truss_model * m, int slot, const int32_t * tokens, int n, const float * embeds, int n_img,
                     int cols, int img_row0, int nh, int nw, float * logits)
{
    return guarded(
        [&] {
            const engine::ImageIn img = image_in(embeds, n_img, cols, img_row0, nh, nw);
            eval(*m, slot, tokens, n, [&] {
                if (logits) TRUSS_CUDA(cudaMemcpyAsync(logits, m->logits, sizeof(float) * m->n_vocab,
                                                       cudaMemcpyDeviceToHost, m->fwd->stream()));
                sync(*m);
            }, 0, &img);
            return 0;
        },
        -1);
}

int truss_eval_img_argmax_s(truss_model * m, int slot, const int32_t * tokens, int n, const float * embeds,
                            int n_img, int cols, int img_row0, int nh, int nw, int32_t * next)
{
    return guarded(
        [&] {
            const engine::ImageIn img = image_in(embeds, n_img, cols, img_row0, nh, nw);
            eval(*m, slot, tokens, n, [&] {
                sampling::argmax(m->logits, m->n_vocab, m->next, m->fwd->stream());
                TRUSS_CUDA(cudaMemcpyAsync(next, m->next, sizeof(int), cudaMemcpyDeviceToHost, m->fwd->stream()));
                sync(*m);
            }, 0, &img);
            return 0;
        },
        -1);
}

int truss_eval_step_s(truss_model * m, int slot, const int32_t * tokens, int n, int step, float * logits)
{
    return guarded(
        [&] {
            eval(*m, slot, tokens, n, [&] {
                if (logits) TRUSS_CUDA(cudaMemcpyAsync(logits, m->logits, sizeof(float) * m->n_vocab,
                                                       cudaMemcpyDeviceToHost, m->fwd->stream()));
                sync(*m);
            }, step);
            return 0;
        },
        -1);
}

int truss_eval_argmax_s(truss_model * m, int slot, const int32_t * tokens, int n, int32_t * next)
{
    return guarded(
        [&] {
            eval(*m, slot, tokens, n, [&] {
                sampling::argmax(m->logits, m->n_vocab, m->next, m->fwd->stream());
                TRUSS_CUDA(cudaMemcpyAsync(next, m->next, sizeof(int), cudaMemcpyDeviceToHost, m->fwd->stream()));
                sync(*m);
            });
            return 0;
        },
        -1);
}

int truss_eval_argmax(truss_model * m, const int32_t * tokens, int n, int32_t * next)
{
    return truss_eval_argmax_s(m, m->cur, tokens, n, next);
}

int truss_drafts(const truss_model * m) { return m->drafts; }

int truss_set_option(truss_model * m, const char * name, double value)
{
    return guarded(
        [&] {
            const std::string k = name ? name : "";
            if (k == "drafts") {
                if (value < 1 || value > m->drafts_max)
                    throw std::invalid_argument("drafts must be 1.." + std::to_string(m->drafts_max) + " (the open value)");
                m->drafts = (int) value;
                return 0;
            }
            if (!m->fwd->set_moe_option(k, value)) throw std::invalid_argument("truss_set_option: not a runtime knob: " + k);
            return 0;
        },
        -1);
}

int truss_wait_ready(truss_model * m)
{
    return guarded([&] { m->fwd->wait_pinned(); return 0; }, -1);
}

void truss_expert_counts(const truss_model * m, long long * out6) { m->fwd->expert_counts(out6); }

int truss_slots(const truss_model * m) { return m->fwd->slots(); }

void truss_lock_ms(truss_model * m, double * out4, int reset) { m->fwd->lock_ms(out4, reset != 0); }

/* TRUSS_ROUND_TRACE=<file> (track U): the interval log is process-global. dump flushes what is buffered; mark
   records a phase boundary (the tag is the harness's). Both do nothing when the variable is unset. */
void truss_round_trace_dump(void) { rt::dump(); }
void truss_round_trace_mark(int tag) { rt::mark(tag); }

int truss_select(truss_model * m, int slot)
{
    return guarded(
        [&] {
            check_slot(*m, slot);
            m->fwd->select(slot);
            m->cur = slot;
            return 0;
        },
        -1);
}

int truss_checkpoint_s(truss_model * m, int slot)
{
    return guarded([&] { check_slot(*m, slot); m->fwd->checkpoint(slot); return 0; }, -1);
}

int truss_checkpoint(truss_model * m) { return truss_checkpoint_s(m, m->cur); }

int truss_restore_s(truss_model * m, int slot)
{
    return guarded([&] { check_slot(*m, slot); return m->fwd->restore(slot); }, -2);
}

int truss_restore(truss_model * m) { return truss_restore_s(m, m->cur); }

}  // extern "C"
