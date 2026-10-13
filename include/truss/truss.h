/* TRUSS C API: one model, one sequence, on one GPU. The server (server/) and other languages bind to this.
 *
 * A model holds the sequence's state (caches, recurrent state). truss_eval appends tokens to the sequence (a prompt,
 * or one decoded token) and returns the next-token logits of the last one; truss_reset starts a new sequence.
 * Errors: functions return a negative value or NULL and truss_last_error() says why (per thread).
 */
#ifndef TRUSS_H
#define TRUSS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct truss_model truss_model;

/* n_ctx: longest sequence; max_chunk: tokens per GPU pass of a prompt (larger = faster prefill, more memory).
   expert_usage: NULL, or a file of routed counts [layer][expert] float32 (flashnext_truss_usage.py); with it the
   most-used experts stay in VRAM, which sets decode speed. */
truss_model * truss_open(const char * gguf_path, int n_ctx, int max_chunk, const char * expert_usage);

/* Every load option. Start from truss_default_params() and set what you need. */
typedef struct truss_params {
    int n_ctx;                  /* longest sequence (default 65536) */
    int max_chunk;              /* prompt tokens per GPU pass (default 8192) */
    const char * expert_usage;  /* routed counts [layer][expert] float32 (tk-profile): the hot set; NULL: index order */
    const char * mtp;           /* MTP draft block GGUF (flashnext_truss_mtp_pack.py): enables truss_spec_step */
    int drafts;                 /* MTP drafts per speculative round, 1..7 (default 3) */
    const char * cpu_dir;       /* CPU expert tier (q4s files; needs expert_usage); NULL: off */
    float cpu_share;            /* share of each layer's non-resident routing mass on the CPU (default 0.5) */
    int cpu_threads;            /* CPU tier threads (default 12) */
    const char * draft_vocab;   /* int32 token ids the MTP draft head scores (data/draft_vocab_en.bin); NULL: all */
    float draft_min_p;          /* drafts stop below this MTP probability (default 0.5; 0: always draft `drafts`) */
} truss_params;
truss_params truss_default_params(void);
truss_model * truss_open_params(const char * gguf_path, const truss_params * params);
void truss_close(truss_model * m);
const char * truss_last_error(void);

int truss_n_vocab(const truss_model * m);
int truss_n_ctx(const truss_model * m);
int truss_position(const truss_model * m);          /* tokens in the sequence so far */
/* progress of the eval running on another thread: tokens run so far / tokens asked (updated per chunk; 0/0 before
   the first eval). Safe to call while an eval runs. */
void truss_progress(const truss_model * m, int * done, int * total);

/* model file metadata: string value (NULL if absent or not a string), integer value (def if absent) */
const char * truss_meta_string(const truss_model * m, const char * key);
int64_t truss_meta_int(const truss_model * m, const char * key, int64_t def);

int truss_reset(truss_model * m);

/* append tokens[0 .. n) and write the last token's next-token logits (n_vocab floats) to logits (NULL: skip) */
int truss_eval(truss_model * m, const int32_t * tokens, int n, float * logits);

/* the same, but only the argmax of those logits (greedy decoding without the logits copy) */
int truss_eval_argmax(truss_model * m, const int32_t * tokens, int n, int32_t * next);

/* Greedy speculative decoding (opened with params.mtp). `next` is the token after the sequence (e.g. from
   truss_eval_argmax). One round: MTP drafts after `next`, one verify pass over [next, drafts], keep the matching
   prefix. Appends emitted[0 .. *n_emitted) to the sequence (emitted[0] == next, then the accepted drafts; at most
   truss_drafts() + 1 tokens) and returns the greedy token after them in *new_next. The output equals plain greedy
   decoding token for token. */
int truss_spec_step(truss_model * m, int32_t next, int32_t * emitted, int * n_emitted, int32_t * new_next);
int truss_drafts(const truss_model * m);            /* 0: opened without MTP */

/* Sequence slots (TRUSS_SLOTS=n at open): the model holds n independent sequences, each with truss_n_ctx()
   positions (n_ctx / n: the K/V memory is shared out, not added). truss_select makes one current; every other call
   (eval, spec steps, reset, checkpoint, position) acts on it. Switch only between calls. Slot 0 is current at open. */
int truss_slots(const truss_model * m);
/* diagnostics: host ms each layer-split stage was held / waited for since the last reset: [front held, front wait,
   last held, last wait] */
void truss_lock_ms(truss_model * m, double * out4, int reset);
/* TRUSS_ROUND_TRACE=<file>: the engine writes an interval log (per stage: host waits, lock hand-offs, and the
   device-busy spans of a layer-split round) for track U's idle attribution. Process-global, and nothing happens
   when the variable is unset. dump flushes the buffer; mark records a phase boundary the reader can key on. */
void truss_round_trace_dump(void);
void truss_round_trace_mark(int tag);
int truss_select(truss_model * m, int slot);

/* The same calls naming their slot. Calls on DIFFERENT slots may run on different threads at once: the engine holds
   each layer-split stage only while one slot's work is enqueued there, so the two GPUs work on two conversations at
   the same time. Calls on one slot must not overlap. */
int truss_position_s(const truss_model * m, int slot);
void truss_progress_s(const truss_model * m, int slot, int * done, int * total);
int truss_reset_s(truss_model * m, int slot);
int truss_eval_s(truss_model * m, int slot, const int32_t * tokens, int n, float * logits);
/* truss_eval_s in prompt chunks of at most `step` tokens (a chunk holds each GPU for its whole pass: smaller chunks let
   another slot's rounds through more often, at some prompt speed) */
int truss_eval_step_s(truss_model * m, int slot, const int32_t * tokens, int n, int step, float * logits);
int truss_eval_argmax_s(truss_model * m, int slot, const int32_t * tokens, int n, int32_t * next);
int truss_spec_step_s(truss_model * m, int slot, int32_t next, int32_t * emitted, int * n_emitted, int32_t * new_next);
int truss_checkpoint_s(truss_model * m, int slot);
int truss_restore_s(truss_model * m, int slot);

/* Sequence checkpoint (one slot, pinned host memory): truss_checkpoint saves the sequence's recurrent state at
   truss_position(); truss_restore returns the sequence to it and returns that position (-1: none taken). A later
   prompt that extends only the checkpointed prefix then reads only its own tail. */
int truss_checkpoint(truss_model * m);
int truss_restore(truss_model * m);

/* Sampling on the device (kernels/sampling/sample.cuh): temperature > 0; top_p >= 1, top_k <= 0, min_p <= 0 are off.
   Each sampled row draws from its own random stream (seed, a per-model counter that every sampled row advances).
   Penalties (kernels/sampling/penalty.cuh, llama.cpp's formulas, applied before the filters): repetition_penalty 1,
   frequency_penalty 0 and presence_penalty 0 are off. They count `history` (n_history token ids, host memory, read
   during the call only), of which each row keeps the last penalty_last_n (<= 0 or > TRUSS_PENALTY_CAP: the cap). */
#define TRUSS_PENALTY_CAP 4096
typedef struct truss_sampling {
    float temperature;
    float top_p;
    int top_k;
    float min_p;
    uint64_t seed;
    float repetition_penalty;
    float frequency_penalty;
    float presence_penalty;
    int penalty_last_n;
    const int32_t * history;
    int n_history;
    /* Tokens no sampled row may produce (n_ban <= 8 ids, host memory, read during the call; 0: off), e.g. the
       end-of-turn tokens while the reply's reasoning block is open. ban_lift >= 0 ends the ban at that token: a row
       whose fed tokens (this call's, up to and including the row's own) contain ban_lift is not banned. -1: every
       row. Zero-initialised callers (n_ban 0) are unaffected. */
    const int32_t * ban;
    int n_ban;
    int32_t ban_lift;
} truss_sampling;

/* truss_eval, then a sample of the last token's next-token distribution in *next. The penalties count `history`
   as given: the caller includes `tokens` in it when they should count. */
int truss_eval_sample(truss_model * m, const int32_t * tokens, int n, const truss_sampling * sp, int32_t * next);

/* truss_spec_step for sampled decoding (Strata's sampled verify): every verify row is sampled with `sp`; drafts are
   accepted while the row's sample equals the draft, and the first row that differs gives *new_next. Each emitted
   token is a sample of the model's own distribution given the tokens before it, whatever the drafts were. Penalties:
   `history` is what counts before `next`; verify row t also counts `next` and drafts 1..t (Strata's per-row rule). */
int truss_spec_step_sampled(truss_model * m, int32_t next, const truss_sampling * sp, int32_t * emitted,
                            int * n_emitted, int32_t * new_next);
int truss_eval_sample_s(truss_model * m, int slot, const int32_t * tokens, int n, const truss_sampling * sp,
                        int32_t * next);
int truss_spec_step_sampled_s(truss_model * m, int slot, int32_t next, const truss_sampling * sp, int32_t * emitted,
                              int * n_emitted, int32_t * new_next);

#ifdef __cplusplus
}
#endif

#endif
