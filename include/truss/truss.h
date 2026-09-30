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
void truss_close(truss_model * m);
const char * truss_last_error(void);

int truss_n_vocab(const truss_model * m);
int truss_n_ctx(const truss_model * m);
int truss_position(const truss_model * m);          /* tokens in the sequence so far */

/* model file metadata: string value (NULL if absent or not a string), integer value (def if absent) */
const char * truss_meta_string(const truss_model * m, const char * key);
int64_t truss_meta_int(const truss_model * m, const char * key, int64_t def);

int truss_reset(truss_model * m);

/* append tokens[0 .. n) and write the last token's next-token logits (n_vocab floats) to logits (NULL: skip) */
int truss_eval(truss_model * m, const int32_t * tokens, int n, float * logits);

/* the same, but only the argmax of those logits (greedy decoding without the logits copy) */
int truss_eval_argmax(truss_model * m, const int32_t * tokens, int n, int32_t * next);

#ifdef __cplusplus
}
#endif

#endif
