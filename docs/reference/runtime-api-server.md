# Runtime, C API and server: src/runtime/, include/truss/ + src/api/, server/

---

## `src/runtime/expert_store.h`, `expert_store.cu` — which experts live where

**Problem.** Flash-Next's routed experts are 37 GiB; after dense weights, caches and scratch a 3090 has ~14–16 GiB
for them. The rest (cold, ~24 GB) must come over PCIe (13.5 GB/s pinned, x8) when used.

**Layout.** One device arena per projection (gate, up, down):

```
[hot experts of layers 0..23][slot 0][slot 1][hot experts of layers 24..47]
                              ^ base (ProjView.trellis for every layer)
```

Hot experts sit at fixed offsets (negative for the first half). Layer l's cold experts always land at the same place
inside slot l % 2 (packed in expert order), so **the per-layer meta tables (K, word offset from base) are built once
and never rewritten**, and the MoE kernels need no change. Offsets are int32 words (±4 GiB of base); the loader
throws if one overflows. Cold experts are copied once at load into pinned host memory, per (layer, projection),
contiguous in slot order.

**API.**

| call | does |
|---|---|
| `ExpertStore::plan(layers, budget, usage = {})` | hot set whose bytes (hot + 2 slots + scales + meta) fit `budget`. Without usage: the first N experts of every layer (index order). With usage (routed counts [layer][expert]): every layer first gets its top `floor` experts by count per byte, then greedy by count per byte over all layers; the floor (swept in steps of 8) with the most hot usage wins — a layer with few hot experts would make both slots large |
| `ExpertStore::load_usage(path, n_layer, n_expert)` | reads a usage file: n_layer · n_expert float32, layer-major (`flashnext_truss_usage.py` writes it from a routing capture) |
| `ExpertStore(layers, hot)` | builds arenas, pinned cold copies, meta tables; uploads suh/svh (all resident) |
| `weights(l)` | `moe::Weights` for layer l |
| `prefetch(l)` | copy all of layer l's cold experts into slot l%2 on the store's copy stream, after the slot's last user released it |
| `fetch(l, ids, n)` | copy only the cold experts among `ids` (host ints; deduplicated) into their slot places |
| `acquire(l, compute)` | compute stream waits for layer l's copy |
| `release(l, compute)` | records that compute is done with slot l%2 |
| `device_bytes()`, `cold_bytes()` | memory used; bytes streamed per full pass |

**Protocol (Forward).** Long chunk: `prefetch(0)`, `prefetch(1)` at chunk start; for each layer `acquire(l)` → MoE
kernel → `release(l)` → `prefetch(l + 2)`. Short chunk (≤ 32 rows): routing to host → `fetch(l, ids)` → `acquire` →
kernel → `release`. Events per slot keep copies and compute ordered across chunks and modes.

**Measured.** Full model, 4K chunk: 204/512 experts hot, 23.6 GB streamed per chunk, PCIe 1.75 s vs compute ~1.8 s,
overlapped → 2,263 tok/s (TRACKER #52). Streamed == resident bit-exact (`qwen4exp_forward stream`).

**Limits / next.** The two slots are sized for whole-layer prefill streaming (~1.2 GB at 150 hot/layer) and are
dead weight in decode, so a usage-ranked set fits fewer experts than it could (TRACKER #56). The fetch path is
synchronous per layer (routing must be read back first).

---

## `include/truss/truss.h`, `src/api/truss_c.cu` — C API (`libtruss.so`)

One `truss_model` = one model file, one sequence, on the current CUDA device (`CUDA_VISIBLE_DEVICES` selects it).

| function | contract |
|---|---|
| `truss_open(gguf, n_ctx, max_chunk, expert_usage)` | load (architecture must be `qwen4exp`); NULL on error. `max_chunk` rounded down to a multiple of 4. `expert_usage`: NULL or a usage file (`ExpertStore::load_usage`) for the hot set |
| `truss_close(m)` | free everything |
| `truss_last_error()` | message of the last failure on this thread |
| `truss_n_vocab`, `truss_n_ctx`, `truss_position` | sizes; tokens in the sequence |
| `truss_meta_string(m, key)` / `truss_meta_int(m, key, def)` | GGUF metadata (chat template, stop ids, ...) |
| `truss_reset(m)` | new sequence |
| `truss_eval(m, tokens, n, logits)` | append n tokens (split into `max_chunk` chunks); write the last token's next-token logits (n_vocab floats; NULL skips) |
| `truss_eval_argmax(m, tokens, n, &next)` | same, greedy token on the device (no logits copy) |

Returns 0 / pointer on success, −1 / NULL on failure (every C++ exception is caught at the boundary). The logits
buffer is allocated before the engine so the expert budget sees the remaining memory. A second architecture becomes a
dispatch on `general.architecture` inside `truss_open`.

---

## `server/` — OpenAI-compatible HTTP server (Python)

| file | role |
|---|---|
| `truss_ctypes.py` | `Model(gguf, n_ctx, max_chunk)`: ctypes signatures for every C function, `eval(tokens) → logits` (numpy view reused per call), `eval_argmax`, `reset`, `meta_string/int`, `position`. ctypes releases the GIL during calls |
| `chat.py` | `Chat(tokenizer_json, template)`: HF `tokenizers` tokenizer (NFC-normalizing like training; llama.cpp skips NFC — the only difference found on real text), chat template from the GGUF rendered by transformers `apply_chat_template`; `Detokenizer` (text deltas, holds back incomplete UTF-8); `sample()` (temperature, top-k, top-p; numpy) |
| `app.py` | `Engine` (model + the sequence in it + lock) and the FastAPI app |

**Endpoints.** `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (messages, tools, `max_tokens`,
`temperature`, `top_p`, `top_k`, `seed`, `stop`, `stream`, `chat_template_kwargs` e.g. `enable_thinking`),
`POST /v1/completions` (raw prompt). With thinking on (template default) the text up to `</think>` is returned as
`reasoning_content`, the rest as `content` (`ThinkSplitter` in streaming, holds back a possible partial tag).

**Engine.generate.** One request at a time (lock). Prefix reuse: if the new prompt starts with every token already in
the engine's sequence, only the new tokens are evaluated; otherwise reset (the GDN state cannot rewind). Greedy
(`temperature ≤ 0`) uses `eval_argmax`; sampling copies logits. Stops on the GGUF EOS id, `<|im_end|>`,
`<|endoftext|>`, a stop string, `max_tokens`, or n_ctx. After each request it logs one line (stdout and `--log`):

```
<log-tag>: prompt N tokens = R reused + P read in X ms (S tok/s), M generated in Y ms (G tok/s), drafts accepted 0 of 0
```

This is the format `flashnext_strata_bench.py` parses; run the server with `--log-tag "strata serve"` and the bench
runs unchanged against it.

**Run.** `CUDA_VISIBLE_DEVICES=2 python3 -m server.app --model <gguf shard 1> --tokenizer <tokenizer.json>
[--n-ctx 65536] [--chunk 8192] [--expert-usage file] [--port 8080] [--log file] [--log-tag tag]`. Usage file in use:
`~/ML_projects/flashnext/20260930_truss_tg/data/usage_chatcode64.f32` (32K chatcode tokens). The Flash-Next tokenizer.json in use:
`/data/www/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/models/Qwen3.8-27B-EXL3-3.5bpw/tokenizer.json` (same 248,077 tokens as
the GGUF; checked token-for-token against llama-tokenize on 7.5K tokens of code with special tokens).

**Known limits.** Single sequence; no rewind (multi-turn chats whose history re-renders differently start over);
decode 13.6–15.5 tok/s (bring-up; the speed work is CP4/CP5).
