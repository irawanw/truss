# Runtime, C API and server: src/runtime/, include/truss/ + src/api/, server/

---

## `src/runtime/expert_store.h`, `expert_store.cu` — which experts live where

**Problem.** Flash-Next's routed experts are 37 GiB; after dense weights, caches and scratch a 3090 has ~13–16 GiB
for them. The rest (cold, ~24 GB) must come over PCIe (13.5 GB/s pinned, x8) when used.

**Tiers.** *hot*: a static usage-ranked set, resident for the run. *ring*: one device region that is either decode's
FIFO cache of recently fetched cold experts, or, during a prompt chunk, two whole-layer stream slots plus the chunk's
own large buffers (`spare()`: Forward's prefill scratch, `moe::prefill` workspace, residual). *host*: every cold
expert in pinned memory, per layer `[expert: gate | up | down]` in expert order.

**Layout.** One device arena:

```
[hot experts, layers 0..23][ring: slot 0 | slot 1 | spare ... ][hot experts, layers 24..47]
                            ^ base = ProjView.trellis of every projection, ProjView.shift = 4
```

Each expert's three projections are contiguous (one copy per fetched expert). Meta offsets count 32-byte units
(`shift` 4), so int32 reaches ±64 GiB of base. Two meta tables per (layer, projection): the **stream** table (cold
experts at their fixed place in slot l % 2, built once) and the **ring** table (rewritten by `fetch()` from a pinned
host mirror, one 4 KB copy per projection per layer that changed). `weights(l)` returns the current mode's table.

**API.**

| call | does |
|---|---|
| `ExpertStore::plan(layers, budget, Sizes{ring_bytes, stream_extra}, usage = {})` | hot set whose `device_bytes` fit `budget`; ring = max(ring_bytes, 2 × largest cold layer + stream_extra). Without usage: the first N experts of every layer. With usage (routed counts [layer][expert]): every layer first gets its top `floor` experts by count per byte, then greedy by count per byte over all layers; the floor (steps of 8) with the most hot usage wins |
| `ExpertStore::load_usage(path, n_layer, n_expert)` | usage file: n_layer · n_expert float32, layer-major (`tk-profile` or `flashnext_truss_usage.py` writes it) |
| `ExpertStore(layers, hot, sizes)` | arena, pinned cold copies, both meta tables, suh/svh |
| `weights(l)` | `moe::Weights` for layer l in the current mode |
| `begin_stream(compute)` | a prompt chunk starts: waits for queued compute, drops the ring's contents, stream mode |
| `spare()` | stream mode: `stream_extra` bytes after the slots, valid until the next `fetch()` |
| `prefetch(l)` | stream mode: all of layer l's cold experts into slot l % 2 (one copy), after the slot's release |
| `fetch(l, ids, n, compute)` | ring mode: makes the cold experts among `ids` (host) resident: FIFO allocation (256-B aligned, wrap at the end, evicting the oldest), one copy each, ring meta updated; waits for queued compute first (the ring may overwrite what earlier kernels read). Refetches if a copy evicted an expert this call needs |
| `prefetch_hint(l, ids, n)` | ring mode, right after `fetch(l)`: starts copying layer l+1's predicted cold experts behind layer l's copies (updates l+1's ring meta). Refuses to evict an expert layer l's fetch needs (stops the hint instead) |
| `acquire(l, compute)` / `release(l, compute)` | compute waits for layer l's copies / records it is done with slot l % 2 |
| `stats()` | ring mode: fetch calls, experts asked, fetched on demand (misses, bytes), prefetched by hints (count, bytes) |
| `device_bytes()`, `cold_bytes()`, `ring_bytes()` | memory used; bytes streamed per prompt chunk; ring size |

**Protocol (Forward).** Prompt chunk (> 32 rows): `begin_stream`, `prefetch(0)`, `prefetch(1)`; per layer
`acquire(l)` → `moe::prefill` → `release(l)` → `prefetch(l + 2)`. Decode step / window (≤ 32 rows): per layer
routing to host → `fetch(l, ids, n, s)` → `acquire` → `moe::window` (≤ 8 rows) or `moe::prefill`.

**Why a ring (TRACKER #57).** On held-out chatcode routing a FIFO of recently fetched experts halves the misses of a
static set of equal size (10,000 slots: 74 → 38 per token; FIFO within 3% of LRU;
`flashnext_truss_cache_sim.py`). Letting prompt chunks borrow the ring for their buffers moved ~3 GB of
prefill-only memory into the cache.

**Measured.** Full model, 4K chunk prefill: 2,263 tok/s (TRACKER #52, before the ring). Decode (real code prompt,
64K context, chunk 8192, profile `usage_code_truss8.f32`): 5,743 hot + 4.6 GB ring, 94 fetches (165 MB) per token,
33 tok/s (#57). Bit-exact vs all resident incl. ring wraps (`qwen4exp_forward stream`).

**Limits / next.** Fetch is synchronous per layer (routing read back to the host); misses cost PCIe time on the
critical path (~9.5 ms per token at 94 misses). Next: MTP windows, a CPU tier for misses.

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
decode 33 tok/s at #57 (TRACKER #56+ for the speed work).

---

## `src/cpu/expert_q4.h`, `expert_q4.cc` — CPU tier for routed experts (decode)

**Why.** Decode misses cost PCIe bytes (x8, 13.5 GB/s): ~200 MB per generated token with the 2.63-bpw pack, far
above what 100+ tok/s allows (TRACKER #59). This box's CPU reads RAM at ~50 GB/s even beside the renters, so the rarest
experts are computed on the host instead. Their 4-bit copies come from the BF16 originals: weight NMSE ~0.009 per
matrix, about the pack's K4 trellis (0.0097) and far below the K1/K2 (0.375/0.100) the pack gives rare experts — the
tier lowers error as well as adding miss bandwidth.

**Format q4s** (`flashnext_truss_cpu_q4.py`, one `L<nn>.q4s` per layer, experts in order, `EXPERT_BYTES` = 2,764,800
each): per matrix [rows][cols] (out × in), blocks of 32 along the input: 16 bytes of nibbles (element i in the low
nibble of byte i for i < 16, the high nibble of byte i − 16 otherwise; value (nibble − 8)·d) and one fp16 d per
block (chosen by squared error among amax/7 × {0.85 … 1.1}). Expert = gate q, gate d, up q, up d (640 × 2560), down
q, down d (2560 × 640).

**API.** `expert_view(bytes)` → `Q4Expert`; `ExpertPool(threads)`; `start(x [T][2560], T ≤ 8, slots {row, expert,
w}, y [T][2560])` returns at once, `wait()` blocks (the caller works on items meanwhile); `run` = both.

**Work split.** Slots are grouped by expert. Phase 0 items: an expert's gate and up rows in chunks of 64 (10 per
expert) → h = silu(gate)·up. Phase 1 items: h quantized, down rows in chunks of 256 (10 per expert). Then y[row] =
Σ over the row's slots in slot order of w·out.

**Numerics.** x and h quantized per 32-block to int8 (d = amax/127, as the GPU's Q8_1), int32 dots (AVX2 maddubs),
fp32 scaling. Rows are independent and summed in slot order, so a verify window equals single steps
(`cpu_expert_test`: rel 2e-7 vs a double reference; each row bit-identical to its 1-row call).

**In Forward (Options::cpu_dir, cpu_share, cpu_threads).** At load, per layer, the non-resident experts in ascending
usage until they hold `cpu_share` of the layer's non-resident routing mass are the CPU set (static: results never
depend on cache state); their q4s bytes are read into RAM. Decode layers (fetch mode): the FFN input rows and routing
weights come to the host in the routing sync; CPU slots go to the pool; the GPU kernel gets the same routing with
those slots masked (weight 0, id of an expert already in the row, so no extra work); the CPU sum is added to the
routed output before the shared-expert combine. Prompt chunks (stream mode) compute every expert on the GPU from the
trellis pack. The MTP block has no CPU tier.

**Speed.** `cpu_expert_test` timing: 4 rows × 8 experts in ~0.93 ms with 8 threads + caller (converter running
beside it). Probe (`tools/tk-bench/cpu_q4.cc`): 8 threads 10.3 experts/ms at 4 rows, 16 threads 16.8.
