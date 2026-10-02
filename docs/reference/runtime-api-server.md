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
| `fetch(l, ids, n, compute)` | ring mode: makes the cold experts among `ids` (host) resident: FIFO allocation (256-B aligned, wrap at the end, evicting the oldest), one copy each, ring meta updated; waits for queued compute first (the ring may overwrite what earlier kernels read). Refetches if a copy evicted an expert this call needs (up to `need + 1` rounds: each re-copied expert lands at the head; a fixed 3-round bound failed on a 4-row verify window, 10-01). Throws outside ring mode. `compute` = nullptr skips the wait (doorbell: the doorbell already orders it) |
| `begin_ring()` | switch to ring mode (ring starts empty); called on the thread that enqueues kernels, before `weights()`, because with a doorbell driver `fetch` runs later on another thread (a mode switch there raced with kernel enqueue, TRACKER #61) |
| `fetch` return value | true when it queued a copy (ring entries or meta tables) or needs an expert that `prefetch_hint` queued for this layer since its last fetch (`hinted_`, the copy may still be in flight); false = every id already on the device, nothing to wait for (the doorbell's `wait_plan` then skips `go`) |
| `admit(le, compute)` / `poll_admitted()` | Strata's adaptive tier (TRACKER #78), between decode passes: copy the cold experts `le` into the ring (FIFO eviction) on the copy stream after `compute`'s queued work; evicted experts leave the device at once, admitted ones are pending (`Layer::pend`, `on_device` false) until the batch's event completes (`poll_admitted`, also called by `fetch`, which treats a pending id like a hinted one). One batch in flight at a time |
| `signal(l, flag, seq)` / `upload(dst, src, bytes)` | queue, behind every copy issued so far, a write of `seq` to device word `flag` (the doorbell's "go"; sources come from a 1,024-entry pinned word ring since a queued copy reads its source when it runs) / a small pinned copy on the copy stream |
| `prefetch_hint(l, ids, n)` | ring mode, right after `fetch(l)`: starts copying layer l+1's predicted cold experts behind layer l's copies (updates l+1's ring meta). Refuses to evict an expert layer l's fetch needs (stops the hint instead) |
| `acquire(l, compute)` / `release(l, compute)` | compute waits for layer l's copies / records it is done with slot l % 2 |
| `stats()` | ring mode: fetch calls, experts asked, fetched on demand (misses, bytes), prefetched by hints (count, bytes) |
| `device_bytes()`, `cold_bytes()`, `ring_bytes()` | memory used; bytes streamed per prompt chunk; ring size |

**Protocol (Forward).** Prompt chunk (> 32 rows): `begin_stream`, `prefetch(0)`, `prefetch(1)`; per layer
`acquire(l)` → `moe::prefill` → `release(l)` → `prefetch(l + 2)`. Decode step / window (≤ 32 rows): per layer
routing to host → `fetch(l, ids, n, s)` → `acquire` → `moe::window` (≤ 8 rows) or `moe::prefill`.

**Doorbell decode (`src/runtime/doorbell.{cuh,cu}`, Forward `Options::doorbell` = true, TRACKER #61).** Strata's
design: the host enqueues a whole pass ahead and never synchronizes per layer. Per decode layer the GPU runs
`publish` (copies routing ids, the next layer's predicted ids, weights and the FFN input rows into mapped pinned host
memory, then raises the layer's doorbell word = the pass's seq), then `spin_until(go, seq)` (a one-thread kernel
holding the stream). A driver thread in `Forward` serves the layers in enqueue order. It spins on its job counter
(`jobs_pushed`, up to 200,000 pauses before the condvar) and on the doorbell, then:
1. starts the CPU pool;
2. calls `fetch(l, ids, n, nullptr)`, which returns whether it queued a copy;
3. only if it did, `signal(go)`;
4. writes the mapped words `need_copy` (= seq when copies were queued) and `plan` (= seq);
5. calls `prefetch_hint`, waits for the pool, and raises `cpu_done`.

The GPU runs `wait_plan`: it spins on the mapped `plan`, also on `go` when `need_copy == seq`, and copies the masked
ids from mapped memory to the device. Then come the expert kernel, `spin_until(cpu_done)` and `add_mapped` of the CPU
rows. Flags only increase, so nothing is reset between passes. Output is the same as the synchronous path
(`tk-bench-spec ... sync`).

This handoff is Strata's: mapped flags and plan, no CUDA call between a layer's routing and its expert kernel when no
copy is needed (TRACKER #77). Before, every layer waited for a `go` written by the copy stream behind its masked-id
upload, ring-meta uploads and the previous layer's hint copies.

`fetch` counts an expert hinted for the layer since its last fetch as a copy (`hinted_`): that copy may still be in
flight. The go word must precede the hint copies (queued after them it cost 49 → 41 tok/s).

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
| `truss_default_params()`, `truss_open_params(gguf, &params)` | every load option: `n_ctx`, `max_chunk`, `expert_usage`, `mtp` (draft block GGUF → speculative decoding), `drafts` (1..7), `cpu_dir` / `cpu_share` / `cpu_threads` (CPU tier). `truss_open` = defaults + its three arguments |
| `truss_spec_step(m, next, emitted, &n, &new_next)` | one greedy speculative round after `next`: drafts, one verify pass, accept the matching prefix; appends `emitted[0..n)` (`next` first, then the accepted drafts) and returns the greedy token after them. Output equals plain greedy decoding. Near n_ctx it falls back to one plain step |
| `truss_drafts(m)` | drafts per round (0: no MTP) |
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
| `truss_ctypes.py` | `Model(gguf, n_ctx, max_chunk, expert_usage, mtp, drafts, cpu_dir, cpu_share, cpu_threads)` (the `truss_params` struct), `spec_step(next) → (tokens appended, next token)`, ctypes signatures for every C function, `eval(tokens) → logits` (numpy view reused per call), `eval_argmax`, `reset`, `meta_string/int`, `position`. ctypes releases the GIL during calls |
| `chat.py` | `Chat(tokenizer_json, template)`: HF `tokenizers` tokenizer (NFC-normalizing like training; llama.cpp skips NFC — the only difference found on real text), chat template from the GGUF rendered by transformers `apply_chat_template`; `Detokenizer` (text deltas, holds back incomplete UTF-8); `sample()` (temperature, top-k, top-p; numpy) |
| `app.py` | `Engine` (model + the sequence in it + lock) and the FastAPI app |

**Endpoints.** `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (messages, tools, `max_tokens`,
`temperature`, `top_p`, `top_k`, `seed`, `stop`, `stream`, `chat_template_kwargs` e.g. `enable_thinking`),
`POST /v1/completions` (raw prompt). With thinking on (template default) the text up to `</think>` is returned as
`reasoning_content`, the rest as `content` (`ThinkSplitter` in streaming, holds back a possible partial tag).

**Engine.generate.** One request at a time (lock). Prefix reuse: if the new prompt starts with every token already in
the engine's sequence, only the new tokens are evaluated; otherwise reset (the GDN state cannot rewind). Greedy
(`temperature ≤ 0`) uses `eval_argmax` after the prompt and then, with `--mtp`, speculative rounds (`spec_step`),
emitting each round's tokens in order (a stop token or string inside a round ends the reply; the engine keeps the
round's later tokens, and prefix reuse tracks them); sampling copies logits. Stops on the GGUF EOS id, `<|im_end|>`,
`<|endoftext|>`, a stop string, `max_tokens`, or n_ctx. After each request it logs one line (stdout and `--log`):

```
<log-tag>: prompt N tokens = R reused + P read in X ms (S tok/s), M generated in Y ms (G tok/s), drafts accepted 0 of 0
```

This is the format `flashnext_strata_bench.py` parses; run the server with `--log-tag "strata serve"` and the bench
runs unchanged against it.

**Run.** `CUDA_VISIBLE_DEVICES=2 python3 -m server.app --model <gguf shard 1> --tokenizer <tokenizer.json>
[--n-ctx 65536] [--chunk 8192] [--expert-usage file] [--mtp file] [--drafts 3] [--cpu-dir dir] [--cpu-share 0.5]
[--cpu-threads 12] [--port 8080] [--log file] [--log-tag tag]`. Usage file in use:
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
expert) → h = silu(gate)·up. Phase 1 items: down rows in chunks of 256 (10 per expert); h is quantized once per group at the phase 0 → 1 flip (AVX2 `quantize`, bit-identical to the scalar loop). fp16 block scales are preconverted to fp32 at load (`Q4Matrix::df`, `preconvert_scales`). Idle workers spin `TRUSS_CPU_SPIN_US` µs before blocking (see the trellis section). Then y[row] =
Σ over the row's slots in slot order of w·out.

**Numerics.** x and h quantized per 32-block to int8 (d = amax/127, as the GPU's Q8_1), int32 dots (AVX2 maddubs),
fp32 scaling. Rows are independent and summed in slot order, so a verify window equals single steps
(`cpu_expert_test`: rel 2e-7 vs a double reference; each row bit-identical to its 1-row call).

**In Forward (Options::cpu_dir or cpu_trellis, cpu_share, cpu_threads, cpu_dynamic).** At load, per layer, the non-resident experts in ascending
usage until they hold `cpu_share` of the layer's non-resident routing mass are the CPU set (static: results never
depend on cache state); their q4s bytes are read into RAM. Decode layers (fetch mode): the FFN input rows and routing
weights come to the host in the routing sync; CPU slots go to the pool; the GPU kernel gets the same routing with
those slots masked (weight 0, id of an expert already in the row, so no extra work); the CPU sum is added to the
routed output before the shared-expert combine. Prompt chunks (stream mode) compute every expert on the GPU from the
trellis pack. The MTP block has no CPU tier.

`Options::cpu_dynamic` (Strata's split, `serve()`): every expert with a CPU copy is eligible; per layer, routed experts
already on the GPU (hot, or in the ring incl. queued copies: `ExpertStore::on_device`) run there, and of the missed
eligible ones the last m in routing order go over PCIe, the rest to the CPU, m minimizing max(CPU time, copy time)
with the CPU's time for n experts = `cpu_ms_call + cpu_ms_expert·n` (a least-squares line through the pool's own call
times, `ExpertPool::last_call_ms`, exponentially weighted with decay 0.98; the slope keeps its last value while the
calls are too alike in n to fit) and copies at `pcie_gbps`. A flat per-expert mean charged each call's ~0.1 ms of
wake/join to its ~1 expert and starved the CPU (TRACKER #75). Doorbell path
only. Results then depend on the cache state; with `cpu_trellis` the CPU's weights equal the GPU's, so only fp32
summation order differs.

**Speed.** The tier is DRAM-bound: 16 threads read q4s at 46.6 GB/s, 84% of this box's roofline; instruction cuts were neutral and more threads slower (TRACKER #62, #64). Its per-layer join is the cost: a verify pass at share 0.7 spent 32 of 75 ms in it (#63). `cpu_expert_test` timing: 4 rows × 8 experts in ~0.93 ms with 8 threads + caller (converter running
beside it). Probe (`tools/tk-bench/cpu_q4.cc`): 8 threads 10.3 experts/ms at 4 rows, 16 threads 16.8.

**Pool scheduling.** Items are taken lock-free from one atomic ticket = phase << 20 | item index; a phase's last item
prepares what the next phase reads (q4s: quantize h; trellis: nothing, its items prepare their own inputs) and then publishes the next phase's
ticket. Before 10-01 the phase and the index were two atomics: a worker could read the old phase, take index 0 of the
new one after a flip, run the wrong item and retire it against the old phase, so a pending count never reached zero
and the call hung (seen with the trellis tier's 3 phases; `cpu_trellis_test` now runs 3,000 small calls).

## `src/cpu/expert_trellis.h`, `expert_trellis.cc` — CPU GEMV from the pack's own trellis bytes

**Why (TRACKER #73).** Strata's CPU computes missed experts from the very bytes the GPU would copy, so its CPU tier
costs no RAM. The q4s tier above needs a second copy (2.76 MB per CPU expert beside the pinned 1.77 MB): 28 GB at
share 0.15, which swapped once the renters grew (they hold 58-77 GB of the 125 GB). `Options::cpu_trellis` points the
CPU at the ExpertStore's pinned pack copies instead (`ExpertStore::host_part`): no file, no extra RAM, and the CPU
computes the GPU's own weights.

**Math.** Per projection a = fp16(H128(suh·x)), c = W·a with W decoded from 16×16 tiles, y = svh·H128(c) — the
reference `qwen4exp::project` and `moe::window`. Weight j of a tile is the 16-bit state ending at bit (j+1)·K of the
tile's cyclic 256·K-bit stream (MSB first), value bytesum(state · 0x83DCD12D) + 1024 as fp16 times 1/147.7 − 10.39.

**Kernel (K ≤ 4, `gemv_tiles4`).** Each nt column's tiles are byte-swapped once into a small buffer (the stream then
reads as big-endian bytes; the tile's last word in front for lane 0). For weight index m the 8 lanes of an octet sit at
byte offsets K·l + const with one shared bit shift, so a pair of 16-byte loads, one `vpshufb` and two immediate shifts
give 8 states; then `vpmulld`, byte sum (`maddubs` + `madd`), convert, FMA to the codebook value, one FMA per row
into the octet's accumulators (activations pre-permuted by `trellis_prep`). K is a template parameter (shifts become
immediates). K 5-6 use the scalar-window kernel `gemv_tiles`. The weights are not rounded to fp16 by default (the
GPU's hfma2 does): rounding cost ~40% of the loop on Zen 2 and moves a weight by at most half an fp16 ulp
(`TRUSS_TRELLIS_ROUND=1` restores it).

**Measured** (`cpu_trellis_test`, K = 2, renters at loadavg ~27): 4.8 cycles per 8 weights at 1 row (first version
8.6: scalar windows, F16C rounding); 0.80 ms per expert on one thread from cache; 12 threads over experts from DRAM
8.5-9.6 experts/ms; the pool on a 4-row call with 37 distinct experts 9.6 experts/ms (PCIe moves ~7.6/ms). Rel. error
vs a dense double reference from the bit-by-bit decode 1-9e-6 (K 1-4, 1-8 rows); the pool equals per-slot experts
exactly and each row equals its 1-row call.

**Kernel (K ≤ 4, default since 10-01: `gemv_i16`, TRACKER #76).** Measured on this Zen 2 per vector op
(`ub` probe, 8 independent chains): `vpmulld` 2.3 cycles; every other integer multiply (`vpmullw`, `vpmulhuw`,
`vpmaddubsw`, `vpmaddwd`) and every shift 1 cycle on one pipe; and/or/add/`vpshufb` several per cycle. `gemv_tiles4`
spends ~4.3 multiply-pipe cycles per 8 weights, which is its 4.8. `gemv_i16` does 16 weights (two m of an octet) per
step:
- states: m's 32-bit window also holds m+1's state (ms + K + 16 ≤ 32), shifted down and up and `vpblendw`-ed:
  16 states in 16-bit lanes from one `vpshufb`;
- hash: s·0x83DCD12D mod 2³² from 16-bit products (low half `mullo(s, 0xD12D)`, high half
  `mulhi(s, 0xD12D) + mullo(s, 0x83DC)`, exact because s < 2¹⁶);
- byte sum: low half by `vpmaddubsw`, high half by and/shift. This is the measured balance between the multiply pipe
  and the ALUs: all-ALU 3.4, both `maddubs` 3.25, mixed 2.8 cycles per 8 weights;
- dot: one `vpmaddwd` of the byte sums (0..1020) against int16 activations into int32 accumulators, flushed to fp32
  every 8 slices (overflow bound 1.1e9).

The codebook's affine part is a per-row term: Σ wᵢaᵢ = kinv·Σ bytesumᵢ·aᵢ + (1024·kinv + kbias)·Σ aᵢ.

Activations are int16 with one scale per prepared row (amax/32767) taken from the fp32 Hadamard output; through the
pool a row is prepared per input block (below), so the scale is per block, which is finer. Weight m and m+4 read
the same inputs, so a prepared slice is the 8 inputs a[16kt+8v ..] written twice for v = 0, 1, then the scale and Σa.
The row stride is still `trellis_prep_floats(in, 1)`.

Error vs exact activations is 2.3-3.2e-5, against 1.7-2.2e-4 for the GPU's own fp16 activations, so about 8x lower.
`cpu_trellis_test` passes a kernel that is within 1e-5 of the fp16 reference (`gemv_tiles4`) or no further from exact
than the GPU is (`gemv_i16`). Slice engine test: CPU-tier KL vs all-resident 2.39e-4 (was 3.35e-4).

Memory order: blocks of 8 slices outer, columns inner, so each block reads contiguous runs of tiles (k-slice major),
with software prefetch one block ahead. Walking a column at a stride of NT tiles (3.8 KB for gate/up) left one thread
at 1.0 ms/expert from DRAM against 0.55 from cache.

Measured (`cpu_trellis_test`, K 3, loadavg ~23):
- 0.55 ms/expert on one thread from cache, 0.68 from DRAM (was 0.83 / 1.0);
- 12 threads over DRAM experts: 12.5-15.6 experts/ms (was 8.1-9.0);
- pool latency for 1 row × 4 experts: 0.29 ms at 12 threads, 0.19 at 23 (was 0.43-0.46).

`TRUSS_TRELLIS_I16=0` (or `TRUSS_TRELLIS_ROUND=1`) selects `gemv_tiles4`. K 5-6 keep `gemv_tiles`.

**Pieces of a product** (for the pool's input-split items, TRACKER #82). `trellis_inputs(W, i0, i1)` is W over inputs
[i0, i1) (multiples of 128): tiles are k-slice major, so it is a pointer offset (+ `suh` offset), and its bytes are one
contiguous run; prepare it with `trellis_prep` on x + i0. `trellis_gemv_raw` writes the raw product c = W·a before the
output Hadamard and svh; partial products over input blocks add up to the whole one. `trellis_out` applies
svh·H128 to columns [c0, c1) held in a buffer (in place Hadamard). `trellis_gemv` = raw + out per 512 columns (same
results as before).

**In the pool.** Slots carry `t` (a `TrellisExpert`) instead of `e`; one call is all one kind. Items are
input-split (TRACKER #82), three phases:
- 0: (expert, gate | up, input block of `TRUSS_CPU_TR_GU_IN` = 256 inputs): prepare the rows over that block only and
  write the raw partial product, all 640 columns (10 items per matrix, 20 per expert);
- 1: (expert, h block of 128, `TRUSS_CPU_TR_DN_COLS` = 640 columns of down): rebuild the h block from the gate/up
  partials (sum in block order, svh·H128 — the output Hadamard is per 128-block, so the block stands alone —
  silu(gate)·up), prepare it, write down's raw partial over it (20 items per expert);
- 2: (128 output columns): for every slot, down's partials summed in block order, svh·H128, added into y with the
  slot's weight in slot order (20 items).

No serial step: the old pool prepared gate/up per expert (phase 0 had G items) and h for down at the flip on one
thread, and read 128-column items that take 768 B of every 3,840 B slice row. D0 measured those as 40 µs of a 211 µs
two-expert call plus 1.25x thread time per item (`flashnext/20261002_truss_d0`). Pool probe, 22 pinned threads, K 3,
1 row: 1 / 2 / 4 / 8 experts per call 0.167 / 0.188 / 0.310 / 0.600 → 0.10 / 0.118 / 0.221 / 0.425 ms. Summation
orders are fixed, so results do not depend on scheduling; error vs exact 4.8e-5 (one scale per row: 6.2e-5; the
GPU's fp16 activations 3.8e-4). Engine (spec, share 0.2, 22 pinned threads): CPU wait 8.4-8.9 → 5.6-5.8 ms/pass,
68.6-79.7 → 77.9-83.6 tok/s. Slice CPU-tier KL vs all-resident 2.09e-4.
`last_call_ms()` is the call's own time (start to its last item), which the dynamic split averages instead of the
driver thread's start-to-wait time (that included issuing the PCIe copies and overstated the CPU's cost).
Idle workers spin `TRUSS_CPU_SPIN_US` µs (default 20,000, Strata's `kSpinBeforeSleep`) and block only when the
spin times out. `start()`, the phase flips and the end of a call take no lock: they store the state (seq_cst) and
call `wake()`, which locks and notifies only when `sleepers_ > 0`. The caller spins in `wait()`.

Until TRACKER #77, a worker that left its spin because items opened re-checked `open()` and slept when the others
had already taken every item. Most of the pool fell asleep each phase, and each `start()` paid futex wakes:
- 0.2 ms per call in the engine, 13.9 ms per pass;
- after the fix, 0.2 ms per pass.

Decode at share 0.2 went from 42-46 to 57-61 tok/s. Pool probe (`pw`, 1-expert calls 0.2-5 ms apart): 0.30-0.82 →
0.12 ms per call, zero wakes.
`TRUSS_CPU_PIN=1` pins worker i to physical core i + 1 (one logical CPU per core from sysfs, as Strata's pool);
off by default, neutral in `cpu_trellis_test` at 6-23 threads under renter load (TRACKER #76).
Row stride of the prepared activations is `trellis_prep_floats(in, 1)` for every kernel (the K 5-6 kernel read rows
at 2·in, wrong for multi-row calls through the pool, which preps rows one at a time).
