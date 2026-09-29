# 04 — Engine architecture ("Loom")

Loom is a decode engine for EXL3-format trellis models (PAW X3 GGUF and EXL3 safetensors) on
Ampere GeForce cards: RTX 3090 (GA102, 82 SMs) and RTX 3060 (GA106, 28 SMs), both sm_86.

## 1. Goals and non-goals

**Goals**

1. Serve PAW-27B-X3.1 (and later EXL3-format models of the same family) on one 3090 with the
   **same accuracy** as llama-paw, faster than every existing engine on the same file.
2. **Exact speculative decoding**: drafted output byte-identical to serial decoding, greedy and
   seeded-sampled.
3. **No host on the per-token path**: the GPU never waits for the CPU inside a step or a
   speculative round.
4. A codebase one person can hold in their head: one weight format, one GPU architecture family,
   one model family at a time, and a small fixed set of fused ops.
5. Room to grow: a new GPU arch, codec rate or model family is a local addition, not a rewrite.

**Non-goals (explicit)**

- General model support, generic tensor graphs, CPU inference, training.
- Non-trellis weight formats, except the Q4_K embedding and Q5_K head that PAW files carry.
- Tensor parallelism over PCIe (no P2P on this box [S13]).
- Hopper/Blackwell features. They may come later as new kernel variants behind the same interfaces.

## 2. What "zero bottleneck, zero latency" means here, measurably

There is always a bottleneck. The design target is that **the only ones left are the unavoidable
ones**: DRAM reads of weights, KV and state, plus the trellis decode ALU work. Measured by:

| metric | target | how it is measured |
|---|---|---|
| GPU idle inside a decode step | ≤ 2% of wall | nsys `--cuda-graph-trace=node`, decode window |
| host time per speculative round | ≤ 0.2 ms | engine timers, the host thread's critical section |
| launches per token (AR) | ≤ ~450 (about one per fused op) | nsys |
| trellis matmul throughput | ≥ 1.15× the best existing kernel at K3.5, per shape | `tools/tk-bench` op level |
| allocations after load | 0 | allocator counters, asserted in debug builds |

## 3. Layers

```
┌───────────────────────────────────────────────────────────────┐
│ server (Python, thin): OpenAI API, tokenizer, chat template,    │  not on the per-token path
│ tool/reasoning parsers, streaming, cancel                       │
└──────────────┬────────────────────────────────────────────────┘
               │ nanobind: submit(request) / pop(tokens) through a lock-free ring
┌──────────────▼────────────────────────────────────────────────┐
│ runtime (C++): engine thread, scheduler, sequences, spec loop │
│   step executor → CUDA graphs per row bucket; device scalars  │
├───────────────────────────────────────────────────────────────┤
│ model/<family> (C++): builds the step as a fixed list of fused │
│   ops with static buffers (qwen35_dense first)                 │
├───────────────────────────────────────────────────────────────┤
│ spec/: dflash2, mtp, ngram drafters; window policy; accept     │
├───────────────────────────────────────────────────────────────┤
│ plan/: shape-keyed kernel plans, per-arch kernel registry,     │
│   measured tune cache (like exllamav3's autotune, keyed on M)  │
├───────────────────────────────────────────────────────────────┤
│ kernels/: trellis, attention, gdn, norm, head, sampling        │
│   each with a slow reference implementation for tests          │
├───────────────────────────────────────────────────────────────┤
│ formats/: gguf reader, exl3 safetensors reader, repacker,      │
│   reference dequant (bit-exact oracle)                         │
├───────────────────────────────────────────────────────────────┤
│ core/: arena allocator, streams/events, graph capture/replay,  │
│   device-scalar block, logging, config/env, timers             │
└───────────────────────────────────────────────────────────────┘
```

**Why C++ for the engine and Python only for the server:** a 27B token takes ~18–20 ms on a 3090.
TensorFold measured 12–14 ms of Python host time for a 918-kernel forward on GB10 [02 §2]. On a
3090 that would not stay hidden. The server's per-request work (tokenize, template, parse) is not
per-token, so Python is fine there, and it keeps the HTTP/tool-parsing code small and easy to
change.

## 4. Weights: formats and the repack

- **Readers:** `formats/gguf` (PAW `paw-dense` arch: `*.m3_trellis` I16, `*.m3_suh`, `*.m3_svh`,
  Q4_K `token_embd`, Q5_K `output`) and `formats/exl3` (safetensors: `trellis`, `suh`, `svh`,
  `mul1` marker, `quantization_config.json`). Both produce one **tensor catalog**:
  `{name, shape, K (1..8 or x.5), codebook, byte span}`.
- **K from tile size:** 16·K uint16 per 16×16 tile for integer K; 16·K + 8 for half-integer K
  (the exllamav3 rule).
- **Repack (lossless):** at load, trellis words are permuted into the order the kernel reads
  (lever A1). The repack is a pure function `repack(catalog entry) → device layout`, with an
  inverse used by the tests. The file on disk never changes, so any tool that reads GGUF/EXL3
  still works.
- **Reference dequant:** `formats/reference` decodes any tile to fp16 exactly as exllamav3's
  `reconstruct` does (cross-checked against the exl3xpu `ref.py` idea, MIT). This is the oracle
  for every kernel test.

## 5. Kernel tiers (trellis linear)

| rows M | kernel | notes |
|---|---|---|
| 1..16 | `trellis_gemv_ri` (new) | row-invariant, repacked layout, multi-stage `cp.async`, no cooperative launch, Hadamard-in prologue and svh epilogue fused, fixed-order split-K reduction. Variants per K ∈ {2, 2.5, 3, 3.5, 4}; multi-output variant for QKV+gate and gate+up |
| 17..1023 | `trellis_gemm_tc` | port of exllamav3's cooperative tensor-core GEMM with `half_k`; fp16 accumulate with fp32 fold |
| ≥ 1024 | `reconstruct` + cuBLAS fp16-accumulate | measured best for prefill [S18c] |

The plan (`plan/`) picks the tier and config per `(arch, K, k, n, M-bucket)` from a measured tune
cache. The tune key **includes M**; llama-paw keyed on `(bits, k, n)` only [S4] §3.

**Row-invariance contract:** for a fixed weight matrix, the output row `r` depends only on input
row `r`. The K split, the per-split reduction order and the epilogue are functions of `(k, n, K)`
only, never of M. The test is in §9.

## 6. One decode step (dense Qwen3.5/3.8 family), as fused ops

Per layer, AR or verify with M rows:

```
GDN layer (48×):
  1  trellis_gemv_ri  [rmsnorm+suh+H prologue]  → qkv (10240) + z/gate (6144)        (multi-output)
  2  gdn_decode_fused   conv1d state update, L2 norm q/k, α/β (small F32 GEMVs), delta-rule
                       recurrence on 128×128 state per head, gated RMSNorm            (one kernel)
  3  trellis_gemv_ri  [H+suh prologue] ssm_out, [svh+H+residual epilogue]
  4  trellis_gemv_ri  [rmsnorm prologue] ffn gate+up (multi-output), [SiLU·up epilogue]
  5  trellis_gemv_ri  ffn_down, [residual epilogue]
Attention layer (16×):
  1  trellis_gemv_ri  q (+gate), k, v   (multi-output), fused q/k RMSNorm + RoPE + KV append
  2  flash_decode_gqa  split-K over the context, 6 q heads per K/V load, q8_0 KV
  3  trellis_gemv_ri  o_proj [residual epilogue]
  4-5 FFN as above
Head:  rmsnorm → Q5_K head GEMV (port of the llama-paw MMQ override) → device argmax or
       Gumbel-keyed sample
```

That is about 5 launches per layer → **~330 launches per token**, against 1987 today [S3].
Each step is captured once per row bucket as a CUDA graph and replayed with device scalars
(positions, lengths, page table).

## 7. The speculative round, on the device

```
           ┌──────────── one graph (or a conditional WHILE body) ─────────────┐
host ──▶   │ draft (DFlash2 / MTP / n-gram) → verify forward (M = window)     │ ──▶ pinned ring
 (seed,    │ → per-row sample (argmax or keyed Gumbel) → accept prefix        │     of committed
  limits)  │ → commit KV pages + GDN state for accepted positions             │     tokens
           │ → catch-up drafter KV (KV-only) → next draft                      │
           └──────────────────────────────────────────────────────────────────┘
```

- **Proposals are trees or copy chains, not only chains.** A best-first tree over DFlash2's top-16
  lattice (~12–16 rows), or, when an 8-gram of the context matches, a verbatim copy chain of up
  to 31 rows. Verify runs the whole tree in one forward: per-node RoPE positions, attention over
  committed keys plus the node's own path, the GDN recurrence along each node's path (a node's
  state = its parent's state + one update), and conv1d over the node's path.
- **GDN state after a round — replay, not checkpoints:** verify records each node's small GDN
  inputs (k, v, g, β per head). The commit replays only the accepted path into the cached state,
  in one launch across all 48 GDN layers, with the same update order as serial steps (TensorFold
  `replay_many_kernel`). No per-node state copies are needed. Storing a full state per node would
  cost 151 MB per node (arithmetic: 12 nodes ≈ 1.8 GB).
- **Exactness:** row-invariant kernels plus keyed sampling mean the token at position p is the
  same whether p was drafted or decoded alone. The test is in §9.
- The host thread only moves tokens to the server and admits or cancels requests.

## 8. Code layout

```
trellis-kernel/
  CMakeLists.txt            CUDA 12.6 (the box's pinned toolkit), C++20, sm_86 default
  include/loom/             public C++ API (engine.h, request.h, config.h)
  src/
    core/                   arena.cc, graph.cc, stream.cc, dscalar.cc, timer.cc, log.cc
    formats/                gguf.cc, exl3.cc, catalog.cc, repack.cc, reference_dequant.cc
    kernels/
      trellis/              codec_<name>.cuh (one per codec: tile format + codebook), mma.cuh,
                            hadamard.cuh, gemv_ri.cu,
                            gemm_tc.cu, reconstruct.cu, hadamard.cuh
      moe/                  moe_window.cu (routed experts of a verify window, TRUSS)
      attention/            flash_decode_gqa.cu, prefill_fa.cu, kv_quant.cu
      gdn/                  gdn_decode_fused.cu, gdn_prefill_chunked.cu
      norm/                 rmsnorm.cu (standalone, for places a prologue cannot take it)
      head/                 q5k_head.cu, hot_vocab_head.cu, topk.cu
      sampling/             argmax.cu, gumbel_keyed.cu, accept.cu
      reference/            slow, exact CUDA/C++ references for every op above
    plan/                   registry.cc (arch → kernel variants), plan_cache.cc, tune_store.cc
    model/qwen35_dense/     weights.cc, step.cc (the fused-op list), prefill.cc
    spec/                   dflash2.cc, mtp.cc, ngram.cc, window_policy.cc
    runtime/                engine.cc, scheduler.cc, sequence.cc, kv_pages.cc, gdn_ckpt.cc
    bindings/               nanobind module for the server and tests
  server/                   Python: app.py (OpenAI API), chat.py, tools.py, stream.py
  tools/
    tk-bench/               op-level microbench at real shapes, warm-clock protocol
    tk-ablate/              base / no-decode / no-load / skeleton variants of any kernel
    tk-parity/              teacher-forced logits, KL vs Q8_0 dump, greedy hash, spec==serial
    tk-repack/              offline repack check and layout dump
    gguf2exl3/              PAW GGUF → EXL3 exporter (Phase 0)
  tests/
    unit/                   decode bit-exactness, repack round-trip, row invariance, per-op refs
    layer/                  one layer vs llama-paw dumps
    e2e/                    greedy hash vs golden, spec == serial, 256k fit
  docs/                     this folder; decisions/ holds dated decision records (D0, D1, ...)
  bench/results/            JSON results keyed by git sha + GPU + clocks, never hand-edited
```

## 9. Tests and measurement (built before the kernels)

**Correctness tests (CI on every commit, on whichever GPU is free):**

1. `decode_bitexact`: for every tile of all 400 X3.1 matrices, repacked-kernel decode ==
   `reference_dequant`, bit for bit.
2. `row_invariance`: for each kernel and M ∈ {1..16}, row r's output bits are identical when the
   other M−1 rows are random, and identical to M = 1.
3. `op_reference`: each fused op vs its slow reference, within a tolerance documented per op
   (bit-equal where the class is E0).
4. `layer_parity`: a layer's outputs vs llama-paw dumps (`GGML_PAW_*` dump hooks exist).
5. `spec_equals_serial`: 20 prompts × {greedy, seed 1234 sampled}; drafted output bytes ==
   serial output bytes.
6. `kl_gate` (not per commit; per milestone): chatcode/rawcode 64×2048 KL vs Q8_0, paired
   bootstrap vs llama-paw X3.1 (protocol and seed from [S15]).

**Measurement protocol (every speed number), taken from this project's own failures:**

- Interleave arms (A, B, A, B …), never A then B [S18c].
- Discard the first run after idle; report median and spread, not one number [S1] §8.
- Log SM clock, memory clock and temperature at each launch.
- nsys always with `--cuda-graph-trace=node` [S3].
- Counters: not available on this box (`RmProfilingAdminOnly=1`, no sudo); ablate instead (TRACKER rule 15).
- Label every figure as **measured** (with its command and artifact path) or **estimate**
  (with its arithmetic). An estimate is never repeated as a result.
- GPU use follows the renter rules: flock plus watchdog, and only the GPU the user names.

## 10. How the codebase grows

- **New GPU arch** (e.g. sm_89): add kernel variants under `kernels/*` and register them in
  `plan/registry.cc` for that arch. The model and runtime layers do not change.
- **New codec rate** (e.g. K2.5 for a 3060 build): add a window extractor in
  its codec header (`kernels/trellis/codec_<name>.cuh`, interface in docs/CODE.md) and a `repack` case, then pass `decode_bitexact` and
  `row_invariance`.
- **New model family** (e.g. `qwen4exp_moe` for Flash-Next): a new `model/<family>/` that
  reuses the op kernels and adds only what is new (the grouped trellis kernel for experts,
  hyper-connections).
- **Concurrency:** the scheduler packs rows from several requests into one M bucket (≤ 16 for the
  row-invariant tier, larger through the GEMM tier). Row-invariance keeps each request's output
  independent of its batch-mates.
- **Two GPUs:** a pipeline-stage abstraction (a layer range per device, host-staged transfers).
  There is no TP, per [S13].

## 11. What we reuse, and licenses

| source | license | what we take |
|---|---|---|
| llama-paw / llama.cpp | MIT | GGUF reader logic, Q4_K/Q5_K dequant and MMQ head, GDN and FA kernels as starting points, dump hooks for parity |
| exllamav3 | MIT | trellis decode primitives, the K3.5 ring, the GEMM tier, DFlash2 selector semantics |
| TensorFold | MIT | row-invariance rules, keyed-sampling rule, window/drafter policies (design, not code) |
| exl3xpu | MIT | reference-decoder approach for the oracle |
| glq | **GPL-3.0** | **nothing** (read-only reference) |
| sglang-exl3 | not public | nothing; only its published numbers as a target |
