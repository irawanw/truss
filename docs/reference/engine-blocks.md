# Engine, blocks, routed-expert runtime: src/engine/, src/blocks/, src/runtime/moe_*

The model-agnostic part of TRUSS (docs/ENGINE.md). A model family (src/model/<family>/) only composes blocks;
everything here is shared by every model. Split out of the old `qwen4exp::Forward` on 2026-10-14 with the output
unchanged byte for byte (`tests/golden/`, 10 engine configurations, TRACKER #160).

## src/engine/

| file | what |
|---|---|
| `model.h` | the family interface: `Family` (hyperparameters the engine needs, `stage(lb, le, n_ctx)`), `StageModel` (one stage's blocks: `add_tensors`, `moe_layers`, `scratch_bytes`, `build`, `bind_experts`, `embed`, `layers`, `head`, `states`, `table_stats`, `graph_info` (B1: the decode kernels taking pos0 by value + the re-capture epoch)), `Drafter` (speculative drafts on the last stage), `Chunk`, `DrafterFiles`, `open_family()` (src/model/registry.cc) |
| `block.h` | `Stateful`: a block's per-slot state - `alloc_slots`, `reset`, `accept(slot, n, window)`, `commit(tokens)`, `ck_buffers` / `ck_save` / `ck_load`, `positional` (E6 lending) |
| `stage.h`, `stage.cu`, `stage_seq.cu` | `Stage`: one device's layer range. Construction = the memory plan (dense weights, small buffers, blocks and their slot states, then the expert store takes the VRAM left, then the prompt buffers carved from its spare region). `run_chunk`, `head`, `accept`, `checkpoint` / `restore` / `reset`, `select` / `settle`, the stage lock and the layer-split hand-off |
| `pipeline.h`, `pipeline.cu`, `pipeline_slots.cu`, `pipeline_stats.cu` | `Pipeline`: the public engine (one stage or a layer split). Selected-slot calls (`run`, `head`, `verify`/`accept`, `draft`, `reset`, `checkpoint`/`restore`, `set_position` (B1: re-align the host cursor, graph mode)), the concurrent slot calls `prompt()` (first stage a chunk ahead on its own thread, decode share) and `round()` (drafts, verify window over both stages, pick, accept; unwind on a throw), diagnostics summed over stages (`split`, `all_resident`, `graph_info` for B1) |
| `graph_step.h`, `graph_step.cu` | `GraphStep` (B1): one all-resident T=1 decode step (run + head + greedy argmax + token copy-back) as a reusable CUDA graph; per step refreshes pos0 in the model's pos-taking nodes (`cudaGraphExecKernelNodeSetParams`), rewrites the pinned token, launches + syncs; re-captures on the model's epoch change; requires no split, all experts resident, same logits/next_d addresses; always `cudaGraphInstantiateWithFlags` (the legacy 3-arg call segfaults, TRACKER #169) |
| `options.h`, `options.cc` | `Options` (every engine knob, `MoeOptions` inside) and `apply_env` (all `TRUSS_*` knobs: one place) |
| `gpu.h` | `Gpu`: a stage's device, stream, cuBLAS handle, current scratch, owned allocations |
| `limits.h` | `FETCH_ROWS` (32): fetch vs stream chunks, row-invariant fp32 gemv size |
| `profile.h`, `profile.cc` | `Profile`: `TRUSS_PROFILE_SECTIONS` / `TRUSS_SECT_CSV` section events per layer (0..6 layer chain, 7..12 doorbell MoE split) |
| `round_trace.h` | `TRUSS_ROUND_TRACE` interval log (track U) |
| `stage_lock.h` | `StageLock` (FIFO, hand-off ticket order, bounded waits), `Held<>` |
| `util.h`, `util.cc` | `require`, `load_mark` (`TRUSS_LOAD_TIMES`), `spin_timeout_s`, `robust_on`, `inject_fault` (`TRUSS_FAULT`), `dmalloc`, `DeviceGuard` |

**Invariants.** Every chunk is one ordered launch sequence on the stage's stream. Per-slot state lives in the blocks
(indexed by slot, nothing swapped on a slot change); position-indexed caches are sub-ranges of one n_ctx array per
block. A verify window runs tentative: recurrent blocks snapshot before it and replay `accept(n)` rows after; the
drafter defers its commit to the next `draft()`. `select()` inside the last stage's open window is an error.

**How to change.** A new engine feature that every model gets goes here; if it needs model knowledge, add a
`StageModel` / `Stateful` hook with a default, never a family name. Gate: `tests/golden/run.sh` identical.

## src/blocks/

| file | block | state per slot |
|---|---|---|
| `weights.h` | the blocks' host weight views (`HcWeights`, `GdnWeights`, `DsaWeights`, `PleWeights`, `MoeWeights`, `SkHcWeights`, `Ds41AttnWeights`, `EngramWeights`) | - |
| `linear.h/.cu` | `Linear`: every projection by the tensor's format (Q8_0 repacked at upload, F32; trellis dense next), `Act` (quantize once per input), `lin`, `lin32` (row-invariant gemv <= 32 rows), `lin_multi` (one launch at decode rows), `lin_multi_silu`; host-resident rows (untied embedding in pinned memory, TRACKER #60) | - |
| `head.h/.cu` | `Head`: output projection; drafter vocab subset (`set_draft_vocab`); FP16 vocab tiling | - |
| `hc.h/.cu` | `Hc`: hyper-connection mix / combine / expand (`TRUSS_HC_SILU`, `TRUSS_HC_COLLAPSE` fusions) | - |
| `gdn.h/.cu` | `Gdn`: gated delta net mixer | state x2 (window flip), conv rows; snapshot + raw rows for replay |
| `attn_dsa.h/.cu` | `Dsa<Shape>`: sparse attention mixer (instantiated for `dsa::FlashNext`) | K/V + indexer blocks (positional), indexer open block + snapshot |
| `ple.h/.cu` | `PleTable` (stage: n-gram window, reader, lookahead) and `Ple` (layer: key/value, gate, conv history) | token window; history + snapshot |
| `ple_hash.h/.cc` | `PleDims`, `ple_rows` (n-gram hash), `ple_gather` (mapping path) | - |
| `ple_reader.h/.cc` | `PleReader`: rows by parallel reads + row cache (TRACKER #88) | - |
| `norm.h` | `RmsNorm`: row RMSNorm (fp32 / fp16 out); `forward_add` fuses the residual add before it (`norm::add_rms`, bit-equal to add + norm) | - |
| `ffn_dense.h` | `DenseFfn`: SwiGLU (gate + up in one launch at decode rows) | - |
| `shortconv.h/.cu` | `ShortConv`: in_proj -> B*x causal depthwise conv -> C* -> out_proj | conv rows; snapshot + window rows (`tail_rows`) |
| `attn_gqa.h/.cu` | `AttnGqa`: q/k/v, per-head norm + rope + cache write, dense causal GQA (kernels/attn/gqa) | K/V (positional) |
| `attn_ds41.h/.cu` | `AttnDs41`: DeepSeek-V4.1 attention (q LoRA, shared latent kv = key and value, window 128 + compressed top-512, sinks, inverse rope, grouped output LoRA as one wo_a of T x 8 rows + `pick_groups`); KV sources compress (ratio 1 projection, ratio 2 gated pooling) into the compressed cache + fp32 index keys; index sources score and select (`IDX_ROWS` = 64 rows at a time), the candidate source also picks 2,048 blocks of 8; consumers get source pointers (`set_sources`); `alloc()` for tests without a stage. Gate: `ds41_attn_parity` (#177) | window ring (256 rows); KV sources: compressed cache, index keys (positional), ratio-2 raw rings; chunk top-k / candidates on the source |
| `engram.h/.cu` | `Engram`: DeepSeek-V4.1 n-gram lookup at its layers (before attention): host hash of the 2-4-grams over the compressed vocab (24 row ids per token), int4 rows by a pread pool (16 threads) into pinned memory, `engram_dequant_q4` -> wkv (Q8_0) -> `engram_gate` per copy. v1 syncs the stream before reusing the pinned rows. Gate: `ds41_engram_parity` (#178) | last 3 committed tokens (`commit`; checkpointed) |
| `moe.h/.cu` | `Moe`: router (fp32) + routed experts through `runtime::MoeExec` + optional gated shared expert; routing rule (`RouteFn`) and shared expert are parameters | - (residency is the runtime's) |

**Adding a block** (a new layer kind: GQA attention, short conv, Mamba-2, dense FFN): a class with a dims struct,
`add_tensors`, `forward(gpu, slot, ...)`, and `Stateful` if it keeps per-sequence state; a unit test vs a reference;
a line in CMake and in this table.

## src/runtime/moe_* — routed experts of a stage

`MoeExec` (moe_exec.h) owns the `ExpertStore`, the doorbell driver, the CPU tier and the residency policies, and runs
one MoE layer per `run(Call)`; it sees expert tables (`MoeLayer`), a kernel table (`MoeKernels::of<Shape>()`,
moe_kernels.h) and router outputs, never a model.

| file | content |
|---|---|
| `moe_exec.h` | `MoeOptions`, `MoeLayer` (tables, model layer, `cpu_ok`, `next_in_stage`, `profiled`), `MoeExec`, `all_resident()` (B1) |
| `moe_exec.cu` | construction (budget, hot-set plan, store, CPU tier, route trace, doorbells), `run()` dispatch (all-resident path: every expert on the device and no CPU tier -> the window kernel right after the routing, no doorbell / plan wait / hint; 10-15) + stream path + joins, adaptive tier, routing profile, E6 lending, stats |
| `moe_paths.cu` | doorbell path (publish, job, shared expert while the driver plans, plan wait, window), fetch path (one sync, CPU split, fetch + hint), split path (E7) |
| `moe_driver.cu` | `setup_doorbells`, the driver thread loop, watchdog (`TRUSS_DRIVER_DEBUG`) |
| `moe_serve.cu` | `serve()` (static / dynamic CPU split, copies, plan, hints, idle-link admission, `TRUSS_DRIVER_TS`), `fit_cpu_cost` |
| `cpu_tier.cc` | CPU tier loaders: q4s files or the pinned pack bytes (the CPU kernels exist for the Flash-Next expert shape only: another shape fails loudly) |
| `moe_kernels.h` | the shape-erased kernel table |
| `moe_exec_state.h` | private `CpuTier` and `Bell` |

Behaviour is the old Forward's, documented per option in `options.h` and runtime-api-server.md (ExpertStore,
doorbell, CPU tier). Known quirk kept for exactness: the idle-link admission's per-pass decay fires on model layer 0
only, so on a layer split's last stage it never decays (admission is off by default).
