# Plan 2026-10-15: LFM2.5-24B-A2B X3.1 on TRUSS - speed (and the loose ends to close the project)

> **STATUS 10-15 06:30 - read section 5 first ("Phase 2: beat llama.cpp").** P1, P2, P3, P4a are DONE (decode
> 117.6 -> 204 tok/s, commits b2e01a4 / 8d3fc4b / c489ee9 / d823c32, TRACKER #163-#166, all on GitHub main).
> P0, P4 rest, P5, P6, P7 are open and folded into section 5. LiquidAI's official llama.cpp GGUFs are still faster
> at decode (+30%) and at 16K prefill (+45%); the user's goal now is to beat them.

State before this plan: `docs/HANDOVER-20261014-engine-lfm.md` (sections 1, 3, 8) and
`~/ML_projects/lfm24b/20261014_x31/reports/RESULTS.md`. Branch `refactor/engine-blocks`.
User decisions (10-15): KL is measured against the **bf16** model (not Q8_0); **no coding evals**; work on speed;
MTP is not available (the checkpoint has no MTP heads).

## 0. Where we start (measured 10-15, one RTX 3090, all experts resident)

| | K2 | K2.5 | K3 | K3.5 |
|---|---|---|---|---|
| pack GB | 6.90 | 8.33 | 9.77 | 11.20 |
| KL vs bf16 (16,352 tokens) | 0.166 | 0.114 | 0.082 | 0.063 |
| prefill tok/s 4K / 16K | 6,719 / 3,528 | - | - | 6,570 / 3,468 |
| decode tok/s (ctx 512) | 121.3 | 117.2 | 120.9 | 117.6 |

Q8_0 alone (llama-paw): KL 0.033. Speed hardly depends on K: no phase is limited by expert bytes.

**Decode profile** (nsys, K3.5, ctx 512, 60 steps; `tk-bench-decode`): 8.42 ms/step = **7.95 ms kernel time + 0.47 ms
gaps**, 642 launches/step. CUDA graphs can win at most the gaps (~6%): not the lever (same verdict as Flash-Next,
TRACKER #122, there for a different reason).

| kernel | ms/step | launches | ideal-ish | note |
|---|---|---|---|---|
| `moe::window_kernel` | 2.85 | 38 | ~1.0 | 75 us/layer for ~16.5 MB of K3.5 experts = 220 GB/s; `Tune<Lfm2>` is Flash-Next's geometry |
| `attn::attn_kernel` (GQA) | **1.95** | 10 | ~0.2 | 195 us/layer at 576 cells: `split_count` = cells/512 -> 1 split -> grid 1 x 8 x 1 = 8 blocks on 82 SMs, T=1 leaves most threads idle |
| `gemv_multi_kernel` (Q8 dense) | 1.55 | 85 | ~0.9 | ~460 GB/s |
| doorbell `wait_plan` + `publish` | 0.42 | 76 | 0 | residency bookkeeping; all experts are resident |
| router (`route_sigmoid`, `f32_gemv_multi`) | 0.46 | 113 | ~0.15 | |
| `rms`, `add`, `quantize`, `conv`, `advance`, `prepare` | 0.69 | 316 | ~0.3 | small ops; fusion |

## 1. Targets and gates

- **Decode >= 200 tok/s** (<= 5.0 ms/step) at ctx 512, and no worse than -10% at ctx 8K. **Prefill 16K >= 5,500 tok/s.**
- Quality gates, every step:
  - kernel unit tests (`gqa_attn_test`, `moe_window_test`, `lfm_small_ops_test`) PASS;
  - LFM output: `tk-parity-kl <K3.5 pack> data/kl/bf16_base_c32.logits 32 0 fp16` stays **0.063 +- 0.005** (and
    the per-row parity gate of step P0 passes);
  - any change in `src/engine`, `src/runtime`, `src/api`, `server/` or a kernel Flash-Next uses: `tests/golden/run.sh`
    10 x IDENTICAL (`TRUSS_EXPERT_GB=7`; split configs `TRUSS_EXPERT_GB=3 TRUSS_RING_GB=0.5`).
- Speed claims only from `tk-bench-decode <pack> 512 128 8192 2048` / `tk-bench-prefill <pack> 16384 2048 0 fp16` on an
  idle GPU 2, 3 repeats, plus an nsys table like section 0 for each kept change.
- Rules: no file over 500 lines (kernels 800); docs in `docs/reference/` in the same commit; TRACKER row per step.

## 2. Steps (in order: biggest saving per hour first)

**P0 - parity gate that this model can pass** (0.5 day). `tests/layer/lfm2moe_parity.py`: record per row the
reference's minimum top-4 router margin over all MoE layers (hook as `PROJ/scripts/d_router_probe.py`); a row passes
if KL <= 1e-4, or its margin < 1e-3 (a tie: report, do not fail); gate = no failing row. Expect: slices <= 11 layers
pass all rows; full packs pass with ties reported. Needed so P1-P5 have a correctness gate besides KL.

**P1 - decode attention: use the GPU** (0.5-1 day, expected -1.7 ms). `src/kernels/attn/gqa.cu`:
- (a) first, the cheap test: `split_count` for small T by cells/64 (cap so splits x HKV >= ~2 x SMs, workspace
  allowed) -> grid 1 x 8 x ~20; combine_kernel already exists. Measure.
- (b) if still > 30 us/layer: a decode variant (T <= 4): one warp per key chunk, all G query heads of a KV head per
  block, K/V read once with 16-byte loads; flash-decoding combine.
- Gate: `gqa_attn_test` (add T=1 at 64/576/8K cells), parity, KL. Also run ctx 8K / 32K decode to show the scaling.

**P2 - MoE window geometry for 2048 x 1536** (1-2 days, expected -1.5 ms). `src/kernels/moe/moe_window.cu`
`Tune<Lfm2>` (BLOCKS_PER_SM, `Geom<>` for GateUp and Down): sweep every geometry that satisfies the tiling asserts,
at 1 / 4 / 8 rows, on the real pack (as TRACKER #13-#16 did for Flash-Next). Shape-only change: Flash-Next's
`Tune<FlashNext>` untouched -> golden not needed, `moe_window_test` for the Lfm2 shape is. Target >= 600 GB/s.

**P3 - all-resident fast path** (0.5 day, expected -0.4 ms). When every expert of the stage is resident (LFM on a
24 GB card), the doorbell publish / wait_plan pair does no work. `src/runtime/moe_paths.cu` / `moe_exec.cu`: if the
plan is static (no cold experts, no CPU tier), call the window directly. Model-agnostic (`MoeExec` decides from its
own store), golden 10 x IDENTICAL (Flash-Next never takes it: experts > VRAM).

**P4 - small-op fusion** (1-2 days, expected -0.5 ms). Read first: Flash-Next small-kernel work
(`~/ML_projects/flashnext/20261013_small_kernels/reports/LOG.md`, parked +2.5-3%, bit-exact) and reuse what applies.
Candidates for LFM: add + rms + quantize in one kernel (residual -> next projection input); router gemv + sigmoid +
top-k in one; conv `advance` folded into `conv`. Blocks change, not the engine. Gate: bit-exact vs before (unit
test) or KL.

**P5 - dense Q8 gemv at decode** (1 day, expected -0.4 ms). `gemv_multi_kernel` at ~460 GB/s: check the split for
LFM's shapes (6144 x 2048 in_proj, 2048 x 2048 out, q/k/v) with `tk-bench-gemv-split`; pick per-shape splits.

Expected after P1-P5: ~8.4 -> ~4.0-4.5 ms/step = **220-250 tok/s**. Re-profile after each step; reorder if the table
changes.

**P6 - prefill attention on tensor cores** (2-3 days). The 4K -> 16K drop (6.6K -> 3.5K tok/s) is GQA prefill
(v1, CUDA cores). QK^T and PV on mma tiles (`src/kernels/trellis/mma.cuh` fragments), causal block skip. Gate:
`gqa_attn_test`, parity, KL. Target 16K >= 5,500 tok/s.

**P7 - later, if wanted:** prompt-lookup (n-gram) drafter as a model-agnostic `engine::Drafter` (no training; good
for code/agent traffic: verify of 4-8 rows costs ~1 row once decode is bandwidth-bound). Only after P1-P5, since it
pays most when a step is cheap.

## 3. Closing steps (not speed)

- **C1 merge** `refactor/engine-blocks` into main after P0 (golden first).
- **C2 Flash-Next serving back**: its shards are on HDD (`/mnt/hdd/ml/flashnext/20261003_x31/data/`); moving them back
  needs ~98 GB NVMe (rule: keep >= 50 GB free) - after the LFM bf16 / Q8_0 moves of 10-15 the NVMe has room. Then
  restart pm2 `truss-x31-eval-g2` only when GPU 2 is not needed for P1-P6.
- **C3** update `docs/reference/` for every changed file, TRACKER rows, RESULTS.md table after each kept step.

## 4. Data locations (after the 10-15 cleanup)

| what | where |
|---|---|
| packs K2 / 2.5 / 3 / 3.5 (active) | `~/ML_projects/lfm24b/20261014_x31/data/gguf/lfm25-24b-x31-k*.gguf` |
| bf16 KL base, 32 x 1024 (active) | `.../data/kl/bf16_base_c32.logits` |
| 4-layer test slice | `.../data/test/lfm-slice4-k2.gguf` |
| Q8_0 GGUF (cold) | `/mnt/hdd/ml/lfm24b/20261014_x31/data/gguf/lfm25-24b-q8_0.gguf` |
| bf16 model (cold; e_bf16_base.py / c2 / c3 take `--model-dir`) | `/mnt/hdd/ml/models/lfm/LFM2.5-24B-A2B` |
| encode output (cold) | `/mnt/hdd/ml/lfm24b/20261014_x31/data/enc/` |
| capture resume state (cold) | `/mnt/hdd/ml/lfm24b/20261014_x31/data/work/calib` |

To rebuild a pack: `c4_pack.py <K> <out> --src /mnt/hdd/ml/lfm24b/20261014_x31/data/gguf/lfm25-24b-q8_0.gguf`.
The PAW agent for this project is retired (worktree and branch `agent/lfm24b` removed, pm2 `paw-brain-g2` deleted).

## 5. Phase 2: beat llama.cpp (user 10-15: "surpass llama speed")

### 5.1 The numbers to beat (one RTX 3090, GPU 2 idle, measured 10-15)

| model | GB | KL vs bf16 | tg @512 | tg @8K | pp 4K | pp 16K |
|---|---|---|---|---|---|---|
| llama.cpp QAD-Q4_0 (official) | 13.58 | 0.0965 | **282** | **268** | 5,511 | **5,179** |
| llama.cpp Q4_K_M (official) | 14.53 | 0.1093 | 262 | 249 | 5,148 | 4,850 |
| TRUSS X3.1 K3.5 | 11.20 | **0.0630** | 203 | 186 | **6,543** | 3,465 |
| TRUSS X3.1 K3 | 9.77 | 0.0817 | 207 | 189 | 6,621 | 3,489 |
| TRUSS X3.1 K2 | 6.90 | 0.1658 | 215 | 196 | 6,796 | 3,545 |

llama: `~/llama-paw/build/bin/llama-bench -m <gguf> -ngl 99 -fa on -r 3 -p 4096,16384 -n 0` and
`-p 0 -n 128 -d 512,8192`; files in `~/ML_projects/lfm24b/20261014_x31/data/ref/`. TRUSS: `build/tk-bench-decode
<pack> 512|8192 128 65536 2048`, `build/tk-bench-prefill <pack> 4096|16384 2048 0 fp16`. (tk-bench-decode copies each
argmax to the host per token; llama-bench's tg does not: a 1-2% handicap for TRUSS, keep it, it is what serving does.)
Quality is already ours (K3.5: KL -35% vs QAD-Q4_0, 18% smaller). Speed must not cost it: K3.5 KL stays 0.063 +- 0.005.

**Targets:** decode >= 290 tok/s at 512 and >= 275 at 8K (<= 3.45 / 3.6 ms per step) on K3.5; prefill 16K >= 5,500.

### 5.2 Where a K3.5 decode step goes now (nsys, 4.90 ms, 539 launches; profile recipe in section 0)

| item | ms | note |
|---|---|---|
| `moe::window_kernel` x38 | 1.82 | 48 us/layer. Hardly depends on K (K2 decode only 0.27 ms/step faster than K3.5): not byte-bound - dependency waits, the gate/up -> down hand-off, Hadamard rotations, per-item setup. `moe_window_test` standalone: ~400 GB/s even for Flash-Next |
| `dense::gemv_multi_kernel` x85 | 1.54 | in_proj 6144x2048 x30 = 0.60 (667 GB/s); head 128000x2048 = 0.32 (870 GB/s); out 2048x2048 x40 = 0.32 (564 GB/s); dense FFN L0-1 = 0.12; attn qkv = 0.11. All Q8_0: ~1.0 GB per token |
| router: `f32_gemv_multi` x38 + `route_sigmoid` x38 | 0.34 | route_sigmoid is ONE block per launch |
| `rms_kernel` x81 (add fused in) | 0.20 | one block per row |
| attn `prepare_kernel` x10 | 0.12 | 12 us each: fp64 pow / cos / sin per lane (GeForce fp64 = 1/64 rate) |
| `quantize_kernel` x85 | 0.12 | activation -> Q8 per projection input |
| conv `conv_kernel` + `advance_kernel` x30 | 0.13 | |
| GQA `decode_kernel` + combine x10 | 0.23 | |
| launch gaps | 0.42 | the all-resident decode step has no host sync inside: graph-capturable |

### 5.3 Steps (each: measure, gate, commit, TRACKER row, docs/reference in the same commit)

**B1 - CUDA graph for the all-resident decode step** (expected -0.4 ms). Was dropped for Flash-Next (TRACKER #122:
CPU-tier waits dominated); LFM has no host sync per layer since P3. Capture `Pipeline::run` for T = 1 (and the verify
sizes) per slot / position-independent args: kernels that take `pos0` / n_cells by value need them from device memory
(or re-capture per call = no gain; use `cudaGraphExecKernelNodeSetParams` / a device-side position word). Start with a
probe: capture one step, replay, compare logits bit-for-bit vs eager, time it. Gate: identical logits, golden.

**B2 - small-op fusion** (expected -0.4 ms): (a) router: f32 gemv + sigmoid + bias + top-4 + renorm in one kernel (64
outputs x 2048: one CTA per 8 outputs, last CTA does the top-k via a counter); (b) `add_rms` also writes the Q8
activation (`Linear::Act`) so the next projection skips `quantize` (needs a Linear entry that accepts a ready Act);
(c) attn prepare: per-position cos / sin computed once per step (fp64 once per (pos, i), 32 values, shared by the 10
layers) or a [n_ctx][32] fp32 table built at load in fp64 - bit-identical to today; (d) conv + advance in one kernel.
Read `~/ML_projects/flashnext/20261013_small_kernels/reports/LOG.md` first (bit-exact fusion patterns). Gate: KL identical.

**B3 - MoE window internals** (expected -0.4..0.8 ms, the biggest lever, hardest). Use `build/tk-bench-moe-trace`
(kernel's own trace: dependency waits vs work per item kind) on the Lfm2 shape at 1 row first - add an Lfm2 mode to it.
Candidates: overlap the down items with the tail of gate/up (finer-grained dependencies per column group instead of
per slot); fold the Hadamard rotation of A_d into the gate/up epilogue; more items per block on the down side; check
register spills of the Lfm2 instantiation (`cuobjdump -res-usage`). Then re-sweep `Tune<Lfm2>` (sweep script recipe:
TRACKER #164; rewrite the Tune block, rebuild tk-bench-decode, time). Flash-Next's window SASS must stay identical
(per-function md5 recipe in TRACKER #164) or the golden must pass.

**B4 - fewer dense bytes** (expected -0.3..0.6 ms; changes the pack: measure KL). The non-expert tensors are Q8_0
(~1.0 GB read per token; llama's Q4 files read ~0.5). Options, in order: (a) the output head 128000x2048 (278 MB, 0.32
ms) as trellis K4 or Q6-like; (b) the conv in_proj / out_proj as trellis K4 - this is the planned **trellis case in
`blocks::Linear`** (also needed for dense models: PAW-27B-X3.1, Qwen4, Nemotron), so build it model-agnostic with a
unit test vs the exllamav3 reference; encode with `c3_encode.py`-style settings on the same calibration. Gate: K3.5-
pack KL rises by <= 0.005 (report the new KL honestly; it is a new pack, name it e.g. K3.5-d4).

**B5 - GQA prefill on tensor cores** (= P6; prefill 16K 3.5K -> >= 5.5K). `src/kernels/attn/gqa.cu` `attn_kernel` is
CUDA-core with 51 KB smem (1 CTA/SM). QK^T and PV on mma tiles (`src/kernels/trellis/mma.cuh`), causal tile skip,
fp32 online softmax. Gate: `gqa_attn_test`, KL (prefill path) +- noise. Check also pp at 32K / 64K.

**B6 - prompt-lookup drafter** (P7): once a step is cheap, verifying 4-8 rows costs ~1 row on this model (the window
already takes up to 8 rows): a model-agnostic `engine::Drafter` from n-gram matches in the context. Only after B1-B4;
it does not change llama-bench-style numbers, it changes served code/agent throughput.

Order: B1 (quick probe) -> B2 -> B3 -> B5 -> B4 -> B6. Stop rule per step: 3 failed attempts -> write why in TRACKER
and move on.

### 5.4 Rules and traps learned 10-15 (do not repeat)

- GPU 2 is ours; GPUs 0, 1, 3 belong to paying tenants. Check `nvidia-smi -i 2` before timing: a leftover process
  (10-15: an orphaned llama-imatrix) silently halves the numbers.
- Never `pkill -f` / `pgrep -f` with a pattern that also appears in your own shell command line: it kills your shell
  (happened 3x). Kill by PID, or use a bracket pattern (`pgrep -f "f_fetc[h]"`).
- NVMe `/` is shared with tenants' docker: keep >= 50 GB free. Cold data -> `/mnt/hdd/ml/...` with
  `~/ML_projects/_shared/scripts/storage_move.py`.
- LFM routing is tie-heavy: KL is the quality gate (8 chunks: `TRUSS_PREFILL_ROWS=1024 build/tk-parity-kl <pack>
  data/kl/bf16_base_c32.logits 8 0 fp16 [step 1|4]` = 0.06742 / 0.06794 / 0.06779 today on K3.5); the mean-KL
  parity script can only pass on slices <= 11 layers (handover section 8).
- Any change in `src/engine`, `src/runtime`, `src/api`, `server/` or a kernel Flash-Next uses: golden 10 x IDENTICAL
  (`CUDA_VISIBLE_DEVICES=2 TRUSS_EXPERT_GB=7 tests/golden/run.sh build <dir>`, then `run.sh compare
  ~/ML_projects/flashnext/20261014_engine_refactor/golden/old_a <dir>`; ~12 min).
- Serve check: `python3 -m server.app --model <pack> --tokenizer /mnt/hdd/ml/models/lfm/LFM2.5-24B-A2B/tokenizer.json
  --n-ctx 32768 --chunk 2048 --port <p>` works (10-15: thinking + answer correct, 191.8 tok/s served on K3.5).
