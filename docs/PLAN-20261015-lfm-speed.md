# Plan 2026-10-15: LFM2.5-24B-A2B X3.1 on TRUSS - speed (and the loose ends to close the project)

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
