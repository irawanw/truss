# Handover 2026-10-14: engine split done, LFM2.5-24B-A2B X3.1 in progress

## 0. LATEST (10-15 06:40) - the next agent starts here

**Goal now (user): make TRUSS faster than llama.cpp on LFM2.5-24B-A2B.** Work order: `docs/PLAN-20261015-lfm-speed.md`
**section 5** (numbers to beat, decode profile, steps B1-B6, gates, traps). Sections 1-8 below are history.

State:
- Branch `refactor/engine-blocks` == GitHub `main` (irawanw/truss) at the last commit; build dir `build/` is current.
- Packs `~/ML_projects/lfm24b/20261014_x31/data/gguf/lfm25-24b-x31-k{2,2.5,3,3.5}.gguf`; results and the comparison
  with LiquidAI's official GGUFs: `~/ML_projects/lfm24b/20261014_x31/reports/RESULTS.md`.
- TRUSS K3.5: KL vs bf16 0.063 (official QAD-Q4_0 0.0965, Q4_K_M 0.109) at 11.2 GB; decode 203 tok/s vs llama 282;
  prefill 4K 6.5K vs 5.5K (ours faster), 16K 3.5K vs 5.2K.
- GPU 2 is free (no brain, no agent; the PAW omp agent was retired 10-15).

Open task besides speed - **Hugging Face upload** (user named the repo `lackonendes/PAW-24B-X3.1`, public):
- Token: `HF_TOKEN` in `~/ML_projects/.env` (the CLI login on the box cannot create repos; this one can).
- `~/ML_projects/lfm24b/20261014_x31/scripts/g_upload.py` (run with the .env loaded: `set -a; . ~/ML_projects/.env;
  set +a; python3 scripts/g_upload.py`) creates the repo and uploads `PAW-24B-X3.1-K{2,2.5,3,3.5}.gguf`. Started 06:20
  10-15, log `data/work/logs/g_upload.log`; upload speed dropped to ~3 MB/s - check whether all four finished
  (`HfApi().list_repo_files("lackonendes/PAW-24B-X3.1")`), rerun for missing files (upload_file re-sends the file).
- **Model card not written yet.** Write `README.md` in the style of `lackonendes/PAW-125B-FLASH-NEXT-X3.1`'s card
  (license: `other`, `license_name: lfm1.0`, base_model `LiquidAI/LFM2.5-24B-A2B`; "follow @nawariokra on X" line;
  "stock llama.cpp cannot read this file, it needs TRUSS https://github.com/irawanw/truss"). Content: the RESULTS.md
  tables (KL vs bf16 per K and vs the official GGUFs, speed), how it was made (X3.1 trellis experts at uniform K,
  code-262k calibration, everything else Q8_0), the serve command (above), and the honest speed statement: llama.cpp's
  official Q4 files decode faster today; ours win on quality per GB and 4K prefill. Upload with `HfApi().upload_file`.

## 1. Where things stand

| track | state | evidence |
|---|---|---|
| Engine split (engine / blocks / runtime / model folders) | **DONE**, committed | golden 10/10 byte-identical to the old engine; qwen4exp_forward short/stream/decode/checkpoint/cpu PASS |
| No big files | **DONE** for engine, blocks, runtime, api, server, model; kernels <= 800 | `wc -l` audit in TRACKER #160 |
| lfm2moe family on TRUSS | **DONE** (works), speed **untuned** | `tests/layer/lfm2moe_parity.py`: mean KL 2.3-5.4e-4 vs HF on the same weights (4-layer slice) |
| LFM X3.1 encode, K 2 / 2.5 / 3 / 3.5 | **DONE** 21:37 10-14 (C-RUN-DONE 2..39, 38 layers x 4 rates) | `~/ML_projects/lfm24b/20261014_x31/data/work/logs/c_run.out` |
| LFM packs (one GGUF per K) | **DONE** 10-15 03:40: K2 6.89 GB, K2.5 8.32, K3 9.76, K3.5 11.19 (tensor data) | `PROJ/data/gguf/lfm25-24b-x31-k*.gguf`, `data/work/logs/c4_pack_all.log` |
| LFM full-model parity | **implementation OK** (exact to ~1e-6 per row); full-model mean KL 3.7e-2 is routing near-tie flips, not a bug; gate to redefine | section 8 | section 8 |
| LFM KL vs bf16 | **DONE** 10-15: K3.5 0.063, K3 0.082, K2.5 0.114, K2 0.166 (Q8_0 alone 0.033) | `PROJ/reports/RESULTS.md` |
| LFM coding evals | TODO | step 4 |
| LFM speed, measured | 10-15, one 3090: prefill 6.6-6.7K tok/s at 4K, 3.5K at 16K; decode 117-121 tok/s; ~same for every K | `PROJ/reports/RESULTS.md` |
| LFM speed work (GQA on tensor cores, decode fixed cost, MoE Tune for 2048/1536) | TODO | step 5 |

## 2. What is running and how to check it

```bash
grep "^\[" ~/ML_projects/lfm24b/20261014_x31/data/work/logs/c_run.out | tail     # one line per finished layer
tail -2 ~/ML_projects/lfm24b/20261014_x31/data/work/logs/enc_L<nn>.log            # the layer in progress
ls /mnt/hdd/ml/lfm24b/20261014_x31/data/enc/L*/DONE | wc -l                       # finished layers (38 = all)
```

- Driver: `~/ML_projects/lfm24b/20261014_x31/scripts/c_run.sh 2 39` (nohup, GPU 2). It is **resumable**: if it
  dies, rerun the same command; finished layers (`enc/L<nn>/DONE`) are skipped, the capture resumes from its
  saved hidden state. It stops itself if NVMe free space drops under 50 GB.
- It uses ~10 GB of GPU 2. Anything else on GPU 2 meanwhile: cap the TRUSS expert budget (`TRUSS_EXPERT_GB=7`).
- Per-layer quality is in `/mnt/hdd/ml/lfm24b/20261014_x31/data/enc/L<nn>/K<k>/manifest.json` (`summary`:
  held-out output error per projection). Layer 2: K2 gate 0.010 / up 0.013 / down 0.035; K3.5 ~0.0012 / 0.0015.

## 3. Next steps, in order (each with its gate)

**Step 1 - wait for `C-RUN-DONE 2..39`** in c_run.out. Then add a TRACKER row with the per-K median held-out
error over all layers (read the manifests). Kill rule (plan): K2 error worse than 2x Flash-Next's K2 -> stop.

**Step 2 - build the four packs** (CPU, ~10 min each, ~7-12 GB each, on NVMe; check `df -h /` >= 50 GB first):
```bash
cd ~/ML_projects/lfm24b/20261014_x31
for K in 2 2.5 3 3.5; do ~/flashnext/venv-tfmain/bin/python scripts/c4_pack.py $K data/gguf/lfm25-24b-x31-k$K.gguf; done
```
Gate: TRUSS loads each (`truss_open` binds every tensor exactly once or fails naming it); sizes ~ the plan table.

**Step 3 - full-model parity on TRUSS** (per pack, GPU 2):
```bash
cd ~/trellis-kernel
CUDA_VISIBLE_DEVICES=2 ~/src/venvs/exl3new/bin/python tests/layer/lfm2moe_parity.py \
  ~/ML_projects/lfm24b/20261014_x31/data/gguf/lfm25-24b-x31-k2.gguf build/libtruss.so \
  ~/ML_projects/lfm24b/20261014_x31/data/corpus/ids_code_heldout_1024.pt --steps 16 --prompt 512
```
Gate: mean KL <= 2e-3. Run with `TRUSS_EXPERT_GB=7` (TRUSS otherwise takes 21 GB of experts and the reference OOMs).
The old note "all 40 layers need ~20 GB" was wrong (fp32 experts are ~2.4 GB per layer, ~90 GB in all): the script
now streams the reference one layer at a time (section 8).

**Step 4 - quality per K** (the user's actual question: which rate is good enough):
- **DONE 10-15, KL vs the bf16 model** (user: "kl against fp16", not Q8_0): `PROJ/scripts/e_bf16_base.py` (HF, fp32
  math, layers streamed) writes the base, `e_kl_chain.sh` scores the packs, `e_q8_ref.sh` the Q8_0 GGUF. Table in
  `PROJ/reports/RESULTS.md`. tk-parity-kl on LFM needs `TRUSS_PREFILL_ROWS=1024` (n_ctx 1024 in one run).
- (old plan, superseded) KL vs the Q8_0 model: make the base with llama-paw (`~/llama-paw/build/bin/llama-perplexity -m
  data/gguf/lfm25-24b-q8_0.gguf --kl-divergence-base base.logits` on held-out code text), then
  `build/tk-parity-kl <pack> base.logits 8 0 fp16` per K (tk-parity-kl is model-agnostic now).
- Coding evals: serve each pack with the TRUSS server (`python3 -m server.app --model <pack> --tokenizer
  ~/ML_projects/models/LFM2.5-24B-A2B/tokenizer.json`) and run the Flash-Next eval scripts
  (`~/ML_projects/flashnext/scripts/x31_eval_run_lcb.sh` pattern; HumanEval / LCB-lite).
- Report a table K vs pack GB vs KL vs pass@1 in TRACKER + the plan doc.

**Step 5 - speed** (after quality is known; measure first with `tk-bench-prefill` / `tk-bench-decode`, both
model-agnostic):
- `kernels/attn/gqa.cu` v1 is CUDA-core: put QK^T and PV on mma tiles (`kernels/trellis/mma.cuh` has the fragments).
  Gate: `gqa_attn_test` still PASS, faster at 4K rows and at decode.
- `Tune<Lfm2>` in `kernels/moe/moe_window.cu` copies Flash-Next's geometry: sweep it (`tk-bench`-style, as TRACKER
  #13-#16 did for Flash-Next).
- The CPU expert tier is Flash-Next-shaped only (`cpu::D_MODEL`); LFM fits in VRAM, so it is not needed here.

**Step 6 - merge** `refactor/engine-blocks` into main once steps 2-3 pass (golden first:
`tests/golden/run.sh build <dir>` + `run.sh compare <golden>/old_a <dir>` must say 10 x IDENTICAL).

## 4. Rules for whoever continues (they are why this split exists)

- **A new model = a folder + a registry line.** Only `src/model/<family>/` names a model. A new layer kind is a new
  block (`src/blocks/`) with a unit test; a new expert shape is a struct + `Tune<>` + instantiation. If a port needs
  an edit in `engine/`, `runtime/`, `api/` or `server/`, make it a model-agnostic hook in its own commit, gated by
  the golden test. Every TRUSS model is trellis-quantized; targets are RTX 3060 / 3080 / 3090 (sm_86).
- **Golden gate for every engine change**: 10 x IDENTICAL. When GPU 2 is shared: `TRUSS_EXPERT_GB=7` (and for the
  `split` / `mtp_split` configs `TRUSS_EXPERT_GB=3 TRUSS_RING_GB=0.5`, via `ONLY="split mtp_split"`): residency
  never changes the numbers. Golden reference: `~/ML_projects/flashnext/20261014_engine_refactor/golden/old_a`.
- **No file over 500 lines** (kernels 800). Docs for every file in `docs/reference/` updated in the same commit.
- Log every experiment in TRACKER.md (question, result, PASS/FAIL, evidence path).

## 5. Known issues / loose ends

- `gemv_multi_test` "router 2 matrices" fails its 1e-6 gate at 1.9e-6 - also on the pre-split build (single-output
  shared-expert gate: fp32 noise). Fix the gate, not the kernel.
- `lfm2moe_parity.py` gates on mean KL only; top-1 over ~10 rows is reported (one near-tie flips it).
- `blocks::Linear` handles Q8_0 / F32 today; dense trellis matrices (PAW-27B-X3.1, future Qwen4 / Nemotron packs)
  need a trellis GEMV/GEMM case there - the next engine-level item for the dense models.
- Weights larger than VRAM for dense models (27B on a 3060, 70B on a 3090) need `runtime::ExpertStore` generalized
  to any trellis matrix - planned, not started.

## 6. Things changed on the box that must be undone

- Flash-Next serving is **stopped**: pm2 `truss-x31-eval-g2` (:8113) and `x31-lcb-eval` (LCB P4 at 12 items,
  resumable per item). Restart only after the LFM work frees GPU 2.
- The Flash-Next X3.1 shards are on HDD: `/mnt/hdd/ml/flashnext/20261003_x31/data/flashnext-x31-0000{1,2}-of-00002.gguf`.
  Move them back to `~/ML_projects/flashnext/20261003_x31/data/` with `~/ML_projects/_shared/scripts/storage_move.py`
  (needs ~98 GB free on NVMe) before restarting :8113 - after the LFM bf16 (48 GB) moves to HDD.
- `~/ML_projects/flashnext/20261012_q4tier` moved to `/mnt/hdd/ml/flashnext/20261012_q4tier` (finished experiment).

## 7. Who continues: the PAW agent (set up 10-14 16:50)

PAW-125B Flash-Next X3.1 (served by TRUSS on GPU 2 from the HDD pack) continues steps 2-5 as an `omp` agent, the
same pattern as the 10-04 self-optimization run:

| piece | where |
|---|---|
| brief (its work order and rules) | `~/ML_projects/lfm24b/20261014_x31/agent/AGENT_BRIEF.md` |
| its memory / results | `agent/NOTES.md`, `agent/LEDGER.tsv`, `agent/notify.txt` (RESULT / DECISION / STUCK for humans) |
| its code | worktree `agent/wt`, branch `agent/lfm24b` (from `refactor/engine-blocks`) |
| its brain | pm2 `paw-brain-g2` (`agent/scripts/brain.sh`, :8113), omp profile `lfmwork` |
| GPU access | `agent/scripts/gpurun` only: stops the brain, runs one job (<= 25 min) on GPU 2, restarts it (`agent/runs/gpurun.tsv`) |
| start | `agent/scripts/start_after_encode.sh` (nohup, running): when the encode prints C-RUN-DONE it starts the brain, then `agent/scripts/loop.sh 24 120` in tmux `lfmagent` |

Watch it: `tmux attach -t lfmagent`, `tail agent/logs/loop_*.log`, `cat agent/notify.txt agent/LEDGER.tsv`.
Stop it: `tmux kill-session -t lfmagent` (then `pm2 stop paw-brain-g2` to free GPU 2).

**Status 10-15: the agent is STOPPED; the lead does the work directly.** What happened: the watcher gave up after
900 s (the brain loads from HDD in ~20 min); the loop started by hand ran omp in print mode (`-p`, no interface), and
the brain hung on the first 21,677-token prompt (0% GPU, no answer to a 4-token request for 90 s). `agent/scripts/
interactive.sh` (interactive omp in tmux, waits for the brain) was added, then the user said to stop omp. The brain
(`paw-brain-g2`) is stopped; GPU 2 is free. Not diagnosed: why the brain stalled in prefill (box load ~95 from other
tenants' processes - xmrig ~975% CPU, two ollama llama-server ~750% each - may starve its HDD page cache).

## 8. Full-model parity: what was found 10-15 (continue here)

**Parity script fixes** (`tests/layer/lfm2moe_parity.py`, uncommitted on `refactor/engine-blocks`):
- Reference streams layers: one causal forward over all ids (row m-1 = last row of the m-prefix), each layer built,
  run, freed. Needs an explicit causal mask for the attention layers: HF eager attention applies NONE when
  `attention_mask=None` (the old per-prefix recompute only read the last row, so it never showed); conv layers get
  `None` (they read it as a padding mask).
- Prints a noise floor: the same reference with the residual rounded to fp16 after every layer (KL vs fp32 per step).
- With the fix the 4-layer slice gives mean KL 4.7e-7 (was 2-5e-4 with the old reference).

**Results, K2, prompt 512, 16 decode steps** (logs `PROJ/data/work/logs/parity_*.log`; slices from
`c4_pack.py 2 data/test/lfm-slice<N>-k2.gguf --layers N`):

| layers | TRUSS mean KL | fp16-residual floor | top-1 |
|---|---|---|---|
| 4 | 8.5e-7 | 3.7e-7 | 17/17 PASS |
| 8 | 2.3e-6 | 1.1e-7 | 17/17 PASS |
| 10 | 2.4e-6 | 2.6e-7 | 17/17 PASS |
| 11 | 2.9e-6 | 3.2e-7 | 17/17 PASS |
| 12 | 3.6e-3 | 7.0e-7 | 16/17 FAIL |
| 14 | 1.3e-2 | 3.6e-7 | 16/17 FAIL |
| 16 | 1.2e-2 | 1.5e-3 | 16/17 FAIL |
| 40 (full) | 3.7e-2 | 6.7e-3 | 17/17 FAIL |

Reading: up to 11 layers TRUSS matches to ~1e-6. The slice-12 failure is **one row**: step 2 has KL 6.1e-2, the other
16 rows ~5e-6. That row has the smallest router margin at layer 11 (4th vs 5th biased sigmoid score 1.8e-4; the other
rows >= 7.7e-4): TRUSS's fp16 activations flip a near-tie expert choice. Not a bug. Ties are common in every MoE layer
(~45 of 528 rows have margin < 1e-3; probe: residual / MoE-output range and margins per layer, `PROJ/scripts/d_router_probe.py <pack> <ids.pt>`),
so over 38 MoE layers about half the rows of the full model take at least one
flip -> full-model per-row KL is bimodal (~1e-6 or 1e-2..0.3) and its mean is dominated by flips.
Also seen: layer 9's MoE writes an outlier channel (|x| 134, residual rms 0.01 -> 0.24); far from fp16 range, harmless.

**Decision needed / next:**
1. Redefine the full-model parity gate (the mean-KL 2e-3 gate cannot pass on this model): e.g. per row, pass if KL
   <= 1e-4 OR the reference's minimum router margin on that row is < some eps (a tie); fail otherwise. Needs the
   reference to record per-row min margin over layers (hook as in the probe). Slices to 11 layers already show the
   arithmetic is exact.
2. Then run it on the four packs (`PROJ/scripts/d_parity_chain.sh`), and go to step 4 (KL vs Q8_0 on real text,
   coding evals) - the quality numbers that matter; routing-tie noise is the same kind llama.cpp has.
3. Step 5 (speed).

**Disk:** NVMe at 54 GB free after the packs + slices (rule: keep >= 50 GB). Test slices
`PROJ/data/test/lfm-slice{4,8,10,11,12}-k2.gguf` (~8 GB; 14/16/24 already deleted) are disposable; delete when the bisect is done.
`d_parity_chain.sh` (`PROJ/scripts`) runs parity on all four packs in sequence (already has `TRUSS_EXPERT_GB=7`).
