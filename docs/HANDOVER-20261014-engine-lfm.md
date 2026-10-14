# Handover 2026-10-14: engine split done, LFM2.5-24B-A2B X3.1 in progress

Read first: `docs/ENGINE.md` (layout contract), `docs/CODE.md` (rules, no file over 500 lines), TRACKER.md rows
#159-161 (local file, not in git) and its "Do not repeat" list. Branch: `refactor/engine-blocks` (not merged to
main yet; see step 6).

## 1. Where things stand

| track | state | evidence |
|---|---|---|
| Engine split (engine / blocks / runtime / model folders) | **DONE**, committed | golden 10/10 byte-identical to the old engine; qwen4exp_forward short/stream/decode/checkpoint/cpu PASS |
| No big files | **DONE** for engine, blocks, runtime, api, server, model; kernels <= 800 | `wc -l` audit in TRACKER #160 |
| lfm2moe family on TRUSS | **DONE** (works), speed **untuned** | `tests/layer/lfm2moe_parity.py`: mean KL 2.3-5.4e-4 vs HF on the same weights (4-layer slice) |
| LFM X3.1 encode, K 2 / 2.5 / 3 / 3.5 | **RUNNING** on GPU 2 since 15:58; ~9.3 min per layer, 38 MoE layers (2..39); ETA ~22:00 10-14 | `~/ML_projects/lfm24b/20261014_x31/data/work/logs/c_run.out` |
| LFM packs (one GGUF per K) | TODO (after the encode) | step 2 |
| LFM full-model quality (KL, coding evals) | TODO | steps 3-4 |
| LFM speed (GQA attention on tensor cores, MoE Tune for 2048/1536) | TODO | step 5 |

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
Gate: mean KL <= 2e-3 (it builds HF reference layers on GPU in fp32: all 40 layers need ~20 GB; run it when the
encode is finished, or reduce `--prompt`).

**Step 4 - quality per K** (the user's actual question: which rate is good enough):
- KL vs the Q8_0 model: make the base with llama-paw (`~/llama-paw/build/bin/llama-perplexity -m
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
