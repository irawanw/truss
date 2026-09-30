# TRUSS handover — decode speed work (2026-09-30)

Read first: [TRACKER.md](../TRACKER.md) rows #56–#60 (every measurement below has a row) and
[docs/reference/](reference/README.md) (every source file). Rules: [CODE.md](CODE.md), `~/ML_projects/STORAGE.md`.

## 1. Goal and user constraints

- **Targets, in order (user):** decode 150 tok/s → prefill 3,000 → decode 200 → prefill 4,000 tok/s, on GPU 2
  (RTX 3090, PCIe x8), with **lower error than Strata** (no lower-bpw pack: the user rejected it).
- Bar on this box: Strata Q2_0 measured 84–107 tok/s decode, 960–1,330 prefill (engine 0.1.19; newer Strata
  estimates ~140 tok/s decode on a 3090).
- Standing rules: GPU 2 only (0/3 are renters; stop if a renter takes 2); no subagents; kill by PID; **active data on
  NVMe, move cold data to HDD with `_shared/scripts/storage_move.py`** (keep ≥ 50 GB free on NVMe); never move
  `paw27b/20260923_x3up/data/hess`; docs updated in the same commit as code; log every step in TRACKER; stage only
  own hunks in ML_projects; measure before claiming.
- User feedback to honor: study competitors' code (Strata) instead of arguing hardware ceilings; optimize each hot
  path to the peak of the hardware we have (AVX2 + DRAM bandwidth, 3090 bandwidth/tensor rate, 13.5 GB/s PCIe).

## 2. Where decode stands (code-agent prompt, 3,398 tokens, greedy, GPU 2)

| step | tok/s | commit / state |
|---|---|---|
| bring-up baseline | 15.3 | #56 |
| ring expert cache, split DSA attention, engine routing profile | 33.3 | 5c2d5ef (#57) |
| MTP speculative decoding (2.9 tok/pass, output identical to plain greedy) | 38.3 | e49f449 (#58) |
| pre-gated prefetch | 43.4 | 2a79705 (#59) |
| token embedding on host, 40K context | 45.7 | 579d73d (#60) |
| Strata draft vocab (40,525 ids) + draft min-p 0.5 | 49.1–49.4 | **uncommitted** |
| CPU tier, 150 rarest experts/layer (NVMe subset), share 0.5 | **59.1** | **uncommitted** |
| CPU tier, full archive (HDD), share 0.7 | **35.6** (plain 13.0): PCIe 50 MB/pass, **CPU now the wall** (~75 ms/pass) | uncommitted |

Per-pass profile at 59 tok/s (≈50 ms/pass, 3.1 tok/pass): GPU waits on PCIe copies ~22 ms (153 demand + 70
prefetched experts = 413 MB), GPU compute ~21 ms (dense GEMV 9, experts 4.6, router 2), GPU waits on CPU ~4 ms.
**The CPU tier is underused; PCIe is the wall.** Estimate if the CPU takes nearly all misses: ~25 ms/pass →
~120 tok/s *(est.)*.

## 3. What is uncommitted (working tree, builds, tests listed below)

- `src/runtime/doorbell.{cuh,cu}` + Forward driver thread (`Options::doorbell`, default on): decode FFNs without a
  host sync (GPU publishes routing to mapped memory, driver thread splits/fetches/runs CPU, spin kernels wait). Two
  bugs found and fixed: (a) ring-mode switch raced with kernel enqueue → `ExpertStore::begin_ring()` on the
  enqueuing thread; (b) the "go" word queued behind the next layer's hint copies → signal before hints. Doorbell and
  sync paths now measure the same (49.3 vs 49.4): no gain yet because PCIe-bound; it is the base for CUDA graphs.
- `ExpertStore::signal/upload/begin_ring`, fetch without compute stream.
- Draft head over a token-id subset (`Options::draft_vocab`, `data/draft_vocab_en.bin` from Strata, MIT, see
  `data/README.md`), `dense::q8_gather`, `sampling::argmax_prob`, `Options::draft_min_p` (draft() returns count).
- C API `truss_params` / `truss_open_params` / `truss_spec_step` / `truss_drafts`; Python `Params`, `Model.spec_step`;
  server `--mtp --drafts --draft-vocab --draft-min-p --cpu-dir --cpu-share --cpu-threads`, speculative loop for greedy
  requests, log line reports real draft counts. **Not yet run end to end in the server.**
- `moe::window`: when every pair of a window is skipped (all experts on the CPU tier) the output is zeroed (it was
  left unwritten → garbage; caused a FAIL at share 0.7). Unit test still passes.
- CPU tier subset loading (`L<nn>.ids` next to `L<nn>.q4s`).
- `tk-bench-spec` args: `[draft vocab file|-] [cpu dir|-] [min p] [sync]`, env `TRUSS_CPU_SHARE`, `TRUSS_CPU_THREADS`.

Before committing: rerun `qwen4exp_forward` modes short/stream/decode/cpu/long on the slice, `cpu_expert_test`,
`moe_window_test`, `dsa_prefill_test`, `tk-bench-spec` (must print PASS), then update docs/reference
(runtime-api-server: doorbell, signal/begin_ring; model-qwen4exp: draft_vocab ids, min-p, doorbell; kernels-small:
argmax_prob; kernels-dense: q8_gather) and add TRACKER #61.

## 4. Data and files

| what | where |
|---|---|
| model pack (X3 2.63 bpw experts + Q8_0 dense) | `~/ML_projects/flashnext/20260918_ngram_q8/data/qwen38-flash-next-paw-x3-q8_0-00001-of-00002.gguf` |
| MTP draft block (X3 K3, identity Hessian) | `~/ML_projects/flashnext/20260930_truss_tg/data/mtp_x3/flashnext-mtp-x3k3.gguf` |
| routing profile (engine, 8 held-out code prompts) | `.../20260930_truss_tg/data/usage_code_truss8.f32` |
| bench prompt (Strata bench code prompt) | `.../20260930_truss_tg/data/prompt_code4k.i32` |
| CPU tier q4s, full archive 68 GB (cold source, HDD) | `/mnt/hdd/ml/flashnext/20260930_truss_tg/data/cpu_q4/` (L00–L47 .q4s, index.json) |
| CPU tier NVMe subset (150 rarest/layer, 19.9 GB) | `.../20260930_truss_tg/data/cpu_q4_rare/` |
| Strata newest source (read-only worktree) | scratchpad `strata_new` (origin/main d6708a4); `~/strata_src` has a local change in `src/core/verify.cpp`, not merged |

NVMe free: 52 GB (other users' growth took ~20 GB today). A full CPU set (~350 experts/layer ≈ 46 GB) does not fit
under the 50 GB rule; nothing large in ML_projects is cold and movable (x3up = hess, never move; slice8 shard 2 is a
hard link). **Decision needed from the user** (options in §6).

Scripts (ML_projects/flashnext/scripts): `flashnext_truss_{usage,prompt_ids,cache_sim,pack_sim,mtp_encode,mtp_pack,
cpu_q4,cpu_subset}.py`. Card: `20260930_truss_tg/README.md` (update it with #60/#61 results).

## 5. Lessons (do not repeat)

1. PCIe misses dominate decode; measure fetched MB per pass (`tk-bench-spec` prints it).
2. cuBLAS SGEMM picks its algorithm by row count: the router flipped experts between 1-row and 4-row windows (3% logit
   difference). Decode-size fp32 GEMMs use `dense::f32_gemv` (row-invariant). Verify == run(4) == steps, bit-exact.
3. The window kernel's pairs with id −1 are skipped; a window with **no** GPU pairs must still write zeros (fixed).
4. With a driver thread, anything the enqueuing thread reads (store mode, meta pointers) must be set on that thread.
5. Order on the copy stream matters: a layer's go word must precede speculative (hint) copies.
6. The q8 GEMM's I2F is not its limit (magic-number conversion was slower, reverted).
7. A first-N-ids draft vocab loses acceptance; Strata's curated 40,525-id subset keeps it exactly (2.98 tok/pass).
8. Background tool tasks are killed after ~30 min: long jobs run detached (`setsid nohup ... &`) with a log.
9. Chatcode routing profiles miss code-decode routing (27% vs engine profile's 40% static hit): profile with the engine.

## 6. Next steps (ranked by expected gain)

1. **Grow the CPU tier to cover the mid-usage experts** (target: CPU takes most misses, PCIe the rest, both overlapped).
   Storage options for the user: (a) allow ~46 GB more on NVMe (below the 50 GB-free rule), (b) load the full q4s
   archive from HDD into RAM at server start (~5 min once), (c) derive the CPU copies of mid-usage experts from the
   trellis pack at load on the GPU (no disk, but +~12% error on those experts vs BF16-q4s). **Measured at share 0.7
   (full archive): PCIe drops to 50 MB/pass but spec falls to 35.6 tok/s — the CPU expert kernel is too slow, so do
   step 2 first**, then grow the CPU share while tok/s rises.
2. **Optimize the CPU expert kernel to DRAM bandwidth (now the top priority)**: `cpu_expert_test` timing 4 rows × 8 experts 0.93 ms (8
   threads); probe 16 threads 52 GB/s. Targets: pin threads, spin briefly before sleeping (wake latency per layer),
   prefetch rows, larger row chunks for 4-row windows; report % of measured DRAM peak.
3. **CUDA graph of the decode pass** on top of the doorbell path (positions from device memory), removing ~2,400
   launches per pass (GPU compute is ~21 ms/pass; launch gaps included).
4. **Adaptive expert cache** (Strata: swap up to 96 experts every 4 rounds during drafting).
5. **Prompt-lookup drafts** for code edits (Strata: +6–11%).
6. Better MTP acceptance: calibrate the MTP experts' Hessian from engine captures (identity Hessian now).
7. Then prefill (3,000 target): the dense Q8 GEMM runs ~70 TOPS vs cuBLAS int8 ~200; prefill regressed to 1,180
   tok/s in one measurement taken while the converter held GPU memory — re-measure clean before trusting it.

## 7. How to run

```bash
cd ~/trellis-kernel && cmake --build build
M=~/ML_projects/flashnext/20260918_ngram_q8/data/qwen38-flash-next-paw-x3-q8_0-00001-of-00002.gguf
D=~/ML_projects/flashnext/20260930_truss_tg/data
# speculative decode bench (PASS = identical to plain greedy)
CUDA_VISIBLE_DEVICES=2 ./build/tk-bench-spec $M $D/mtp_x3/flashnext-mtp-x3k3.gguf @$D/prompt_code4k.i32 128 3 \
  $D/usage_code_truss8.f32 40960 8192 $PWD/data/draft_vocab_en.bin $D/cpu_q4_rare 0.5
# server (Strata bench compatible log line)
CUDA_VISIBLE_DEVICES=2 python3 -m server.app --model $M --tokenizer \
  /data/www/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/models/Qwen3.8-27B-EXL3-3.5bpw/tokenizer.json --n-ctx 40960 \
  --expert-usage $D/usage_code_truss8.f32 --mtp $D/mtp_x3/flashnext-mtp-x3k3.gguf --draft-vocab data/draft_vocab_en.bin \
  --cpu-dir $D/cpu_q4_rare --port 8090 --log-tag "strata serve"
```
