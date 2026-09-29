# TRUSS: our own engine, one GPU, trellis on the hot path

Written 2026-09-29. Numbers marked *(est.)* are arithmetic, not measurements.

TRUSS is our own inference engine (own code base, no ggml runtime). Target models, in order:
1. **Qwen3.8 Flash-Next PAW X3.1** (125B MoE, 48 layers = 36 GDN + 12 DSA attention, 512 experts, top-10) on
   **one 3090**, 256K context, MTP. Hardest case: it needs a cold tier. First and main target.
2. **PAW X3.1 27B** (dense). All weights resident; exercises dense trellis + spec.
3. **PAW 35B** (Qwen3.6-35B-A3B MoE). All experts resident; exercises MoE without a cold tier.

If the engine runs model 1, models 2 and 3 are subsets (no cold tier, no PLE, no DSA).

## 1. The bar (what "same speed" means)

| | TG 4K | TG 32K | TG 262K | PP 4K | PP 32K | PP 262K |
|---|---|---|---|---|---|---|
| Strata Q2_0, our 3090 (GPU 2), engine 0.1.21, measured 2026-09-29 | 84–94 | 86–107 (26–29K) | — | 960–1,140 | ~1,320 (26–29K) | — |
| Strata Q2_0, RTX 5070, engine 0.1.22, their table | 88 | 77 | 61 | 1,226 | 1,844 | 1,304 |

TRUSS gate: TG and PP ≥ the Strata row on the same box, same prompts (`flashnext_strata_bench.py`), and **KL vs
the Q8 reference lower than Strata Q2_0**. Strata moves fast (0.1.24 now); re-measure the bar before the final gate.

## 2. What the numbers say (the design follows from these)

1. **Trellis does not fit more experts at equal bits.** Q2_0 = 1.382 MB/expert (17.82 GiB / 13,843 slots);
   our X3 pack = 1.403 MB/expert (34.48 GB / 24,576). Same slot count at the same rate. Trellis buys *quality per byte*, which is the goal.
2. **Hit rate is a weak lever.** Strata's layer-split run: hits 59–71% → 96–97% gave decode ×1.14–1.31.
3. **The round is mostly not weight bytes.** 4-row window reads ~3.5 GB dense + ~2.3 GB experts ≈ 6–7 ms at 3090
   bandwidth; the measured round is 28.9 ms *(est. split)*. Speed comes from the round structure (miss wait, syncs,
   drafts, PLE), which is exactly the part we must own. That slack is also where trellis ALU cost can hide — only if
   the kernel reaches ≥ 80% of the read ceiling at M ≤ 5.
4. **256K costs almost no VRAM.** DSA reads 2,048 selected positions per query, so the KV (3.2 GiB int8 at 256K)
   lives in pinned RAM; VRAM keeps the indexer keys (~0.1 GB), a resident KV window, GDN state (113 MB) and its
   spec snapshots.
5. **CPU trellis is dead** (0.13 tok/s measured). The cold tier needs a CPU format (Q4_K candidate).
6. **RAM is the tight resource.** 125 GB total, ~88 GB available with renters. Our Q8 PLE shard is 49 GB (Strata's
   is 28.8 GB). Q4 of *all* experts (68 GB) does not fit; Q4 of the cold set (~11.5k × 2.76 MB ≈ 32 GB) does, with a
   smaller PLE.
7. **One format per expert.** An expert is either trellis-on-GPU or Q4-on-CPU, fixed in the pack. Adaptive
   residency would change an expert's weights with cache history. The hot set is static per pack (from the routing
   census); a promotion band holding both copies is an option only if RAM allows.
8. **PP streams the cold set over PCIe** per 8,192-token chunk: Q4 cold ≈ 32 GB / ~12.5 GB/s (GPU 2, x8) ≈ 2.5 s ⇒
   PP cap ~3,200 tok/s *(est.)*, above the bar but not by much on x8.

## 2b. Codec: X3.1, not X3

The newest codec is **X3.1** = the x3up encoder (exllamav3 v1.5.1 refit/drift + our v2) + fractional rates
(K3.5 on PAW-27B; decoded by llama-paw's x3v batched GEMV, commits 77c82f660 / 873c8b4f5). PAW-27B-X3.1: KL vs
BF16 0.0231 FineWeb / 0.0383 WikiText-2, top-1 92.5%.

**Flash-Next has no X3.1 encode yet**; the pack on disk (`20260918_ngram_q8`) is X3 experts + Q8_0 dense.
Re-encoding 24,576 experts at ~28.5 s each is ~190 GPU-h on one GPU *(est., from the 320-expert band run)*, which
competes with engine work for GPU 2. So: build and gate the engine on the X3 experts (same kernel family), and
schedule the X3.1 expert encode when a GPU is free. The dense part is re-encoded with X3.1 in CP1 (small).

## 2c. The TRUSS re-encode: X3.1 with a calculated fit (user, 2026-09-29)

Flash-Next is re-encoded to X3.1 with a structure **computed for the 1-GPU box**, the way Strata sizes its pack,
instead of one uniform rate:

- **Only the hot set is trellis.** Cold experts go to Q4_K (from bf16, imatrix) for the CPU pool. Trellis encode
  work shrinks to the hot set (~11–15k of 24,576 experts), roughly halving the ~190 GPU-h *(est.)*.
- **A solver picks, per expert, residency and rate**, minimizing usage-weighted error:
  `sum_e freq(e) * err(e, fmt(e))` subject to
  - VRAM: dense + hot experts + KV window + GDN/spec state + PP buffers ≤ 24 GB (budget sheet, CP0e);
  - RAM: cold Q4 + PLE + KV + pinned buffers ≤ ~80 GB (renters use the rest);
  - speed: expected misses per window × CPU miss cost (CP0c) + hot bytes / kernel rate ≤ the round budget that
    meets the Strata TG bar.
  Inputs: `freq` from the routing census (CP0b), `err` per expert and rate from encode MSE × Hessian (existing band
  runs give the curve), costs from CP0a/CP0c. Frequent experts earn higher K; the tail sits in Q4 on the CPU.
- **Dense part**: X3.1 mixed 3–5 bit, gated by KL (shared with a future 2-GPU pack).
- **PLE table**: smaller than today's 49 GB Q8 (Strata: 28.8 GB) so RAM fits; format gated by KL.
- Output: one TRUSS pack for "1×3090 24 GB"; the solver re-runs for other boxes (3060 12 GB, 2-GPU D2) with no
  new encode for experts whose rate is unchanged.

Order: CP0 measures the solver inputs → solver → encode hot set (GPU 2, long job) while the engine is built on the
existing X3 experts (same kernel family) → swap in the X3.1 pack at CP4.

## 3. Engine shape (fixed now so models 2 and 3 fit)

- **Model descriptor**: per layer {mixer: gdn | dsa_attn | attn} × {ffn: moe(n, k, shared) | dense}, tensor → codec.
- **Codecs**: trellis X3.1 (GPU only; fractional rates, x3v kernels), Q4_K / Q8_0 (CPU and GPU), f16/f32. A kernel is chosen per (codec, M range).
- **Pack**: TRUSS native file converted from our GGUFs. Per-expert contiguous blobs (one DMA per expert), hot/cold
  tag, dense tiles, residency map. Cold experts are re-quantized **from bf16**, not from trellis.
- **Runtime**: one captured CUDA graph per window size; routing decided on the device; hits computed on the GPU;
  misses posted to a CPU pool through flags in mapped pinned memory (no `cudaMemcpy`/sync on the token path); MTP
  draft inside the graph; GDN rollback by snapshot.

## 4. Checkpoints (each ends with a report, measured numbers and a go / no-go)

| CP | What | Gate |
|---|---|---|
| **0** | **Ground truth, no engine code.** (a) trellis expert GEMV at M=1..5 on GPU 2 vs read ceiling; (b) routing census → static hot-set coverage and unique-experts-per-window curve; (c) CPU Q4_K / Q3_K / Q2_K expert GEMV time on this CPU under renter load; (d) quality anchor: existing X3 KL numbers + a way to get Strata Q2_0 KL; (e) VRAM/RAM budget sheet at 256K | numbers in a report; pick hot-set size, cold format, PLE format |
| 1 | Fit solver (§2c) → X3.1 re-encode plan; pack converter (GGUF + bf16 → TRUSS pack) and loader. Engine development runs on the existing **X3** experts until the X3.1 hot-set encode lands | byte counts = budget sheet; checksums |
| 2 | Reference forward, one token, layer by layer | each layer's output matches llama-paw on the same weights (rel. err ≤ 1e-3) |
| 3 | Kernels: trellis window GEMV, dense trellis, GDN + snapshots, DSA indexer + sparse attn with KV in RAM, head + sampling, CPU Q4 pool | each ≥ 80% of its read ceiling (or best measured) |
| 4 | Decode engine, no spec: device routing, GPU hits, CPU misses via doorbells, one graph per token | KL vs Q8 ≤ CP0 anchor; raw TG ≥ 50 |
| 5 | MTP spec window (4–5 rows), GDN rollback | TG ≥ Strata on GPU 2 at 4K and 32K |
| 6 | PP: 8,192-token chunks, trellis GEMM on tensor cores, cold Q4 streamed + overlapped | PP ≥ Strata at 4K and 32K |
| 7 | 256K: KV in pinned RAM + resident window | needles 5/5 to 262K; TG ≥ 61, PP ≥ 1,300 at 262K |
| 8 | Server (OpenAI/Anthropic API, prompt-cache checkpoints) | the bench script runs unchanged against it |
| 9 | PAW X3.1 27B, PAW 35B on the same engine | each ≥ its current llama-paw speed at equal KL |

Kill rules:
- CP0a trellis GEMV < 60% of the read ceiling at M=4 and no fix in sight: the hot path is slower than Q2_0 by
  construction; stop and re-think the codec for the hot set.
- CP5 TG < 0.8× Strata after the round is profiled: stop, report where the time goes.
- KL not better than Strata Q2_0 at CP4: the accuracy reason for TRUSS is gone; report before continuing.

## 5. Where things live

- Engine code and design: `~/trellis-kernel` (this repo).
- Experiments and data: `~/ML_projects/flashnext/<YYYYMMDD>_truss_cp<N>/`, scripts `flashnext/scripts/flashnext_truss_*`.
- GPU: GPU 2 only (user, 2026-09-29). Check it is free before every run; stop if a renter takes it.

## CP4 note (2026-09-30): fill the MoE window's bubbles inside the layer kernel

v7a (CP3a) runs each GEMV item at the decode ceiling. A 4-row K2 window still loses ~30 µs to organisation:
routing plus input Hadamard at the start (~12–15 µs) and the dependency tail (~10–15 µs); see TRACKER #26. None of
this can be removed inside a stand-alone MoE kernel. The layer kernel reuses v7a's queue and counters and adds
items that don't depend on routing:

| Work (per layer, Flash-Next GGUF) | Size | Depends on |
|---|---|---|
| Router `ffn_gate_inp`, 2560×512, **F32** | 5.2 MB (2.6 MB as fp16 in the TRUSS pack) | FFN input x |
| Shared expert, 2560×640 ×3, Q8_0 | 5.2 MB | x |
| `hc_ffn_down` / `hc_ffn_up`, 10240×320 each, Q8_0 | 3.5 MB each | before x / after combine |

Order of work:
1. Routing, then the Hadamard of the routed inputs, run on a few blocks.
2. Meanwhile, the other blocks stream the shared expert.
3. Routed gate/up and down items follow.
4. The shared expert's output joins the combine.
