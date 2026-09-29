# 02 — Landscape: who runs trellis on a 3090, how fast, and why

Checked 2026-09-27. "Measured here" means on this box. Everything else is published by the named
source and has **not** been reproduced here.

## 1. Speed table: Qwen3.8-27B on one RTX 3090

| engine | weights | no drafter (tok/s) | with drafter (tok/s) | source |
|---|---|---:|---:|---|
| llama-paw (ours) | PAW B3.5, 3.5 bpw | 40.68 (llama-bench tg128); ~33.5 in the 262k serving config | DFlash2: **100.45** (8k code, greedy) | measured here [S16], [S11] |
| llama-paw (ours) | PAW-27B-X3.1, K3.5 | **never measured** | never measured | [S15] |
| exllamav3 (Mia fork 1.4.2) | EXL3 3.5 bpw, 14.2 GB | **45.6** (3 runs); 42.6 at 1k via server | DFlash2: 139.6 (11k code), 164.4 (short) | measured here [S8], [S9], [S5] |
| exllamav3 community branch `355c6ee` | EXL3 4.0 bpw | 42.8 | MTP 116.3; DFlash2 **162.9** (GSM8K, 8k ctx, 5.66 tokens/round) | [r0b0tlab/qwen38-exl3-dflash2](https://github.com/r0b0tlab/qwen38-exl3-dflash2) |
| SGLang + sglang-exl3 plugin (0xSero) | turboderp EXL3 **3.0 bpw**, 13.84 GB | **52.4** | MTP code 141 / prose 96; DFlash2 **code 225** / prose 98 | [local-ai-registry recipes](https://github.com/0xSero/local-ai-registry/pull/112), [PR #96](https://github.com/0xSero/local-ai-registry/pull/96) |
| llama.cpp, UD-Q4_K_XL | 17.56 GB | 41.55 | MTP 66.4 | [HackMD study](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090) |
| vLLM (HyperQwen), W4A16 ~4.5 bpw | — | — | 127 single-user with MTP | [syv-ai/HyperQwen](https://github.com/syv-ai/qwen38-27b-rtx3090) |

**About "exllamav3 already reaches 50–60 tok/s":**

- The highest no-drafter number I can trace to a real measurement is **52.4 tok/s**, from SGLang
  with the sglang-exl3 plugin on a **3.0 bpw** EXL3 file.
- Stock exllamav3 measured here was 45.6 at 3.5 bpw; the community branch reports 42.8 at 4.0 bpw.
- A "58–69 tok/s TabbyAPI" figure appeared in a search summary, but I could not trace it to a
  measurement, so treat it as unverified.
- Either way, the direction is right: **the best Ampere EXL3 stack is ~29% ahead of llama-paw on
  plain decode**, and more than 2× ahead on code with DFlash2.

Normalized to our bit rate (arithmetic, assuming its linear time scales with bytes):

- sglang-exl3 at 3.5 bpw would take ~15.5 ms of linears + 5.8 ms other ≈ 21.3 ms, i.e. **~47 tok/s**.
- At equal bits, the ranking is roughly sglang-exl3 ~47, exllamav3 ~45.6, llama-paw ~40.7.

## 2. What each one does that matters

### exllamav3 (v1.5.2, 2026-09-26, MIT)

- **Three kernel tiers by row count M** (`exl3_gemm.cu`):
  - int8 GEMV for M ≤ 2;
  - fp16 QTIP-style GEMV for M ≤ 8 (no dynamic shared memory, 4–6 blocks/SM, warps split K,
    trellis windows resolved by lane shuffles);
  - autotuned cooperative tensor-core GEMM above 8.
- **K3.5 (fractional trellis) in every tier:** int8 GEMV (`exl3_gemv_int8_inst_*_h3`), fp16 GEMV
  (`exl3_gemv_half_inst.cu`), GEMM and MGEMM (`get_gemm_kernel_ptr(..., half_k)`) and MoE coop
  (`*_h3`). llama-paw has only the first. Our X3.1 file uses the **same format**: same mul1
  codebook, same K3.5 ring, same suh/svh basis [S14]. So X3.1 could run in exllamav3 once exported.
- **fp16 accumulation with a periodic fp32 fold** (`EXL3_GEMM_H_ACC`), about 14% at batch 1 on a
  3090 per their comment.
- **Whole blocks captured as C++ graphs** (`EXL3_BC_ATTN`, `EXL3_BC_GDN`, `EXL3_BC_MLA`): each
  decode attention or GDN block is one graph call with patched pointers, which removes Python
  host time.
- **DFlash2:** KV-only catch-up (`update_kv_from_target`), a top-K plus selector walk in CUDA,
  and the target head over the block rows.
- Their own note: "Ampere GPU performance at low bitrates needs work" (v1.5.0 release).

### sglang-exl3 (0xSero; recipes and container image are public; plugin source is not)

- "**Marlin-template EXL3 kernels for sm_86** at K=3/4/5 (ported from Hopper work)". Per their
  report, **all 27B linears per decode step take 13.3 ms against ExLlamaV3's 19.2 ms**
  ([PR #83](https://github.com/0xSero/local-ai-registry/pull/83)).
  - Arithmetic: 9.12 GB / 13.3 ms ≈ **686 GB/s at 3 bpw**, well above our 3-bit 428–471.
  - It is the only published evidence that the 3-bit decode penalty on Ampere can be beaten.
  - **K3.5 support is not stated.**
- MTP draft head over a **32k hot-token** vocabulary, with the target verifying on the full head.
- bf16 token embedding in pinned host memory (`SGLANG_EXL3_EMBED_HOST=1`) to free VRAM for context.
- fp16-accumulate cuBLAS on prefill: +46–49%, top-20 logprob KL 0.0015 nats vs fp32 accumulate.
- Full-decode CUDA graphs at batch 1; prefill graphs up to 240 tokens.
- The plugin repo `github.com/0xSero/sglang-exl3` asked for credentials when cloned, so its source
  is not available to us. Only the image digest and recipes are.

### TensorFold (MIT, Python + Triton + CUDA per model family)

- **Row-invariant kernels**: a row's bits do not depend on how many rows share the pass. The K
  split depends only on the shape, sums run in fixed order, there are no atomics, and there is no
  reduction across rows. Drafted output is byte-identical to serial decoding.
- **Re-laying weights at load for the read pattern:** 107–130 → 200–220 GB/s of GB10's 240.
- EXL3 experiment: a ~100-line grouped kernel decodes each 16×16 tile straight into MMA B
  fragments, with Hadamards as warp butterflies. It reads experts at **208–220 GB/s vs
  exllamav3's 127–158** on GB10.
- Draft head over part of the vocabulary: the first 98,304 ids cover 99.6% of committed tokens.
  A 4-bit copy of the head is used for drafting, and the full head for verifying.
- Window width chosen by measurement: 12 rows matched 16 on acceptance and saved 3 ms per round.
- Per-request drafter choice by tokens per millisecond: GLM greedy code went 52.9 → 66.3.
- Host vs GPU check: the 27B's 918 kernels take 12–14 ms of host time, hidden behind 50–85 ms of
  GPU time on GB10. **On a 3090, where a token is ~20 ms, a Python host loop would not stay hidden.**

### Why a DGX Spark reaches ~100 tok/s, and what that means for a 3090

The claim: TensorFold, Qwen3.8-27B, one Spark, DFlash2 drafts, **102.9 tok/s** against 12.9 serial
(8×). Per prompt: sequence 126.9, json 106.2, code 75.7. It is single pass, with long structured
outputs. On their standard bench (64-token replies, median of 5 seeds) the same engine does
**49.6 code / 45.8 chat** on one Spark ([recipe](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/qwen3.8-27b.md#dgx-spark-cuda)).

The **3090 has ~3.7× the Spark's read bandwidth**: 892.7 GB/s measured here, against ~240 GB/s
measured by TensorFold. Serial decode shows it: 40–52 tok/s on a 3090 against 12.9–13.1 on the
Spark. The Spark's 100 is not more hardware. It is a bigger **speculative multiplier**:

`tok/s = tokens per round ÷ (verify(rows) + draft + commit + host)`

| term | TensorFold, one Spark (measured by them) | llama-paw, one 3090 (measured here) |
|---|---|---|
| one-row forward | 76 ms (13.1 tok/s) | 24.6–29.9 ms |
| verify cost at 12 rows | **83 ms = 1.09× one row** | **1.52× one row** (x3g, 16 rows 1.58×) [S6]; EXL3 at 16 rows 1.65× [S5] |
| draft per round | 8 ms (DFlash2, 4-bit, 98k-id draft vocabulary, 288 kernels) | 6.3 ms (select) [S10] |
| commit | < 1 ms (in-place, one-launch GDN path replay) | inside decode_tgt |
| host per round | 12–14 ms, **hidden** behind 76–83 ms of GPU time | **6.7 ms exposed** [S10] |
| proposal | **trees**: best-first over DFlash2's top-16 lattice, 4 children per node, ~12 rows, scores carrying the target's Gumbel noise; plus a **copy rule** (8-gram match → verbatim chain) | a chain, n_max 5 |
| tokens per round | 4.7 (standard code bench) to ~7–12 (the tweet's long structured prompts; arithmetic from tok/s × round time) | 3.48 |

Three things make the Spark multiplier large:

1. **Rows are nearly free** there. At 76 ms per forward the weight stream hides everything else.
2. **Trees and copies commit more tokens per round from the same drafter.** On M5 Max, trees took
   code 76.5 → 99.1 tok/s (sampled); tiled weights and a draft vocabulary → 126. Whole-file edits
   reached 295–388 tok/s.
3. **Structured prompts:** sequence and json outputs are highly predictable.

On a 3090 the same design meets a harder physics problem. The forward is 3.7× shorter, so the
extra rows' compute and the per-row work (attention, GDN, head, activation transforms) are *not*
hidden unless the kernels are built for it. **Flattening the verify curve on a 3090 is the
central kernel problem for speculative decoding, and neither llama-paw (1.52×) nor exllamav3
(1.65× at 16) has solved it.**

Arithmetic for the ideal: 11 extra rows of tensor-core math on the 27B cost 2 × 24.3e9 × 11 =
0.53 TFLOP. That is 3.8 ms at fp16-accumulate peak, or 7.5 ms at fp32-accumulate, against ~13–16 ms
of weight streaming that it can overlap. So **verify(12) ≈ 1.1–1.25× one row is physically
available on a 3090**, if the kernel decodes each trellis tile once for all rows and overlaps the
MMA with the stream.

**What a TensorFold-class engine should do on a 3090 (arithmetic, not measured):**

- one row 18–20 ms; verify(12) at 1.2× = 22–24 ms; draft ~2.5–4 ms (bandwidth-bound, so ~3×
  faster than the Spark's 8 ms); commit ~0.5 ms; host ~0 with graphs. Round ≈ **25–29 ms**.
- Standard bench, 4.7 tokens/round → **~160–190 tok/s** (Spark: 49.6).
- Tweet-like prompts: code 7.2 → ~250–290, json 10.1 → ~350–400, sequence 12.1 → ~420–480
  (Spark: 75.7 / 106.2 / 126.9).
- That is ~3–3.5× the Spark on the same prompts, the same ratio as bandwidth. Existing 3090 bests
  are sglang-exl3 225 (code, DFlash2, 3.0 bpw) and llama-paw 100.45.

**Could TensorFold itself run on a 3090?**

- Probably. Its CUDA path is PyTorch + Triton (bf16, which sm_86 has) plus one small CUDA
  extension. It reads the MLX 4-bit checkpoint (~14.4 GB), which fits in 24 GB with its 4-bit
  drafter and moderate context.
- Two caveats: its Triton kernels were tuned on GB10, and its 12–14 ms Python host time per
  forward would **not** hide behind a ~20 ms 3090 forward.
- Running it is the fastest way to measure its design on our hardware. Added as Phase 0 item P0.7.

### Megakernel data points (for the deferred lever)

- **Lucebox megakernel** (local copy at `~/luce-megakernel`): Qwen3.5-0.8B bf16 on a 3090,
  413 vs 267 tok/s for llama.cpp (1.55×). It uses cooperative `grid.sync` *between layers*; a
  sync inside the GDN loop deadlocked. The model is small, so launch overhead dominates.
- **Mirage Persistent Kernel** (compiler-generated megakernel): Qwen3-8B on A100 went 14.5 → 12.5 ms
  (1.16×) ([MPK paper](https://arxiv.org/pdf/2512.22219)).
- **Hazy Research "No Bubbles"**: Llama-1B on H100 at 78% of memory bandwidth, >1.5× over
  vLLM/SGLang ([blog](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)).
- Reading these together: the gain shrinks as the model grows. For a 27B at ~18 ms/token,
  launch-related cost is ~2–3 ms. That is worth it only after everything cheaper is done.

### Other trellis runtimes (for reference and oracles)

- [exl3xpu](https://github.com/0xSero/exl3xpu) (MIT): EXL3 on Intel Arc, bit-exact vs exllamav3.
  It ships a **bit-exact PyTorch reference decoder** (`exl3xpu/ref.py`), useful as a test oracle.
- [glq](https://github.com/cnygaard/glq): QTIP-derived trellis in vLLM, validated on sm_86.
  **GPL-3.0, so do not copy code.**

## 3. What this means for the plan

1. **The kernel headroom is proven by someone else.** A 3-bit trellis matmul at ~686 GB/s on
   sm_86 exists (sglang-exl3). Ours runs 3-bit at 428–471. That gap is the largest single item,
   and it is not a codec property.
2. **The framework headroom is proven by our own profiles:** ~7 ms/token of non-matmul time plus
   idle, and 6.7 ms of host time per speculative round.
3. **No existing engine serves X3.1 at its K3.5 rate with fast kernels.** exllamav3 has the
   kernels but does not read our GGUF. llama-paw reads it but lacks the kernels.
4. **Serial decode has little headroom left**: the ceiling is 74.9 tok/s on X3.1, and the best
   existing stack is at an equal-bit ~47.
5. **Speculative decode has a lot.** The 3090's verify curve (1.52× at 12 rows) is far from the
   ~1.1–1.25× that physics allows. Chains leave tokens on the table that trees and copies take.
   The host round is exposed. A TensorFold-class design on a 3090 points to ~160–190 tok/s on
   ordinary coding replies and 250+ on long structured output (arithmetic above). **This, not
   serial decode, is where a new engine can make a large difference.**
