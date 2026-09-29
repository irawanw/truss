# 03 — Levers: what can speed up trellis decode without changing accuracy

"Same accuracy" means **the same weights** (the X3.1 file, unchanged) and no change to what the
model computes, except rounding order where explicitly allowed. Nothing here re-encodes, prunes
or re-quantizes the model.

## Exactness classes

Every lever carries one of these labels, and its gate follows from the label.

| class | meaning | gate |
|---|---|---|
| **E0** | bit-identical outputs to the reference path | greedy output hash identical; per-layer outputs bit-equal |
| **E1** | identical weights and math, different floating-point summation order (split-K, fused reductions) | chatcode KL vs Q8_0 (64×2048, paired bootstrap, the protocol in [S15]): the CI of the difference vs llama-paw X3.1 must include 0 |
| **E2** | a numerics change: fp16 accumulate, int8 activations | E1 gate **plus** a paired benchmark A/B (HumanEval+, MBPP+, MMLU-Pro subset), as done for fp16-accumulate GEMM [S18c] |

The 0.2% relative-RMS block gate is **not** the right gate for accumulator changes. fp16
accumulate failed it (2–4e-3) yet was null on three benchmarks (0 discordant HumanEval+ items)
[S18c].

## A. Kernel levers (the trellis matmul)

| # | lever | evidence | class | expected effect | phase |
|---|---|---|---|---|---|
| A1 | **Load-time repack of trellis words into the kernel's read order.** A lossless permutation: each lane's two window words land adjacent and 16-byte aligned, so K3/K3.5 windows use fixed shifts, not per-lane variable shifts. | 3-bit decode costs 2.3× 4-bit, and the variable shift is the cause [S1] §3. TensorFold's regrouping took GB10 from 107–130 to 200–220 GB/s. | E0 (same values) | 3-bit/K3.5 decode toward the 4-bit cost. [S1] §6 estimated ~6% of AR for 3-bit; **K3.5 is unmeasured** | 1 |
| A2 | **Marlin-style structure**: stripes of weight tiles per SM, multi-stage `cp.async` pipeline with an L2 `evict_first` policy on weights, activations staged once per block, in-order reduction. | The sglang-exl3 Marlin-template kernel: 13.3 vs 19.2 ms for all 27B linears (≈686 GB/s at 3 bpw). Marlin's design reaches near-ideal speed at batch 1–16 on A10 (sm_86). | E1 | The largest item. Target ≥ 1.15× the best existing kernel per shape at K3.5 | 1 |
| A3 | **No cooperative launch.** Do the input Hadamard per block (each block transforms the 128-wide slices it needs) and the output Hadamard/svh in the epilogue, so there is no `grid.sync`. | grid.sync + Hadamard folds ≈5% of the kernel [S1] §7; `grid.sync` ~7 µs [S7]; the cooperative launch also caps the grid [S1] §4. | E0 if the same FWHT order is kept | ~5% plus freedom to size the grid | 1 |
| A4 | **Row-invariant M=1..16 kernel**: the K split depends only on the shape; fixed-order sums; no atomics; no cross-row reduction. | TensorFold keeps drafted output byte-identical; our verify tier (x3g) is *not* bit-identical to the GEMV (0.2–1.0% RMS) [S18e]. | E1 vs llama-paw, and **E0 between serial and drafted** | Makes speculative decoding exact. Its cost at M=1 must be measured (the "exactness tax") | 1 |
| A5 | **Bit-width dispatch by measurement**: int8 activations for the 3-bit/K3.5 m=1 shapes where they win. | int8 +11–37% on 3-bit at m=1, fp16 wins 4-bit, and int8 degrades with M [S2] E1. | E2 (activation quantization) | −10.6% GEMV time on B3.5 [S2] (K3.5 unknown) | 1 (A/B arm) |
| A6 | **fp16 accumulate, folded to fp32 on a cadence.** | exllamav3 reports ~14% at batch 1 on a 3090 (code comment); prefill A/B quality-null [S18c]. | E2 | Reduces MMA/fold cost, which is 8–24% of the skeleton [S2] | 1 |
| A7 | **Fused multi-output launches**: QKV+gate in one launch, gate+up in one launch sharing the input Hadamard, SiLU·up folded into down's prologue. | exllamav3 MGEMM "sliced mode"; the "fuse to remove a round trip" rule [S7]. | E0/E1 | Fewer launches (397 → ~200) and one activation transform instead of 2–3 | 2 |
| A8 | **Fold RMSNorm + f32→f16 + suh into the prologue** and **svh + residual add into the epilogue**. | The cast fold was measured bit-identical, 0.48–0.56 ms/token [S2] E4, [S11]. | E0 | ~0.5–1 ms/token | 2 |
| A9 | **K3.5 tensor-core GEMM for 17..1023 rows** (port exllamav3 `half_k` GEMM). | Missing for K3.5 [S15]; x3g took nt=8 from 2.57× to 1.42× for K3/K4 [S6]. | E1 | Verify windows >16 and mid-size prefill chunks | 4 |
| A10 | **Reconstruct + fp16-acc cuBLAS for ≥1024 rows.** | PP8192 1215 tok/s, quality-null [S18c]. | E2 (already accepted) | Keep as is | 4 |

Deliberately not pursued as kernel levers, with the reason:

- **Codebook changes or re-encoding** — not bytes-bound [S1] §6, and the format must stay
  EXL3-compatible.
- **LUT codebooks** — mul1 is computed; the LUT result was for the older codec.
- **Grid sizing** — closed [S1] §4.
- **Hopper/Blackwell features** (TMA, wgmma, programmatic dependent launch, FP8/NVFP4 tensor
  cores) — none exist on sm_86.

## B. Framework levers (everything around the matmul)

| # | lever | evidence | class | expected effect | phase |
|---|---|---|---|---|---|
| B1 | **One CUDA graph per decode step**, per row bucket (1, 2, 4, 8, 16). Position and lengths live in device scalars; no host sync inside a step. | llama-paw: 1987 launches/token, 2.57 ms idle/token [S3]; exllamav3 BC_* graphs. | E0 | Removes most idle time | 2 |
| B2 | **Device-side speculative round**: draft → verify → accept → commit KV/GDN state → next draft, all on the GPU. The host only reads committed tokens from a pinned ring. | Host work is 6.72 ms of a 48.49 ms round (14%) [S10] §2. | E0 | Up to ~6.7 ms per round | 3 |
| B3 | **CUDA conditional WHILE node** (CUDA ≥ 12.3) to run N rounds without the host. | Documented in the [CUDA graphs guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html); **sm_86 support on this driver is unverified**, so Phase 0 probe P0.5 checks it. | E0 | Removes the remaining per-round host latency | 3 (if the probe passes) |
| B4 | **Static memory arena**: every buffer planned at load; no allocation, pool assertions or LIFO rules at run time. | llama-paw's pool-LIFO crash while restructuring `xh` [S11] §1. | E0 | Reliability; enables B1 | 2 |
| B5 | **Pinned host embedding** (optional) to free ~0.7 GB of VRAM for context. | sglang-exl3 `EMBED_HOST`. | E0 | VRAM only | 5 |

## C. Model-architecture levers (same math, fewer bytes moved or kernels issued)

| # | lever | evidence | class | expected effect | phase |
|---|---|---|---|---|---|
| C1 | **GQA-batched flash-decode** (K/V loaded once for the query heads that share them; ratio 6). | +30.2% decode at 242k [S18d]. | E1 | Long-context decode | 2 |
| C2 | **Fused GDN decode step**: conv1d update + L2 norms + gated delta recurrence + gated RMSNorm in one kernel, state in registers/shared memory per head. | GDN is 1.35 ms/round in 48 launches [S12]; exllamav3 BC_GDN. | E1 | ~0.5–1 ms/token | 2 |
| C3 | **KV cache at q8_0, with int4-Hadamard as an option** for 256k with a drafter, or for the 3060. | q8_0 is quality-free [S18]; EXL3 cq4 vs cq8 moved TG 0.7% [S9]; a drafter at 256k needs ~4.5-bit KV on 24 GB [S9]. | E2 for int4 | Context capacity | 4, 6a |
| C4 | **GDN state checkpoints for rollback and prefix reuse** (per verify position, and every N tokens for the prefix cache). | Speculative rejection needs GDN rewind; exllamav3 "GDN: Optimize rewind". | E0 | Correctness for B2; faster continuation TTFT | 3, 5 |

## D. Speculative-decoding levers (the output never changes; only speed moves)

| # | lever | evidence | class | expected effect | phase |
|---|---|---|---|---|---|
| D1 | **DFlash2 with KV-only catch-up** (already in llama-paw, measured cheap), running on our kernels. | [S11]; exllamav3 `update_kv_from_target`. | E0 by construction (target verifies) | Baseline drafter | 3 |
| D2 | **Hot-vocabulary draft head**: the drafter reads only the ids that cover ≥99.5% of committed tokens; verification keeps the full head. | TensorFold 98,304 ids = 99.6%; sglang-exl3 32k hot ids. The drafter head costs 0.626 ms per block row [S10] §3. | E0 | Most of the 6.3 ms select cost | 3 |
| D3 | **Window width chosen by measurement**, per request (6/8/12/16). | TensorFold: 12 rows = 16 rows in acceptance, 3 ms cheaper. | E0 | Round cost | 3 |
| D4 | **MTP head as a second drafter, chosen per request by committed tokens per ms.** | TensorFold GLM: 52.9 → 66.3 greedy code. | E0 | Workload-dependent | 3 |
| D5 | **n-gram / copy drafts from the context** (file edits, repeated code). | TensorFold: file-edit prompts reach 190 tok/s on Flash Next. | E0 | Large on edit-heavy agent traffic | 3 |
| D6 | **Gumbel-keyed sampling** (`argmax(logit/T + g(seed, pos, token))`) so that sampled decoding is exact under drafting too. | TensorFold "Exact means byte-identical". | E0 (for a given seed) | Sampled requests keep the speculative speedup | 3 |
| D7 | **Draft trees verified in one forward**: best-first over DFlash2's top-16 candidate lattice, 4 children per node, ~12–16 rows, scores weighted by the target's own Gumbel noise (0.7) and calibrated on traced rounds. Needs tree attention (each node sees committed keys plus its own path), a **GDN recurrence along tree paths** (a node's state = its parent's state + one update), a tree conv1d, and per-node RoPE positions. | TensorFold M5: code 76.5 → 99.1 tok/s from trees alone; calibration +6.5% tokens; `gdn_tree.cu`, `attention.py` `_paths`. | E0 (the target verifies) | +20–30% tokens/round from the same drafter | 3 |
| D8 | **Copy rule**: an 8-gram match in the context proposes a verbatim continuation as a chain of up to 31 rows (TC tier) instead of a tree. | TensorFold: such a match existed in ~25% of tree rounds and was right 94% of the time; whole-file edits 239 → 293 tok/s with 31-node chains; 4,567-token copy task 319 tok/s. | E0 | Very large on agent edit traffic | 3 |
| D9 | **In-place commits + one-launch GDN replay** of the accepted path across all 48 GDN layers. | TensorFold: KV copies cost 7 ms/round at 20k and 12 ms at 40k before; 2.7 after. | E0 | Keeps rounds flat with context | 3 |
| D10 | **Session n-gram prior on tree scores** (weight 0.1, kept only while it helps). | TensorFold: 4.49 → 4.62 tokens/pass. | E0 | +3% | 3 |

**Correction to the previous version of this file:** it said tokens per round are "a property
of the drafter, not of the engine". That was wrong.

- The drafter bounds the **candidates**: TensorFold found the true token in the drafter's top-16
  for 6.5–7.0 tokens/round, and no scorer picked right more than ~75% of the time past depth 1.
- **How many of those candidates get verified** is set by the engine: tree vs chain, width, copy
  rule, and the cost per extra row.

Our 3.48 tokens/round was a *chain* at n_max 5 [S10]. It is not the drafter's ceiling.

**The central kernel requirement for speculative speed:** verify(12–16 rows) must cost ≤ ~1.2×
one row on the 3090. Today it is 1.52–1.58× (llama-paw x3g) and 1.65× at 16 (EXL3). The
arithmetic for why ~1.1–1.25× is physically available is in `02-landscape.md`. This makes lever
A4 (the row-invariant M ≤ 16 kernel) the most important kernel item, ahead of shaving M = 1.

## E. Deferred, and gated on evidence

| # | lever | why deferred | gate to start |
|---|---|---|---|
| E-1 | Persistent decode megakernel / fine-grained dataflow (counter-based dependencies, not `grid.sync`) | Megakernel gains shrink with model size (0.8B 1.55×, 8B 1.16×); `grid.sync` ~7 µs measured here | After Phase 3, nsys shows non-matmul + idle ≥ 3 ms/token |
| E-2 | Green contexts / SM partitioning so drafter and target run concurrently | sm_86 support not verified; a round is sequential by nature | A measured idle gap between draft and verify ≥ 1 ms |
| E-3 | L2 persistence window for GDN state / KV hot pages | 6 MB L2, and GDN state is 151 MB per token, so it cannot fit | Probe P0.5 shows a measurable gain on attention at ≥32k |
| E-4 | 2×3090 pipeline for MoE | No P2P; 30 µs round trip [S13] | Phase 6b only |

## Sources

[S1]–[S18] as listed in `01-evidence-the-wall.md`. [S18e] = memory `x3-tensorcore-gemm.md`.
External links are inline.
