# PLAN: DeepSeek-V4.1 pack on TRUSS: 70 tok/s decode, 2,000 tok/s prefill, GPUs 0 + 2 (10-19)

User (10-19 10:12): pivot to speed first (pack quality fix F1-F4 paused, `PLAN-20261019-dsv41.md`). Take the levers
from Strata (`~/strata_latest`), mmnga/Strata `deepseek` (`~/strata_mmnga_deepseek`, our exact model) and PolyStrata
(`~/polystrata`). **No lever is called infeasible before it is measured on this box.** Every step = one TRACKER row
(#185 on). Golden gates every shared-code change; KL (`ds41_kl`, 0.0578 now) gates every change that touches math.

## Where we start (measured)

| | now | target | gap |
|---|---|---|---|
| prefill, 2,048-token prompt | **247 tok/s** (8.3 s) | 2,000 at 256K | 8x |
| decode, no spec | **19.3 tok/s** mean, 25 median (40 ms), spikes to 242 ms | 70 with spec | 3.6x |

That run (#185) was a crude baseline: random-token prompt, index-order residency (not our hot set), no CPU tier,
one 64-step series. M0 (running) redoes it right.

## What others get (their box vs ours)

| engine | model / box | decode | prefill |
|---|---|---|---|
| mmnga/Strata `deepseek` | **our model**, Q4_K_M all 384 experts, 2x 5090, 410 GB RAM at ~300 GB/s | **60.7** (DSpark depth 5, 256K) | not reported |
| PolyStrata | V4 Flash 0731 (sibling), 91 GB, one 24 GB card + 128 GB RAM | 29-36, no drafter | 688-805 (10K), 393 (252K) |
| PolyStrata | same, one 5090 + 120 GB | 48-53 | 1,003-1,147 |
| Strata main | Flash-Next, 5070 12 GB | 94 | 2,650 |

Our box: 2x 3090 (GPU 0 x8 ~13 GB/s, **GPU 2 linked x4 ~6.7 GB/s**, slot is x8), 3960X 24 cores AVX2, 125 GB RAM
~55 GB/s shared with the renters (load avg 15-19 from them), CPU trellis 42 GB/s from RAM (R6).

## Budget arithmetic (*est.*, M0 replaces these with measurements)

**Decode, per token.** 70 tok/s = 14.3 ms.
- GPU side: dense ~5 GB Q8_0 + ~65 resident routed experts per token -> ~12 ms at 850 GB/s if kernels hit
  bandwidth. Today a step is ~35-40 ms with almost no fetches, so **the GPU side itself is ~3x too slow** (v1
  attention / indexer / mHC kernels, launches, the GPU 0 <-> 2 hand-off). M0-C (nsys) says which.
- Cold side: ~29% of 240 routed slots miss with the 73-hot set (R1 held-out coverage 0.713) = ~70 misses x 8.85 MB
  (K2) = ~620 MB/token. At 70 tok/s that is 43 GB/s, against CPU 42 + PCIe 13 + 6.7 GB/s. Tail skip >=4 (-41%,
  R4) -> ~370 MB/token = 26 GB/s. Window heat / locality (R3 +6 points, PolyStrata 20% -> 50% on code) lowers it
  more. **Feasible on paper only if CPU, PCIe and GPU overlap fully.**
- Spec (DSpark): amortizes the GPU side over accepted tokens, but distinct misses per row only drop 13% at
  window 6 (R5), so cold bytes per accepted token go **up** with wide windows. The window width is a trade we
  measure, not assume (mmnga uses depth 5 only because their RAM is 6x ours).

**Prefill, per 2,048-token chunk.** ~65 GB of experts are not in VRAM; stream mode copies every layer's cold
experts per chunk: GPU 2 (layers 0-19, ~33 GB) at 6.7 GB/s = ~4.9 s, GPU 0 (~32 GB) at 13 GB/s = ~2.5 s. Run one
after the other = ~7.4 s: **that is our 8.3 s. Prefill is PCIe-bound, not compute-bound.** 2,000 tok/s needs the
copies amortized over more tokens and both links busy at the same time.

## Phases

### Measured 10-19 (slice bench, L0-5, GPU 0; TRACKER #188-#192)

| | decode ms/layer | TG at 40 layers (est.) | prefill ms/layer (2,048) | PP (est.) |
|---|---|---|---|---|
| start, all resident | 0.811 | 30.8 | 114 | 449 |
| + attention v2 (#189) + sk_final warp (#190) | **0.625** | **40.0** | **70.7** | **724** |
| 32% resident, PCIe only | 1.350 | 18.5 | 131 | 391 |
| 32% resident, CPU tier + hints (#191) | 1.080 | 23.1 | | |
| + tail skip R5 (#192) | 1.009 | 24.8 | | |

What this says about 70 tok/s: the budget is 0.357 ms per layer per token. All-resident is 0.625 (dense gemv 0.27 at ~75%
of bandwidth, experts 0.17 at ~41%, attention projections + norms ~0.1, mHC 0.09). So the GPU side needs every kernel
near peak AND the cold path (~1.1 misses x 8.9 MB per layer per token) fully hidden behind it. Speculation amortizes
the GPU side but multiplies distinct misses per round, so it only pays once misses are cheap. Biggest levers, in
order: D9 (dense to ~4.5 bits: -4.5 ms/token and +~400 resident experts), expert window kernel to ~85% bandwidth,
CPU misses overlapped with the GPU (late join), weight-share skip (PolyStrata `--skip-weight`, 10-12% over rank skip
at equal KL on GLM), then DSpark. Every quality-changing lever is gated by `ds41_kl`.

| id | lever | source | gate |
|---|---|---|---|
| D9 | dense weights (attention projections, shared expert; 8 GB/token at Q8_0) to trellis ~4.5 bpw | PAW dense recipe | `ds41_kl` cost; TG |
| D10 | skip by weight share (an expert outside VRAM under F of its row's weight in every row) | PolyStrata #34 | `ds41_kl` at equal speed vs rank skip |

### M0 Measure (running 10:16, `20261019_truss_port/data/chain_1016.sh`)
| id | what | output |
|---|---|---|
| M0-A | real code prompt (2,048 held-out code tokens), calib-usage hot set (`usage_calib256.f32`), GPU tiers only | PP, TG, misses/step |
| M0-B | + CPU trellis tier and Flash-Next's served knobs (CPU dynamic, PCIe frac 0.2, hint 3, admit idle 64, split rows) | same |
| M0-C | nsys trace of A | per-kernel and per-section ms of one decode step, prefill copy vs compute |
| M0-D | then: prefill at 8K / 32K / 128K / 256K (chunk 2K / 8K / 16K), so the O(T^2) indexer and attention share is known | PP curve |

### D Decode (to 70)
| id | lever | source | gate |
|---|---|---|---|
| D1 | GPU-side kernels from the M0-C top list: attention / indexer / compressor decode kernels, mHC fused (Sinkhorn + collapse + post in one), Engram row read off the critical path, CUDA graph of the step (`TRUSS_DECODE_GRAPH`) | ours; mmnga N25/N26/N32 (graph-captured FFN, score-grid QK, paired-head values) | GPU-only step <= 14 ms (all-resident proxy) |
| D2 | overlap of the two stages: GPU 2 starts token t+1's layers 0-19 only after GPU 0 ends t; in a spec window both stages hold rows, so pipeline the verify rows across the split | ours, Strata SECOND_GPU.md | step = max(stage), not sum |
| D3 | tail skip (missed expert of rank >= 4/5 -> 0) | Strata default 0.1.42 | KL (`ds41_kl`) cost <= +0.005 |
| D4 | lookahead: next layer's router on this input -> prefetch / CPU start one layer early (`TRUSS_HINT_K` exists, two-layer like mmnga) | mmnga | hint hit >= 74% (R5) |
| D5 | window-heat cache + persisted profile across starts | mmnga N40, PolyStrata | +hit points on a code answer |
| D6 | CPU tier tuned under renter load: threads / pinning to cores renters leave free, light experts joining one layer late (Strata #2690), mapped-memory signalling | Strata, mmnga N49 | CPU tier >= 35 GB/s in the engine |
| D7 | DSpark drafter (3 layers, block of 5, MXFP4 experts, port from mmnga's verified code), window width chosen by measured cold bytes per accepted token | mmnga | accepted tokens per round; tok/s |
| D8 | VRAM for experts: KV budget at 256K vs resident experts, the ring size, int8 KV, lend the KV tail (`TRUSS_KV_LEND`) while the context is short | ours #114 | resident experts +N |

### P Prefill (to 2,000)
| id | lever | source | gate |
|---|---|---|---|
| P1 | bigger chunks (8K-16K): the cold-expert copy per chunk is fixed, so tok/s scales with chunk size until compute dominates | PolyStrata (1,024-token sub-chunks, one copy per layer) | PP at 8K, 16K |
| P2 | both links at once: GPU 2 streams + computes chunk k+1's layers 0-19 while GPU 0 does chunk k's layers 20-39 (pipeline over chunks) | ours | PP = 1 / max(stage) |
| P3 | rebalance the split by link speed (GPU 0 x8 takes more cold layers; GPU 2 x4 fewer). **GPU 2 stays x4**: no reboot / reseat while the renter runs (user 10-19), so plan for 6.7 GB/s on GPU 2 | measured R-link | stage times equal |
| P4 | cold experts during prefill on the CPU for the rows that route to them sparsely (few tokens per expert) instead of copying the whole expert | Strata split path (`TRUSS_SPLIT_ROWS`) | bytes over PCIe per chunk |
| P5 | tensor-core prefill kernels: attention (64 heads x 512 latent), indexer O(T^2) at 256K, grouped o-LoRA as 8 small GEMMs not a dense T x 8 | ours; mmnga score-grid | PP at 128K / 256K |
| P6 | prompt cache: reuse KV of the previous turn (coding agent: most of a 256K prompt repeats) | Strata PROMPT_CACHE_TAIL, MESSAGE_BOUNDARY_CACHE | re-read tokens per turn |

### Order
M0 -> D1 + P1 (biggest, cheapest) -> D2 + P2/P3 -> D3-D6 -> D7 -> P5 -> P6. Re-plan after each with the measured
numbers. Pack quality (F1, layer-20 three-way check) resumes after D1/P1 or whenever the user says.

## Rules carried
- At most GPUs 0 + 2, every GPU job under `run_watched2.sh` (renter wins). CPU tier competes with renters for cores
  and RAM bandwidth: measure under their load, never assume an idle box.
- Judge quality only with TRUSS (`ds41_kl`) or exact math, never the Python swap simulation.
- Files < 500 lines, model code in `src/model/deepseek41/`, docs/reference updated with each change.
