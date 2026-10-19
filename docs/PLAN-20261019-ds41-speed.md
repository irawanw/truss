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

### Measured 10-19 (slice bench, L0-5, GPU 0; TRACKER #188-#195)

| | decode ms/layer | TG at 40 layers (est.) | prefill ms/layer (2,048) | PP (est.) |
|---|---|---|---|---|
| start, all resident | 0.811 | 30.8 | 114 | 449 |
| + attention v2 (#189) + sk_final warp (#190) | **0.625** | **40.0** | **70.7** | **724** |
| 32% resident, PCIe only | 1.350 | 18.5 | 131 | 391 |
| 32% resident, CPU tier + hints (#191) | 1.080 | 23.1 | | |
| + tail skip R5 (#192) | 1.009 | 24.8 | | |
| all resident + grouped wo_a GEMM (#194) | 0.624 | 40.1 | **56.5** | **906** |
| 8,192-token chunk (1,176 of 1,536 resident) | | | 215 per 8K | 953 |

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

## Compared with Flash-Next on TRUSS (4,363 tok/s, 2 GPUs) and tqLLM (10-19, user: "check deeply")

**Work per token** (FLOPs, from the shapes; Flash-Next from `ledger-prefill-20261006.md`):
| per token per layer | Flash-Next (48 layers) | DS41 (40 layers) |
|---|---|---|
| routed experts | 10 x 3 x 640 x 2,560 x 2 = 98 M | 6 x 3 x 2,304 x 5,120 x 2 = 425 M |
| dense Q8 (attention projections, shared expert, indexer, mHC) | 3.6 G params / 48 x 2 = 150 M | ~7.6 GB Q8 / 40 x 2 = ~380 M |
| attention | DSA in 12 of 48 layers, 2,048 keys x 256 x 24 heads: ~13 M average | every layer, ~640 keys x 512 x 64 heads: ~84 M + indexer |
| **whole model** | **~13 GFLOP / token** | **~36 GFLOP / token (2.8x)** |

So Flash-Next's 2,259 (one GPU) / 4,363 (two) at 150K is worth ~810 / ~1,560 tok/s of DS41 work at the same
efficiency. The DS41 slice already does ~1,000 per GPU (#196, E3 32-head attention 51 ms/layer): per GPU we are
at ~1.2x Flash-Next's efficiency. **2,000 = both GPUs pipelined (Flash-Next's 1.93x, #155) x ~1,050 per GPU
with the PCIe streaming hidden.** Items, by what they cost:

1. **The recipe that gave Flash-Next 4,363 was never run on DS41.** #185/#187 (247 tok/s) = one 2,048-token chunk
   through `tk-bench-decode`: no chunk pipelining (needs many chunks, `Pipeline::run_prompt` in `tk-bench-spec`),
   no copy/compute overlap across chunks, and 2K chunks re-stream every cold expert per 2K tokens. Flash-Next's
   number: 150K prompt, 8,192-token chunks, run_prompt pipelined, link busy 75% of a compute-bound chunk.
   -> **P9: one full-model prefill run in that recipe** (32K prompt, chunk 8K; ~3 min on GPUs 0 + 2, asks user).
2. **PCIe at 2,000 tok/s** (*est.*): ~67 GB cold experts per 8K chunk = 4.1 s per chunk at 2,000. Split 14:
   GPU 2 (x4, 6.7 GB/s) 14 layers ~23 GB = 3.5 s; GPU 0 (x8) 26 layers ~44 GB = 3.3 s -> fits, both links ~85%
   busy. 16K chunks halve it but cost VRAM (residency). Compute per GPU at split 14: GPU 0 needs 26 layers in
   4.1 s = 19 us/token/layer vs 25 today -> the kernel work below is still needed.
3. **Dense prompt GEMMs are fp16 HMMA (W8A16, dequant + cuBLAS: 47-71 TFLOPS, #44); Flash-Next's ledger ran
   int8 IMMA (Q8_1 activations, `q8_gemm.cu` 65-94 TOPS, 120 peak at 1,440 MHz).** -> **P7: int8 prompt rows,
   fp16 decode** (`TRUSS_ACT=q8` exists for both; a prompt-only switch is new). Dense is 11.6 of 51 ms/layer;
   ~-5 ms if 2x. Quality gate `ds41_kl` (Flash-Next: KL 0.065 vs 0.040 for activations, both phases).
4. **Experts in prefill: trellis -> fp16 HMMA at 0.69-0.78 of peak** (ledger: "at target"; DS41 18.4 ms per 2K =
   ~47 TFLOPS). Only int8 doubles that ceiling. tqLLM's X3 prefill reconstructs trellis weights straight to int8
   with load-time row scales into the GEMM layout (+18-22% then +7% pp, commits 367611d / 30c8d50 / c3b611f).
   -> **P8: trellis -> int8 tile + int8 activations for expert prompt rows** (error: weight row-scale rounding +
   activation int8; gate `ds41_kl`).
5. Attention: Flash-Next's DSA attention was its worst kernel too (eta 0.11, ledger item 4a); our E3 32-head
   kernel continues that line (56.8 -> 51.0 ms/layer, parity PASS).
6. **tqLLM decode lever: retrieval drafts** (`--draft-retrieve 32`: draft tokens copied from the prompt / history,
   no drafter model). Coding agents re-emit file text, so it is a free drafter while DSpark / MTP is not ported.
   -> **D11** (spec gate: accepted tokens per round on a coding transcript; cold-expert bytes per round as D7).

## CurtisAccelerate 1017 EXL3 on 4x CMP 170HX (user link 10-19)
github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX (+ `DeepSeekv41-Flash-EXL3-TC-MoE` kernels):
our exact model, routed experts EXL3 2.9 bpw (trellis, K3, layers 18-22 K2), everything else official FP8, vLLM
TP4 on 4 x 64 GB = **all 384 experts resident**, Engram on the host (265 GB pinned), DSpark depth 5.
- Decode (one user, 140 W): structured 180 / **code 137** / prose 97 tok/s; cycle ~34 ms; accepted tokens per
  step 5.97 structured, ~4.7 code (137 x 0.034), 3.31 prose. **This is the first measured DSpark acceptance for
  our model: ~4.7 tokens per round on code.**
- Prefill 1,520 tok/s flat 2K-32K, 1,460 at 64K, 1,330 at 128K (chunk 2,048, `--max-num-batched-tokens 2048`).
- Kernel rule by rows: cooperative kernel 1-42 rows, `moe_out_v2` 43-64, `moe_out_prefill` 65-2,048 (tensor-core
  trellis decode to fp16 fragments, hi/lo fp16 split of bf16 activations, fp32 accumulate); one shared workspace.
What it changes for us: with all experts resident they spend ~34 ms per 6-row round. Our round = GPU side (~25-30
ms for 6 rows if the window kernels hold) + the cold bytes of ~6 x 0.87 x the misses per token. At ~4.7 accepted
per round on code, **70 tok/s on code = a round of <= 67 ms** -> the cold bytes per round must fit in ~35-40 ms
of CPU + PCIe (~1.5-2 GB per round): needs higher residency (D8 ring fix, D9 dense 4.5 bpw), skips (D3/D10) and
lookahead. **D7 DSpark moves up: it is the multiplier that makes 70 possible on code** (prose at ~3.3 per round
stays nearer 45).

## Execution from 10-19 evening: planner + executor
User (10-19): run the work through opencode2 with `opencode/deepseek-v4.1-flash-free` as the executor; Claude plans,
writes one brief per batch, reviews the diff and report, runs the gates it doubts, commits and writes TRACKER rows.
Briefs and reports: `~/ML_projects/dsv41flash/20261019_truss_port/exec/` (`E<n>_brief.md` -> `E<n>_report.md`).
Only slice benches on GPU 0 (no full-model runs, user 10-19: "use more cheap pp and tg").

| batch | what | status |
|---|---|---|
| E1 | slice prefill / decode at 8K / 16K / 32K context, chunk 2K / 8K (M0-D, P1, P5 sizing) | done #196: 906 tok/s at 32K, attention flat per token, no O(T^2) yet |
| E2 | D10 weight-share skip `TRUSS_SKIP_WEIGHT=F`, golden gate, slice speed vs tail skip R5 | done #197: golden identical; F=0.12 27.8 vs R5 24.9 tok/s, KL open |
| E3 | prefill sparse attention: cp.async, 32 heads per CTA | done #198: v5 default, 1,042 tok/s est. |
| E4 | P7 speed side: slice prefill with int8 dense prompt rows (`TRUSS_ACT=q8`) vs fp16; then a prompt-only switch | queued |
| next | expert window decode kernel to ~85% bandwidth; P8 int8 expert prefill; prefill attention 32 heads/CTA + cp.async; sk_part on tensor cores; D9 4.5-bit dense; split at 14 + pipelined prefill | queued |

KL for E2 / D3 (the full-model `ds41_kl`, two GPUs) waits for the user's OK.

## Rules carried
- At most GPUs 0 + 2, every GPU job under `run_watched2.sh` (renter wins). CPU tier competes with renters for cores
  and RAM bandwidth: measure under their load, never assume an idle box.
- Judge quality only with TRUSS (`ds41_kl`) or exact math, never the Python swap simulation.
- Files < 500 lines, model code in `src/model/deepseek41/`, docs/reference updated with each change.

## Prefill budget after #194 (slice, per layer at 2,048 rows: 56.5 ms; 2,000 tok/s needs <= 25.6)

experts 18.4 (gate_up 12.5 + down 5.9), sparse attention 12.5 (~10 TFLOPS: each CTA re-reads its row's 640 keys,
5.4 GB/layer of key traffic; next: 32 heads per CTA + cp.async), dense GEMMs 11.6, mHC sk_part 3.2 (fp32 CUDA cores
-> tensor cores), rest ~4. PCIe (*est.*): ~65 GB of cold experts per chunk; split at 14 (GPU 2 x4 takes layers 0-13)
balances the links at ~3.5 s per chunk -> 8,192-token chunks cap at ~2,300 tok/s with both stages pipelined.
256K: the indexer's O(T^2) scores (~8.6 GFLOP/token at the end of a 256K prompt over 8 index layers) must run on
tensor cores; not measured yet (slice at 32K next).
