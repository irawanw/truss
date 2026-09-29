# Loom plan — a trellis-only decode engine for RTX 3090 / 3060

Written 2026-09-27. Plan only; no code exists yet. Read `README.md` for the verdict, then
`docs/01…04` for the evidence behind every number below.

## 0. Rules this plan runs under (each learned the hard way on this project)

1. **Measure before claiming.** Every number here is labeled *measured* (with its source) or
   *estimate* (with its arithmetic). Estimates are never repeated as results.
2. **Every phase has a pass gate and a kill gate, written before the work starts.** A phase that
   misses its kill gate stops. Its useful parts are upstreamed into llama-paw or exllamav3 instead.
3. **Same accuracy.** Only exactness classes E0/E1/E2 from `docs/03-levers.md`, each with its
   gate. No re-encoding, no pruning, no new quantization.
4. **Ablate before rewriting.** Any "the cost is X" claim gets a one-hour ablation before code is
   restructured around it.
5. **Do not reopen closed levers** (`docs/01` §10) without new evidence.
6. **GPUs:** kernel iteration at op level on the RTX 3060 box (192.168.18.18, validated to
   reproduce 3090 encoder results). Every gate is confirmed on a 3090, only on the GPU the user
   names, under flock plus the renter watchdog. No subagents. Long jobs are resumable and
   detached (`setsid`, logs under `bench/results/`).

## 1. Scope

- **In:** EXL3-format trellis (mul1 codebook, K ∈ {2, 2.5, 3, 3.5, 4}); Qwen3.5/3.8 dense
  hybrid (GDN + full attention); PAW-27B-X3.1 first; one RTX 3090; then an RTX 3060 profile; then
  the Flash-Next MoE on 2×3090.
- **Out:** other formats, other model families (until Phase 6), CPU inference, TP over PCIe,
  non-Ampere tuning.

## 2. Targets and what physics allows (arithmetic, not measurement)

X3.1 per-token read at short context ≈ 11.9 GB (`docs/01` §2).

| quantity | value | how |
|---|---:|---|
| Hard ceiling, any engine, AR | **74.9 tok/s** | 11.9 GB ÷ 892.7 GB/s measured read ceiling |
| Loom AR estimate, low | 50 tok/s | linears 10.64 GB at 686 GB/s (the sglang-exl3 3-bit rate) = 15.5 ms, head 1.25 ms, rest 3.0 ms → 19.8 ms |
| Loom AR estimate, high | 58 tok/s | linears at 760 GB/s (85% of read ceiling) = 14.0 ms, head 1.25 ms, rest 2.0 ms → 17.3 ms |
| llama-paw today | 40.68 (B3.5, tg128) / X3.1 unmeasured | measured [S16] |
| best existing, equal bits | ~47 | sglang-exl3's 52.4 at 3.0 bpw, rescaled to 3.5 bpw (`docs/02` §1) |
| llama-paw speculative today | 100.45 | measured [S11]: chain, n_max 5, 3.48 tokens/round, verify 1.52× at 12 rows, host 6.7 ms/round |
| TensorFold, one DGX Spark | 49.6 code / 45.8 chat (standard bench); 75.7–126.9 on long structured prompts | published; a Spark has ~1/3.7 of the 3090's read bandwidth |
| **Loom speculative estimate, standard coding replies** | **~160–190 tok/s** | round = verify(12) at 1.2× one row (22–24 ms) + draft 2.5–4 ms + commit 0.5 ms = 25–29 ms; 4.7 tokens/round (TensorFold's tree + copy on the same DFlash2 checkpoint) |
| **Loom speculative estimate, long structured output / edits** | **~250–450 tok/s** | same round; 7–12 tokens/round (TensorFold's per-prompt numbers × round time) |
| existing 3090 best, speculative | 225 (sglang-exl3, code, 3.0 bpw) | published |

**Trellis-specific sensitivity (arithmetic).** The ~160–190 row above borrows TensorFold's
tokens/round and verify flatness, which were measured on **affine 4-bit**. For our trellis
file:

`tok/s = τ ÷ (r × T1 + ~3.5 ms draft+commit)`

where T1 = one-row forward, r = verify(12 rows) ÷ T1, and τ = tokens per round.

| scenario | T1 | r | τ = 4.7 (TF trees) | τ = 4.0 (weaker drafter) |
|---|---:|---:|---:|---:|
| today's kernels, trees added | 24.6 ms | 1.52 (measured) | 115 | 98 |
| kernel at sglang-exl3's trellis rate (~686 GB/s), exllamav3-like verify | 19.8 ms | 1.35 | 156 | 132 |
| same, flat verify | 19.8 ms | 1.20 | 172 | 147 |
| best case: 85% of the read ceiling, very flat verify | 17.3 ms | 1.15 | 201 | 171 |
| for reference: affine 4-bit (14.4 GB) on a 3090 at ~750 GB/s | ~19 ms | ~1.15 | ~187 | ~159 |

- **Trellis reads 17% fewer bytes** than MLX 4-bit (11.9 vs 14.4 GB/token) but costs far more
  instructions per weight to decode. At one row our kernel is instruction-bound (delete all loads
  → 73–88% of the time remains [S1]), so today it loses that byte advantage.
- **At 12–16 rows trellis should gain more than affine does.** A tile is decoded once into MMA
  fragments and shared by all rows. The one-row kernel already issues full 16-row MMAs with 15
  empty rows. The extra per-row trellis work is the activation Hadamard and the suh/svh scales,
  ~rows × (k + n) against k × n weights. This is an argument, not a measurement (G1d measures it).
- **Measured evidence that trellis on a 3090 can already do this class:** sglang-exl3 reaches
  225 tok/s on code with DFlash2 on a 3.0 bpw EXL3 trellis 27B.
- **Honest trellis number:** ~150–175 tok/s on standard coding replies, if Phase 1 reaches the
  sglang-exl3 kernel rate and G1d's flat verify. 190–200 is the best case, not the plan.

**Read this table honestly:**

- **Serial decode:** Loom should land at roughly the best existing stack's speed per bit (+5% to
  +25%) while running the more accurate X3.1 file. It will not be 2× anything there; the ceiling
  (74.9) does not allow it.
- **Speculative decode is where the large gain is.** In the previous version of this plan I capped
  it at 109–135 by treating our chain drafter's 3.48 tokens/round as fixed. That was playing it
  safe, and it was wrong. Trees, copy chains and a flat verify curve are engine features.
  - On a Spark they give 3.8× (standard) to 8× (structured) over serial.
  - On a 3090 the rounds are ~3.5× shorter, so the same design should land around 3× the Spark's
    numbers on the same prompts.
- **What could stop it:**
  - the verify curve does not flatten on a 3090 (we measure 1.52× at 12 rows; ≤ 1.2× is
    physically available but unproven);
  - our Q2_K DFlash2 drafter's candidates are worse than the checkpoint TensorFold uses.
  Both are measured in Phase 0 before any engine code.

## 3. Phases

### Phase 0 — Measure what we do not know (2–4 days; ~4 GPU-hours on one 3090 + the 3060 box)

No engine code. The output is a decision record, `docs/decisions/D0.md`.

| id | task | artifact | done when |
|---|---|---|---|
| P0.1 | X3.1 speed in llama-paw `master`, interleaved with B3.5 in the same session. (1) `llama-bench -m PAW-27B-X3.1.gguf -ngl 99 -p 512 -n 128 -r 5 -sm none`. (2) Row curve `-p 1,2,4,8,16 -ub N -b N -n 0 -r 5`. (3) 8k code AR through the server with the 256k serving flags. | `bench/results/p0_1_llamapaw_x31.json` | numbers with spread; clocks logged |
| P0.2 | **Exporter `tools/gguf2exl3`**: PAW GGUF → EXL3 safetensors dir. `m3_trellis/suh/svh` → `trellis/suh/svh` + `mul1`; K from tile width (56 u16 = K3.5); Q4_K embed and Q5_K head dequantized to bf16. Invert the PAW GDN layout permutation if one exists (check the `20260902_exl3_parity` phase-5 handover first). | exporter + a converted X3.1 dir on NVMe | exllamav3 v1.5.2 loads it; logits sane |
| P0.3 | Accuracy check of the export: chatcode + rawcode KL vs Q8_0 (64×2048), paired vs llama-paw X3.1 (protocol [S15]). | `bench/results/p0_3_export_kl.json` | the CI of the difference includes 0. If not, the exporter is wrong; fix it before anything else |
| P0.4 | exllamav3 v1.5.2 speed on the X3.1 export: `perf.py` AR + rows 1..16; DFlash2 on the same 8k code contract as [S11]. Linear-only ms per forward via nsys. | `bench/results/p0_4_exl3_x31.json` | numbers with spread |
| P0.5 | **Per-shape baseline table**: 7 linear shapes of the 27B × K ∈ {3, 3.5, 4} × M ∈ {1, 2, 4, 8, 16}. Kernels: llama-paw `sq` / x3v / x3g, exllamav3 int8 GEMV / fp16 GEMV / GEMM. Warm-clock protocol. | `bench/results/p0_5_shape_table.json` | the table Phase 1 must beat |
| P0.6 | Platform probes (tiny programs, 3060 and 3090): read ceiling on the 3060 (`bw.cu` from [S1]); a conditional WHILE graph node on this driver; `createpolicy`/`L2::evict_first` loads; `cudaAccessPolicyWindow` set-aside size. | `bench/results/p0_6_probes.json` | yes/no plus numbers |
| P0.7 | **Run TensorFold itself on one 3090.** Install it from pip in its own venv, pull `Vontra/Qwen3.8-27B-MLX-4bit` (~14.4 GB) + `z-lab/Qwen3.8-27B-DFlash2`, and serve at a context that fits 24 GB. Run its `tools/bench_openai.py` (code/chat × sampled/greedy, seeds 1234–1238) plus `draft_decode(trace=…)`: one-row forward ms, verify ms at 1/4/8/12/16/32 rows, draft ms, host ms, tokens per round, stop reasons, candidate hit rate. | `bench/results/p0_7_tensorfold_3090.json` | Its design's real 3090 numbers and round anatomy. This is the most direct answer to "what can a 3090 do" |
| P0.8 | **Tree ceiling for our own drafter, offline:** on ~200 traced llama-paw DFlash2 rounds (8k code contract), record the Q2_K drafter's top-16 candidates per block position. Compute tokens/round for chain n=5, best-first trees of 12/16 rows, and the copy rule. | `bench/results/p0_8_tree_ceiling.json` | Tells us whether our drafter supports 4.7+ tokens/round, or whether the drafter (not the engine) must change first |

**Decision D0 (written, with the numbers):**

- If exllamav3-on-X3.1 is ≥ 1.15× llama-paw on AR (P0.4 vs P0.1) and its DFlash2 path works,
  then **serve X3.1 on exllamav3 in production now** (the zero-engineering win). Loom continues
  only against exllamav3 as the bar.
- Phase 1–3 gates are set as multiples of **max(llama-paw, exllamav3, TensorFold-on-3090)
  measured in P0**. They are not taken from this document's estimates.
- If TensorFold on a 3090 (P0.7) already reaches most of §2's speculative estimate, D0 says so
  plainly. The options are then: serve with it, contribute a trellis matmul to it (MIT), or build
  Loom. D0 records which, and why.
- If P0.8 shows our Q2_K drafter cannot support ≥ 4.5 tokens/round even with trees, a drafter
  track (better-quantized or distilled DFlash2) runs alongside Phase 1.

### Phase 1 — The kernel: `trellis_gemv_ri` (2–4 weeks, op level, 3060 first then 3090)

The one thing that decides whether Loom is worth building. Levers A1–A6 (`docs/03`).

| step | what | gate |
|---|---|---|
| 1.1 | `formats/reference_dequant` + `repack` + inverse; test over every tile of all 400 X3.1 matrices | **G1a:** 100% bit-exact |
| 1.2 | `tools/tk-ablate` variants (base / no-decode / no-load / skeleton) exist before the first kernel | harness reproduces [S1]'s numbers on x3v within 3% |
| 1.3 | Iteration 1: repacked layout + multi-stage `cp.async` + no cooperative launch, K3.5, M = 1 | ablation shows skeleton ≤ 35% of time (today 47–53%) |
| 1.4 | Iteration 2: row-invariant M = 1..16 with a fixed K split; measure the exactness tax at M = 1 | **G1b:** row-invariance test passes for M = 1..16 |
| 1.5 | Iteration 3: fp16-accumulate fold (A6) and the int8-activation arm (A5) as measured A/B arms | each arm keeps or drops by the numbers |
| 1.6 | K3 and K4 variants; multi-output variant (QKV+gate, gate+up) | G1a/G1b for each |

- **Pass G1c (serial):** summed over the 27B shape mix, at K3.5, M = 1: **≥ 1.15× the best
  kernel in P0.5.**
- **Pass G1d (verify flatness, the gate that matters most for speed):** summed linears at
  M = 12 ≤ **1.15×** M = 1 and at M = 16 ≤ **1.20×**. Today: llama-paw whole forward 1.52× / 1.58×;
  exllamav3 linears 1.275× already at M = 8 [S6].
  - If G1d passes but G1c misses, keep going. Speculative decoding is the main prize.
  - If G1d misses, the speculative targets in §2 are off and the plan is re-scoped in D1.
- **Kill:** after three design iterations, if < 1.10× the best existing kernel → **stop Loom.**
  Upstream the best kernel variant into llama-paw (and offer it to exllamav3), and record why in
  `docs/decisions/D1.md`.

### Phase 2 — Dense forward, AR only (3–4 weeks)

- **2.1** `core/` (arena, graph, device scalars) and `formats/` loaders for GGUF and EXL3. Static
  memory plan for 262,144 context at q8_0 KV on 24 GB (estimated 21.5 GB [S18b]).
- **2.2** Ops per `docs/04` §6:
  - fused prologues and epilogues (A7, A8);
  - fused GDN decode (C2);
  - GQA-batched flash-decode with q8_0 KV (C1);
  - Q5_K head (port the llama-paw MMQ override);
  - device argmax and Gumbel-keyed sampling.
- **2.3** Step executor: one graph per row bucket; zero host sync inside a step.
- **G2a (accuracy):** chatcode/rawcode KL vs Q8_0, paired vs llama-paw X3.1. The CI of the
  difference includes 0 (E1). Greedy 400-token output on 8 prompts is recorded as golden.
- **G2b (speed):** AR ≥ **1.15× max(llama-paw, exllamav3) on X3.1** from P0, measured
  interleaved on the same 3090. GPU idle ≤ 2% (nsys); launches ≤ 450 per token.
- **G2c (fit):** 262,144 context loads and decodes on 24 GB.
- **Kill:** if G2b misses by more than 5% after one profile-driven fix of the largest measured
  item, freeze Loom at the kernel library. Ship the kernels into llama-paw and stop.

### Phase 3 — Exact speculative decoding with trees (3–4 weeks)

This is the phase with the largest expected payoff (§2). Build order:

- **3.1 Tree verify forward.** Per-node RoPE positions. Tree attention: each node attends the
  committed keys plus its own root-to-node path, chunked by absolute key position so the bits do
  not depend on the tree shape. **GDN along tree paths**: each node's state is its parent's state
  plus one update, in fp32, chains taking a one-state path; ported from the idea of TensorFold's
  `gdn_tree.cu`, written fresh for our layouts. Tree conv1d over each node's own path. Test: every
  tree node equals serial steps along its path, bit for bit.
- **3.2 Commit in place + one-launch GDN replay** of the accepted path over all 48 GDN layers (D9).
- **3.3 DFlash2 drafter on Loom kernels:** KV-only catch-up (D1), hot-vocabulary draft head (D2;
  the id set measured to cover ≥ 99.5% of committed tokens), best-first tree over the top-16
  lattice with target-Gumbel-weighted, calibrated scores (D7).
- **3.4 Copy rule** (D8): an 8-gram index, verbatim chains up to 31 rows through the TC tier.
- **3.5 Device-side round** (B2): draft → verify → sample → accept → commit → catch-up in one
  graph, with the host only reading committed tokens. Conditional WHILE node (B3) if probe P0.6
  passed. Window width and tree size chosen per request by committed tokens per ms (D3), MTP as a
  second drafter (D4), keyed sampling (D6).

Gates:

- **G3a (exactness):** `spec_equals_serial` passes for 20/20 prompts, greedy and seed-1234
  sampled, with trees and copies on.
- **G3b (speed, standard):** TensorFold's bench (`bench_openai.py`, 64-token replies, code/chat ×
  sampled/greedy, median of seeds 1234–1238) on one 3090: **≥ 1.5× TensorFold's own 3090 numbers
  from P0.7**, and ≥ 150 tok/s on code.
- **G3c (speed, our contract):** the [S11] contract (8k code context, greedy, 400 tokens):
  **≥ max(200 tok/s, 1.10× exllamav3-on-X3.1 DFlash2 from P0.4)**. Host time per round ≤ 0.2 ms.
- **G3d:** a drafter at 262,144 context fits on 24 GB (KV int4 for the drafter if needed, C3).
- **Kill / re-scope:** if G3b/G3c miss, trace rounds (the same record as TensorFold's
  `_round_record`). If tokens/round is short, the drafter is the limit: move to a drafter track
  (distill DFlash2 on the target's own tokens; TensorFold's gate is 6.5 tokens/round offline). If
  round time is long, the verify curve is the limit: back to G1d.

### Phase 4 — Prefill and long context (2 weeks)

- K3.5 tensor-core GEMM tier (A9); reconstruct + fp16-acc cuBLAS at ≥ 1024 rows (A10); chunked
  prefill interleaved with decode.
- Attention at 256k: split-K flash-decode tuned for q8_0; int4-Hadamard KV as an E2 option.
- **Gate:** PP8192 ≥ 1.25× llama-paw (1215 → ≥ 1520). TG at 200k ≥ 20.6 tok/s (exllamav3
  measured here [S9]).

### Phase 5 — Serving (2 weeks)

- Python server over nanobind: OpenAI chat/completions with streaming and cancel, qwen3 reasoning
  and qwen3_coder tool parsers, chat template.
- Prefix cache with GDN state checkpoints (C4); concurrency 1–4 by row packing.
- **Gate:** the C1 numbers from Phases 2–3 through HTTP within 3%. C4 aggregate ≥ 1.25× C1.
  2-hour mixed soak with zero failed requests and zero cross-request state leaks.

### Phase 6 — Beyond one 3090, dense (each separately gated, after Phase 5)

- **6a RTX 3060 12 GB profile:** X3.1 (11.5 GiB) does not fit with KV, so this needs a smaller
  build (K2.5/K3 class, e.g. the B2.5-class 8.67 GiB). That is a **quality trade the user
  decides**; it is not an engine matter. Target: 64k context at q8_0/int4 KV. Gate: AR ≥ 1.15×
  llama-paw on the same file and card.
- **6b′ → moved to Phase 7** (Flash-Next trellis, 1-GPU and 2-GPU packs, the 2× rule).
  The Strata measured bar and the doorbell design now live there and in `docs/05`.
- **6b (old; kept for reference) 2×3090 Flash-Next MoE:** grouped trellis kernel (TensorFold `grouped_kernel` shape,
  exllamav3 `moe_coop`), device routing, layer-range pipeline (no TP). Dense tensors stay in
  their measured best codec (`docs/01` §9). Gate: ≥ 1.15× llama-paw's 48.6 raw / 89.2 with MTP [S13].
- **6c Persistent-decode megakernel** (E-1): only if, after Phase 3, nsys shows non-matmul plus
  idle ≥ 3 ms/token. Gate: +8% end to end.

### Phase 7 — Flash-Next trellis (FNT): 1-GPU pack S1, 2-GPU pack D2 (design: `docs/05`)

**Rule (user, 2026-09-29): D2 ≥ 2.0× S1 on PP and on TG, same prompts, same engine, or D2 is
not shipped** and the box runs two S1 instances instead. A layer split is banned for TG: it is
the old paradigm, where the second card adds capacity and no speed.

**Measured bar (2026-09-29), Strata 0.1.19 on ONE 3090 (GPU 2, x8, 275 W):**
- Setup: GSQ-RCO Q2_0, calibrated `--pcie-frac 0.20 --spec-min-p 0.70`, MTP spec 4, int8 KV,
  renter load ~22.
- Code prompts of 4.2–5.4K tokens: decode 88.5 / 94.2 / 84.4 tok/s, prompt 959–1140 tok/s.
- Code prompts of 26–29K tokens: decode 106.7 / 97.8 / 85.7 tok/s, prompt ~1320 tok/s.
- Round = 28.9 ms for 2.56 tokens at 4K. Draft acceptance 80–91%.
- Report: `~/ML_projects/flashnext/20260929_strata_q2_0/reports/RESULTS.md`.
- Ours today on 2×3090: 46–48 raw, 79–89 with MTP. **Output quality vs PAW X3 is not compared
  yet** (P7.0d).

**Targets:**
- S1 ≥ Strata on the same prompts: TG ≥ 90 at 4K, PP ≥ 1300.
- D2 ≥ 2× S1: TG ≥ 180, PP ≥ 2600.
- D2 physics ceiling ~400 tok/s *(est., docs/05 §4)*, so 180 needs ~45% of it.

**P7.0 — probes, before any engine code** (≈ 1 day, two GPUs for a short slot):
- **a. Doorbell ping-pong (was P6.0).** GPU→mapped pinned host→GPU with device-side flags,
  at 5, 20, 80 and 320 KB, on the x8 and x16 pairs, 10k iterations, median and p99.
  **Gate ≤ 8 µs median at 20 KB.**
- **b. Strata round breakdown.** From its logs and nsys, how much of the 28.9 ms round is GPU
  busy versus waiting on CPU experts versus PCIe copies. This says how much D2 gains by having
  no misses.
- **c. Load balance, offline, no GPU.** From real Flash-Next routing traces (code prompts),
  bin-pack experts into 2 ranks by frequency × bytes. Report the per-window max/mean
  expert-byte ratio. **Gate ≤ 1.10 at p50, ≤ 1.25 at p95.**
- **d. Quality anchor.** KL and top-1 agreement of Strata Q2_0 and of our PAW X3 build against
  the bf16 reference, same prompts. This fixes what "same accuracy" means for the dense
  re-encode.

**P7.1 — dense re-encode, shared by S1 and D2.**
- Q8_0 (4.45 GiB) → mixed 3–5 bit trellis with Hessian-allocated rates (~3.2 GiB *(est.)*).
- Use sequential Hessians (`flash-next-codec-ceiling`).
- **Gate:** KL versus the current Q8_0 build ≤ the P7.0d gap between Strata and bf16. If it
  fails, D2 keeps Q8_0 dense.

**P7.2 — kernels (reuse Loom Phase 1).**
- Row-invariant trellis GEMV for the dense tiles at M = 1..16.
- Grouped trellis expert kernel with device routing (TensorFold `grouped_kernel` shape).
- Head-sliced GDN and attention.
- **Gate:** each op ≥ 80% of its read ceiling at M = 4.

**P7.3 — S1 engine (one 3090).**
- One graph per window.
- Hot-expert VRAM cache from the routing profile.
- Cold tier: choose after measuring (i) a CPU-codec copy of the cold experts, or (ii) PCIe
  trellis copies. POC-1 found CPU trellis decode too slow (0.13 tok/s).
- MTP window with a confidence floor, plus suffix/copy drafts.
- **Gate:** ≥ Strata's measured numbers above.

**P7.4 — D2 engine (two 3090s).**
1. **Repack, not re-quantize:** per-rank tile slices, and the expert-to-rank map from P7.0c.
2. **One layer:** heads split, experts split, 2 doorbell exchanges, fixed-order sum.
   **Bit-exact** against the serial layer.
3. **Full model:** one graph per rank per window; the drafter on the rank that finishes first.
4. **A-dup variant** (mixer duplicated, 1 exchange per layer): measure against 2 exchanges and
   keep the faster.
5. **PP:** the same layout, sub-chunk overlap of the exchange.

- **Gate G7:** D2/S1 ≥ 2.0 on TG and on PP, at 4K and 26–29K, same prompts. Output identical to
  serial.

**Kill rules:**
- P7.0a > 8 µs, and A-dup is also < 2× on paper: drop D2 and serve 2× S1.
- G7 fails after P7.4: same.

## 4. Calendar (rough; GPU availability is the main risk)

| phase | effort | cumulative |
|---|---|---|
| 0 | 2–4 days | ~1 week |
| 1 | 2–4 weeks | ~1–1.5 months |
| 2 | 3–4 weeks | ~2–2.5 months |
| 3 | 3–4 weeks | ~3 months |
| 4–5 | 4 weeks | ~3.5–4 months |
| 6 | per sub-phase | after that |

## 5. Risks, with the mitigation built into the plan

| risk | likelihood | mitigation |
|---|---|---|
| K3.5 decode ALU is worse than K3 and caps the kernel | medium | P0.5 measures it first; A1 repack targets exactly this |
| The row-invariance tax at M = 1 eats the kernel gain | medium | measured in step 1.4; kept only if G1c still passes |
| The sglang-exl3 "13.3 ms" does not transfer to K3.5 or our shapes | medium | it is only a target; the gates use our own P0 numbers |
| GPUs are rented and unavailable | high | op-level work on the 3060; 3090 only for gates |
| The exporter (P0.2) changes numerics (GDN layout, head dtype) | low–medium | P0.3 KL gate before any speed comparison |
| Maintenance: a second engine to keep alive | real | narrow scope, one family, a reference for every op, tests before kernels; kill gates stop sunk cost |
| The verify curve does not flatten on a 3090 (1.52× today at 12 rows) | medium | gate G1d in Phase 1, before any model code; P0.7 shows what TensorFold's kernels achieve on a 3090 |
| Our Q2_K drafter's candidates cap tokens/round below TensorFold's | medium | P0.8 measures it offline; drafter track if needed |
| Tree GDN / tree attention correctness | medium | bit-exact node-vs-serial-path tests before any speed work (3.1) |

## 6. Deliverables per phase

Each phase ends with:

1. a dated decision record `docs/decisions/Dn.md` (numbers, commands, artifacts, keep or kill);
2. results JSON under `bench/results/` keyed by git sha, GPU and clocks;
3. an update to `docs/01-evidence-the-wall.md` if the wall moved.
