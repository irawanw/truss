# Plan v3: X3.1 on one RTX 3090 (GPU 1) - from 55 served to 100 tok/s, prefill 1,925 to 3,000 (2026-10-06)

Executor: PAW (omp, selfopt pipeline). Lead: reviews every TRACKER row and notify line, owns gates and decisions.
Basis: `docs/RESEARCH-20261006-decode-ceiling.md` (pass formula, byte floors). Supersedes the step lists of
PLAN-20261005 and PLAN-20261006-x31-gpu1-100-3000 (their rules and gates still apply unless changed here).

## 0. The truth we optimize (user, 10-06: "never seen 77; 55 steady; prefill below 2000")

**KPI = the served brain log, not the bench.** `scripts/served_stats.py` (Phase 0.2) over the pm2 log:

| served, 10-06 (284 requests >= 200 gen tokens, 90 fresh prompts >= 30K) | p10 | p50 | p90 |
|---|---:|---:|---:|
| decode tok/s | 49.4 | **55.4** | 62.4 |
| decode tok/s, context > 120K | 49.9 | 54.9 | 62.0 |
| drafts accepted per draft (temp 0.8) | 0.40 | **0.50** | 0.70 |
| prefill tok/s, fresh prompts >= 30K | 1,886 | **1,925** | 2,033 |

The greedy bench (77.7 tok/s, 2.89 tokens/pass, 0.78 accepted/draft) overstates serving by ~40%: served
sampling accepts 0.50/draft, so ~2.3 tokens/pass. **Every result row reports bench AND (after landing) served p50.**

## 1. The formula (targets are bandwidth efficiencies, not guesses)

RTX 3090: 936 GB/s peak. Achievable for streaming kernels: 650-750 GB/s (PAW-27B's kernels reached 600-700).
Define per kernel k: bytes_k (weights + activations + state it must read/write, from shapes), t_k (nsys),
**eta_k = bytes_k / (t_k x 936 GB/s)**. Target eta >= 0.65 for every kernel that moves > 1 MB.

  T_pass = sum_l [ sum_{k in layer l, GPU non-expert} t_k + max(t_gpu_exp(l), t_cpu(l), t_copy(l)) ] + t_head + t_draft + t_host
  Recoverable_k = t_k - bytes_k / (0.65 x 936 GB/s)      (rank all work by this; it is the ms on the table)

Floors (research doc §5): GPU non-expert ~4.6 GB/pass -> 7.6 ms at eta 0.65 (today ~13); resident experts ~1.7 GB ->
2.8 ms (today ~5.3); CPU misses ~0.36 GB at ~45 GB/s -> 8 ms (today 14.65); head ~1.2 GB -> 2 ms.
Pass at targets ~ 48 x (0.16 + max(0.06, 0.17)) + 4 = ~20 ms -> served ~2.3 tokens/pass = **~115 tok/s served**.
Prefill (kernel-bound, 275 W): chunk time = sum of kernel times; targets per kernel class in Phase 5.

## 2. Rules for this plan

- GPU 1 only. Every GPU run: `/tmp/selfopt_gpu2.lock`, `GPU START/END paw` lines, brain down as short as possible,
  one backgrounded script with a trap that restarts the brain; never poll the model while it is down.
- Host RAM: check `free -g` available >= 20 GB before a run; earlyoom now kills tk-* and python3 first.
- Bench: `kbench` (fixed 10-06: served env, no section timers unless `--sections`; exactness runs without
  admission). Never compare a `--sections` row with a plain row.
- **Reporting (mandatory, lead reads these):** every measured step = one TRACKER.md row (id, date, hypothesis,
  bytes/eta before, change, after, verdict, log path) + one `notify.txt` line (RESULT / STUCK / DECISION) + NOTES.md.
  STUCK after 3 failed attempts on one idea. DECISION for anything touching weights, served config, gates.
- Gates: G-X (greedy tokens identical) for pure scheduling/fusion that keeps the arithmetic. **G-N (new, numerics):**
  a kernel that changes summation order or precision passes if, with placement fixed (`TRUSS_CPU_DYNAMIC=0`,
  no admission), tk-parity-kl vs the previous build gives KL <= 0.002 and top-1 >= 99.0% on 8 x 2048 tokens, and
  PPL within +-0.3%. G-S: interleaved A/B, 2 pairs, mean beats the pair spread. Served p50 confirms after landing.

## 3. Phases (in order; each step names its deliverable)

### Phase 0 - instruments (no engine change)
0.1 Re-dump the golden with the fixed kbench (lead approves in notify), one `--quick` + one full row = new baseline.
0.2 `scripts/served_stats.py`: parse the brain log -> p10/p50/p90 decode (by context band), accepted/draft, prefill
    (fresh >= 30K), per day. Row in TRACKER = served baseline above.
0.3 **Served-pattern bench on GPU 1**: port `20261003_x31/scripts/e7_steps.py` + `e7_serve.sh` (GPU 2, X3) to GPU 1,
    X3.1, served sampling (T 0.8, top_p 0.95, top_k 40, min_p 0.05, draft-min-p 0.5, fixed seeds), 8 agent steps of the
    150K session. Output tok/s + accepted/draft. Check it reproduces served p50 within 10%; this becomes G-S.
0.4 **Bandwidth ledger (decode)**: nsys, 20 clean decode passes at 150K (no section timers). For every kernel:
    count/pass, t/pass, bytes/pass (from shapes in the code; write the formula per kernel in the table), eta,
    Recoverable. Plus total idle gap time between kernels per pass. TRACKER row + `docs/ledger-decode-20261006.md`.
0.5 **Bandwidth ledger (prefill)**: same for one 8K chunk at 150K: per kernel TFLOPS or GB/s vs 3090 limits at the
    measured clock (int8 IMMA, fp16 HMMA, memory), recoverable ms per chunk.
Deliverable: the two ledgers. **Lead reviews them before Phase 1 starts.**

### Phase 1 - GPU decode, non-expert (target ~13 -> ~7.6 ms/pass)
1.1 Top-5 kernels by Recoverable: per kernel a standalone microbench on the real shapes (rows 1-5), variants until
    eta >= 0.65 or 3 attempts (STUCK). Typical causes to test, in order: too few CTAs for 4-row GEMV (grid < 2x82 SMs),
    non-vectorized loads (< 16 B), Q8 scale loads uncoalesced, fp32 router weights (0.25 GB/pass: f32_gemv at eta?).
1.2 Fuse launch chains where the ledger shows gaps: hc read (norm+down+up) -> mixer input; router + shared expert;
    quantize kernels (427 launches/pass) folded into their producers.
1.3 Only if 1.1+1.2 leave > 2 ms of gaps: persistent per-layer kernel (megakernel), Hazy "no bubbles" design.
Gate: G-X where arithmetic is unchanged, else G-N; G-S.

### Phase 2 - CPU tier (target 14.65 -> ~8 ms/pass)
2.1 In-process timers (no perf available: linux-tools mismatch): per pool call wake latency, per-phase time, per-item
    ns per byte; compare with `tools/tk-bench/cpu_quant.cc` pool numbers under the same load. Write the 1.5x gap down.
2.2 Fix what 2.1 names, one at a time: TLB (MADV_HUGEPAGE on the pinned arena where THP gives it; measure
    AnonHugePages), the 3 barriers per call, thread wake (spin vs futex), NUMA/CCX placement of experts' rows.
Gate: kmicro checksums unchanged + G-X + G-S.

### Phase 3 - fewer misses (target CPU experts 193 -> ~130/pass)
3.1 Replay (c1/c4 scripts) of the engine's actual admission (<= 2/layer, <= 64/pass, min count 1, idle link only) vs
    the replay's admit-all (-31%): find which cap loses the gain. 3.2 Implement the replay-best policy. Gate G-S + G-Q
    (placement: PPL band) + one agent replay.

### Phase 4 - tokens per pass at served sampling (0.50 accepted/draft)
4.1 Measure acceptance by draft position and by draft-min-p on the served bench (0.3). 4.2 Adaptive draft count from
    running acceptance; top-2 tree at position 1 (R_REPORT). MTP weights: re-encode gave nothing (10-06), not again.

### Phase 5 - prefill (1,925 -> 3,000; kernel-bound, 275 W)
5.1 From ledger 0.5: MoE gate_up/down (34 TFLOPS vs 57-66 cuBLAS-class), dense gemm (69-87 TOPS int8), attention
    int8 (39.7 ms/launch). Non-bit-exact rewrites are now allowed under G-N (rule change 10-06).
5.2 Order by recoverable s per 150K prompt; same microbench-first discipline as 1.1.

## 4. Stop/escalate
Three failed attempts -> STUCK + TRACKER why. A kernel at eta >= 0.65 is done: move on. If Phase 0 ledgers show the
floors above are wrong, PAW writes the corrected numbers and the lead re-budgets before more building.

## 5. Lead review of the Phase 0 ledgers (2026-10-06 ~21:45) -> Phase 1 re-ordered

Ledgers (`work/agent/docs/ledger-{decode,prefill}-20261006.md`, TRACKER 124-127) accepted, with one correction:
**the PCIe "peak 13.4 GB/s" in the prefill ledger is not the peak.** Lead's test on GPU 1 (scratch `bw.cu`, 64 MB,
brain idle, H2D, pieces 4 KB .. 64 MB): `cudaMallocHost` 21.7 GB/s at 256 KB, **24-25 GB/s at >= 1 MB**;
malloc + `cudaHostRegister` (= the engine's cold arena) **18-20 GB/s**; 64 KB pieces 13-15; 4 KB pieces 1.9.
IOMMU runs in translated mode (dmesg) - registered 4 KB pages cost ~20%.
So: prefill streams 38.5 GB/chunk at 13.4 GB/s = 2.87 s, while the arena could give ~19.5 (1.97 s) and pinned-alloc
memory ~24.5 (1.57 s). Once kernels reach the 2.77 s/chunk target, **the link becomes the wall unless fixed**.
Decode: 1,107 copies/pass, 217 MB in 26.2 ms busy = **8.3 GB/s effective** (avg ~196 KB per copy, plus 4 KB meta
tables): the copy stream's per-copy overhead, not the link.

**Phase 1 (decode) new order - the expert supply chain first (29 of 53 ms/pass is waiting for it):**
1.A Copy path: histogram copies/pass by size and by source (demand / hint / admission pieces / meta tables / other);
    then batch: meta tables updated on the device from mapped host words (or one copy per layer), admission pieces
    256 KB -> >= 1 MB, demand+hint of one layer in as few copies as the ring layout allows. Target: copy busy
    <= 12 ms/pass at >= 18 GB/s effective. Gate G-X.
1.B The 7 host stalls/pass at the MoE boundary (7.45 ms): instrument what the host thread blocks on (rdtsc around
    every CUDA call/sync/event query in the enqueue path, per layer); fix the cause. Gate G-X.
1.C wait_plan (11 ms) vs spin (10.5 ms): show with driver timestamps how much of wait_plan is the driver still
    inside the previous layer's ct->pool->wait() (CPU tier) vs its own plan work. That splits Phase 1 from Phase 2.
1.D Then window_kernel eta 0.33 -> >= 0.6 (2.6 ms), then glue fusion (~4.5 ms), as before.
**Phase 5 (prefill) addition:** 5.0 cold arena in `cudaHostAlloc`'d chunks (e.g. 1 GiB each) instead of
malloc+register: measure link GB/s and startup time and host RAM (must stay within the renter budget); if startup
regresses, allocate in the background as today. Do before kernel work so the link stops hiding behind compute.
Router fp32 -> Q8 is a DECISION (precision): not now.
