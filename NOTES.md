# Agent notes (survive context compaction: keep current)

## Ground rules I work under (added 10-04 by lead, rule 9 in AGENT_BRIEF.md)
- Keep a change only if: (a) decode AND prefill not slower than the best kept kbench row (`--repeat 2`),
  (b) tokens EXACT, (c) weights/pack/quantization/expert error UNTOUCHED (no re-encode/re-quant/prune/skip/approximate).
  Anything that changes expert arithmetic -> notify DECISION, do not commit.
- Notify file: `echo "date | KIND | msg" >> /home/green-gpu/ML_projects/flashnext/20261004_selfopt/data/notify.txt`
  (RESULT on a new best row, DECISION on quality/config/weights change, STUCK after 3 attempts on one idea).

## Baseline facts (kbench row 1004_220442, commit 8af5ead)
- decode 52.3 tok/s, prefill 2043 tok/s, cpu_ms/slot 1.176, load ~24.
- Exactness: the golden `data/ledger/ref_tokens.i32` was RE-BASELINED by the lead (10-04 22:23) and now equals the
  baseline run's tokens: rows 1004_224142+ compare against it and my PLE row came out EXACT. Gate = kbench's
  `exact` column (tokens.i32 vs golden); the old DIFF rows (220442/221407/221706) predate the re-baseline.
  The served config still runs TRUSS_CPU_DYNAMIC=1 (timing-based split, do-not-repeat 33), so keep an eye on it,
  but as measured, tokens reproduce EXACT across runs so far.
- VRAM (X3.1, served): 23.56 GiB total, 9.27 in use before experts (ctx 0.69 + weights 4.60 + buffers 0.13 +
  KV/state 3.85), routed experts 13.54 GiB (margin 0.75). Experts: **3,813 resident of 24,576**, ring 6.02 GB
  (a prompt chunk takes all 6.02 GB of it). CPU tier: 21,275 eligible experts, dynamic split.
- Decode pass (48.7 ms, 2.75 tok/pass): GPU compute ~21, CPU join 16.5, copy wait 10.0, host plan wait 3.2,
  expert kernel 5.3. 337 CPU slots + 54 PCIe fetches (107-132 MB)/pass; 441 cold experts routed/pass.
  **Total miss bytes ~445 MB/pass read once from pinned DRAM (~28 GB/s observed) = ~16 ms floor**, link 13.3 GB/s.
- Prefill (2043): 149,183 tok / 18.2 chunks of 8192 = 4.05 s/chunk. Cold stream ≈ 36 GB/chunk at 13.3 GB/s = 2.7 s
  (the hard cap: 8192x10 routings cover ~all 512 experts/layer). PLE host gather ~5 s/run.

## Tried
| commit | change | kbench result | verdict |
|---|---|---|---|
| (8af5ead, pre-me) | CPU frac kernel gemv_i16f | baseline row 52.3 / 2043 | kept (it is HEAD) |
| 4f75948 (branch `ple-lookahead`) | PLE lookahead: next chunk's reads issued during current chunk | row 1004_233016: decode 43.5/26.2, prefill 1494/2024, EXACT, load 49 | NOT kept (rule 9); reverted 10ecc55; RETRY on a quiet box (load < ~28) |

## Lead standing instructions (10-05)
- Call kbench with bash `timeout: 1200` and WAIT for it (early return => omp retry storm while the brain is down).
- After EVERY kbench row: append a RESULT line to `SELFOPT/data/notify.txt`.
- Revert a change unless a `--repeat 2` on a quieter box beats the baseline row.
- Box noise: baseline row ran at load ~24; rows at load 40+ show +-20 tok/s decode swings (43.5 vs 26.2 same code).
  Do not burn kbench runs at load > ~28-30.

## Now (10-06 ~11:45) - SERVING GPU 1 (x16), tg up, pp kernel-bound + power-capped
**Serving: brain on GPU 1 (user order 10-06 GPU2->GPU1; edited scripts kbench:67,70 + selfopt_brain.sh:8; pm2
saved). Served config now TRUSS_HINT_K=3 (was 2), frac 0.2. Branch a0479b6 = main d8457ab engine.**
- GPU 1 link = x16 16.0 GT/s, holds under sustained load (GPU 2 was gen4 x8 = the old 13.3 GB/s wall).
  Decode demand copies 2x faster: wait 9.7 -> 5.3-6.1 ms/pass. tg: K=2 -> 62.7; K=3 -> 67.9 @load17
  (final row 1006_113004 EXACT 61.7/60.8 @21-25 vs kept row D 55.8/57.3 = +4-5 @matched load). K=4 worse
  (59.9: prefetch 215 MB/pass ring pressure, join 14.7). frac 0.35 worse (60.5: demand bytes double 110->215
  MB, wait-copies 10.4). frac 0.2 optimal. K=3 wins because hints LAND in time on x16 (accept 77.6% vs 73.9;
  #85 "K=2 best" was an x8-era fact).
- pp on GPU 1 = 1930-1952 every row: prefill is KERNEL-bound (stream exposed 0.64 s; sections 73.1 ~= wall
  76.8; chunk = kernel chain ~87 ms/layer). Row D's 2132 on GPU 2 was copy-paced at ~equal wall (kernels ~=
  copies at x8). Kernels are SW-Power-Cap throttled: flag ACTIVE (nvidia-smi -q -d PERFORMANCE), 246-248 W
  pinned at the 250 W cap (default was 370 - someone set 250), SM 1200-1740 MHz (max 2115), temps 74-80 < 83
  target. CEILING (corrected, own the over-promise): kernel-bound pp ~= 8192/(48 x t_layer); at 2115 MHz
  t_layer ~57 ms -> pp ~2820 max; 3000 needs ~2245 MHz > boost. At 370 W expect ~2450-2650. sudo nvidia-smi
  -pl 370 -i 1 = human action. "x16 -> 3000" claim was wrong twice: prefill kernel-bound, and 250 W cap.
- GOLDEN MOVED (user decision): ref_tokens.i32 = GPU 1 K=3 dump (112555 == 113004 byte-identical = GPU 1 IS
  deterministic; the earlier "self-divergence" was comparing different split modes - invalid test). Old golden
  backed up: ledger/ref_tokens.i32.gpu2-backup-20261006. Lead: G-Q the new dump vs Q8 teacher (old-golden
  DIFF root cause: x16 timing flips ring-admission -> CPU-vs-GPU expert placement; results differ ~2e-4
  (cpu_trellis_test) -> argmax flip at token 137).
- Served truth (brain log, GPU 1): tg real traffic 48-58 (long gens decay with ctx), accept 34-43% temp-0.8;
  pp ~1900-1950 fresh full prompts, reuse turns 60-760 tok/s by suffix size.

## Main repo status (10-06 ~10:00)
- main = d8457ab MERGED agent/x31-speed (row-D engine already in via 49b2fee; this merge adds truss_c.cu
  server PLE lookahead + NOTES; handoff net-reverted pre-merge). Conflict in truss_c.cu resolved keeping
  main's prog_done counters + the lookahead args. Built -j8 clean, `pm2 restart truss-x31-brain`,
  /v1/models answers, greedy completion verified end-to-end (prompt 59 in 1323 ms, 12 gen, drafts 10/15).
  Earlier lead commits on main: 1c585aa fast startup 8.3 s (#120), 1d38be9 tk-parity-kl decode-step mode,
  cc53231 live pp/tg console. Main HEAD may be DIFF vs the old golden (startup/decode-step changes) - the
  golden is the lead's/human's to move (G-Q vs Q8 teacher, plan §3).
- Lead's master plan: docs/PLAN-20261005-x31-decode100.md. My lane W4.1 concluded (mechanism works, reverted
  as copy-bound); W4.2 16K chunks = VRAM/human; W1-W3 = lead + user gates.

## Slots-depth idea: BLOCKED on VRAM (from failed run 1004_224142; keep for later)
Real sizes: chunk buffers `big_bytes()` = **4.06 GiB** (scratch ~2.48 = 8192x285KB/row + 268MB dsa select ws;
moe::prefill workspace ~1.36 GiB (A_gu 839 MB [pairs=81920][2*2560] half + A_d 105 + C_d 419); residual 3x336 MB);
one slot (largest cold layer) ~0.98 GiB; ring 6.02 GiB = 2*slot + extra. 4 slots need extra <= 2.1 GiB: would need
~2 GiB trimmed from scratch/moe/residual (sub-batching moe::prefill by tokens could save ~0.7; aliasing one MTP
residual ~0.34; the rest is per-layer peak, hard). The reverted attempt also crashed (cuBLAS 13): my edit deleted
the `slot_` computation loop -> slot_=0 -> ring undersized. Design to reuse: `git show 3ea909e` (N slots,
`eff_slots = min(req,(ring_base-extra)/slot)` with ring_base = max(z.ring_bytes, extra+slot) UNALIGNED, computed
identically in plan() and ctor; compute slot_ from layers_ FIRST; all `%2` -> `%slots_`).

## Next ideas (after this one)
1. If PLE pipelining lands: also move the collect dequant off the critical path (collect on a helper thread + event;
   ~0.5 s/run left on the table), and pipeline the 42 MB H2D on the copy stream behind an event.
2. Per-chunk breakdown of link vs compute (instrument prefetch/acquire) — the ~1.4 s/chunk of chunk time unexplained
   by stream(2.7)+compute(1.3)+CPU(0.7)+PLE(0.37)+mixer(0.27).
3. Decode: plan wait 3.2 ms/pass is the host round trip for the split decision (driver: split 0.18 + CPU start 0.92
   + copies/plan 1.27). A GPU-side split with a fixed pcie_frac (do-not-repeat 33 says fixed fraction is the stable
   method) could remove the host decision from the critical path.
4. Decode acceptance: 2.75 tok/pass; coupled draft sampling (TRACKER #109e) was measured but never shipped.
5. CPU join 16.5 ms/pass is a pinned-DRAM-read floor (~445 MB @ 28 GB/s) unless miss bytes drop; bytes are pack
   arithmetic -> DECISION territory, do not touch.

## 2026-10-06 afternoon: attention dequant workstream (USER RULE: NO SUBAGENTS - limited resources; all inline)
- nsys prefill breakdown (GPU1, K=3, int8 KV): attn_kernel<FlashNext,0> avg 44.7 ms x ~28/chunk = ~1.25 s/chunk
  (biggest single kernel); gate_up 1.20 s; down 0.61 s. ncu is ERR_NVGPUCTRPERM (admin-blocked) - classify by probes.
- fp16-KV probe (TRUSS_KV_INT8=0, /tmp/pf_f16_out.log): attn avg 28.9 ms (-35%) BUT KV 3.85->7.00 GiB, experts
  13.50->10.35 GiB, spec 64.6->51.6 tok/s. int8 KV saves 3.15 GiB (=+13 tg) but COSTS attn ~16 ms/launch:
  the dequant8 staging loop (sync LDG + int->float cvt + scalar muls, ~20+ instr per 8 codes) is the critical path.
- FIX (this commit): dequant8 rewritten bit-exact with fp16x2 SIMD: XOR 0x80 (two's complement -> offset binary),
  PRMT into 0x6400|u8 (= fp16 1024+u8 exactly), HSUB2 1152 (= exact s8), HMUL2 scale (single RN, identical to the
  fp32 path it replaces). Exhaustive proof: all 256 codes x all 65536 finite half scales, 0 mismatches (/tmp/dq_bit.cu).
  Same loads, same tile order, same bits -> EXACT must hold.
- Decode attn_kernel<,1> also pays it: 37.0 us int8 vs 18.8 us fp16 per launch - tg gains too if the fix works.
- If fast dequant doesn't close the gap: next candidate is cp.async staging of codes+scales (smem 42->58.5 KB,
  2 CTAs/SM) - measure before building; and per-layer fp16/int8 KV split knob (allocation is already per-layer).

## Row 1006_153824 (commit 53b59b9): fast dequant + staged decode = EXACT, pp 1975/1968 @load 27.2 (kept 113004: 1944/1930 @21.3)
- attn<,0> 44.7 -> 39.7 ms; attn<,1> 37.0 -> 25.7 us; tg 57.7/60.6 vs 61.7/60.8 (load-unmatched: 27 vs 21).
- Staged cp.async for PREFILL int8 FAILED (attn 61 ms: 2 CTAs/SM occupancy loss > async gain) -> staged only in SPLIT
  (decode) instantiation: AttnSmem<Shape, STAGED>; prefill keeps direct fast dequant at 3 CTAs/SM.
- Matched-load tg confirmation row pending (box flaps 27-35 all day; poller pattern in prior sections).
- MoE scout map partly stale: silu+mul ALREADY fused in gate_up epilogue (moe_prefill.cu:319-321); prep->A_gu round
  trip is justified by measured TRACKER #42 (23 vs 36 TFLOPS). Do not re-litigate.
- Next targets by size (nsys): dense gemm_kernel Q8 6.5 s (14%), gate_up 4.4 s, decode waits (spin+plan ~9.8 s),
  hc small-kernel chains decode-side. Per-layer fp16/int8 KV split knob = +45 pp on fp16 layers but costs expert
  budget (tg) -> DECISION + lead G-Q territory.

## 2026-10-06 evening: D0 decode-pipeline simulator - calibrated PASS (tools/sim/decode_pipeline.py)
- Discrete-event model {GPU serial chain, driver serial, copy-engine FIFO, CPU pool serial-at-DRAM-floor}.
  Gate on six GPU1 rows: max |err| 1.4% on wall AND tg, tg order MATCH on all six
  (probe 68.4 > K2 62.4 > final 61.5 > admit2 61.2 > f0.35 61.1 > K4 59.1).
- Key mechanisms (each traceable to code, no fudge): dem_pressure 0.032 ms per demand expert above
  1.6/layer (expert_store.cu ring_put/wait_compute slot contention - this is what lands f0.35 and makes
  the ORDER match); host_gap = 3.7 + 0.02*copy_busy (driver publishes + issues copies on same host);
  per-row cpu_slow in [0.95, 1.22] = pool wall band (DRAM floor 0.056 ms/slot .. engine charge 0.085).
- Lever verdicts at served point (base sim 61.3, row 61.7): a_hint2ahead -0.0 DEAD while CPU binds (link
  FIFO total unchanged; freed go-wait absorbed by CPU wait); d_speccpu -0.8 DEAD (0.06 ms/layer early
  start < 12% mispredict DRAM waste); c_frac005 -5.1 => frac 0.2 IS the engine balance point (both
  directions worse; consistent with measured f0.35 -1.2); b_ringslots +1.4; g_graphs +0.2
  (launch_gap 0.005->0.002); combo_bg +1.6 = best attainable with weights/placement untouched (~63 tg).
- Model verdict: CPU tier is the binding wait at the served point (~0.29 ms/layer = 14 ms/pass); link is
  secondary (3.7 ms/pass); the 3.3 ms/pass "wait for host plan" is host-side slack absorbed by both.
  tg gains must remove CPU-tier bytes (placement = D1, or DECISION line) or cold volume; plan D0(e)
  answer: ~63 tg combo b+g, i.e. +1.6 - the plan's 75-85 needs D1 placement + D2, not scheduling.
- Next per plan section 4: D2 nsys 20 decode passes (top-10 kernels to TRACKER + measured launch-gap
  total = true graphs value), P2 SASS attn int8-vs-fp16 per tile, P4 inter-kernel gap sum (drop P4 <1 s).

## P2 SASS study attn int8 (CLOSED 10-06, no buildable variant)
- cuobjdump attn_kernel<FlashNext,false> (runtime dtype: int8 AND fp16 paths live in ONE instantiation):
  4728 instr; main KV-tile loop 477 instr (backward branch 0xfb0->0x2d80), nested staging loop 102.
- Staging loop = LDGSTS x2 (codes+scale cp.async) -> LDS -> 8x PRMT + 8x HADD2 + 8x HMUL2 (the shipped
  fp16x2 fast dequant) -> 4x STS.128: ~1.5 ALU ops per code = floor; nothing left in the dequant itself.
- Main loop per tile: 64 HMMA.16816.F32 + 48 LDSM = 112 (23%); staging 102; softmax/rescale ~180
  FMUL/EX2. Tensor work is a MINORITY - the int8 tax vs fp16 (39.7 vs 28.9 ms/launch) is the
  LDS->dequant->STS round-trip + cp.async wait, structural at this design point.
- The plan's "dequant once into fp16 smem shared by query rows" variant IS the shipped design
  (staging once per tile per CTA, all warps LDSM the same fp16 tile, dsa_prefill.cu:351-402).
  Register-dequant-at-ldsm stays rejected (+640 instr/tile, analyzed prior session).
- Verdict: 39.7 ms = floor for int8 prefill attn here. Going below needs fp16 KV (VRAM -> tg) =
  DECISION/G-Q territory, not a kernel item. No build, no kbench.

## 2026-10-06 late-evening: D2 nsys decode profile + P3/P4/graphs verdicts (ALL closed, engine untouched)
- Bench gotcha: "spec : 7.73 s" sums BOTH prompts' spec loops; the naive last-7.73 s nsys window swallowed
  p1's prefill tail (phantom 16x39 ms attn<,0>). Clean window = last prompt tokens/tok/s = 4.28 s
  (TRACKER lesson 37; proper hook: TRUSS_BENCH_PROFRANGE=1 + nsys --capture-range=cudaProfilerApi, spec.cu:252).
- D2 run (load ~35 recorded; brain down ~2.5 min, lock held, auto-restarted): p0 4063 tok pp 2077 spec 74.3
  (3.37 tpp); p1 21708 tok pp 2079 spec 59.8 (2.56 tpp); device draft 1.68 + verify 40.18 + head 0.97 = 42.83
  ms/pass; CPU tier 18.33 ms/pass wait; driver split 0.25 + CPU start 0.81 + copies/plan 2.43 (51.7 layers/pass).
- Decode top-10 ms/pass (p1 window ~100 passes): spin 8.11, wait_plan 7.66, window 5.35, gemv_multi<4,32,1> 2.22,
  gemv<1,32,1> 1.00, f32_gemv<2> 0.81, gemv<4,32,2> 0.80, route<16> 0.67, quantize 0.64 (427 launches/pass),
  attn<,1> 0.61. 2201 launches/pass; gaps <10us = 2.13 ms/pass; >=10us = 5.22 ms/pass (waits + context slices).
- Verdicts (plan stop rules -> NO builds, NO kbench): CUDA graphs ceiling = 2.13 + ~half of scheduling gaps
  ~ 2.7 ms/pass = +3.8 tg (6.4%) < 10% gate => DROP. P4 prefill: sections ~= wall (exposed 0.64 s < 1 s) => DROP.
  P3: P1 microbench already proved gemm_kernel issue-bound at 69-87 TOPS on ALL real shapes incl. gate_up =>
  within 15% => DROP. P2: staging 102 instr/tile, fp16x2 dequant ~1.5 ops/code = floor; dequant-once-into-smem
  IS the shipped design; 39.7 ms = int8 attn floor (below it = fp16 KV = VRAM = DECISION).
- spin+wait_plan = 15.8 ms/pass = 39% of device busy => CPU tier is the binding wait at the served point
  (sim + nsys agree). Remaining tg levers: placement/residency (lead lane, D1) and power limit (human lane).
- Plan section 4 COMPLETE: P1 negative, D0 sim calibrated, D2 measured, P2/P3/P4/graphs all closed by stop rules.

## 2026-10-06 evening: Order 2 CPU standard-quant verdict - NO encode (bench + sim, CPU-only)
- Added tools/tk-bench/cpu_quant.cc (tk-bench-cpu-quant): one expert (gate/up 640x2560, down 2560x640,
  down zero-padded to 768 for K-quants), formats trellis K2.5/K3.5 (truss_cpu's own kernels) + llama-paw
  libggml-cpu AVX2 vec_dot (Q3_K/Q4_K/Q8_0), 1 thread + 22 pinned (skip core 0, pool scheme), 64 experts
  (DRAM-bound), mlocked arenas, 5 interleaved rounds at load 33-42.
- Pool ms/expert R=1.2: trellis_mix 0.0452 (41.0 GB/s) | Q3_K 0.0478 (47.1) | Q4_K 0.0574 (51.4) |
  Q8_0 0.0922 (56.6); DRAM ref 57.1. BYTES rule: Q3_K +21pct bytes buys only +15pct GB/s. Padding trap:
  down in=640 = 2.5 blocks of 256 -> 768 (+20pct down, +6.7pct expert); Q8_0 blocks by 32, no pad.
  Bytes/expert 1.855 (3.02 bpw = today, matches pack-inspect) / 2.253 / 2.949 / 5.222 MB;
  RAM x21296 = 39.5 / 48.0 / 62.8 / 111.2 GB - Q3_K padded BREAKS the 40-45 budget (45.0 unpadded).
- Sim swap (ratio-scaled cpu_per_slot, sanity 61.3 vs row 61.7): tg -2.7 / -11.6 / -33.5 pct.
  Rule (>= +8pct decode AND bpw >= 3.02): ALL FAIL. Verdict: no standard quant worth encoding; today's
  trellis mix is simultaneously the fastest AND the smallest CPU tier. ggml dots scale per row
  (R2 +70-80pct) while trellis amortizes row 2 in its decode loop (+10-15pct). Notified 18:46.
- Coordination (lead 18:15): lead owns D1 placement/KV-lend/ADMIT/residency - do NOT touch those knobs.
  ANY GPU run: flock /tmp/selfopt_gpu2.lock + one-line "GPU START/END <who> <what>" to data/notify.txt.
  HDD /mnt/hdd READ-ONLY + flaky: never run from it; copy small pieces to NVMe only.

## 2026-10-06 ~19:00: Order 3 - MTP re-encode, PAW step 1 = TRUSS_MTP_CALIB capture
- User approved MTP re-encode with real Hessians (goal >2.8 tokens/pass; greedy tokens stay EXACT - drafts
  are verified by the main model). Lead encodes + packs in ~/ML_projects/flashnext/20261006_mtp_reencode/
  (data/mtp_bf16.safetensors = 31 mtp.* BF16 tensors already on NVMe; venv ~/src/venvs/exl3new).
- PAW step 1: capture on GPU 1 under /tmp/selfopt_gpu2.lock, served env (NOW: TRUSS_KV_LEND=1
  TRUSS_ADMIT_IDLE=64 + selfopt_brain.sh env), brain down < 15 min, GPU START/END notify lines.
  Corpus /home/green-gpu/flashnext/x31/corpus/ids_code_1024.pt tensor [256,1024] -> .i32 prompt files,
  SKIP the last 16 sequences (held out) -> 240 x 1024 = ~245k rows (~2.5 GB fp32) into
  ~/ML_projects/flashnext/20261006_mtp_reencode/data/mtp_calib.f32 + README line (rows, sequences,
  commit, env). MAIN repo build has the capture compiled (forward.cu, verified in build via strings);
  use main's build unless the capture path needs my branch (it does not).
- MEMORY CAUTION (box froze ~18:46 as my bench finished; brain 38 GB + renters, RAM tight): keep run host
  footprint small, stream capture to disk, NO big mlocks; check free -g available >= 20 GB before starting.
- PAW step 2 (later, after lead posts packed MTP GGUF): bench new MTP vs mtp_x3k3 same prompt set -
  tokens/pass, tok/s, VRAM/resident experts, EXACT greedy tokens; interleaved A/B, 2 pairs.
- STEP 1 DONE 10-06 19:2x: capture rc=0, mtp_calib.f32 = 245,520 rows (2,514,124,800 B = 240x1023x2560x4),
  main build 5798d3c, served env replicated, brain down ~3 min, lock held, START/END notified; prompts
  kept in data/prompts/; README written (data/README.md). cpu_quant.cc already committed (b3e97cb).
  IDLE on Order 3 until lead posts the packed MTP GGUF -> then step 2 A/B bench.
- LESSON (10-06): pm2 stop brain kills MY OWN serving connection - every GPU run must be ONE backgrounded
  script that takes the lock, notifies, stops the brain, runs, and RESTARTS the brain via an EXIT trap
  (scripts/capture_calib.sh pattern); then wait for the script/user notice - NEVER poll the model while
  the brain is down.

## 2026-10-06 ~20:30: ORDER 4 = Plan v3 (docs/PLAN-20261006-x31-bandwidth.md), Phase 0 instruments
- Order 3 CLOSED: lead verdict keep A (served MTP); my phase-1 A/B/C numbers -> TRACKER #123; phase 2 skipped.
- Plan v3 KPI = SERVED brain log (scripts/served_stats.py), not the greedy bench: served p50 55.4 tok/s,
  accept 0.50/draft (greedy bench 77.7 overstates ~40%). Targets: eta_k = bytes_k/(t_k x 936 GB/s) >= 0.65
  for every kernel moving >1 MB; floors GPU non-expert 4.6 GB -> 7.6 ms (now ~13), resident experts 1.7 GB
  -> 2.8 ms (now ~5.3), CPU misses 8 ms (now 14.65), head ~2 ms; pass ~20 ms -> ~115 served. G-N gate NEW:
  kernel changes numerics => must pass tk-parity-kl <= 0.002 + top-1 >= 99% + PPL +-0.3% with placement fixed.
- PHASE 0 DONE so far: 0.1 golden re-dumped (md5 a7ae0e42, kbench_1006_201716, exactness WITHOUT admission) +
  baseline row 1006_202143 decode 66.6 / pp 2,136 / EXACT (determinism holds admission-off). 0.2 served_stats.py
  written (scripts/, committed) reproduces plan §0 table. TRACKER #124.
- 0.3 DONE (e7_03_203346, TRACKER #125): dedicated :8193 server brain-exact config + 8-step 150K replay seed=1.
  Decode p50 63.7 vs served 55.4 (+15%, outside plan 10% band - warm single session; e7 = relative A/B harness
  only), accept 0.570, step-0 prefill 2,130, reuse confirmed. Ports scripts/e7_serve_x31.sh + e7_steps_x31.py.
- 0.4 DONE (TRACKER #126): docs/ledger-decode-20261006.md (worktree) from ledger04_204049 nsys (20 passes,
  53.0 ms/pass, gaps 11.20). Headline ~29 ms/pass = expert-supply-chain WAIT (spin 10.5 + wait_plan 11.0 +
  7.45 in 7 MoE-boundary host stalls after hc::combine before router quantize). Recoverable: window 2.60
  (eta 0.33), DSA glue 1.71 (0.10), hc glue 1.10 (0.08), fusion class 2.54, router-f32 0.40 (Q8 = DECISION);
  dense gemv eta 0.63 (GDN qkv 0.83 / DSA qkv 0.86 / heads ~0.9 AT target, do not touch). Bytes method: GGUF
  tensor table /tmp/gguf_infos.pkl + grid decode per=4x32/lpo (lpo16 -> x8!). Evidence in data/logs/ledger04_204049.
- 0.5 DONE (TRACKER #127): docs/ledger-prefill-20261006.md from ledger05_211615 (whole-proc nsys + clock logger;
  2,052 tok/s; SM 1,440 MHz @ 277 W TDP -> peaks at 1440: int8 120.2 TOPS / fp16-HMMA-fp32acc 60.1 TF / fp32 30.2).
  Chunk9 T=8192: 3,847.7 ms, kernels 98.5% busy, expert H2D 38.5 GB overlapped (75% link) = COMPUTE-BOUND, 25%
  PCIe headroom. moe trellis eta 0.69 AT TARGET (decode window 0.33 = batch-4 occupancy, not codec); dsa attn
  Rec 412-641 (biggest); delta_rule 220; int8 gemm 185; quantize 75; hc glue 70. Rec sum ~1,075 -> ~2,950 tok/s:
  3,000 reachable by efficiency alone. **PHASE 0 COMPLETE 10-06 21:31 — sent tell_lead DONE; awaiting lead review
  of the two ledgers before ANY Phase 1+ work. Do not start Phases 1-5 without lead go.**
- tell_lead.sh DONE|STUCK|DECISION "one line" = two-way channel (also logs notify.txt); use ONLY phase DONE /
  brick wall / needs-lead. First use: DONE at end of Phase 0. Never while brain down mid-run unless the wall itself.
- kbench fixed by lead (hash c438f387 in data/ledger/scripts.sha256): served env, --sections opt-in ONLY,
  exactness admission-off. Never compare --sections rows with plain rows.

## 2026-10-06 ~22:00: ORDER 5 = Phase 1 in lead's NEW order: 1.A -> 1.B -> 1.C, tell_lead DONE before 1.D
- Power cap now 250 W (user, grid tripping): write "250W" in every row/note. Old baseline 1006_202143 (~275W)
  NOT comparable; eta RATIOS still fine. BASELINE-250W = row 1006_213540 @71a821e: decode 67.9, prefill 1,945,
  CPU 1.336 ms/slot, EXACT, load 21.76 (decode up / prefill down vs 275W: prefill TDP-throttles, decode has gaps).
- PCIe (TRACKER 128): peak 24-25 GB/s cudaMallocHost >=1MB; 18-20 malloc+cudaHostRegister (engine arena);
  13-15 at 64KB, 1.9 at 4KB. 13.4 = engine-achieved, not peak.
- 1.A MEASURED (from /tmp/kern_trace.csv = ledger04_204049 trace; nsys Bytes = DECIMAL MB):
  per pass: 155 whole demand+hint experts (2.150M K3.5 x115.5 + 1.536M K2.5 x39.5) = 309 MB 18.5ms 16.7 GB/s;
  454 admission pieces 0.262M (256KiB PIECE) = 119 MB 6.5ms 18.3; 393 ring_meta copies 4KB = 1.6 MB 0.76ms 2.1;
  49x53KB + 12x225KB misc (PLE/DSA state-ish) 5.4MB; 43x4B signal. TOTAL 1,107 copies 435 MB 26.2ms busy =
  16.6 GB/s. Engine counters (bench.log) confirm 433.4 MB/pass: demand 65.3=128.2MB + hint 89.8=180.9MB +
  admission 61.3 issued=124.3MB. THE 0.4 LEDGER's "217 MB" WAS AN UNDERCOUNT -> lead's "busy <=12ms @18GB/s"
  target assumed 217MB; honest floor = 435MB @ 19.6 (registered >=1MB) = 22.2ms. Realistic 1.A goal:
  >=18.5 GB/s effective, busy ~22-24ms, copies 1,107 -> ~600. REPORT correction to lead.
- 1.A DESIGN (implemented, see commits): (1) PIECE 256KiB -> 1MiB (expert_store.h); (2) meta: ring_put sets
  Layer::meta_dirty; copies at fetch/hint/claim_issue/admit gated on dirty + clear (dedup: once per layer per
  dirty period; invariant pend==0 => meta on device KEPT since landing/adm events record behind the gated copy);
  acquire(l) uploads if still dirty + re-records copied_[l%2] (covers admit_step puts during CPU wait). All ring
  mutations are engine-thread-only (forward.cu 560/575/707 + admit() between passes) => no host-meta races.
  Kernel only reads meta for plan slots; protect_layer_/protect_ blocks evicting in-use experts.
  (3) demand+hint MERGE into one copy: BLOCKED by layout - ALIGN=256 but K3.5 bytes 2,150,000 not div by 256
  -> padding between adjacent ring slots -> host runs not device-contiguous. Skip, note in TRACKER.
- 1.B root cause CONFIRMED (decode-window sqlite, ledger04_204049): 19 gaps >5 ms avg 8.18 ms ALL
  hc::combine_kernel -> dense::quantize_kernel with ZERO overlap = pass boundary; host accept + verify-enqueue
  then layer-1 ple() blocks on contended SSD PLE preads (7.85 ms/pass @275W, 1.69 at baseline instant - varies
  with tenant I/O). FIX = early-issue ticket across the draft chain (implemented; see postmortem below).
- ORDER 6/7 VERDICT (10-07): 1.A KEPT, 1.B REVERTED IN FULL (revert 4775617 = a077865+0712b86).
  Interleaved A/B (2 rounds x 4 arms, 250W, order B0 B1 B2 B3 B3 B2 B1 B0, worktrees ab_b0/ab_b1 +
  --env TRUSS_PLE_NOINC=1): means B0 base 71.0, B1 1.A 71.9, B2 1.B-NOINC 63.3, B3 1.B 62.6.
  1.B costs 8.6 tok/s EVEN WITH TICKETS OFF -> the upfront draft-chain enqueue itself (draft window
  1.7 -> 5 ms, verify ~+2 ms; driver can't pipeline it). Confirm row 1007_064447: 72.8/1974/EXACT,
  draft-window 1.65, PLE wait 0.51, driver 0.18/1.03. TRACKER 129/130.
- 1.B postmortem (lessons): (1) HANG: workers read plain n_pages_ while append() grew it -> early-exit ->
  finished_ never reached n_pages_; fixed by handing pages out under mu_. (2) TOKEN DIFF token 55: prefill
  lookahead ticket consumed by FIRST decode ple() left the draft ticket open; next ple_prefetch appended
  across windows; fixed by fresh-ticket guard (repro then IDENTICAL-TO-GOLDEN). (3) 2-slot variant
  instrumentation: begin_wait 0.1 ms vs append 1431 ms/run (~2.5 ms/pass) = mu_ convoy from per-page lock
  handoffs, NOT SSD tail. (4) Baseline PLE wait was only ~1.24 ms/pass -> 1.B ceiling ~1 ms, the 7.45 ms
  MoE-boundary stall is NOT mostly PLE. All tweaks saved UNMEASURED on branch agent/1b-experiments (c5c5b08).
- NOW: 1.C ATTRIBUTION-FIRST (Order 6 step 4): nsys -t cuda,osrt,nvtx PROFRANGE capture on the KEPT build
  (4775617 = 1.A only) -> top 3 host causes of the 7.45 ms MoE-boundary stall + wait_plan 11.0/spin 10.5
  split IN MS. Script /tmp/run_order6_1c.sh (lock+trap). No fix before the attribution table. Then
  tell_lead DONE with the table. Phase 5.0 (cudaHostAlloc arena) after 1.C. 1.D only after lead go.

## 2026-10-07 ~07:20: ORDER 7 1.C ATTRIBUTION TABLE (TRACKER 131) - DONE, told lead
- Method (analyze_stall.py): engine-thread gating overlap only (worker/pool overlap = concurrent, NOT causal);
  wait_plan split by 4B plan-signal memcpy (pre = prep + signal queued before EXECUTING, post = PCIe visibility).
- MoE-boundary stall (combine->quantize >5ms): 7.26 ms/pass @275W (7x20.7) / 3.69 @250W-traced (6x13.5).
  Top-3 gating (275W -> 250W ms/pass): (1) engine PLE-collect cond_wait+mu_ 4.4 -> 1.6; (2) engine accept+
  verify-enqueue CPU (no event covers it) 2.9 -> 2.1; (3) PLE pread busy in gaps 0.67 -> 0.85 (SSD NOT the wall).
- wait_plan 11.01 -> 12.03 = ~94% PRE-SIGNAL: driver API prep 2.25/2.63 + copy-backlog drain ~8.2/8.6 before
  the 4B signal executes; post-signal 0.08. spin 10.52 -> 23.81 (osrt tax inflates CPU tier).
- 1.A inventory CONFIRMED: 1,107 -> 652 copies/pass, 435 -> 415 MB, busy 26.2 -> 22.5 ms, eff 16.6 -> 18.4 GB/s
  -> wait_plan NOT copy-busy-bound. LEVER: early/separate-stream plan write (~9-10 ms device wait); boundary
  stall = PLE collect + accept/enqueue CPU.
- CAVEAT: osrt trace tax - ledger07 decode 42.4 tok/s vs 72.8 clean (ledger04 56.4 vs ~66): absolutes upper bounds.
- NEXT per lead: tell_lead DONE (done 07:2x) -> await; Phase 5.0 arena after ack; 1.D only on go.

## 2026-10-07 ~08:45: ORDER 8 = SIM FIRST (done, TRACKER 132, told lead) -> await lever choice
- Sim recalibrated on clean 1007_064447 (-0.2%) WITH ENGINE-CODE STRUCTURE (forward.cu serve 595-717):
  plan = MAPPED HOST FLAG already (1.C traced wait_plan 12 ms = osrt tax on driver; clean ~0.4);
  go = copy-stream behind THIS demand, BEFORE hints; **pool->wait INSIDE serve -> driver serialized
  by CPU tier**. Model A closes: wall 35.78/35.85, verify 32.36/32.43, busy 20.3/20.5 (go-spin 4.8
  INSIDE window kernel 5.3), spin 11.64 (pool wall/call 0.514 = 0.136 ms/expert x 3.68), copy 17.1.
- LEVERS (tok/s vs 72.9): L1 spec-cap +0.0 (ALREADY IMPLEMENTED: mapped plan + demand-first FIFO);
  L2 window-eta +0.0 (window time IS go-spin -> freed time extends spin; lead's absorption risk CONFIRMED);
  5.0 arena +0.0 (absorbed); L3 glue -3ms +6.7 (pre-doorbell work converts); L4 CPU -30% +19.0;
  L2+L3+L4 +29.9 (L2 still absorbed); all +29.9. RANKING L4 >> L3 >> L1=L2=5.0=0. Model-B sens: L4 tops both.
- Phase 2.0 memory floor (/tmp/memfloor.cc, 22 pins = TRUSS_CPU_PIN scheme, no SMT overlap, renters
  load 43.6): 57.4 GB/s vs in-situ 16.7 (412 MB/pool-busy 24.7ms; lead basis 34.5) -> CPU tier NOT
  DRAM-bound; -30% = kernel ALU work (Zen2 int-mul pipe), Phase 2/3 program.
- DONE msg sent asking: build L4-first-attempt (gemv_i16f instruction-mix, kmicro CPU-only) vs L3
  (ready single build, +6.7 converts). A/B 2 rounds vs 4775617 after choice.
- Files: tools/sim/decode_pipeline.py --x31/--x31-levers (commit ba0136b); /tmp/memfloor.cc.

## 2026-10-07 ~09:30: ORDER 9 L4 ATTRIBUTION (TRACKER 133, told lead) -> await go on kmicro ALU program
- Harness tools/cpu/callshape.cc (commit 7beac17) on the REAL pool; CPU-only. TRAP FOUND: pthread_create
  inherits creator affinity -> pinning caller to core 0 BEFORE pool creation collapses all 22 workers onto
  core 0 (their setaffinity EINVAL unchecked). Engine never pins caller -> in-situ clean. Create pool FIRST.
- Reproduced: wall 0.286 (p50 0.263 p95 0.342) = engine fit 0.334; paced 0.675/1.5 unchanged (workers hot).
- DECOMP (sums exact): compute 0.213 (74%) + fork/join 0.073 (26%); 1x1 = 0.076 = lead 0.075 EXACT;
  overhead constant per call (60 vs 167 items) = phase flips + wake; knob grid GU_INxDN_COLS: default best.
- Suspects: (a) small+constant; (b) DOMINATES: ALL 48 CPUs 100% tenant-busy (no free core, siblings of all
  22 worker pins busy; brain nice 0 SCHED_OTHER = no priority edge) -> lost 0.142/call = 6.8 ms/pass;
  (c) clock 3.8-4.0 GHz clear; TLB clear (TLB-hot = random, 7.3MB/call fits L2 TLB); BW clear (57.9 vs 34.4).
- Lead's 3x reconciles: compute-only 34.5 GB/s (= 445/12.9 basis); per-thread 1.5 vs kmicro 2.3-3.6 = SMT+cold.
- FIX PROPOSED: (d) kmicro ALU-mix program (kernel vpmulld-free already; ~5-6 mul-pipe ops/16w). Sim:
  compute -30% -> wall x0.875 -> +6.9 tok/s (79.8); -40% -> +9.5; (a) per-group phase deps +2-3 second priority.
- kmicro baseline 10-07: K35 R1 197.9 us (checksum = ref 4a32754e93e35f86). Build: g++ -O2 -g -mavx2 -mfma
  -mf16c -std=c++20 -Isrc /tmp/callshape.cc build/libtruss_cpu.a -lpthread -o build/callshape.
- NEXT: lead go -> kmicro ALU experiments on gemv_i16f (src/cpu/expert_trellis.cc), checksums frozen.
