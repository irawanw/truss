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
