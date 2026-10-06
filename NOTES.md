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

## Now (10-06 ~10:00) - FINAL: merged to main and serving (user instruction "if all pass merge + serve")
**Branch `agent/x31-speed` = 654c31a; main = d8457ab (merge of it); brain restarted on main build = serving it.**
- Kept-best engine = row D 216d52a family (PLE prefill pipelining + parallel collect; row 1005_032028:
  decode 55.8/57.3, prefill 2132/2128 EXACT @23.8/26.3). Final branch row 1006_094357 @load 22.4/17.1:
  decode 61.3/58.2 (>= kept), prefill 2014/2006 EXACT. The prefill delta vs 2132 is BOX DMA contention
  (GPU0 neighbor 99% util, GPU3 81%; today's own d9b6c0b reference row also 2012 @21.9; PLE gather 0.96 vs
  0.64 s, copies ~+3.7 s/150K slower), not code: the branch engine is row D's + env-gated profiler (unset =>
  zero behavior change) + server-only truss_c.cu. Post-revert quick gate 1006_093951 EXACT too.
- Warm handoff (85d791a) REVERTED (654c31a) on matched-load A/B: ref 1006_093116 (d9b6c0b, profiler) pp 2012
  exposed 2.41 s vs candidate 1006_092012 pp 1991 exposed 1.93 s. Mechanism verified (exposed -20%) but pp
  unchanged => chunks are copy-stream-bound; boundary waits are off the critical path. Re-land only on an x16
  link (compute-bound regime) or with deeper staging (VRAM-blocked, see Slots-depth). truss_c.cu server PLE
  lookahead KEPT and merged: serving gets the pipelined gather on full-prompt turns.
- Profiler caveat (measured): TRUSS_STREAM_PROFILE=1 costs ~5% pp (prof_harvest host-syncs at every chunk
  boundary). Read exposed only as a relative number; pp with profiler on is not the served number.
- PCIe: GPU 2 link gen4 x8 of max x16 (4-way TRX40 lane split) => ~13.3 GB/s ceiling = THE wall for both
  targets (prefill chunks copy-paced at ~46 GB/chunk; decode demand copies 10 ms/pass of 44). DECISION
  notified 09:12: physical x16 ~doubles copy throughput -> pp ceiling ~3000+ (brief target), tg +10-12.
- Remaining levers all gated (reported to lead/human): KL-gated hints K=3/4 (DIFF vs golden; self-reference
  KL cannot gate placement - needs the Q8-teacher G-Q, plan §3), bigger ring (VRAM), 16K chunks (VRAM
  +2.6-5 GB vs 0.75 margin), E7 past 2K (short prompts only), MTP re-encode W3/B6 (weights = user decision),
  and the lead's own lanes W1 (residency replay) / W2 (device-ms) / W0.1 (interleaved --ab kbench).
- Quiet-box truth kept for reference: decode 62.0 (1005_033555); lead's merge row 1005_165231 showed
  71.4/78.9 EXISTS at load 6 (DIFF - his KL/startup territory). Served tg on real traffic 48-60, accept
  34-43% by temperature; served pp ~1900-2130 by link contention, higher on prefix-reuse turns.

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
