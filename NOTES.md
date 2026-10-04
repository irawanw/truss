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

## Now (10-05 ~04:15) - HUMAN-PAUSED
**PAUSED by human (via lead). `agent/x31-speed` = c165e46, tree CLEAN, `ninja -j8` builds all targets. NO kbench runs
or experiments until the lead says unpaused.**
- Commits since kept row D (216d52a): c90a254 bench per-pass host-window timers + decode PLE wait/phases print;
  cb65afd fix (I had dropped section/moe/driver reads in c90a254 -> stale prefill values in the decode profile);
  f057ea6 NOTES; c165e46 stream profiler (TRUSS_STREAM_PROFILE=1: per-chunk prefetch copy-BUSY + acquire
  exposed-WAIT events, stream_stats(); bench prints "prefill sections ms" + "prefill stream: ..."). Profiler is
  measurement-only, zero behavior when the env is unset, compiles clean - the step in progress when paused;
  NOT yet kbench-run.
- Ledger facts: kept best = row D 1005_032028 (216d52a) decode 55.8/57.3, prefill 2132/2128 EXACT @23.8/26.3.
  Instrumentation row 1005_033555: decode 62.0 (best sample) / prefill 2132 EXACT @19-20. hint_k=4 row
  1005_024923 DIFF, hint_k=3 row 1005_025943 DIFF (placement rounding moves tokens) -> keep K=2; lead took the
  gated decode options (hints, ring, placement, coupled sampling) to the human. Lead is building a KL gate into
  kbench (human-approved: placement-only changes may ship if KL vs reference stays at the noise floor).
- Decode truth (instrumented): ~97% device-bound. Host windows/pass: draft 1.67 + verify 40.93 + head 0.91 +
  accept 0.87 = 44.39 vs device 43.50. Floor ~= 44 ms/pass ~= 62-63 tok/s at load<=20. accept 0.87 = per-layer
  GDN replay on partial accept (structural, forward.cu accept()); PLE decode gather wait 0.34 (every pass reads
  NEW sliding-window trigram rows; reads pipelined, tail exposed). moe_window.cu is a tuned persistent dataflow
  kernel (TRACKER #25) and CPU gemv_i16 is op-count-optimized (#76) + at the DRAM floor - no rewrite win in sight.
  Priority demand-copy stream skipped with math (wait-for-copies == demand MB / link rate; NOTES history).
- Prefill truth (code read, not yet measured): stream mode = 2 whole-layer slots; prefetch(l+2) issued after
  release(l) -> stream already runs one layer ahead WITHIN a chunk. split_rows default 0 -> served 8192-row
  chunks stream WHOLE cold layers (~avg 0.24s copies vs ~3.85s compute per chunk -> copies should be hidden).
  The open question the profiler answers: how much stream is EXPOSED at acquire() and at chunk boundaries.

### Next steps on resume (project A: cross-chunk expert-stream pipelining, lead-approved, multi-day OK)
1. At load<28: `kbench --env TRUSS_STREAM_PROFILE=1 --note "stream profile"`; read "prefill stream: copies X busy,
   acquire waits Y exposed" + "prefill sections ms". If Y ~= 0 (stream hidden): cross-chunk pipelining only saves
   the inter-chunk bubble (first-2-layer copies + host glue, est ~2-3 s/run = 3-4%) -> report to lead BEFORE
   building. If Y is seconds: build as pitched (+17-27%, prefill 70 -> ~55-60 s).
2. Then per lead: design + unit test for ring/protect under cross-chunk eviction; implement in small kbench-gated
   steps (EXACT, no regression), RESULT notify per keep; keep 0.75 GB VRAM margin + host RAM budget.
3. Server was down ~15 min for the lead's KL measurement (back up). Load rule: no rows at load >~28.

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
