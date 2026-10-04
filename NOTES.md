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

## Now (10-05 ~03:40)
**`agent/x31-speed` = 216d52a (row D parallel collect MERGED, lead approved). Best rows: 1005_032028 decode 55.8/57.3,
prefill 2132/2128 EXACT @23.8/26.3.** Gather copy+put 0.92->0.15s (total 1.50->0.64). ple_reader_test extended:
parallel mixed + all-miss rows bit-identical (run `./build/ple_reader_test <gguf>` after any reader change).
- **hint_k tested (lead --env): K=4 row 1005_024923 DIFF 51.9/62.1, 2124/2119 (wait-copies UP 10->10.9: prefetch
  98->214MB congests the one copy stream); K=3 row 1005_025943 DIFF 58.6/52.2, 2129/2125 (profile better, join 9.7,
  but decode in noise band). ANY hint-width change moves CPU/GPU placement -> DIFF vs golden -> human-gated.
  RECOMMENDED keep K=2 (matches #85).** Priority demand stream SKIPPED with math (wait ~= demand MB / link rate;
  reordering cannot recover; told lead 03:10).
- **Decode gap measured from bench logs: device draft+verify+head = 43-52 ms/pass vs wall 47.6-52.9 -> host
  stalls/gaps ~= 4-4.5 ms/pass steady (~8-9% of decode wall). Inside it: PLE host gather 1.9-2.7 ms/pass for only
  54 rows = ~46 us/row (SSD-ish per row!) - decode rows are NEW trigram rows every pass (sliding window), so every
  pass pays page copies; reads should be pipelined a pass ahead (wait~0) - NOT yet instrumented for decode: NEXT
  STEP print decode gather wait+phases (ple_host_ms already carries them), one full row to read.**
- Lead steer 03:35 (post row D): exact-safe decode levers = CPU expert kernel (kmicro, checksums MUST NOT change)
  + host pass overhead (plan/readback/launchs). kmicro baseline now: K3 R1 174.6us (ref 168.2), K25 199.4 (192.6),
  K35 203.7 (197.2) - same checksums as main ref; branch ~3-4% over ref = load noise or small real regression, recheck.
  compare vs a11d378 rows (decode 36.8-62.6, 52.3-56.6 at load<=31; prefill 2107-2125). Win+EXACT -> DECISION
  with rows (human changes served config). Mechanism: hinted experts land before the next layer's split -> GPU hits
  at split time -> demand PCIe (132MB/pass, 67.8 fetches) + CPU slots shrink; wait-for-copies 10ms/pass ~= demand
  bytes/link-rate -> K=4 converts demand into early-issued copies (prefetch ~98->~200MB, link capacity ~650MB).
- **Priority demand-copy stream: analyzed, SKIP** (lead approved but math says ~0 gain: all copies share one copy_
  stream; measured wait-for-copies 10ms/pass == 132.4MB demand / 13.3GB/s -> the wait IS the demand link time,
  not queue drain; admit claims already yield via demand_ev_/piece_out_; hints subsume it). Cross-stream meta-table
  races make it risky anyway. Told lead in 02:36 DECISION note.
- Row D candidate (code-only, if hint rows don't satisfy): parallel collect copy+put (load-47 run showed 15.9s
  copy+put under CPU starvation; reader-pool fan-out; hits-first barrier only needed when cache_misses).

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
