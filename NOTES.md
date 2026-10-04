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
- **GAP CORRECTED (instrumented row 1005_033555 @load 19-20, decode 62.0 = new best sample, prefill 2132, EXACT;
  row had a self-inflicted bug: section/moe/driver reads dropped -> layer line showed stale prefill values; fixed
  c90a254+fix commit). TRUE per-pass split: host windows draft 1.67 + verify 40.93 + head 0.91 + ACCEPT 0.87 =
  44.39 vs device 43.50 -> host overhead ~= accept 0.87 (per-layer GDN replay on partial accept - structural) +
  PLE decode gather wait 0.34 (rows are new sliding-window rows every pass, reads pipelined, tail exposed).
  Decode is ~97% device-bound; earlier "4.5ms gap" was my arithmetic error (wrong tokens/pass divisor).**
- Decode floor at load<=20 ~= 44.4ms/pass (62 tok/s). Device verify 40.9: mixer ~7.1 (GDN state r/w BW-bound),
  routed MoE ~20 (copies ~10 = demand link time; expert kernel ~5.25 GPU-resident experts), join ~11-16 (DRAM),
  plan-wait ~2.7-3.6 (doorbell->split->plan dependency), ple+hc ~2.7, hc ~1.7.
- Remaining decode levers: (a) hints/ring/coupled-sampling = config+DIFF, human-gated (reported); (b) CPU kernel
  already at DRAM floor + op-count optimized (#76 comments in expert_trellis.cc: gemv_i16 5 mul-ops/16w, 2.8 cyc/8w,
  bound by total ops; kmicro mine-vs-ref 3-4% = load noise, checksums identical); (c) CANDIDATE: GPU expert-kernel
  batching (moe_window.cu, 5.25ms/pass ~36 GPU experts/pass - launch/tail overhead? placement+order neutral if
  combine order kept) - reading moe_window.cu now.
- Lead steer 03:35 (post row D): exact-safe decode levers = CPU expert kernel (kmicro, checksums MUST NOT change)
  + host pass overhead (plan/readback/launchs). kmicro baseline now: K3 R1 174.6us (ref 168.2), K25 199.4 (192.6),
  K35 203.7 (197.2) - same checksums as main ref; branch ~3-4% over ref = load noise or small real regression, recheck.
- **Priority demand-copy stream: analyzed, SKIP** (lead approved but math says ~0 gain: all copies share one copy_
  stream; measured wait-for-copies 10ms/pass == 132.4MB demand / 13.3GB/s -> the wait IS the demand link time,
  not queue drain; admit claims already yield via demand_ev_/piece_out_; hints subsume it). Cross-stream meta-table
  races make it risky anyway. Told lead in 02:36 DECISION note.

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
