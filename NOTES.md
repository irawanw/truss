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

## Now (10-05 late)
**Row = branch `ple-dequant` (3f6850a): pipelining + AVX2 put + skip-insert for prefill tickets + phase stats.**
Quick row 1005_013552 QUICK_OK EXACT at load 50 (skip-insert keeps tokens identical; decode tickets still cache).
- Rows so far: 1005_010443 (688cfc4, single run, no --repeat2): decode 61.8 prefill 2057 EXACT load 27.9, gather
  5.82s (reads 2.00). 1005_011323 (688cfc4 --repeat2): decode 44.9/62.6, prefill 1694/979, loads 28.4/31.9 —
  gather 19.48/76.74s but **reads only 0.58/0.16s => pipelining hides the disk; the HOST work in collect/issue
  (hash, page-dedup, copies, map+cache-insert) is the wall and scales ~70x under tenant CPU/RAM pressure**.
- Phase split (long20k, ~2.5 chunks, skip-insert build, load ~50): gather 1.18s = reads 0.52 + copy+put 0.26 +
  insert 0.02 + ~0.38 hash/issue (ple_rows + issue page-dedup, timed in ple() not collect). Insert is now ~nil.
- Next: kbench --repeat 2 on 3f6850a at load < ~28 (poller bg_1); keep iff decode >= 52.3 AND prefill > 2043
  both repeats, EXACT. If phases show hash/issue dominant after this, instrument ple_rows vs issue() separately;
  parallel collect (worker threads, hits-first barrier) stays the follow-on lever.

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
