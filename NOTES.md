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

## Now (10-06 ~09:15) - RESUMED, goal: pp+tg as high as possible (human via chat; pm2 brain restarts allowed)
**`agent/x31-speed` = 85d791a (warm handoff, quick-gate EXACT). Kept best stays row D 216d52a (2132/2128).**
- Stream profile verdict (row 1006_082932, d9b6c0b + TRUSS_STREAM_PROFILE=1 @load 31; decode 27.6 there was
  contention garbage): **prefill stream 68.9 s BUSY vs 2.40 s EXPOSED over 74.4 s = 92% busy / 3.2% exposed.**
  Within-chunk pipelining already works; prefill is LINK-BOUND: every chunk streams its whole cold-layer set
  (~46 GB = 47 x ~0.98 GB) at ~13.3 GB/s -> the copy stream (48 x ~74 ms = 3.8 s) paces the 4.08 s chunk; GPU
  sections per 149k prompt: mixer 34.1 s + moe 22.4 + ple+hc 7.4 + hc 5.4 + shared 1.3 + join 0.27 (join ~0:
  prefill MoE is GPU-only, the CPU tier serves decode). Kernel ~58 ms/layer rides inside the ~74 ms copy cadence.
  Project A as pitched (+17-27%) DEAD; only the ~132 ms/chunk boundary bubble (2.4 s/run) is software-addressable.
- **PCIe FACT (nvidia-smi, DECISION-notified 09:12): GPU 2 link is gen4 x8 of max x16** (4-way TRX40 lane split).
  13.3 GB/s = the x8 ceiling. This is THE wall for both targets: pp copy-bound as above; decode demand copies
  132 MB/pass = 10 ms/pass of the 44 ms pass. Physical x16 would ~double copy throughput: pp ceiling ~3000+
  (brief target reachable), tg +10-12. Human/hardware action, nothing in-code.
- **A-lite 'warm handoff' implemented (85d791a):** chunk N's Forward::run queues chunk N+1's layers 0/1 copies on
  copy_ behind released_[0]/[1] (= release(46)/(47), the slots' last readers); begin_stream returns handed ->
  successor chunk skips its prefetch(0)/(1). Same bytes/placement/arithmetic -> EXACT-safe (quick gate EXACT
  1006_085008). Gate: only when successor is a streaming non-E7 chunk. truss_c.cu server loop now passes PLE
  lookahead (next chunk) so SERVING gets both PLE pipelining and warm handoff. Bench row 1006_085508: prefill
  1995/1990 @run-loads 23.9/28.3 = -6% vs kept 2132/2128 @23.8/26.3, decode 51.8/40.2 - SUSPECTED contention
  (box load swung to 45 mid-row); re-measure with profiler at load <24: if exposed < ~1.0 s and pp >= kept ->
  KEEP; else git revert 85d791a (rule: revert what doesn't help).
- Unit test for ring/protect: DROPPED for A-lite - the change moves copy issue-time only, ring/protect/eviction
  untouched; the event chain is 3 CUDA calls and the EXACT gate catches any stale-slot read (hint rows proved it
  has teeth: placement moves -> DIFF). Told lead in the row RESULT.
- Decode stays ~97% device-bound (unchanged from 10-05): floor ~44 ms/pass ~ 62-63 tok/s @load<=20; lead's own
  quiet merge row 1005_165231 showed 71.4/78.9 EXISTS at load 6 (DIFF, his KL territory). tg levers left are
  human-gated: KL hints (K=3/4 DIFF vs golden), ring size (VRAM), coupled sampling (#109e, +~16% via accept),
  and the same x8 link (demand 10 ms/pass). Quiet-box EXACT row for my branch still owed (run with the re-measure).

### Next (in order)
1. At load <24: `kbench --env TRUSS_STREAM_PROFILE=1 --repeat 2 --note "warm handoff profile row"` -> read
   "prefill stream" line + pp numbers vs kept. KEEP or revert 85d791a per the rule above; RESULT notify either way.
2. If kept: brain serves it only after main merge (lead/human action - state in RESULT).
3. Final ladder report to lead: software within ~3% of x8-link wall; remaining gains = x16 hardware, KL gate,
   VRAM ring, sampling gate. NOTES + notify per row.

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
