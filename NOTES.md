# Agent notes (survive context compaction: keep current)

## Ground rules I work under (added 10-04 by lead, rule 9 in AGENT_BRIEF.md)
- Keep a change only if: (a) decode AND prefill not slower than the best kept kbench row (`--repeat 2`),
  (b) tokens EXACT, (c) weights/pack/quantization/expert error UNTOUCHED (no re-encode/re-quant/prune/skip/approximate).
  Anything that changes expert arithmetic -> notify DECISION, do not commit.
- Notify file: `echo "date | KIND | msg" >> /home/green-gpu/ML_projects/flashnext/20261004_selfopt/data/notify.txt`
  (RESULT on a new best row, DECISION on quality/config/weights change, STUCK after 3 attempts on one idea).

## Baseline facts (kbench row 1004_220442, commit 8af5ead)
- decode 52.3 tok/s, prefill 2043 tok/s, cpu_ms/slot 1.176, exact=DIFF, load ~24.
- **The baseline itself is DIFF** vs `data/ledger/ref_tokens.i32` (golden made by selfopt_setup from main's build).
  Cause: the served config runs `TRUSS_CPU_DYNAMIC=1` (split decisions are timing-based) and the dynamic split is
  known-unstable (TRACKER do-not-repeat 33). So exactness here is only controllable as *tokens identical to the
  previous build's tokens*, not to the golden. I compare my run's `tokens.i32` against the baseline run's
  (`logs/kbench_1004_220442/tokens.i32`) and treat that as the EXACT gate; DIFF-vs-golden status alone is not my regression.
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

## Now (hypothesis under test)
**Stream-slot depth (prefill).** `ExpertStore` stream mode used exactly 2 whole-layer slots: `prefetch(l+2)` into
slot `l%2`, gated by `released_[l%2]`. With stream/layer (~56 ms, link-bound) ≈ compute/layer (~50-84 ms), the copy
engine idles on the release gate every layer, so chunk time ≈ Σ_l max(stream_l, compute_l) ≈ 2x the max, instead of
max(Σ stream, Σ compute). Measured 4.05 s/chunk vs 2.7 s stream bound and ~2.7-4 s compute: consistent with that
serialization. Fix: N slots (default 4) + `prefetch(l+N)`; the copy stream then holds up to N-1 layers of copies
queued and stays saturated; chunk time → max(total compute, total stream) + PLE host time.
Clamp: `eff_slots = min(asked, (ring - chunk_buffers)/slot)` computed with identical inputs in plan() and the
constructor, so the ring never grows past what the hot-set plan reserved => hot set (and decode) unchanged; the
request is free when the ring already holds N slots + buffers.
Env knob `TRUSS_STREAM_SLOTS` (default 4). Startup prints slots/slot GB/ring/buffers.
Expected: prefill 2043 -> ~2600-3000, decode neutral.

## Next ideas (after this one)
1. If slots clamps to 3 and prefill still < 2600: per-chunk breakdown of link vs compute (instrument prefetch/acquire).
2. PLE host gather (5 s of 73 s): the H2D of the gathered rows is issued on the *compute* stream (forward.cu ple()),
   so it serializes with the expert stream on the same copy engine; pipeline the gather one chunk ahead
   (PleReader issue/collect already supports it) so the H2D is small and ready.
3. Decode: plan wait 3.2 ms/pass is the host round trip for the split decision (driver: split 0.18 + CPU start 0.92
   + copies/plan 1.27). A GPU-side split with a fixed pcie_frac (do-not-repeat 33 says fixed fraction is the stable
   method) could remove the host decision from the critical path.
4. Decode acceptance: 2.75 tok/pass; coupled draft sampling (TRACKER #109e) was measured but never shipped.
5. CPU join 16.5 ms/pass is a pinned-DRAM-read floor (~445 MB @ 28 GB/s) unless miss bytes drop; bytes are pack
   arithmetic -> DECISION territory, do not touch.
