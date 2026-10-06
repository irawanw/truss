# Why X3.1 decodes at ~60 tok/s on one 3090, and what actually moves it (2026-10-06)

Requested by the user after "we built it from the ground up, why only 60 / 2,000". Everything below is either read
from the code, a measured log, or an offline replay of recorded routing; estimates are marked **est.**

## 1. Where a 150K decode pass goes (measured, kbench row 1006_153824, served config, GPU 1)

pass 44.4 ms for 2.93 tokens: GPU work ~20 ms (mixer 7.2, hc/PLE 3.7, router+publish 2.1, expert kernel 5.2,
shared 0.2) and **~25 ms waiting on expert misses**: CPU join 16.8, copy wait 5.3, host plan 3.3.
Per pass: 487.7 cold experts the GPU side needs, of which 423 already in the ring (ring hits) and 64.6 copied;
261 distinct experts (314 slots) computed on the CPU. So ~325 experts per pass are not in VRAM.

Strata on this box (logs `flashnext/20261003_x31/data/a7`, `20261001_strata_profile`): 23.6-26 ms/window,
GPU-reach 14.4-15 ms, CPU 3 ms, hit rate 89.5-93.4%, 89-106 tok/s (4K prompts; no 150K Strata run exists).
**The GPU work is comparable; the difference is the miss wait (ours ~25 ms, Strata's ~6 ms).**

## 2. Corrections to what I told the user earlier today

- "The 6 GB ring is wasted in decode": wrong. The ring is a CLOCK cache of fetched experts and serves ~423 hits
  per pass (forward.cu:640-690, expert_store.cu fetch()).
- "Token embedding could move to host": already on host (forward.cu:1082, TRACKER #60).
- Dense weights on GPU are ~3.9 GB, KV 3.85 GB (262K reservation; KV lending already returns the unused tail).
  There is no large single-GPU VRAM win left; at most ~1.5-2 GB more for experts.

## 3. Replay: misses per pass vs expert VRAM (real agent traffic, 3,919 passes, 1,071 distinct experts/pass)

`20261003_x31/scripts/c1_residency_sweep.py route_r3_eval.bin`, uniform sizes; 1.95 MB = X3.1's hot-set average,
1.38 MB = Strata's Q2_0. File: `docs/replay_r3.tsv`. (Replay "engine" at 13.5 GB = 264/pass vs 325 measured on the
150K code bench: same order, different text.)

| expert VRAM | size | slots | engine (today's policy) | admit every CPU miss | static oracle |
|---:|---:|---:|---:|---:|---:|
| 13.5 GB (today) | 1.95 | 6,923 | 264 | 181 | 272 |
| 18 GB (one card, everything else gone: not reachable) | 1.95 | 9,230 | 166 | 98 | 167 |
| **33 GB (two 3090s)** | 1.95 | 16,923 | **40** | **14** | 16 |
| 13.5 GB, Strata's 2-bit size | 1.38 | 9,782 | 148 | 85 | 148 |

## 4. What follows

1. **One card, weights/quality untouched:** the ceiling is set by misses. Best policy (admit every CPU miss into the
   cache, which the engine's ADMIT_IDLE only partly does: replay -31%, engine measured -8%) gives 264 -> 181
   misses. **est.** CPU join 16.8 -> ~11.5 ms, pass ~38 ms, **~75 tok/s**. Not 100.
2. **Two cards, layer-split (pipeline) across GPU 1 + GPU 2 (no NVLink needed: one 10 KB hidden-state hand-off per
   pass):** each card holds half the layers' dense weights and KV, so ~19 GB of experts each (~38 GB total, ~80% of
   all experts). Replay at 33 GB: **misses 264 -> 14-40 per pass (7-19x fewer)** = Strata-class hit rates without
   changing a weight. **est.** pass 44 -> ~25-28 ms -> **~100-115 tok/s** at 150K.
   Prefill: both cards' kernels run (chunk N on layers 24-47 while chunk N+1 is on 0-23) and far less cold streaming:
   **est. ~3,500-4,000 tok/s**. Cost: GPU 2 is no longer rentable.
3. Faster CPU tier (engine 0.068 ms/slot vs PAW's pool microbench 0.045): **est.** -4 to -5 ms/pass on one card.
4. Strata itself is not an option at our quality: it needs 2-bit experts (1.38 MB) for its hit rate.
