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

## 5. The pass formula and its floors (one RTX 3090, X3.1, 150K, served config)

Clean measurement (PAW's MTP A/B run A1, no section timers, served env): **77.7 tok/s, 2.89 tokens/pass, pass 37.3 ms**,
CPU wait 14.65 ms, 193 distinct CPU experts (229 slots), 47 demand + 63 hinted copies, 61 admitted/pass.
(kbench rows set TRUSS_PROFILE_SECTIONS=1: that turns admission off in practice and slows decode; the "~60 tok/s"
rows are biased low. Fix kbench before more decode A/Bs.)

Layers are sequential, and inside a layer the CPU's misses run in parallel with the GPU's resident experts only:

  T_pass = sum over 48 layers [ t_mix(l) + t_hc(l) + t_route(l) + max( t_gpu_exp(l), t_cpu(l), t_copy(l) ) + t_comb(l) ]
           + t_head + t_draft + t_host

Floors = bytes / bandwidth (3090: 936 GB/s HBM-class GDDR6X, PCIe x16 ~21 GB/s measured, host DRAM ~40-57 GB/s shared):

| term (per pass) | bytes | floor | measured | x floor |
|---|---:|---:|---:|---:|
| dense weights (Q8: GDN/DSA projections, hc, shared, router) | ~4.1 GB | ~4.4 ms | | |
| mixer state + KV at 150K (GDN states r+w, DSA indexer keys, 2,048 selected K/V) | ~0.5 GB | ~0.5 ms | | |
| **GPU non-expert total (mix + hc + route + combine)** | ~4.6 GB | **~5 ms** | **~13 ms** | **~2.6x** |
| GPU resident experts (~880 distinct x 1.95 MB) | ~1.7 GB | ~1.9 ms | ~5.3 ms | ~2.8x |
| CPU misses (193 x 1.85 MB, read in place) | ~0.36 GB | ~7 ms at 50 GB/s | 14.65 ms wait | ~2x |
| head (output 0.675 GB) + MTP drafts | ~1.2 GB | ~1.3 ms | ~2.7 ms | ~2x |

Per layer today: GPU non-expert ~0.27 ms, then max(GPU experts 0.11, CPU 0.30) -> the CPU binds every layer.
If the GPU non-expert part reached ~80% of bandwidth (0.12 ms/layer) AND the CPU tier reached its DRAM floor
(~0.15-0.2 ms/layer): pass ~0.32 x 48 + ~3.5 = **~19 ms -> ~150 tok/s at 2.9 tokens/pass** (upper bound).
Neither alone is enough: GPU-only work stops at ~0.27->0.12 but the CPU still binds (pass ~31 ms, ~93 tok/s);
CPU-only work stops when the GPU non-expert time binds.

**Why the GPU is slow at decode:** not FLOPs or bytes, but latency: ~2,200 launches per pass (46 per layer), small
grids (4 rows), each kernel's ramp-up/tail and the dependency chain between them. The mixer moves ~0.5 GB yet takes
7.2 ms (14x its byte floor). This is exactly the regime of the Hazy Research "no bubbles" megakernel (batch-1 Llama-1B:
existing engines <= 50% of bandwidth on H100, one persistent kernel 78%).

**Why the CPU is used for misses (and is right to be):** a cold expert lives in host RAM; the GPU can only read it at
PCIe speed (21 GB/s) while the CPU reads it in place at DRAM speed (~2x). This is Fiddler's result (ICLR 2025).
The CPU tier runs at ~half its DRAM floor today (engine 0.068 ms/slot vs PAW's pool microbench 0.045).

## 6. Workstreams after the MTP A/B (each measured before building)

G1. **Decode megakernel / persistent layer kernel for sm_86** (GPU non-expert 13 -> ~6 ms est.): one persistent
    kernel per pass (or per layer) running the hc -> norm -> GDN/DSA -> hc -> router -> shared chain from a device-side
    instruction queue, CTAs pinned per SM, weights streamed with cp.async, doorbell to the CPU tier unchanged.
    Step 0: nsys of 20 clean decode passes (no section timers): per-kernel bytes/time = achieved GB/s, and the
    idle gaps between kernels; the megakernel's gain is bounded by (sum of gaps + per-kernel ramp losses).
G2. **CPU tier to its floor** (14.65 -> ~8-9 ms est.): find the engine-vs-microbench 1.5x (perf needs the matching
    linux-tools package; else in-process rdtsc timers per phase): candidates are the 4K-page TLB walks on the 38 GB
    pinned arena (1.85 MB expert = 452 pages), the 3-phase barriers (gate/up -> h-quant -> down), and thread wakeups.
G3. **Trellis GPU expert kernel** (5.3 -> ~2.5 ms est.): window kernel at ~320 GB/s; QTIP's bitshift-trellis kernels
    run near the byte bound at batch 1. Profile occupancy/stalls first.
G4. Admission policy as replayed (every CPU miss, link budget permitting): replay -31% misses vs today's capped 61/pass.
