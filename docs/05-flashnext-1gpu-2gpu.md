# Flash-Next trellis (FNT): one quantization, a 1-GPU pack and a 2-GPU pack

Written 2026-09-29, after Strata was measured on one 3090. Numbers marked *(est.)* are
arithmetic, not measurements.

## 1. The rule (user, 2026-09-29)

**A 2-GPU build must be ≥ 2.0× the 1-GPU build of the same engine, on the same prompts, for
both prefill (PP) and decode (TG).** If it is not, the second card is wasted: two independent
1-GPU instances already give 2× aggregate throughput for free, with zero engineering.

That makes the 1-GPU build the reference, and the 2-GPU build has to earn its existence against
it. Today's llama-paw layer split fails this rule. The second card adds memory, not speed: 46–48
raw and 79–89 with MTP on two cards, against Strata's 84–107 on one.

## 2. What Strata proves, and what it does not

Measured here (`~/ML_projects/flashnext/20260929_strata_q2_0/reports/RESULTS.md`):
- one 3090, 23 CPU workers;
- decode 84–94 tok/s at 4–5K tokens and 86–107 at 26–29K;
- prompt processing 960–1330 tok/s.

**It proves:**
1. **Bytes per token decide TG, and the dense part is most of them.** Strata keeps the dense
   part at ~3.1–3.5 GiB of mixed 2–5 bit. We read 4.45 GiB of Q8_0 dense per token, 86% of our
   bytes. We lost on bytes before any kernel ran.
2. **A second processor helps only if nobody waits on a driver.** The CPU computes expert misses
   while the GPU runs; the handoff is a flag in mapped pinned memory, and one graph covers each
   window. No `cudaMemcpy` and no `cudaStreamSynchronize` sit on the token path.
3. **Speculation carries the speed.** Its round costs 28.9 ms: 256 tokens came from 100 rounds,
   2.56 tokens per round at 4K. The raw forward is slow (~50 tok/s, per its paper). The 84–94
   comes from speculation.

**It does not prove "tensor parallel is stupid".** Strata runs on one GPU, so it never tests
splitting a layer across two GPUs. What *is* proven stupid on this box is **blocking** TP:
NCCL/driver collectives, 2 per layer, on PCIe 3090s with no P2P. Our own "TP closed, 30 µs"
measured `cudaMemcpy` staging, which is exactly that kind of driver wait.

## 3. The only two ways two GPUs make ONE stream faster

A token's forward pass is a chain of 48 layers. For two GPUs to finish it in half the time,
there are only two options.

**(A) Split the work of every layer: both GPUs work on the same token at the same moment.**
- Attention and GDN are split by heads. Heads are independent, so there is no traffic inside the
  mixer.
- Experts are split by id.
- Each half is exchanged once the block output is ready. Whatever name it gets, this is the only
  way the *latency of one token* halves.
- The design question is only how the exchange is done: non-blocking, device-side, and 1–2
  times per layer.

**(B) Split the tokens: GPUs work on different windows at the same moment** (a pipeline of
layer ranges). For one stream, window k+1 depends on what window k accepts. It can only be
started early *speculatively*, betting that window k's best path is fully accepted.
- The gain is capped at 1 + P(full accept) ≈ 1.4–1.6× *(est.)*.
- **It cannot reach 2× for one stream.** It is fine for prefill, where chunks are independent.

So:
- **TG uses (A).**
- **PP can use either.** (A) keeps one layout for both.
- The layer split we run today is (B) with no speculation, which is why it gives 1.0×.

## 4. Why ≥ 2× is reachable, and can be superlinear

A 1-GPU pack **cannot hold all experts**: 34.5 GB of trellis experts against 24 GB of VRAM. So
the 1-GPU build pays for misses (CPU compute or PCIe copies) and for streaming every expert
during prefill.

The 2-GPU pack holds **everything resident** (~17 GB of experts per card). The miss path
disappears entirely. So the 2-GPU build removes a cost that the 1-GPU build has, on top of
halving the bytes. That is the source of > 2×.

**Arithmetic for one 4-row verify window on the 2-GPU pack (est.):**

| per GPU, per window | bytes | time at 650 GB/s effective |
|---|---|---|
| half the re-encoded dense part (~3.2 GiB total) | ~1.7 GB | 2.6 ms |
| half the unique routed experts for 4 rows (~2.4 GiB total) | ~1.3 GB | 2.0 ms |
| 96 exchanges × ~5 µs (P6.0 gate is ≤ 8 µs) | — | 0.5 ms |
| launch-free graph overhead, norms, sampling | — | ~1.2 ms |
| **round** | | **~6.3 ms** |

- At Strata's 2.56 tokens per round, that gives a physics ceiling of **~400 tok/s**.
- The 2× rule needs **≥ 180** (2 × a Strata-class 1-GPU build), which means reaching ~45% of
  that ceiling.
- Against today's round: Strata's 1-GPU round is 28.9 ms. The 2-GPU round must be ≤ 14.5 ms.

**PP (est.):**
- Prefill is tensor-core-bound, and the 2-GPU pack streams no experts.
- The exchange is ~5–20 KB per row per layer. That is 10–40 MB per layer for a 2048-row chunk,
  at ~12 GB/s (gen4 x8, GPU 2).
- The exchange is hidden by overlapping sub-chunks: exchange sub-chunk i while computing i+1.
- **2× Strata = ≥ 2,600 tok/s** is well under the link cap.

## 5. One quantization, two packs

| | 1-GPU pack "S1" (3090 24 GB; a 3060 12 GB variant later) | 2-GPU pack "D2" (2×3090) |
|---|---|---|
| dense | re-encoded mixed 3–5 bit trellis (shared by both packs) | same weights, **tiles pre-split per rank**: q/k/v/GDN-in by head, o/out by row, lm_head by vocab rows |
| experts on GPU | hot set from the routing profile, trellis | **all**, trellis, assigned by id with **hot-balanced bin packing** so expected per-token expert bytes are equal on both cards |
| experts off GPU | cold set in pinned RAM (see below) | none |
| exchange | GPU↔CPU doorbell (Strata's) | GPU↔GPU doorbell through mapped pinned host memory, device-side flags, fixed-order sum (bit-exact vs serial) |
| draft head | on the GPU in the window graph | on the less-loaded rank, overlapped with the other rank's tail |

- **The re-encode we need is one, not two.** The quantized weights are the same. D2 is a
  **repack**: 16×16 tiles split at tile boundaries, plus an expert-to-rank map. The only new
  quantization is the dense part (Q8_0 → mixed trellis), and it serves both packs.
- **The S1 cold tier is a real problem for trellis.** POC-1 measured CPU trellis expert decode at
  **0.13 tok/s** on this Threadripper. Trellis is a GPU codec. So S1's cold tier needs either:
  - (i) a CPU-friendly codec for RAM-resident experts (a second encode of the cold experts only,
    with a quality gate); or
  - (ii) PCIe copies of trellis experts to the GPU (Strata's `--pcie-frac`). Calibration here
    picked 0.20, so copies are not free.

  This is the one place where "two encodes" may truly be needed. D2 has no cold tier and does
  not care.

## 6. The design rules for D2

1. No `cudaMemcpy`, `cudaStreamSynchronize`, NCCL or host thread on the token path. One captured
   graph per rank per window size, with the spin-waits on the flags inside it.
2. At most 2 exchanges per layer: after the mixer's output projection, and after the MoE (with
   its shared part). Variant **A-dup**: compute the mixer on both ranks, duplicating those bytes,
   so only 1 exchange per layer is needed. Choose by measurement, because it trades duplicated
   bytes against 48 exchanges.
3. Load balance is part of the encode. The rank that reads more expert bytes sets the round time.
   The routing profile drives the expert-to-rank map, and the imbalance is measured offline
   before any engine code (P7.0c).
4. Sums are exact: fixed order rank 0 then rank 1, fp32 accumulate. The output must match the
   serial reference bit for bit in the one-layer test.
5. The drafter never costs verify time: it runs on the rank that finishes its half first.
6. GPU 2 has an x8 link. Put the smaller-traffic role on it, or pair GPU 1 and GPU 3 (both x16)
   when renters allow.

## 7. Kill rules

- P6.0 doorbell median > 8 µs at 20 KB: D2 route (A) is dead as designed. Try A-dup (48
  exchanges); if that also misses 2× on paper, D2 is dropped.
- D2 < 2.0× S1 on TG or PP at the same prompts: D2 is not shipped, and the 2-GPU box serves two
  S1 instances.
- Dense re-encode fails the KL gate against today's Q8_0 build: keep Q8_0 dense in D2 (the bytes
  halve anyway) and re-derive the arithmetic.
