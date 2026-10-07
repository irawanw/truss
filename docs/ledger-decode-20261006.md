# Decode bandwidth ledger — X3.1, 150K, served config (Plan v3 Phase 0.4, 2026-10-06)

Capture: `data/logs/ledger04_204049/ledger_decode.nsys-rep` (nsys 2024.4.2, `-c cudaProfilerApi`,
`TRUSS_BENCH_PROFRANGE=1` brackets exactly the decode loops; served env incl. `KV_LEND=1 ADMIT_IDLE=64`;
NO section timers; `tk-bench-spec` eval_clean_150k 60 tok). Result: **20 passes, 1.06 s = 53.0 ms/pass wall,
3.00 tokens/pass (R~4 rows), 56.4 tok/s under nsys** (56.4 vs 66.6 baseline = tracing overhead ~15%, ordering
preserved). Kernel busy 42.02 ms/pass; **compute-stream (13) serial gaps 11.20 ms/pass**; stream 14 = copies
(22,053 rows/20 pass = 1,103/pass). Shapes are measured (GGUF tensor table + grid-outs decode below), not assumed:
d_model 2560, hc_dim 10240 rank 320 (2 hc_mix/layer), GDN Hk16 Hv48 S128 conv4 (qkv 10240 + gate 6144 + α48 + β48
= 16480 outs — 36 launches/pass ✓), DSA H24 HKV2 D256 IH4 ID128 RATIO4 top-2048 cells (q|gate 12288 + k512 + v512
+ idx_q512 + idx_k128 = 13952 ✓), experts 512 top-10 d_ff640 (1.8552 MB/expert), router fp32 2560×512.
Grid decode per q8_gemm.cu:268-284: `per = 4×32/lpo`; outs = grid × per (lpo32/wpo1 ×4, wpo2 ×2, wpo4 ×1,
lpo16 ×8). Weight bytes: Q8_0 = 1.0625 B/w, F32 = 4 B/w.
`eta = bytes/(t×936 GB/s)`, `Rec = t − bytes/(0.65×936)` ms/pass.

## Per-kernel (per pass, R~4 rows)

| kernel | n | t ms | GB | bytes formula | GB/s | eta | Rec ms |
|---|---:|---:|---:|---|---:|---:|---:|
| wait_plan_kernel | 51.5 | 11.01 | 0 | device-side wait for expert plan — NOT bandwidth | - | - | (21.5 wait total) |
| spin_kernel | 48.0 | 10.52 | 0 | device-side wait for CPU tier — NOT bandwidth | - | - | |
| moe window_kernel (resident experts) | 51.5 | 5.28 | 1.63 | ~1426 resident routings, ~880 distinct × 1.8552 MB | 309 | 0.33 | 2.60 |
| dense gemv_multi ALL | ~590 | 7.71 | 4.52 | sum of rows below | 586 | 0.63 | 0.28 |
|  ├ GDN qkv+gate+α+β (16480×2560) | 36.0 | 2.08 | 1.61 | 44.8 MB × 36 GDN layers | 774 | 0.83 | 0.00 |
|  ├ mixer out-proj (2560×6144 wpo2) | 54.3 | 1.40 | 0.91 | 16.7 MB × (36 GDN + 12 DSA + MTP) | 648 | 0.69 | 0.04 |
|  ├ lm_head (248320×2560) | ~1.0 | 0.78 | 0.675 | Q8 output.weight, once/pass | ~870 | 0.90 | 0.00 |
|  ├ hc.up (10240×320 lpo16) | 106.5 | 0.87 | 0.37 | 3.48 MB × 96 mix + MTP + head | 426 | 0.46 | 0.30 |
|  ├ hc {down,inject} (324×10240 wpo4) | 102.9 | 0.79 | 0.36 | 3.48 MB × 103 | 453 | 0.48 | 0.26 |
|  ├ DSA q|gate+k+v+idx_q+idx_k (13952×2560) | 15.4 | 0.73 | 0.59 | 38.0 MB × (12 DSA + MTP) | 808 | 0.86 | 0.02 |
|  ├ shared {gate,up} (1280×2560) | 51.6 | 0.42 | 0.18 | 3.48 MB × 51.5 | 429 | 0.46 | 0.12 |
|  ├ shared down (2560×640 lpo16) | 51.5 | 0.31 | 0.09 | 1.74 MB × 51.5 | 290 | 0.31 | 0.09 |
|  ├ draft head (40528×2560) | 2.5 | 0.32 | 0.28 | 110 MB × 2.5 (draft_vocab) | 859 | 0.92 | 0.00 |
|  ├ MTP qkv (12800×2560) | ~1.0 | 0.04 | 0.03 | 34.8 MB × 1 | ~700 | 0.75 | 0.00 |
|  └ misc singletons (<0.05 ms ea) | ~30 | 0.07 | ~0.03 | PLE key/value (in=H·D), embd gather | - | - | 0.02 |
| f32_gemv_multi<2> router (48+MTP) | 51.5 | 0.82 | 0.252 | 2560×512×**4 B** × 51.5 (fp32 weights!) | 308 | 0.33 | 0.40 |
| DSA glue: score 15.5 + select 14.7 + attn 14.7 + rope 15.5 + index/qkv/partial/combine ~28 | ~130 | 2.02 | ~0.18 | score reads idx_k fp16 [37.5K blk][128] 9.6 MB×15.5 = 149 + scores 2.3 + attn KV 512 blk×2×256 B×R×15.5 ≈ 25 | 89 | 0.10 | 1.71 |
| GDN glue: delta_rule 41.4 + state 41.4 + conv 41.4 + l2 41.4 + gates 41.4 + out_norm 36 | ~240 | 0.82 | ~0.24 | state r+w [Hv48][128][128] fp32 ×2 ×36 = 226 + conv_state 9 + acts | 288 | 0.31 | 0.43 |
| hc glue: norm 110 + collapse 106.5 + combine 103 + silu 106.5 + expand 1 | ~426 | 1.26 | ~0.10 | hc streams 4×2560 f32 r+w per row per launch (≈100 MB total) | 79 | 0.08 | 1.10 |
| quantize_kernel<half> | 427 | 0.62 | ~0.013 | fp16 read + int8/scale write, R×K×3 B | 21 | 0.02 | 0.59 |
| route_kernel<16> | 98.5 | 0.65 | ~0.001 | 512 logits read + top-10 write ×R | 1.5 | 0.00 | 0.64 |
| add_mapped_kernel (PLE add) | 48.0 | 0.33 | ~0.008 | 2560 f32 ×R r+w | 24 | 0.02 | 0.32 |
| publish_kernel | 51.5 | 0.56 | ~0 | plan publish (scheduling) | - | - | 0.56 |
| shared_add + swiglu | 103 | 0.15 | ~0.005 | shared-expert act rows | 33 | 0.04 | 0.14 |
| ple (norm/gate/conv/hist) + sampling + mtp join + rows + tail | ~15 | 0.19 | ~0.005 | head/sampling/PLE glue | - | 0.05 | 0.18 |
| **sum kernels** | ~2224 | **42.02** | | | | | |
| **stream gaps** | | **11.20** | | see below | | | ~11.2 |
| H2D memcpy (stream 14, overlapped) | 1106.7 | 26.20 | 0.217 | (47 demand + 63 hint + ~61 admit) experts ×1.95 MB | ~13 GB/s link | | ~10 exposed |
| D2D memcpy | 184.4 | 0.28 | ~0.03 | internal state copies | - | | |

## Gaps (11.20 ms/pass on the compute stream)

7 stalls/pass account for **7.45 ms/pass**: each sits **after `hc::combine` and before the next `quantize`**
(11.5–25.1 ms per 20-pass capture) — the host thread is blocked at the MoE boundary (expert results /
admission / copy completion) before it enqueues the router quantize. Remaining ~3.8 ms/pass: argmax→argmax
(3.7 ms/20p), rows_kernel and silu stalls (3.8/20p each) — sampling + PLE host reads.

## Reading

1. **~29 ms of the 53 ms pass is waiting for the expert supply chain**: 21.5 ms as device-side wait kernels
   (spin 10.5 + wait_plan 11.0) + ~7.5 ms as pure host-stall gaps at the MoE boundary. No bandwidth is being
   spent there — it is the CPU tier + PCIe + admission pipeline (Phase 2/3 lane).
2. **Dense gemv is already near target in aggregate (eta 0.63)**; GDN qkv 0.83, DSA qkv 0.86, heads 0.9 are at
   target — do not touch. Recoverable inside dense: hc.up/down (0.56) + shared (0.21) ≈ 0.8 ms.
3. **The one big real-bandwidth kernel is `window_kernel` (resident experts): eta 0.33, 2.6 ms recoverable**
   — G3 bitshift-trellis/occupancy work, not fusion.
4. **Latency/fusion class (all small bytes, ~4.5 ms recoverable)**: DSA glue 1.71 + hc glue 1.10 + quantize
   0.59 + route 0.64 + add_mapped 0.32 + publish 0.56 + GDN glue 0.43 + router-f32 0.40 — ~2,224 launches/pass,
   hundreds of 1–10 µs kernels: megakernel/fusion territory (Phase 1.2/1.3). Router is fp32 (4 B/w): moving it
   to Q8 would ~halve its 0.82 ms — **precision decision, not a silent commit**.
5. Cross-check vs research doc §5 floors (non-expert 4.6 GB→7.6 ms): measured non-expert kernels ≈ 15 ms at
   ~5.2 GB (eta 0.35 incl. glue); excluding glue the pure weight-stream rows are ~0.6 — the floor gap is
   latency+launch, consistent with the fusion plan.
6. Uncertainties: window bytes use ~880 distinct resident experts (engine stats + research doc; ±10%); DSA idx_k
   assumed fp16 (int8 KV would halve score bytes → DSA eta ~0.15, Rec ~1.5); GDN state assumed fp32 [48][128][128]
   (from ssm config keys; kernel args show fp32 ✓); "misc singletons" bucket <0.1 ms.

Method files: `/tmp/kern_sum.csv`, `/tmp/kern_trace.csv` (nsys stats exports; trace Name=field 20, GrdX=3);
GGUF tensor table via header parse (1657 tensors, `/tmp/gguf_infos.pkl`); capture script
`scripts/run_ledger_04.sh` (worktree). Reproduce: `nsys stats --report cuda_gpu_kern_sum|cuda_gpu_trace
--format csv --force-export true <rep>`.
