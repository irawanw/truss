# Prefill bandwidth ledger — X3.1, 150K prompt, served config (Plan v3 Phase 0.5, 2026-10-06)

Capture: `data/logs/ledger05_211615/ledger_prefill.nsys-rep` (whole-process nsys — `TRUSS_BENCH_PROFRANGE`
brackets only decode, so no capture-range here; served env, NO section timers; `tk-bench-spec` eval_clean_150k
N=1). Result: load 9.11 s; **prefill 149,183 tok in 72.72 s = 2,052 tok/s** (PLE host gather 1.49 s);
chunk = 8,192 tokens (served) -> **20 chunk passes** (18 full + 1 partial 1,727 + tail), markers: ple hist /
delta_rule / route_* counts (931 = 19x49, 720 = 36x20).

**Measured clock during prefill (clocks.csv, 200 ms cadence, GPU idx 1): SM median 1,440 MHz @ 277 W (TDP cap,
util 100%)** — all peaks below are at 1,440 MHz, NOT boost: int8 IMMA **120.2 TOPS**, fp16 HMMA fp32-acc
**60.1 TF**, fp16 HMMA fp16-acc 120.2, fp32 CUDA-core **30.2 TF**, DRAM **936 GB/s**, PCIe gen4x16 **13.4 GB/s**.
`eta = work/(t x peak)`, `Rec = t - work/(0.65 x peak)`.

## Middle chunk (chunk 9, T=8,192): window 3,847.7 ms, kernels 2,574, busy 3,790.5 ms (98.5%), gaps 57.2 ms

| kernel | n | ms | % | work (formula) | rate | peak | eta | Rec ms |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| dsa attn_kernel (12 DSA + MTP) | 13 | 779.2 | 20.6 | 5.4 TF (2x2xTx2048x256x24x13) + KV read upper bound 223 GB (Tx2048x1KB fp16x13, no L2 reuse credit) | 6.9 TF / 287 GB/s | 60.1 / 936 | 0.11 / 0.31 | **412-641** |
| moe gate_up+down (trellis, HMMA m16n8k16 fp32-acc per moe_prefill.cu:14) | 98 | 933.9 | 24.7 | 38.7 TF (Tx10x3x640x2560x2x48) | 41.4 TF | 60.1 | **0.69 AT TARGET** | ~0 |
| dense gemm_kernel (int8 IMMA) | 702 | 941.9 | 24.8 | 59.1 TF (2xTx3.604G Q8 weights = mixer+hc+shared+indexer, GGUF sum) | 62.7 TOPS | 120.2 | 0.52 | 185 |
| gdn delta_rule | 36 | 281.7 | 7.4 | ~1.2 TF fp32 (chunked rule ~2.5 passes x Tx48x128x128) + 19.6 GB (state r+w 226 MB-equiv + fp32 acts T x (10240+6144) x 4) | 4.1 TF / 69 GB/s | 30.2 / 936 | 0.14 / 0.07 | **220** |
| hc glue (norm+collapse+silu) | 296 | 173.1 | 4.6 | 62.8 GB (99 mixes x ~650 MB: res 84 + xn16 20 + gate 335 + down-in 87 + mixed 126) | 363 GB/s | 936 | 0.39 | 70 |
| dsa select+score | 208 | 131.9 | 3.5 | score reads idx_k fp16 [n_blk][128] ~8/layer x 12 x growing n_ctx/4 + select top-512 | ~latency | 936 | ~0.1 | ~90 |
| quantize_kernel<half> | 394 | 116.6 | 3.1 | 25.3 GB (fp16 in + Q8 out, T x ~2560 x 3.06 B x n) | 217 GB/s | 936 | 0.23 | 75 |
| gdn conv | 36 | 57.4 | 1.5 | qkv fp32 r+w ~24 GB (T x 10240 x 4 x 2 x 36) | 418 GB/s | 936 | 0.45 | 18 |
| ampere_sgemm (router fp32, cuBLAS) | 49 | 58.5 | 1.5 | 1.03 TF fp32 (2xTx512x2560x48) | 17.6 TF | 30.2 | 0.58 | 5 |
| moe prep (token gather into expert order) | 49 | 72.3 | 1.9 | T x topk x hidden fp16 scatter ~13 GB | 180 GB/s | 936 | 0.19 | 51 |
| gdn out_norm + l2 + gates + state | 145 | 34.2 | 0.9 | fp32 act r/w ~12 GB | 351 GB/s | 936 | 0.38 | 14 |
| moe combine | 49 | 28.5 | 0.8 | expert out gather T x topk x 2560 | ~150 GB/s | 936 | ~0.16 | 20 |
| ffn shared_add + swiglu + route | 147 | 23.6 | 0.6 | shared expert acts | ~100 GB/s | 936 | ~0.1 | 15 |
| dsa qkv/index/rope/partial | 40 | 25.5 | 0.7 | rope+index prep fp16 | - | - | - | 15 |
| cublas gemvx (shared-expert gate f32) | 49 | 5.0 | 0.1 | T x 2560 f32 | - | - | - | 3 |
| ple (hist/norm/gate/conv) + rows + mtp join + expand | 11 | 14.6 | 0.4 | PLE gather+conv, embedding rows | - | - | - | 10 |
| **sum kernels** | 2,574 | **3,790.5** | 98.5 | | | | | |
| **gaps (host: PLE gather, stream scheduling)** | | **57.2** | 1.5 | | | | | ~57 |
| H2D expert stream (overlapped, batched copies) | ~51 | 38.5 GB | | 48 layers x ~433 non-resident experts x 1.855 MB (residency 3,813/24,576 = 15.5%) | 13.4 link | | | 2,874 ms link-busy (75% of window) |

## Reading

1. **Prefill is COMPUTE-bound, not link-bound**: kernels 98.5% busy while PCIe runs 75% of the window
   (2.87 s of 3.85 s) — 25% PCIe headroom. Chunk wall = max(compute 3.79 s, link 2.87 s).
2. **Expert streaming is architectural at chunk=8192**: ~all 512 experts/layer are touched per 8K chunk
   (81,920 routings), so ~38.5 GB re-streams per chunk -> ~760 GB per 150K prompt. Only 15.5% resident.
   Raising residency cuts link time but not the compute wall; it matters if compute drops below ~2.9 s.
3. **Already at target — do not touch: moe trellis gate_up/down (eta 0.69 fp16-HMMA)**. This is the plan's
   G3 evidence: the trellis path reaches 0.69 of the HMMA peak at batch T=8192; the DECODE window kernel
   (eta 0.33 at R~4) is the anomaly to attack (occupancy at batch 4, not the codec).
4. **Top recoverable per chunk (at 0.65-of-peak floors): dsa attn 412-641, gdn delta_rule 220, dense int8
   gemm 185, quantize 75, hc glue 70, dsa select/score ~90, moe prep 51** — sum ~1,075 ms -> chunk 2.77 s
   = ~2,950 tok/s, i.e. **the 3,000 tok/s target is reachable by kernel efficiency alone, no weight changes**,
   dominated by (a) DSA attention TC utilization + KV reuse, (b) GDN delta-rule fp32 -> TC, (c) int8 GEMM
   pipeline (cp.async/occupancy).
5. Router fp32 GEMM eta 0.58 is fine at 1,440 MHz; Q8 router would trade ~5 ms/chunk for a precision decision
   (same DECISION as decode ledger).
6. Uncertainties: attn KV bytes are an upper bound (no L2-reuse credit; true DRAM traffic needs ncu — rec
   quoted as range); delta_rule FLOPs estimated from the chunked-rule structure (~2.5 passes over state);
   "misc" rows are upper-bound-light estimates from launch counts; last chunk (1,727 tok) excluded.

Method: trace `/tmp/prefill_trace_raw.csv` (exported; copy in data/logs/ledger05_211615/), chunk windows by
`ple::hist_kernel` timestamps, grids from trace field 3; weights from GGUF tensor table (`/tmp/gguf_infos.pkl`).
Reproduce: `scripts/run_ledger_05.sh` (worktree) + `nsys stats --report cuda_gpu_trace --format csv
--force-export true <rep>`.
