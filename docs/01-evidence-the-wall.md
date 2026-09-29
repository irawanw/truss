# 01 — The wall: what we have measured

Everything in this file was measured on this box or read from our own repos. Where a number is
arithmetic or a claim is not measured, it says so. Sources are listed at the bottom as [Sn].

## 1. Hardware facts (measured here)

| fact | value | source |
|---|---|---|
| RTX 3090 pure streaming-read ceiling | **892.7 GB/s** (95.4% of the 936 pin rate), reached with 1 block/SM | [S1] §1 |
| RTX 3090 SM clock under load | steady 1785 MHz; 2115 MHz cold; seen at 1365 MHz at 71 C | [S1] §8, [S18c] |
| CUDA graph node cost | ~1.1 µs per node, whatever the tensor size | [S13] §3, llama-paw `53395ff92` |
| `grid.sync` in a cooperative kernel | ~7 µs each (earlier session, PAW walk kernel) | [S7] |
| GPU0 ↔ GPU3 peer access | none; a 5 KiB round trip costs 30.05 µs, and copies stage through the host | [S13] §1 |
| Hardware counters | `ncu` needs root (`RmProfilingAdminOnly: 1`); use `/usr/local/cuda-12.6/bin/ncu` under sudo | [S1] §2, [S18a] |
| RTX 3060 12 GB | 28 SMs, 360 GB/s spec. **Its read ceiling has not been measured.** | — |

## 2. Model facts (Qwen3.8-27B, read from the GGUF header)

- 64 blocks: 48 gated-delta-net (GDN) layers + 16 full-attention layers (`full_attention_interval = 4`).
- Hidden 5120, FFN 17408, 24 query heads / 4 KV heads, head dim 256. GDN: 48 value heads, 128×128 state.
- 400 trellis matrices, **24,326,963,200 weights**.
- **B3.5** (earlier file): 234 matrices at K3 (4.603 GB) + 166 at K4 (6.027 GB) = **10.629 GB** of trellis, 3.495 bpw [S1] §5.
- **PAW-27B-X3.1** (current best): 395 matrices at **K3.5** + 5 at K3. File 12,346,158,560 bytes [S15].
  From the file size minus embed and head, that is about 10.64 GB of trellis (arithmetic).
- Head is Q5_K (0.874 GB); embedding is Q4_K (0.715 GB) [S14].
- KV cache: 64 KB per token at f16, 34 KB at q8_0 [S18b].
- GDN state (arithmetic): 48 layers × 48 heads × 128 × 128 × 4 B = 151 MB, read and written each token.

**Per-token read at short context, X3.1 (arithmetic):** 10.64 trellis + 0.874 head + ~0.1 other
F32 + 0.30 GDN state ≈ **11.9 GB**. At the 892.7 GB/s read ceiling that is 13.4 ms, i.e.
**74.9 tok/s is a hard upper bound for any engine** on this file and card. No kernel can beat a
kernel that only reads.

## 3. Where autoregressive decode time goes (B3.5, nsys with `--cuda-graph-trace=node`)

30.12 ms/token wall, 27.55 ms GPU busy, **1987 kernel launches per token** [S3]:

| | ms/token | share |
|---|---:|---:|
| `x3v_gemv_kernel` (397 calls) | **22.60** | 82.0% |
| every other kernel (1194 launches) | 4.46 | 16.2% |
| GPU idle | 2.57 | 8.5% |

Trellis matmul reads 10.629 GB in 22.60 ms, i.e. **470 GB/s**, half the card's peak.
nsys drops kernels inside graph replays unless you pass `--cuda-graph-trace=node`. Two earlier
profiles were garbage because of this [S3] §2.

## 4. Why the GEMV is slow: it is structure-bound, not bandwidth-bound

The kernel was ablated in a standalone harness [S1] §2:

| shape | bits | base ms | no decode | no load | **skeleton only** |
|---|---|---:|---:|---:|---:|
| ffn_up 5120×17408 | 4 | 0.0711 | 0.0625 | 0.0639 | **0.0376** |
| ffn_gate 5120×17408 | 3 | 0.0780 | 0.0583 | 0.0686 | **0.0366** |
| ffn_down 17408×5120 | 3 | 0.0709 | 0.0513 | 0.0586 | **0.0335** |
| attn_qkv 5120×10240 | 3 | 0.0422 | 0.0332 | 0.0369 | **0.0206** |

- **Delete every weight byte and the kernel still takes 73–88% of its time.** The skeleton
  (MMA, k-split reduction, 2 grid syncs, Hadamard folds) is 47–53%.
- 4-bit shapes run at 626–670 GB/s. **3-bit shapes run at 428–471 GB/s.** The cost is 3-bit
  decode math (a per-lane variable shift for windows that straddle 32-bit words): 0.0197 ms vs
  4-bit's 0.0086 ms on the same shape [S1] §3.
- Inside the skeleton [S2] E2:
  - The k-split reduction is free (0.0–0.6%).
  - The MMA is 8–24%.
  - The remaining ~25% is spread thin: shared-memory stores, the prefetch ring, address math, the
    epilogue and loop overhead. There is no single target.
- At m=1, the int8 GEMV beats fp16 on every 3-bit shape (+11% to +37%), and fp16 wins every
  4-bit shape. Routing by bit width alone was worth −10.6% GEMV time [S2] E1.
- Launch overhead is ~2.85 ms/token: 397 GEMV launches, plus 397 cast kernels at 1.2 µs each
  (the casts were later folded away) [S1] §5, [S11].
- Closed by measurement: grid size (a cooperative launch already sits at its maximum), load-lane
  idling at 3-bit (~0.002 ms), the k-split reduction, and codec byte count [S1], [S2].

## 5. Speculative decoding: what the round costs

- **Verify cost vs rows** (whole forward, nt = rows): before the tensor-core GEMM port, nt=8
  cost 2.57× one token. After it, **1.42×**; EXL3 costs 1.39× [S6] §4.
  - Trellis matmul alone: EXL3 goes 16.70 → 21.29 ms from nt=1 to nt=8 (1.275×) [S6] §1.
- **Drafter ceiling, fitted:** verify = 29.81 ms + 0.994 ms per token, and select = 2.84 ms +
  0.626 ms per token [S10] §1. At the time, n_max 5 was optimal and the ceiling was ~98 tok/s
  with everything else at zero cost.
- **Round budget at n_max 5** [S10] §2:

  | item | ms/round | share |
  |---|---:|---:|
  | verify | 35.47 | 73% |
  | select (drafter plus the target head over all block rows) | 6.30 | 13% |
  | host work (decode_tgt 2.49 + unaccounted 2.03 + sample 1.08 + misc 1.12) | **6.72** | **14%** |

- **End state: 100.45 tok/s** (8k code context, greedy, 78/120 drafts accepted, output hash
  exact), with `GGML_PAW_X3_GEMV=2 GGML_PAW_MMQ_HEAD=1 GGML_PAW_GREEDY_IDS=1 GGML_PAW_DQ4=0` [S11].
  The KV-only drafter catch-up had already been implemented and measured cheap.
- **Maintenance cost, measured:** on current llama-paw `master`, `common/speculative.cpp` and
  `src/models/dflash.cpp` are **upstream versions**. The PAW DFlash2 customizations were damaged
  in the 2026-09-18 upstream port and wait in `TODO_paw_dflash_spec.patch` [S17] §319. The same
  port silently compiled a 228-line MoE op into 190 bytes of machine code (`GGML_MAX_SRC`)
  [S17] §8.

## 6. Long context

- At 256k, attention is **64%** of the decode forward and reads KV at ~206 GB/s [S18b].
- Batching 2 query heads per block in the flash-attention vec kernel gave **+30.2%** decode at
  242k [S18d].
- Against EXL3 on the same card, we are 21% slower at 1k and 31% slower at 242k [S9].

## 7. Prefill

- fp16-accumulate GEMM (cuBLAS `COMPUTE_16F`): PP8192 915 → **1215 tok/s**, with no quality
  change on MMLU-Pro, HumanEval+ or MBPP+ [S18c].
- Attention in prefill is not tensor-bound: fp16 QKᵀ measured −1.9% [S18c].

## 8. K3.5, the new model's rate: the kernel gap

X3.1 stores 395 of 400 matrices at K3.5. For K3.5, llama-paw has only:

- the int8 `sq` GEMV (up to 8 rows), and
- reconstruct + cuBLAS (9 rows and up).

The x3v fp16 GEMV and the x3g tensor-core GEMM **do not exist for K3.5** [S15], llama-paw
`4b936837b`. Verify windows of 3–8 rows therefore take the int8 kernel, which is the one that
cost 2.57–3.0× at nt=8 before the GEMM port [S5], [S6].

**X3.1 decode and verify speed have never been measured** [S15]. This is the first unknown to
close.

## 9. MoE and two GPUs (Flash-Next 125B, for later)

- On dense tensors at batch 1, X3 trellis is **15–24% slower than Q8_0** (compute-bound), and
  7 shapes cannot run on the X3 kernel at all [S13] §0.1.
- Dense tensors carry 86% of per-token traffic; experts, at 2.6 bpw, carry 14% [S13] §0.
- The decode graph has 2972 nodes, a 3.3 ms/token floor at 1.1 µs each [S13] §3.
- Tensor parallelism across 2×3090 was closed on 30 µs `cudaMemcpy` round trips [S13] §1. **That measured the wrong mechanism** (driver calls). Reopened: PLAN.md 6b′ uses a mapped-pinned-memory doorbell, as Strata does GPU↔CPU.

## 10. Closed levers — do not reopen without new evidence

| lever | result | source |
|---|---|---|
| re-encode / change bit rate for speed | the kernel is not bytes-bound | [S1] §6 |
| grid sizing of x3v | already at the cooperative maximum; `grid_mult ≥ 3` fails | [S1] §4 |
| restructuring `x3g_gemm` for AR | it runs 0% of AR decode | [S3] §5 |
| k-split reduction | free | [S2] |
| 3-bit load-lane idling | ~0.002 ms | [S1] §3 |
| computed codebook vs LUT (older codec) | null or negative | [S7], [S8] |
| register-resident walk (EXL3 shape, older codec) | −1.6% | [S8] |
| megakernel (older codec) | rejected: codec ops were 16% of launches, best case +3% | [S7] |
| naive cooperative `grid.sync` design | ~7 µs per sync, dead on arrival | [S7] |
| trimming graph nodes | 70 nodes removed, zero gain | [S7] |
| deeper DFlash drafts | slower; draft cost grows linearly | [S7], [S10] |
| fp16 QKᵀ in prefill attention | −1.9% | [S18c] |
| MoE group size 16 | −12% | flashnext sprint report |
| TP over PCIe on this box | **REOPENED 2026-09-29**: the 30 µs figure measured `cudaMemcpy` host staging (driver latency), not a mapped-pinned-memory doorbell exchange. Re-test with PLAN P6.0 | [S13] |

**The rule that every win so far obeyed** [S7]: *fuse to remove a global-memory round trip or a
serialized dependency stall, not to reduce node count.*

## 11. The wall in one paragraph

Single-stream decode of a 3.5-bpw trellis 27B on a 3090 is limited by **how many instructions the
trellis matmul issues per weight byte**, not by DRAM. Specifically: the skeleton, the 3-bit and
K3.5 window extraction, and 397 separate launches. Around that sit **~7 ms/token of framework
cost**: other kernels, idle gaps and host work per round.

For speculative decoding, the round is verify plus draft plus host, and host alone is 14%. At
long context, attention takes over. For X3.1, the K3.5 fast kernels do not exist yet.

The ceiling for any engine on X3.1 is **74.9 tok/s** (reads only). Every existing engine sits
between 40 and 55% below it on this card.

## Sources

- [S1] `~/ML_projects/paw27b/20260907_kernel_speed/reports/paw27b_gemv_ceiling_20260907.md`
- [S2] `…/20260907_kernel_speed/reports/paw27b_phase0_results_20260907.md`
- [S3] `…/20260907_kernel_speed/reports/paw27b_forward_localized_20260907.md`
- [S4] `…/20260907_kernel_speed/reports/paw27b_target_forward_20260907.md`
- [S5] `…/20260907_kernel_speed/reports/paw27b_x3_verify_bottleneck_20260906.md`
- [S6] `…/20260907_kernel_speed/reports/paw27b_x3_tensorcore_gemm_20260906.md`
- [S7] `…/20260907_kernel_speed/reports/kernel_closed_levers_20260902.md`
- [S8] `…/20260907_kernel_speed/reports/paw_speed_postmortem_20260902.md`
- [S9] `…/20260907_kernel_speed/reports/paw27b_vs_exl3_longctx_20260907.md`
- [S10] `~/ML_projects/paw27b/20260906_speculative/reports/paw27b_drafter_ceiling_20260907.md`
- [S11] `…/20260906_speculative/reports/paw27b_spec100_v2_residual_closed_20260908.md`
- [S12] `…/20260906_speculative/reports/paw27b_spec100_v2_nsys_profile_20260908.md`
- [S13] `~/ML_projects/flashnext/20260920_pp_tg_kernels/reports/REPORT-FLASHNEXT-PP-TG.md`
- [S14] `~/ML_projects/paw27b/20260826_evals_parity/reports/paw_b35_vs_exl3_3.5bpw_codec_diff_20260919.md`
- [S15] `~/ML_projects/paw27b/20260924_thinking_ladder/reports/PAW-27B-X3.1-README.md`
- [S16] `~/llama-paw/README.md` (bit-rate sweep table, B3.5 speed table)
- [S17] `~/llama-paw/docs/paw/CATCHUP.md`
- [S18] Claude project memory `~/.claude/projects/-home-green-gpu-bonsai-pilot/memory/`:
  - [S18a] `paw-walk-lsu-bound.md`
  - [S18b] `paw27b-longctx-is-attention.md`
  - [S18c] `paw27b-fp16-accumulate-gemm.md` and `paw27b-attention-not-tensor-bound.md`
  - [S18d] `paw27b-attention-gqa-redundancy.md`
