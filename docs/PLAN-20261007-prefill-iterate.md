# Plan 2026-10-07: prefill iteration on GPU 1 (executor: PAW, lead: Claude)

Goal: 150K prefill 2,135 -> ~2,400 tok/s at 250 W with weights untouched, then hand back to decode.
Every item: microbench first (seconds), then ONE short end-to-end check, then a TRACKER row. No 25-min kbench rows
except the P0 baseline. Stop rules are hard: when one fires, report with numbers (tell_lead STUCK), do not keep trying.

## State at hand-off (all on main, deployed to the brain)

| commit | what | effect |
|---|---|---|
| 2aacd52 | prompt buffers overran the stream area by ~T x 83 KB with MTP (hot experts overwritten every prompt chunk) | correctness (TRACKER 140) |
| 6212f99 | hot-expert upload race at load (random resident experts corrupt every start) | engine now DETERMINISTIC: KL self 0.00000 / top-1 100% (TRACKER 141) |
| 0a0c8b1 | dense Q8 GEMM grouped raster (bit-exact) | prefill +2.1% (142) |
| 34afe51 | DSA multi-query prefill attention (TRUSS_DSA_MQ=0 = old) | prefill +4.0%, KL 0.00000 (143) |

- Golden `data/ledger/ref_tokens.i32` re-dumped from 34afe51 (md5 ab93028b; two runs identical). Old one carried the bugs.
- Lesson: every KL / EXACT result between 10-01 and 10-07 compared corrupted against corrupted. TRACKER #121's
  "placement KL 0.08" was most likely bug 6212f99, not placement.
- Fresh 250 W profile of one 150K prefill (s): dense gemm 18.9, MoE gate_up+down 18.7, attention 9.5, GDN delta_rule
  5.6, glue ~8 (near bandwidth floor: hc stream fp32 8K x 4 x 2560 = 335 MB per pass), select+score 2.5, gaps ~3.5.
  PCIe 23.8 GB/s, busy 43%: not binding. Dense GEMM is at its bit-exact floor (Q8 per-32 fold, issue-bound); MoE is
  within 13% of its no-decode ceiling.

## Tools (scripts/, all take the GPU lock and restart the brain on exit)

- `gpu_quick.sh <cmd...>`: any microbench / test on GPU 1 (~1 min of brain downtime).
- `pp_ab.sh name=<worktree>[:ENV=V] ...`: one 150K prefill per arm, in the order given (~2.5 min per arm).
  Standard check = 2 arms (base, new): ~5 min. Use 4 arms (A B B A) only when the 2-arm gap is < 2%.
- `kl_gate.sh <outdir> <wt A>[:ENV] <wt B>[:ENV]`: prompt-path KL, 8 x 2048 tokens (~4 min). Gate KL <= 0.002 AND
  top-1 >= 99%. Bit-exact changes skip it (say so in the row).
- Kernel tests: `gdn_prefill_test 8192`, `moe_prefill_test 8192` (TRUSS_REPS=10), `dsa_prefill_test`,
  `dsa_mq_bench /tmp/dsa_sel.0 out.f16` (real 150K selections; re-dump with TRUSS_DSA_DUMP if /tmp was cleaned).
- NEVER pkill -f (it kills your own shell): kill by PID from `ps -eo pid,cmd`.

## P0 sync + baseline (30 min)

1. Worktree: `git -C work/agent fetch` is not needed (same repo): `git -C work/agent merge --ff-only main` or reset
   agent/x31-speed to main (your 1.A commit is in main already). Build.
2. One full kbench row on main (the only long run): decode, prefill, EXACT vs the new golden. Row "P0 baseline 250W".
3. Served decode after the fixes: `served_stats.py` over requests since 2026-10-07 16:10 (brain restart with 6212f99)
   vs the 10-06 table. Report p10/p50/p90 decode and fresh-prefill p50. This decides whether decode gained from the fixes.
TRACKER row, tell_lead DONE with the 3 numbers.

## P1 GDN chunked delta rule (biggest item, ~1 day)

Today: sequential kernel, 6.29 ms/layer at T=8192 (0.77 us/token, one wave of 192 blocks x 4 warps). Cheap variants
all failed (TRACKER 144): the per-token chain is the floor. Rewrite in chunk form (Yang et al. 2024, gated delta rule).

Math (per value head; state S [dk=128][dv=128] as the kernel's s[row=key][col=value]; token t: a_t = exp(g_t),
delta_t = beta_t (v_t - a_t S_{t-1}^T k_t), S_t = a_t S_{t-1} + k_t delta_t^T, o_t = scale * S_t^T q_t).
Chunk of C = 64 tokens, S0 = state entering the chunk, G = inclusive cumsum of g inside the chunk, Gamma_i = exp(G_i):
- A[i][j] = beta_i * exp(G_i - G_j) * (k_i . k_j) for j < i, else 0 (C x C, strictly lower)
- Tm = (I + A)^-1 (lower triangular; forward substitution, fp32)
- U = Tm (beta . V)            [C x dv]       W = Tm (beta . Gamma . K)   [C x dk]
- Delta = U - W S0             [C x dv]       (the chunk's deltas)
- O = scale * ( diag(Gamma) Q S0 + (Mask . exp(G_i - G_j) . (Q K^T)) Delta ),  Mask: j <= i (diagonal included)
- S_C = Gamma_C S0 + (diag(exp(G_C - G)) K)^T Delta
All ratios through exp(G_i - G_j) with j <= i (g <= 0: values <= 1, no overflow).

Kernels:
- K1, fully parallel over (head, chunk): A, Tm, W, U, and P = Mask . exp(dG) . (Q K^T) to a workspace (fp32);
  process T in slices of 2048 tokens so the workspace stays ~0.25 GB (48 heads x 32 chunks x (W+U+P) floats).
- K2, sequential over chunks per (head, 32-column block) like today: Delta = U - W S0; O; S update. ~0.85 M MAC per
  chunk per block instead of 64 sequential token steps.
- fp32 everywhere first (correctness), then consider tf32/fp16 mma for K2 only if K2 dominates.
Tests: gdn_prefill_test (it compares with ref::gated_delta_rule; keep its tolerances: out and state rel <= 1e-4 at
T=8192 and the short/odd T cases it runs, incl. T not a multiple of 64 and state_in != 0).
Target: <= 3.0 ms/layer at T=8192 (from 6.29). STOP RULE: implemented and correct but > 4.5 ms -> report, keep old.
Then: kl_gate (changes fp32 order) + pp_ab 2 arms. Expected end-to-end +4-6%.

## P2 attention MQ = 8 (~half a day)

Union of 8 consecutive queries = 2.8x one query's blocks (vs 1.95x for 4): gather per query 0.35x (vs 0.49x).
Smem is the limit (q tile 96 x 264 halfs = 50 KB): keep q fragments in registers (each warp owns one 16-row tile:
16 x 256 halfs = 64 regs/lane) or split D into two passes. 12 warps (6 row tiles x 2 cell halves) or 6 warps with
both halves per warp. Target dsa_mq_bench <= 27 ms/layer (from 32.0). STOP if > 30. kl_gate + pp_ab.

## P3 prefill launch gaps (~half a day)

Profile shows ~3.5 s of 70 s idle between kernels. `nsys profile -t cuda` one 150K prefill (prof.sh pattern:
tk-bench-spec, /usr/local/cuda-12.6/bin/nsys), list gaps > 20 us on the compute stream with the host API call
around them (cudaStreamSynchronize / cudaMemcpy D2H / event syncs in the prompt path: E7 split, PLE collect, ids D2H).
Remove the syncs that are not needed for correctness. Target: gaps < 1.5 s per 150K prefill. Bit-exact expected.

## P4 MoE: decoded weight tiles shared across 2 row-groups (~half a day, lower value)

No-decode ceiling (TRACKER/lead 10-07): moe_prefill_test 21.2 -> 18.4 ms/layer. Decoding each 16-deep tile once per
128 rows (two warps share through smem) instead of per 64 rows halves the decode share: expected ~-6% of MoE (~+1.5%
prefill). Must stay bit-exact (same fragments, same mma order). STOP if < 3% on moe_prefill_test.

## P5 decode: keep the expert ring across agent requests (lead measured +4-5% served decode est.)

Every prompt over 96 tokens (83% of served requests) runs as a stream chunk whose area [0, 2 x slot + big_bytes)
is the WHOLE 6 GB ring: admission's work is dropped every request (replay: wipe every 150 passes = +12% CPU slots).
Fix: size the stream area by the chunk: begin_stream(compute, 2 x slot + big_bytes(T)) and carve the big buffers for
T rows (scratch_bytes(c, T, n_ctx, mtp != nullptr) - the MTP term, lesson of 2aacd52), only for T <= split_rows.
Check with the lead_check pattern (checksum of the 800 MB after the area across a chunk must not change).
Measure with the e7 served-pattern replay (run_e7_03.sh), 2 runs per arm.

## Reporting

One TRACKER row per item (numbers: microbench before/after, end-to-end, KL or "bit-exact"), notify line, tell_lead DONE
after each P-item. tell_lead STUCK when a stop rule fires. Power stays 250 W; GPU 1 only; no root changes.
