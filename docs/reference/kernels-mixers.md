# Token mixers: kernels/dsa/ (sparse attention, 12 layers) and kernels/gdn/ (gated delta net, 36 layers)

Both mixers take the attention-side hc mix `mixed [T][2560]` and return `out [T][2560]`; the projections around
them are dense GEMMs run by `Forward` (kernels-dense.md). Both carry state across chunks, so a prompt can be fed in
pieces and a decode step is just a chunk of 1.

---

## DSA: the math (llama-paw `build_layer_attn` + `build_qsa_top_k`; TRUSS rule from TRACKER #47)

Flash-Next DSA layer: 24 query heads, 2 KV heads (q head h reads KV head h / 12), head dim 256; the q projection
holds `[q (256) | gate (256)]` per head. Per-head RMSNorm on q and k, NEOX rope on the first 64 dims (base 1e7).
Indexer: 4 heads × 128; raw indexer keys are **mean-pooled per complete block of 4 cells**, then RMS-normed and
roped **at the block start**; queries normed and roped at their position.

```
score(t, b) = Σ_h relu(idx_q[t][h] · idx_k[b])          over complete blocks b the query sees (4b + 3 ≤ t)
query t attends: the 512 best blocks (ties → the later block) + its tail cells 4⌊(t+1)/4⌋ .. t
out = softmax(q·kᵀ / 16) · v  over those cells, then out ⊙ sigmoid(gate)
```

When a query sees ≤ 512 blocks it attends to everything (plain causal attention). llama-paw keeps 2051 cells
(idx_top_k + ratio − 1) with nondeterministic ties and runs dense-with-mask; TRUSS keeps whole blocks, deterministic
— identical when every block fits, and a documented superset/subset otherwise (#47).

## `src/kernels/dsa/dsa_prepare.{cuh,cu}` — inputs to the selection and attention

| function | in → out |
|---|---|
| `rope_table<Shape>(pos0, T, base, cs)` | cs [T][32] float2 = (cos, sin) of pos·base^(−2p/64), **angles in fp64** (an fp32 angle is off by ~0.02 rad at 256K) |
| `prepare_qkv<Shape>(qfull, k, v, q_norm, k_norm, cs, pos0, T, eps, q16, gate, k_cache, v_cache, stream)` | qfull [T][24][512] raw → q16 [T][24][256] fp16 (normed, roped), gate [T][24][256] fp32; k, v [T][2][256] raw → K/V cache rows pos0.. (fp16; k normed, roped) |
| `prepare_qkv<Shape>(..., gate, const KvCache & kv, stream)` | the same into either cache format: `dsa::KvCache` fp16 (`k16`, `v16`) or **int8** (`kq`, `vq` codes + `ks`, `vs` fp16 scale per `KV_GROUP` = 64 values, scale = max\|x\|/127 rounded to fp16 before encoding): Strata's int8 KV, 1,056 B per cell per layer vs 2,048 (TRACKER #87) |
| `prepare_index<Shape>(idx_q, idx_k, q_norm, k_norm, cs, rope_base, pos0, T, eps, partial, idx_q16, idx_k_cache, stream)` | idx_q [T][4][128] → idx_q16 (normed, roped); every block that **completes** in the chunk is pooled from raw keys (cells before pos0 come from `partial`), normed, roped at its start (own fp64 angle) into idx_k_cache [n_ctx/4][128]; `partial` [3][128] fp32 carries the open block's raw keys to the next chunk |

One warp per normalized row; element i of a row lives in lane i % 32, so the rotary pair (p, p+32) is in one lane
(`static_assert(ROPE_DIMS == 64)`).

`carry_partial<Shape>(idx_k raw [T][ID], pos0, T, partial)`: the indexer's open-block rows for cells in the chunk
(what `prepare_index` does at a chunk's end), used by `blocks::Dsa::accept` over a restored snapshot.

## `src/kernels/dsa/dsa_prefill.cuh`, `dsa_select.cu` (scores + top blocks), `dsa_prefill.cu` (attention) — selection and attention

`dsa::FlashNext { H = 24, HKV = 2, D = 256, IH = 4, ID = 128, RATIO = 4, TOP_BLOCKS = 512, ROPE_DIMS = 64 }`.

| function | contract |
|---|---|
| `select_workspace_bytes<Shape>(max_queries, n_ctx)` | score rows kept at once (≤ 1024) × n_ctx/4 floats |
| `select<Shape>(idx_q16, idx_k_cache, pos0, T, blocks [T][512], n_blocks [T], ws, ws_bytes, stream)` | per query the chosen block ids, **ascending**, first `n_blocks[t]` used |
| `attention<Shape>(q16, gate, k_cache, v_cache, blocks, n_blocks, pos0, T, out16 [T][24·256], ws, ws_bytes, stream)` | gated attention output in fp16 (the out projection's input); `ws` = `attention_workspace_bytes<Shape>(T)` (0 above `SPLIT_ROWS` = 32) |
| `attention<Shape>(q16, gate, const KvCache & kv, ...)` | the same over either cache; int8: the gather dequantizes 8 codes × scale into the fp16 tile (plain loads, not `cp.async`), the math after it unchanged. Measured: rel 6.9-8.5e-3 vs the fp32 reference on N(0,1) data (fp16 2.2-2.9e-4); +30-65% time at 8K-row chunks, 0.08 → 0.11 ms/layer at T 4 |

**select.** `score_kernel`: CTA = 64 queries × 64 blocks, 8 warps of 16 queries × 32 blocks, all 4 indexer heads'
accumulators in registers so relu and the head sum happen before the store; fragments straight from global (the op is
~2% of prefill FLOPs); skipped entirely for chunks where every query sees ≤ 512 blocks. `select_kernel`: one CTA
per query, 4 radix passes of 8 bits over the score bits (scores ≥ 0, so fp32 bits order like values) find the
512th key, then one pass from the last block down keeps keys above it plus the latest equal ones and writes ids from
the end so they come out ascending (two `cub::BlockScan`s per 256-block step).

**attention.** One CTA (2 warps) per (query, KV head): the 12 query heads of a KV head fill one m16 tile (rows
12..15 zero). Cells are gathered 32 per tile (whole 4-cell blocks from the list, then the tail) into shared memory
with cp.async (K and V rows are 512 B contiguous); each warp owns 16 cells of a tile and its own online-softmax
state (fp32, base 2), S = q·kᵀ and P·V on `mma_f32` with `ldmatrix` (V transposed), P rounded to fp16, the rowsum
taken over the rounded P. The two warps' states merge through the K/V buffers; the gate is applied in the epilogue.
Smem row stride 264 halfs (528 B) keeps ldmatrix conflict-free.

**Split form (T ≤ 32: decode steps, verify windows).** One query has only 2 CTAs otherwise (227 µs per decode
layer at 3.4K context, TRACKER #57). A third grid dimension cuts each query's cells into up to 32 contiguous ranges of
whole 32-cell tiles; each CTA writes its unnormalized state (o [12][256], m, l) to `ws`, and `combine_kernel` merges
the ranges in order and applies the gate. Differs from the unsplit form by fp32 reassociation only (same test gates).

**Numerics vs llama-paw.** llama-paw's flash attention accumulates P·V in fp16 (`fattn-mma-f16.cuh`,
`T_C_VKQ = half2`), ~1e-3..2e-3 from exact at ~2K cells; TRUSS accumulates in fp32 and is 2.5–2.8e-4 from the fp32
reference on real data (TRACKER #48, #49).

**Speed (v1).** select + attention per DSA layer: 2.8 µs/token at a 4K prompt, 3.2 at 8K, 5.0 for an 8K chunk at
24K–32K (TRACKER #49). Known headroom: 2 warps/CTA and single-buffered tiles; K/V gathered once per query (neighbouring
queries could share tiles).

**Tested by.** `dsa_prefill_test` (also the int8 cache, quantized on the host in the writer's format, against the same reference: gate rel <= 2e-2; and its attention time in the timed cases). (random fp16-exact inputs vs `ref::qsa_select` / `ref::masked_attention`: selection
equal up to fp64 near-ties ≤ 1e-5; gated attention rel ≤ 5e-4, worst query ≤ 2e-3; includes a later chunk
pos0 > 0), `qwen4exp_dsa_long` D (real 3,659-token data), `qwen4exp_forward decode` (blocks spanning chunks).

---

## GDN: the math (llama-paw `build_layer_attn_linear`)

qkv projection [T][10240] = q [16][128] | k [16][128] | v [48][128]; z [T][6144]; alpha, beta [T][48].
Causal depthwise conv (kernel 4) over the qkv channels + SiLU; q, k l2-normalized per key head (llama:
rms_norm(eps/n)/√n); gate g = softplus(alpha + dt_bias)·a (softplus(x) = x for x > 20), β = sigmoid(beta).
Value head h reads key head h % 16. Per head, state S [128 key][128 value]:

```
S ← e^g · S ;   S ← S + k (β (v − Sᵀk))ᵀ ;   o = Sᵀ q / √128
out = RMSNorm_head(o) ⊙ γ ⊙ sigmoid(z)   → out projection
```

## `src/kernels/gdn/gdn_prepare.{cuh,cu}`

`gdn::Dims { Hk = 16, Hv = 48, S = 128, K = 4 }`.

| function | contract |
|---|---|
| `prepare(dims, qkv, conv_w [C][K], conv_state [K−1][C], alpha, beta_raw, dt_bias, a, T, eps, q, k, v, g, beta, stream)` | conv + silu with the previous chunk's last K−1 raw rows from `conv_state` (zeros at sequence start), then `conv_state` ← this chunk's last K−1 rows; q, k l2-normed; g, β |
| `output_norm(dims, core, z, gamma, T, eps, out16, stream)` | RMSNorm per value head · γ · sigmoid(z) → fp16 |

## `src/kernels/gdn/gdn_prefill.{cuh,cu}` — the recurrence

`gdn::delta_rule(q, k, v, g, beta, state, out, T, Hk, Hv, stream)`: exact token-sequential recurrence (fp32), the
contract of `ref::gated_delta_rule`; `state` [Hv][128][128] in/out (nullptr: zero start, discarded). Overload
`delta_rule(..., state_in, state_out, ...)` reads one buffer and writes another (a verify window keeps the committed
state; `blocks::Gdn::accept`). Block = (value
head, 32 state columns) = 4 warps × 8 columns; a column is held by 4 lanes × 32 rows, so reductions are 2 shuffle
levels. Per token: s·k and s·q reduce together, k·q is computed once per token at staging;
o = a (s·q) + (k·q) δ. k/q/v/g/β are staged through shared memory in cp.async chunks of 8 tokens.

**Speed.** 0.76 µs/token/layer at 8K tokens (from 2.5 for the first version; TRACKER #46); FLOPs are negligible, the
cost is per-step latency. Headroom: a chunked (WY) formulation on tensor cores.

**Tested by.** `gdn_prefill_test` (vs the reference: out rel 2.3e-7, state 6e-8), the parity test (GDN blocks vs
llama-paw), `qwen4exp_forward` (conv/recurrent state across chunks).

## DeepSeek-V4.1 attention kernels: `src/kernels/ds41/` (math: dsref/model.py Attention / Compressor / Indexer)

| file | ops | notes |
|---|---|---|
| `attn_ops.{cuh,cu}` | `rope_inv_freq` (host, fp32 like torch, YaRN when orig_ctx > 0), `rope_tail` (last rd dims of each head, adjacent pairs, position `pos_base + row * pos_mul`, inverse = conjugate; fp32 angle = float(pos) * inv_freq, `sincosf`), `ring_write` / `ring_write_f32` (last R rows of a chunk at p % R), `groups_done` + `pool` (ratio-r softmax-gated pooling per dim; earlier positions of a group from the raw rings), `pick_groups` (block-diagonal wo_a from a dense T x G projection; decode rows only since #194, prompt rows use `dense::q8_gemm_a16_grouped`), `to_half` | elementwise, deterministic |
| `indexer.{cuh,cu}` | `idx_scores` (32 heads x 128; one thread per compressed position, query row in shared memory; -inf past the row's visible count (pos + 1) / ratio; weight scale folded), `select_topk` (one CTA per row: 4-pass radix select on the order-preserving uint32 image, entries > -inf only, ties to the lower index, output in ascending index order, -1 padded), `block_max` (+inf pin on the newest block), `mask_blocks` (shared-memory bitmap) | v1 CUDA-core; prefill at long context wants tensor-core scores |
| `engram.{cuh,cu}` | `engram_dequant_q4` (one warp per 144-byte row: 8 fp16 scales + 128 nibble bytes -> 256 fp16), `engram_gate` (CTA per (row, copy): three sums, signed-sqrt sigmoid gate, h += gate x value in place), `mul` | `ds41_engram_parity` |
| `sparse_attn.{cuh,cu}` | `sparse_attn` (v1): CTA (row, head); window keys from the chunk or the ring, compressed keys by index; softmax max over the keys, sink only in the denominator (as the reference); P.V in key order | reference kernel; `TRUSS_DS41_ATTN_V1=1` selects it in `AttnDs41` |
| `sparse_attn_tc.cu` | `sparse_attn_tc` (v2): CTA (row, 16 heads, key split), 128 threads; q as fp16 in shared memory, keys in tiles of 32 (`uint4` loads, zero + -inf for missing entries), S = QK^T and O += PK with `mma.m16n8k16` (fp16 in, fp32 accumulate; ldmatrix, `.trans` for the PK operand), online softmax per head (one thread per head, keys in order); one split: final output with the sink; several (rows x 4 groups < 160 CTAs, e.g. decode): partial (max, sum, O) to a workspace `sparse_attn_tc_ws_floats` floats, then `sparse_attn_combine` (splits in order, sink, deterministic). `sparse_attn_tc_ok`: D = 512, H % 16 == 0. Dynamic shared memory ~53 KB (opt-in set once per device) | #189: `ds41_attn_parity` rel errors within 1% of v1's on every layer and step; L0-5 slice: decode attention kernel 0.164 -> 0.020 ms/layer, prefill 2,048 rows 50 -> 12.7 ms/layer |
| `sparse_attn_tc_async.cu` | `sparse_attn_tc_async` (v3): v2 with the key tile double buffered through `cp.async` - tile k+1 loads into the second shared buffer while tile k computes (a zero store where the key is missing, `valid` per buffer), so the global key reads no longer stall the tensor-core work across the two `__syncthreads`; same CTA shape (16 heads, KT = 32), same math, rounding and split/combine as v2. Dynamic shared memory ~85 KB. `TRUSS_DS41_ATTN_TC=2` selects it in `AttnDs41`, else v2 | #197: `ds41_attn_parity` numbers below; L0-5 slice numbers below |
| `sparse_attn_tc_h32.cu` | `sparse_attn_tc_h32` (v4): v2 with 32 heads per CTA and 8 warps (256 threads), so a row's keys are read twice instead of four times; S (QK^T, 32 x 32) gives warp (head half, key octet), O (32 x 512) gives warp (head half, 128 dims); the softmax uses all 8 warps (8 threads per head, 4 keys each, shuffle reduce). Same tiles (KT = 32), mma and split/combine as v2. Dynamic shared memory ~72 KB. `TRUSS_DS41_ATTN_TC=3` selects it in `AttnDs41` | #197: `ds41_attn_parity` rel within 1% of v2's on every layer and step; L0-5 slice prefill 56.8 -> 51.0 ms/layer |
| `sparse_attn_tc_16w.cu` | `sparse_attn_tc_16w` (v5, the default): v4 with 16 warps (512 threads); S is split over the two halves of the 512-dim reduction (warp = head half, key octet, k half), each writing one of two partial buffers that the softmax adds, and O gives each warp 16 heads x 64 dims; the extra warps and the halved registers (121 -> 64) hide the ldmatrix / mma latency under one CTA per SM. Same tiles, softmax (16 threads per head, 2 keys each), mma and split/combine as v2/v4. Dynamic shared memory ~78 KB. `TRUSS_DS41_ATTN_TC=4` selects it (else v5); 1 = v2, 2 = v3, 3 = v4 | #197: `ds41_attn_parity` rel within 1% of v2's on every layer and step; L0-5 slice prefill 51.0 -> 49.4 ms/layer |

**Tested.** `ds41_attn_parity` (block level, real weights, #177). Index keys are fp32 (selection is rounding-sensitive).
