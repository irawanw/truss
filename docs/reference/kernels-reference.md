# Reference ops: kernels/reference/ (ref.cuh, ref_linear.cu, ref_norm.cu, ref_ssm.cu, ref_attn.cu, ref_trellis.cu)

**What.** Plain CUDA fp32 (fp64 where accumulation order matters) with one obvious thread mapping each and fixed
reduction order. They are the *definition* of every op: fast kernels are tested against them, and the model reference
forward (`qwen4exp::reference`) is built from them. Speed is not a goal (the attention reference is O(T²·D) with fp64
exps). Activations fp32 row-major `[rows][n]`; weights are `DTensor`s in ggml layout (`ne[0]` = input dim).

## Numerics switch: `ref::Numerics { FP32, LLAMA }`

`FP32` is exact math on the stored weights. `LLAMA` reproduces where llama-paw's CUDA ops round, so parity tests
can show the math matches beyond llama's own rounding: Q8_0 matmuls quantize each 32-block of activations to int8
(Q8_1) and do integer block dots; trellis matmuls round activations to fp16; `masked_attention` emulates the
flash-attention kernel's fp16 P·V accumulation (64-cell tiles, max offset 3·ln2, fp16 accumulator per 16 cells).

## Functions (ref.cuh)

| function | definition |
|---|---|
| `linear(W, x, y, rows, s, num)` | y[r][o] = Σ_i W[o][i]·x[r][i]; W F32 / F16 / Q8_0; one warp per (row, output), xor-tree reduction |
| `rms_norm(x, gamma, gamma_rows, y, rows, n, eps, s)` | y = x / √(mean x² + eps) · γ[r % gamma_rows] (gamma_rows = hc for per-stream norms) |
| `hadamard128(x, n, s)` | Sylvester-order WHT of each 128 block, 1/√128 |
| `trellis_dequant(words, K, in, out, W, s)` | **the bitstream definition of mul1**: tile (kt, nt) at word (kt·out/16 + nt)·8K; weight j's state = the 16 bits ending at circular stream bit (j+1)·K, MSB-first; value = fp16 fma(bytesum(state·0x83DCD12D) + 0x6400, 1/147.7, −10.39); j = 8·lane + i at (n, k) per the fragment order. No Hadamard, no suh/svh |
| `causal_conv_silu(x, w [C][kc], state, y, T, C, kc, s)` | y[t][c] = silu(Σ_j w[c][j]·x[t−(kc−1)+j][c]), earlier rows from `state` |
| `l2_norm(x, y, rows, n, eps, s)` | x / √(Σx² + eps) as llama (rms_norm(eps/n)/√n) |
| `gated_delta_rule(q, k, v, g, beta, state, out, T, Hk, Hv, S, s)` | the GDN recurrence (kernels-mixers.md), one thread per state column |
| `rope_neox(x, rows, heads, hd, n_rot, pos, base, s)` | NEOX pairs (i, i + n_rot/2), angle in fp64 |
| `qsa_select(idx_q, idx_k, T, heads, d, r, top_blocks, sel [T][T], s)` | the DSA block selection rule (rank count, ties → later block, + tail); one sequence from position 0 |
| `masked_attention(q, k, v, sel, out, T, Hq, Hkv, d, scale, s, num)` | softmax attention over the cells with sel = 1; FP32 = fp32 scores + fp64 accumulation; LLAMA = llama-paw's flash-attention numerics |

## Change it

A new op gets its reference here first (definition + a comment with the source it follows, e.g. the llama-paw
function), then a parity check in `tests/layer/` if llama-paw has the op, then the fast kernel and its unit test
against this reference.
