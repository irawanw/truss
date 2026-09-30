# Dense projections: kernels/dense/q8_gemm.{cuh,cu}

All non-expert matrices of Flash-Next are Q8_0 (llama.cpp: blocks of 32 int8 with one fp16 scale), ~3.3 GiB: the
hyper-connection projections (10240→320, 320→10240, 10240→4), GDN qkv/z/alpha/beta/out, DSA q/k/v/idx/out, the
shared expert, PLE key/value, token_embd and the output head. The router (2560→512) and the shared-expert gate
(2560→1) are F32 and run as cuBLAS SGEMM in `Forward::lin32`, not here.

## Storage: `dense::Q8Matrix`

```
struct Q8Matrix { const int8_t * q;   // [out][in]      row o = output o, contiguous inputs
                  const half * d;     // [out][in / 32] one fp16 scale per 32-block
                  int in, out; };
```

GGUF stores Q8_0 as interleaved 34-byte blocks (fp16 d + 32 int8), which cannot be read with 16-byte vector loads.
`q8_repack` splits them once at load time into the aligned `q`/`d` arrays above. A contiguous run of rows is itself a
`Q8Matrix` (`Forward::head` tiles the 248320-row output matrix this way).

## Functions

| function | contract | numerics | speed (GPU 2) |
|---|---|---|---|
| `q8_repack(blocks, in, out, q, d, stream)` | GGUF Q8_0 blocks (device) → `Q8Matrix` storage; `in % 64 == 0` | exact | load time only |
| `q8_quantize_act(x, rows, in, xq, xd, stream)` (fp32 or **fp16** x overloads) | x [rows][in] → xq int8 [rows][in], xd fp16 [rows][in/32] | llama's Q8_1: d = amax/127, q = round(x/d) | 0.02–0.6 ms per matmul at 8K rows |
| `q8_gemm(W, xq, xd, rows, y, stream)` | y fp32 [rows][out] = W·x; any rows/out, `in % 64 == 0` | **exactly llama-paw's** (int32 block dot, fp32 fold `acc += c·d_w·d_x`), = `ref::linear(LLAMA)` to 3e-7 | 65–94 TOPS at 8K rows (TRACKER #43) |
| `q8_gemv(W, xq, xd, rows ≤ GEMV_ROWS = 8, y, stream)` | same as q8_gemm for decode rows | same, only fp32 summation order differs (= q8_gemm to 1.8e-7) | 740–790 GB/s on the big shapes (~85% of peak) (TRACKER #54) |
| `q8_gemm_a16(W, x_half, rows, y, w16, cublas, stream)` | fp16 activations: dequantize W to `w16` (scratch, in·out halfs) then `cublasGemmEx` fp16 in / fp32 accumulate | one fp16 rounding of each weight (≤ 2^-12 rel); no activation quantization | 47–71 TFLOPS (TRACKER #44) |
| `q8_rows(W, ids, n, out, stream)` | out fp32 [n][in] = rows `ids` of W (token embedding) | exact | — |

## Which one runs (`Forward::lin`)

- `Activations::Q8_1` (default): `q8_quantize_act` (from the fp16 GEMM input), then `q8_gemv` if rows ≤ 8, else
  `q8_gemm`. Full-model KL to llama-paw's logits 0.0131 = llama's own run-to-run floor (TRACKER #53).
- `Activations::FP16`: `q8_gemm_a16`. Closer to exact math per matmul (~0.5% less error, TRACKER #33) but +3%
  perplexity on chat-format text vs Q8_1 (TRACKER #53), and slow for decode (it dequantizes the whole matrix per
  call). Used by the tests that compare against the fp32 reference.

## Kernel design (`q8_gemm`)

Block tile 128 outputs × 128 token rows, k chunks of 64 (two Q8 blocks) double-buffered with cp.async; 8 warps as
2 (outputs) × 4 (tokens), warp tile 64 × 32 = 4 × 4 `mma.m16n8k32.s8` (weights as A). One mma k-step is exactly one
Q8 block, so each int32 result is scaled once per block: `acc += float(c) · d_w · d_x`. Shared row stride `SK = 80`
bytes makes the fragment loads conflict-free. The per-block fold (~12 ALU ops per mma) is what limits it; tried and
rejected: magic-number int→float, 2 blocks/SM (spills), 64×128 tiles (all slower, TRACKER #43).

`q8_gemv`: one warp per output row; lane l takes 32-blocks l, l+32, ...; two 16-byte loads of weights and 8 `dp4a`
per activation row per block; warp shuffle reduction. `ROWS` is a template parameter (1..8).

## Invariants

- `in % 64 == 0` for repack/gemm/a16 (all Flash-Next shapes satisfy it; 320 = 5·64), `in % 32 == 0` for gemv.
- Outputs are written, never accumulated: callers need no zeroing.
- **Row invariance:** `q8_gemm` and `q8_gemv` give each row a result independent of the other rows. cuBLAS
  (`q8_gemm_a16`) does not: it picks algorithms by row count (2.5e-6 rel between row counts, `q8_gemm_test` prefix
  line, TRACKER #51).

## Tested by

`tests/unit/q8_gemm_test` on the 8 Flash-Next dense shapes, random weights: q8_gemm vs `ref::linear(LLAMA)` (gate
1e-5), a16 vs `ref::linear(FP32)` on fp16-exact inputs (gate 3e-4), gemv vs q8_gemm at 1/4/8 rows (gate 1e-6),
a16 prefix variance printed; timing for all paths.

## Change it

A faster dense path (e.g. a Q8 GEMM with fp16 activations on int8 tensor cores, or FP8): keep `Q8Matrix` and the
function contract, add the new function here, route to it in `Forward::lin`, and gate it in `q8_gemm_test` against
the matching `ref::linear` numerics.
