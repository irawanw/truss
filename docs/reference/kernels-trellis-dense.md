# Trellis-coded dense matrices: src/kernels/trellis/dense/

A dense projection whose weights are PAW X3.1 trellis tiles (the mul1 codec) instead of a quantised block format.
The weights stay in the trellis form in VRAM: the decode path reads the tiles in place (3.5 bits per weight at K3.5
instead of 8 for Q8_0), the prompt path reconstructs the matrix once per chunk.

The format is model-agnostic: a model family binds its weights by name and the file decides the format
(`docs/reference/formats-core.md`, the binding rule below). Blocks call `lin(T, ...)` as before.

## The format on disk

A dense matrix named `<base>` may be stored as four GGUF tensors (the X3.1 convention, the same names the routed
experts use):

| tensor | type | shape | what |
|---|---|---|---|
| `<base>.m3_trellis` | I16 | [words per tile][tiles] | the trellis words: tile (k tile, row tile) at word `(k_tile * rows/16 + row_tile) * words` (k-tile major, row-tile minor) |
| `<base>.m3_suh` | F16 | [in] | the input-side scales |
| `<base>.m3_svh` | F16 | [out] | the output-side scales |
| `<base>.m3_meta` | I32 | [2][1] | optional: (rate code, 0). Without it the rate comes from the tile width |

`words` = `k_tile_u16(k)` (`src/formats/trellis_k.h`): 16 (K1), 24 (K1.5), 32 (K2), 40 (K2.5), 48 (K3), 56 (K3.5),
64 (K4). The kernels support the integer rates 1..4 and the half-integer 1.5 / 2.5 / 3.5 (the codec headers).

**Binding rule** (`src/formats/binder.h`): `Binder::need(name, dims)` returns the plain tensor `<name>` if the file has
it, else the trellis group `<name>.m3_trellis` (the handle is the group's words tensor, the parts are bound so that
every tensor of the file is still used exactly once). A model family therefore names its weights the same way whatever
format the file holds - no per-format code in `src/model/*/weights.cc`.

**Transform.** The stored matrix is in the transformed basis: `W = diag(svh) . H128 . W_hat . H128 . diag(suh)` per
128x128 block (`H128` = the 128-point Hadamard, natural order, scaled by 1/sqrt(128)). The runtime path transforms the
*activations* instead (the same trick the routed experts use): `A = H128(x * suh)` on the input side, `y = svh *
H128(sum)` on the output side.

## The paths

| rows | path | what |
|---|---|---|
| 1..16 | the window GEMV (`gemv_window.cu`) | the weights are read in place; three launches per matvec (see below) |
| > 16 | reconstruct + GEMM (`reconstruct.cu`) | the matrix is dequantised to fp16 [out][in] (a row chunk at a time) and multiplied by `cublasGemmEx` (fp16 in, fp32 accumulate) |

The 16-column limit is the window GEMV's batch width (`TD_MAX_N`): a drafter's verify window is at most 16 columns.

### The window GEMV (`src/kernels/trellis/dense/gemv_window.cu`)

One matvec = three launches over all the row parts of the matrix (e.g. ffn_gate + ffn_up):

```
had_in  A_p[t]           = H128(x[t] * suh_p)        one warp per (part, t, 128 block)
gemv    P_p[kp][t][cols] = A_p[t][k range kp] . W_p  persistent grid over the items
out     y[t][c]          = svh_p * H128(sum_kp P_p[kp][t])   one warp per (part, t, 128 block)
```

An item is (part, 64-column group, k part); the geometry is 8 warps per item (WK), 4 tiles per warp (WNT), 4 k slices
in flight (PF), fp16 accumulators folded to fp32 every 2 slices (FOLD). The decoded tile is the A operand of one
`mma.m16n8k16` per 16x16 tile (`mma.cuh`'s `mma_w`), the batch is the B operand: one mma per tile, every batch slot
used. `n > 8` adds a second mma per decoded tile (batch half NB = 2), so a 12-column verify decodes the weights once.

The k split of a part depends on its shape only, never on `n` (the drafter's verify batch must equal plain decode:
column t is bit-identical for every n - the test checks it).

The codec is TRUSS's own (`src/kernels/trellis/codec_mul1.cuh`, the same mul1 windows and codebook as the routed
experts): the states it decodes are bit-identical to tqLLM's `Mul1Codec` (the same window ends, the same codebook),
so this kernel and tqLLM's `gemv_window.cu` compute the same product.

### The reconstruct (`src/kernels/trellis/dense/reconstruct.cu`)

`reconstruct_had_kernel`: dequant + both-side Hadamard, written out-major (`[out][in]`, the engine's GEMM layout), one
128x128 tile per block. Ported from tqLLM (which ported it from llama-paw's `paw-x3.cu`); the Hadamard of the weights
is done in half2 arithmetic here (the runtime path transforms the activations in fp32 instead), so the two paths agree
to ~1e-3, not bit-exactly. Against tqLLM's own `reconstruct_had_rows` the result is **bit-exact** (measured: 0 of
89,128,960 values differ on `blk.0.ffn_gate`, K3.5).

**Row chunks.** The prompt path reconstructs a row chunk at a time into a buffer that holds *only that chunk's rows*:

| what | how |
|---|---|
| the tiles of the chunk | the matrix's row tiles: the chunk's first block starts at row tile `row0 / 16`, the 8 sub-tiles of a block are consecutive (the row tiles are minor in the file) |
| the output scales | `svh[row0 ...]`: the chunk's rows are `row0` rows into the matrix, so the scales are offset by `row0` |
| the write | the chunk's local row (`block * 128 + row`), not the matrix's row |

`row0` must be a multiple of 16 (the tile granularity); the engine uses 1024-row chunks.

**Why fp16 and not int8 for the prompt path.** The engine's int8 GEMM (`dense::q8_gemm`) needs the weights in the
Q8_0 layout (int8 with one fp16 scale per 32 weights); the trellis reconstruction produces a continuous fp16 matrix,
and the per-row scale tqLLM uses does not fit that layout. Reconstructing straight into Q8_0 blocks would need a
per-32-block reduction over the reconstructed rows (tqLLM's kernel keeps the whole 128x128 tile in shared memory, so
it could do it, at the cost of a second pass over the tile). The fp16 path was chosen: it is exact with respect to
the reconstructed weights, needs no extra kernel, and the prompt path is not the decode wall (the dense prompt GEMM
is 3x faster in int8, which Phase C can revisit if prefill becomes the wall).

## Scratch and memory

| buffer | size | when |
|---|---|---|
| the words + the scales | the matrix's own bytes (3.5 bpw at K3.5) | upload |
| the window GEMV scratch | `TD_MAX_PARTS * TD_MAX_N * max_in` halfs (the staged activations) + the partial sums of the `TD_MAX_PARTS` largest matrices | upload |
| the reconstruct buffer | 1024 rows x `max_in` halfs (one row chunk) | upload |

All three are allocated at upload (nothing is allocated during a call: the decode step is captured in a CUDA graph,
TRACKER #169).

## Files

| file | what |
|---|---|
| `src/kernels/trellis/dense/plan.hpp` | the host-visible plan of the window GEMV: the parts of one matvec, the k split, the scratch size, the tile order |
| `src/kernels/trellis/dense/gemv_window.cu` | the window GEMV (the three launches) and the rate dispatch |
| `src/kernels/trellis/dense/reconstruct.cu` | the reconstruct to fp16 rows (the prompt path) |
| `src/kernels/trellis/dense/had128.cuh` | the half2 / float-pair Hadamard forms the reconstruct needs (the runtime path uses `src/kernels/trellis/hadamard.cuh`) |
| `src/kernels/trellis/dense/trellis_dense.h` | the format's API: `Matrix` (the device-side matrix), `matvec`, `reconstruct` |

Derived from tqLLM (Apache-2.0) `kernels/formats/x3/{gemv_window.cu, window.hpp, reconstruct.cu, had128.cuh}`, which
ported llama-paw's `paw-x3.cu` (MIT, exllamav3 lineage). The MIT notice of the original applies to the derived files.

## Tests and tools

| what | where |
|---|---|
| `trellis_dense_test` | real PAW-27B-X3.1 matrices (K3.5 and K3): the decode path vs an fp64 reference, the columns 1..16 bit-identical to n = 1, the prompt path vs the decode path |
| `tk-bench-trellis-dense` | the same 9 shape groups as tqLLM's `x3_window_bench`, graph-timed at n = 1/4/8/16, for a same-session A/B against tqLLM's X3_STATUS "Phase T1/T2" numbers |

Both tools use `cudaGraphInstantiateWithFlags`: the legacy three-argument `cudaGraphInstantiate` segfaults in libcuda
on this box (TRACKER #169).

## Measured (2026-10-20, GPU 2, the PAW-27B-X3.1 gguf)

| check | result |
|---|---|
| the decode path vs an fp64 reference (n = 1) | 4.09e-4 .. 4.77e-4 rel (limit 5e-3) on the seven matrices |
| the columns 1..16 vs n = 1 | bit-identical (0.0e+00) on all seven |
| the prompt path vs the decode path | 1.029e-3 .. 1.037e-3 rel (limit 5e-3) |
| the reconstruct vs tqLLM's `reconstruct_had_rows` | 0 of 89,128,960 values differ (bit-exact) |
| the bench, n = 1 over the 9 groups | 523.7 us (tqLLM's window in the same session: 572.4 us) |
| the bench, one 12-column call vs 8 + 4 | 1.28x .. 1.43x per group (tqLLM: 1.31x .. 1.42x) |
| the engine golden | 7/7 IDENTICAL vs `golden_1019b` (the Q8 paths are untouched) |

The per-shape numbers of the two benchmarks drift by up to 15% between runs on this box (machine drift, TRACKER #27):
the same kernels timed back to back in one process (`/tmp/opencode/e13/ab.cu`, ours vs tqLLM's) agree within 2% per
shape, so the sum is the number to compare.
