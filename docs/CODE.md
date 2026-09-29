# Code rules

The layout is `docs/04-architecture.md` §8. These rules keep it from needing a refactor as models, codecs and GPUs
are added. A change that breaks one of them needs a line in `TRACKER.md` saying why.

## Where things go

| what | where | example |
|---|---|---|
| a trellis codec (tile format + codebook) | `src/kernels/trellis/codec_<name>.cuh` | `codec_mul1.cuh` |
| shared device helpers | `src/kernels/trellis/` | `mma.cuh`, `hadamard.cuh` |
| one fused op | `src/kernels/<op>/<op>.{cuh,cu}` | `moe/moe_window.cu` |
| correctness test, one binary per op | `tests/unit/<op>_test.cu` | `moe_window_test.cu` |
| microbenchmark or ablation | `tools/tk-bench/<name>.cu` | `ceiling.cu` |
| codec study or prototype codec (not used by `src/`) | `tools/codec-lab/` | `viterbi_mse.cu`, `proto_v2pair.cuh` |
| results, logs, reports | `~/ML_projects/flashnext/<date>_truss_cpN/` (STORAGE.md), never in this repo | |

Build: `cmake -S . -B build -G Ninja && cmake --build build`. A new file is one line in `CMakeLists.txt`.

## Extension points (add, don't edit)

- **Codec.** Every codec header provides `template <int K> struct Codec` with `NAME`, `TILE_WORDS`, `VEC_WORDS`,
  `TILES_PER_VEC`, a per-lane constructor, `loads(lane)` and `tile(w, sub, f0, f1)` (contract in
  `codec_mul1.cuh`). Kernels take the codec as a template parameter and never call codebook functions directly.
  A new codec = a new header + a case in the rate dispatch + a row in `tools/tk-bench/ceiling.cu`.
- **Model shape.** Ops take a shape struct (`moe::FlashNext`). A new model = a new struct, a `Tune<>`
  specialization, and an explicit instantiation at the bottom of the op's `.cu`.
- **Kernel geometry.** Tunables live in one `Tune<Shape>` per op, set from a measured log that is named in a
  comment. No `#define` knobs.

## Kernel code

- No experiment switches (`#if BISECT`, `NO_DECODE`) in `src/`. Ablations are separate kernels in `tools/tk-bench/`.
- `static_assert` every tiling assumption (tracker rule 14).
- Deterministic: fixed reduction order, no float atomics.
- Device-only work on the caller's stream, so every op can be captured in a CUDA graph.
- Host entry points check their limits and fail loudly. They never clamp.
- A comment says why, not what. Each file starts with what it is and why it is shaped that way.

## Tests

- Compare against an independent reference (llama-paw ggml ops), same device buffers.
- NaN-fill outputs and workspaces before the call (rule 13).
- Exit code is the verdict; print one line per case with the error and the timing.

## Naming

- `namespace truss`, one sub-namespace per op (`truss::moe`). Types `CamelCase`, functions and variables
  `snake_case`, compile-time constants `UPPER_CASE`.
- Files `snake_case`; tool binaries `tk-<tool>-<name>`.
