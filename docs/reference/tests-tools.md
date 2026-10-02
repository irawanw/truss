# Tests and tools: tests/, tools/

Conventions (CODE.md): one binary per op; compare against an independent reference on the same device buffers;
NaN-fill outputs first; **exit code is the verdict**; one line per case with error and timing. Run on GPU 2:
`CUDA_VISIBLE_DEVICES=2 ./build/<name> ...`.

Data paths used below:

```
S    = ~/ML_projects/flashnext/20260930_truss_cp2/data/slice8/flashnext-x3-slice8-00001-of-00002.gguf   (8 layers)
D    = ~/ML_projects/flashnext/20260930_truss_cp2/data         (llama_dump_v1: 12 tokens; llama_dump_long3k: 3,659;
                                                                llama_dump_chunk8: 256 chat tokens)
M    = ~/ML_projects/flashnext/20260918_ngram_q8/data/qwen38-flash-next-paw-x3-q8_0-00001-of-00002.gguf  (full model)
BASE = /mnt/hdd/ml/flashnext/20260920_dense_codec_requant/data/20260923_q4k_retention_logits/q8.logits  (llama Q8 logits)
```

## Unit tests (tests/unit/)

| binary | checks | gate | run |
|---|---|---|---|
| `moe_window_test` | `moe::window` vs llama-paw's unfused PAW X3 chain, random trellis weights (all K or `TRUSS_KFIX`); timing in a CUDA graph | rel within llama's own rounding | `moe_window_test [rows...]` |
| `moe_prefill_test` | `moe::prefill` vs `moe::window` 8 rows at a time; row prefix invariance; TFLOPS | rel ≤ 5e-3, worst token ≤ 2e-2, prefix bit-identical | `moe_prefill_test [T...]` (default 64 512 2048 8192) |
| `q8_gemm_test` | 8 dense shapes: q8_gemm vs ref LLAMA, a16 vs ref FP32, gemv vs q8_gemm; a16 prefix variance; timings | 1e-5 / 3e-4 / 1e-6 | `q8_gemm_test [timing rows]` |
| `gdn_prefill_test` | `gdn::delta_rule` vs `ref::gated_delta_rule` (outputs and final state) | 1e-5 | `gdn_prefill_test [T...]` |
| `dsa_prefill_test` | `dsa::select` / `dsa::attention` vs reference (fp16 and int8 KV caches), incl. a later chunk and the split (≤ 32 rows) path at 1 / 13 / 32 rows; timing to 32K incl. decode steps | see kernels-mixers.md | `dsa_prefill_test` |
| `cpu_expert_test` | `cpu::ExpertPool` vs a double reference on random q4s experts (1/4/8 rows, shared experts, empty rows); each row equals its 1-row call; timing of 4 rows × 8 experts | rel ≤ 2e-3 (measured 2e-7), rows bit-identical | `cpu_expert_test` |
| `gemv_multi_test` | `q8_gemv_multi` / `f32_gemv_multi` on the engine's decode groups vs an fp64 CPU reference; row 0 bit-identical for rows 1..8 and each matrix bit-identical alone vs fused; time of one fused launch vs one launch per matrix (4 rows) | rel < 1e-6, both invariances | `gemv_multi_test` |
| `cpu_trellis_test [threads] [K]` | `cpu::trellis_gemv` vs a dense double reference from the bit-by-bit decode (`trellis_weight_ref`), K 1-4, 1/3/8 rows; `ExpertPool` on trellis slots vs an fp64 reference (bit-decoded weights) and rows vs 1-row calls; 3,000 small calls (phase-flip race); speed single thread / N threads from DRAM / pool | rel < 1e-5 vs the fp16-activation reference, or (int16 kernel) error vs exact activations ≤ the GPU fp16 reference's own; pool no further from exact than the GPU's fp16 activations, rows bit-identical, no hang | `cpu_trellis_test 12 2` |
| `codec_v2one_test` | v2one pack/decode/encode bit-exact; encoder MSE | lab + 10% | `codec_v2one_test [tiles]` |
| `gguf_reader_check.py` | `gguf::File` vs gguf-py, every tensor | exact | `python3 tests/unit/gguf_reader_check.py build/tk-pack-inspect $M` |

## Layer / model tests (tests/layer/)

| binary | checks | run |
|---|---|---|
| `qwen4exp_parity` | every reference block fed llama-paw's own input vs llama's output (hc, GDN, DSA, PLE, router, experts, shared, combine), plus `moe_window` vs llama and fp32; gates 1e-3 / 3e-3 (FFN) / 1e-2 (amplified chains), TRACKER #34 | `qwen4exp_parity $S $D/llama_dump_v1` → 160/160 |
| `qwen4exp_dsa_long` | DSA at 3,659 tokens: A selection rule on llama's indexer (near-tie swaps ≤ 1e-3), B attention vs llama within 2× its fp16 noise, C end to end, D the fast op on real data | `qwen4exp_dsa_long $S $D/llama_dump_long3k` |
| `qwen4exp_forward` | the engine chain (modes `short`, `stream` (needs `llama_dump_chunk8`: 64-token prompt chunks through the ring slots, then 48 decode steps through the ring; all resident == 3 GiB budget with index-order + 1 GiB ring and scattered usage-ranked hot set + smallest ring (wraps), residuals and logits bit-exact), `decode` (needs > 32 tokens: `llama_dump_chunk8`), `cpu` (CPU tier, argv[4] = q4s dir; `TRUSS_TEST_NO_CPU=1` without it: 1-token steps, 4-token runs and 4-row verify windows must give bit-identical logits; KL vs all-resident), `long`; model-qwen4exp.md) | `qwen4exp_forward $S short $D/llama_dump_v1`, `... decode $D/llama_dump_chunk8` etc. |
| `dump.h` | reader for llama_dump output (`get(name)`, `tokens()`) | — |

## Tools

| tool | what | run |
|---|---|---|
| `tk-bench-prefill` (`tools/tk-bench/prefill.cu`) | full-model or slice prefill throughput (warm-up + timed pass, random tokens); prints hot experts and streamed GB | `tk-bench-prefill $M 4096 4096 [budget MiB] [q8\|fp16]`; under `nsys profile` for the per-kernel split |
| `tk-bench-decode` (`tools/tk-bench/decode.cu`) | greedy decode speed: prompt (N random or `@ids.i32`, e.g. `20260930_truss_tg/data/prompt_code4k.i32` from `flashnext_truss_prompt_ids.py`), then 1-token steps with device argmax; mean/median/min/max ms | `tk-bench-decode $M @prompt.i32 64 65536 8192 [usage\|-] [budget MiB] [ring MiB]`; prints fetched experts and MB per step |
| `tk-profile` (`tools/tk-bench/profile.cu`) | routing profile for the hot set (the calibration step): prompts (`@ids.i32`) + greedy generated tokens through the engine, counts every routed (layer, expert), writes the usage file | `tk-profile $M out.f32 256 @p0.i32 @p1.i32 ...` (profile prompts: `20260930_truss_tg/data/profile_prompts/`, held out from the bench) |
| `tk-bench-spec` (`tools/tk-bench/spec.cu`) | MTP speculative greedy decode vs plain greedy on the full model: token sequences must be identical (exit 1 otherwise); several prompts `@a.i32,@b.i32,...` run in turn in one process (one line each, with the prompt read's tok/s and its PLE host gather time; the breakdown sums their spec phases; `TRUSS_BENCH_PLAIN=0` skips the plain runs and the identity check), e.g. the 8 held-out code prompts `20260930_truss_tg/data/profile_prompts/p0-p7.i32` (TRACKER #84: one prompt's tok/s moves with its greedy text when numerics change); also prints the PLE host gather per pass (`Forward::ple_host_ms`); tok/s both ways, tokens per pass, drafts accepted, experts fetched per pass; env `TRUSS_CPU_TRELLIS`, `TRUSS_CPU_DYNAMIC`, `TRUSS_CPU_THREADS`, `TRUSS_CPU_SHARE`, `TRUSS_PCIE_GBPS`, `TRUSS_RING_GB`, `TRUSS_HINT_K` (pre-gated prefetch width), `TRUSS_PCIE_FRAC` (fixed PCIe share of misses, `Options::pcie_frac`), `TRUSS_ADAPT_EVERY` / `TRUSS_ADAPT_SWAPS` (adaptive tier), `TRUSS_PREFILL_ROWS`, `TRUSS_KV_INT8` (all read by `qwen4exp::apply_env`, shared with the C API); usage argument: `data/usage_strata_rank.f32` (Strata's profile) is the best measured; with `TRUSS_PROFILE_SECTIONS=1` it also prints routed MoE split into router + publish / wait for the host plan / wait for copies / expert kernel (`Forward::section_moe_ms`) and the driver thread's split / CPU start / copies + plan (`Forward::driver_ms`); the CPU pool also reads `TRUSS_CPU_PIN`, `TRUSS_CPU_SPIN_US`, and the trellis kernel `TRUSS_TRELLIS_I16` (0: `gemv_tiles4`), `TRUSS_TRELLIS_ROUND` | `tk-bench-spec $M $MTP @prompt.i32 [tokens=128] [drafts=3] [usage\|-]`; MTP = `20260930_truss_tg/data/mtp_x3/flashnext-mtp-x3k3.gguf` |
| `tools/tk-bench/cpu_q4.cc` | CPU expert throughput probe (4-bit weights, int8 activations, AVX2; experts cycled from DRAM) for a CPU miss tier; not built by CMake | `g++ -O3 -march=znver2 -pthread tools/tk-bench/cpu_q4.cc -o cpu_q4 && ./cpu_q4 [threads] [rows] [pool] [s]` |
| `tk-bench-ceiling` | decode ceiling per codec rate from registers; stream ceiling per access pattern | `tk-bench-decode` (`tools/tk-bench/decode.cu`) | greedy decode speed: prompt (N random or `@ids.i32`, e.g. `20260930_truss_tg/data/prompt_code4k.i32` from `flashnext_truss_prompt_ids.py`), then 1-token steps with device argmax; mean/median/min/max ms | `tk-bench-decode $M @prompt.i32 64 65536 8192 [usage\|-] [budget MiB] [ring MiB]`; prints fetched experts and MB per step |
| `tk-profile` (`tools/tk-bench/profile.cu`) | routing profile for the hot set (the calibration step): prompts (`@ids.i32`) + greedy generated tokens through the engine, counts every routed (layer, expert), writes the usage file | `tk-profile $M out.f32 256 @p0.i32 @p1.i32 ...` (profile prompts: `20260930_truss_tg/data/profile_prompts/`, held out from the bench) |
| `tk-bench-spec` (`tools/tk-bench/spec.cu`) | MTP speculative greedy decode vs plain greedy on the full model: token sequences must be identical (exit 1 otherwise); tok/s both ways, tokens per pass, drafts accepted, experts fetched per pass; env `TRUSS_CPU_TRELLIS`, `TRUSS_CPU_DYNAMIC`, `TRUSS_CPU_THREADS`, `TRUSS_CPU_SHARE`, `TRUSS_PCIE_GBPS`, `TRUSS_RING_GB`, `TRUSS_HINT_K` (pre-gated prefetch width) | `tk-bench-spec $M $MTP @prompt.i32 [tokens=128] [drafts=3] [usage\|-]`; MTP = `20260930_truss_tg/data/mtp_x3/flashnext-mtp-x3k3.gguf` |
| `tk-bench-ceiling` |
| `tk-bench-moe-trace` | timeline of one `moe::window` launch from the kernel's own trace | `tk-bench-moe-trace [rows] [K]` |
| `tk-parity-kl` (`tools/tk-parity/kl_vs_base.cu`) | full-model KL / top-1 / perplexity vs a llama-perplexity `--kl-divergence-base` file; `step` > 0 scores the second half of each chunk in runs of `step` tokens (≤ 32: the decode path with fetch, ring and CPU tier) | `tk-parity-kl $M $BASE [chunks] [first] [q8\|fp16] [step] [usage\|-] [cpu dir\|-]` (~8 s per 2048-token chunk) |
| `tk-parity-llama-dump` | llama-paw activations for parity (one ubatch, every row an output) from a prompt or `@ids.i32` | `tk-parity-llama-dump $S out_dir "<prompt>"` or `@file.i32` |
| `slice_gguf.py` | cut an N-layer slice of a GGUF (hard-links the PLE shard) | `python3 tools/tk-parity/slice_gguf.py ...` |
| `tk-pack-inspect` | catalog and budget of a model file, trellis table checks, binding check; `list` for the reader check | `tk-pack-inspect $M [list]` |
| `tk-codec-lab` (`tools/codec-lab/viterbi_mse.cu`) | trellis MSE of candidate codebooks on iid data; `search` for the v2one family | `tk-codec-lab [tiles]` |
| `proto_v2one.cuh`, `proto_v2pair.cuh` | codec prototypes for the ceiling bench (not used by src/) | — |

## Speed measurement rules (TRACKER protocol)

Discard the first run after load (lazy module loading, cuBLAS heuristics: 12 s on the first slice pass); interleave
A/B in one session (renters on GPUs 0/3 change clocks, TRACKER #27); nsys + sqlite for kernel splits (no perf
counters without root); label estimates as *(est.)*.
