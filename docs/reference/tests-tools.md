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
| `dsa_prefill_test` | `dsa::select` / `dsa::attention` vs reference, incl. a later chunk and the split (≤ 32 rows) path at 1 / 13 / 32 rows; timing to 32K incl. decode steps | see kernels-mixers.md | `dsa_prefill_test` |
| `codec_v2one_test` | v2one pack/decode/encode bit-exact; encoder MSE | lab + 10% | `codec_v2one_test [tiles]` |
| `gguf_reader_check.py` | `gguf::File` vs gguf-py, every tensor | exact | `python3 tests/unit/gguf_reader_check.py build/tk-pack-inspect $M` |

## Layer / model tests (tests/layer/)

| binary | checks | run |
|---|---|---|
| `qwen4exp_parity` | every reference block fed llama-paw's own input vs llama's output (hc, GDN, DSA, PLE, router, experts, shared, combine), plus `moe_window` vs llama and fp32; gates 1e-3 / 3e-3 (FFN) / 1e-2 (amplified chains), TRACKER #34 | `qwen4exp_parity $S $D/llama_dump_v1` → 160/160 |
| `qwen4exp_dsa_long` | DSA at 3,659 tokens: A selection rule on llama's indexer (near-tie swaps ≤ 1e-3), B attention vs llama within 2× its fp16 noise, C end to end, D the fast op on real data | `qwen4exp_dsa_long $S $D/llama_dump_long3k` |
| `qwen4exp_forward` | the engine chain (modes `short`, `stream` (needs `llama_dump_chunk8`: 64-token prompt chunks through the ring slots, then 48 decode steps through the ring; all resident == 3 GiB budget with index-order + 1 GiB ring and scattered usage-ranked hot set + smallest ring (wraps), residuals and logits bit-exact), `decode` (needs > 32 tokens: `llama_dump_chunk8`), `long`; model-qwen4exp.md) | `qwen4exp_forward $S short $D/llama_dump_v1`, `... decode $D/llama_dump_chunk8` etc. |
| `dump.h` | reader for llama_dump output (`get(name)`, `tokens()`) | — |

## Tools

| tool | what | run |
|---|---|---|
| `tk-bench-prefill` (`tools/tk-bench/prefill.cu`) | full-model or slice prefill throughput (warm-up + timed pass, random tokens); prints hot experts and streamed GB | `tk-bench-prefill $M 4096 4096 [budget MiB] [q8\|fp16]`; under `nsys profile` for the per-kernel split |
| `tk-bench-decode` (`tools/tk-bench/decode.cu`) | greedy decode speed: prompt (N random or `@ids.i32`, e.g. `20260930_truss_tg/data/prompt_code4k.i32` from `flashnext_truss_prompt_ids.py`), then 1-token steps with device argmax; mean/median/min/max ms | `tk-bench-decode $M @prompt.i32 64 65536 8192 [usage\|-] [budget MiB] [ring MiB]`; prints fetched experts and MB per step |
| `tk-profile` (`tools/tk-bench/profile.cu`) | routing profile for the hot set (the calibration step): prompts (`@ids.i32`) + greedy generated tokens through the engine, counts every routed (layer, expert), writes the usage file | `tk-profile $M out.f32 256 @p0.i32 @p1.i32 ...` (profile prompts: `20260930_truss_tg/data/profile_prompts/`, held out from the bench) |
| `tk-bench-spec` (`tools/tk-bench/spec.cu`) | MTP speculative greedy decode vs plain greedy on the full model: token sequences must be identical (exit 1 otherwise); tok/s both ways, tokens per pass, drafts accepted, experts fetched per pass | `tk-bench-spec $M $MTP @prompt.i32 [tokens=128] [drafts=3] [usage\|-]`; MTP = `20260930_truss_tg/data/mtp_x3/flashnext-mtp-x3k3.gguf` |
| `tools/tk-bench/cpu_q4.cc` | CPU expert throughput probe (4-bit weights, int8 activations, AVX2; experts cycled from DRAM) for a CPU miss tier; not built by CMake | `g++ -O3 -march=znver2 -pthread tools/tk-bench/cpu_q4.cc -o cpu_q4 && ./cpu_q4 [threads] [rows] [pool] [s]` |
| `tk-bench-ceiling` | decode ceiling per codec rate from registers; stream ceiling per access pattern | `tk-bench-decode` (`tools/tk-bench/decode.cu`) | greedy decode speed: prompt (N random or `@ids.i32`, e.g. `20260930_truss_tg/data/prompt_code4k.i32` from `flashnext_truss_prompt_ids.py`), then 1-token steps with device argmax; mean/median/min/max ms | `tk-bench-decode $M @prompt.i32 64 65536 8192 [usage\|-] [budget MiB] [ring MiB]`; prints fetched experts and MB per step |
| `tk-profile` (`tools/tk-bench/profile.cu`) | routing profile for the hot set (the calibration step): prompts (`@ids.i32`) + greedy generated tokens through the engine, counts every routed (layer, expert), writes the usage file | `tk-profile $M out.f32 256 @p0.i32 @p1.i32 ...` (profile prompts: `20260930_truss_tg/data/profile_prompts/`, held out from the bench) |
| `tk-bench-spec` (`tools/tk-bench/spec.cu`) | MTP speculative greedy decode vs plain greedy on the full model: token sequences must be identical (exit 1 otherwise); tok/s both ways, tokens per pass, drafts accepted, experts fetched per pass | `tk-bench-spec $M $MTP @prompt.i32 [tokens=128] [drafts=3] [usage\|-]`; MTP = `20260930_truss_tg/data/mtp_x3/flashnext-mtp-x3k3.gguf` |
| `tk-bench-ceiling` |
| `tk-bench-moe-trace` | timeline of one `moe::window` launch from the kernel's own trace | `tk-bench-moe-trace [rows] [K]` |
| `tk-parity-kl` (`tools/tk-parity/kl_vs_base.cu`) | full-model KL / top-1 / perplexity vs a llama-perplexity `--kl-divergence-base` file | `tk-parity-kl $M $BASE [chunks] [first] [q8\|fp16]` (~8 s per 2048-token chunk) |
| `tk-parity-llama-dump` | llama-paw activations for parity (one ubatch, every row an output) from a prompt or `@ids.i32` | `tk-parity-llama-dump $S out_dir "<prompt>"` or `@file.i32` |
| `slice_gguf.py` | cut an N-layer slice of a GGUF (hard-links the PLE shard) | `python3 tools/tk-parity/slice_gguf.py ...` |
| `tk-pack-inspect` | catalog and budget of a model file, trellis table checks, binding check; `list` for the reader check | `tk-pack-inspect $M [list]` |
| `tk-codec-lab` (`tools/codec-lab/viterbi_mse.cu`) | trellis MSE of candidate codebooks on iid data; `search` for the v2one family | `tk-codec-lab [tiles]` |
| `proto_v2one.cuh`, `proto_v2pair.cuh` | codec prototypes for the ceiling bench (not used by src/) | — |

## Speed measurement rules (TRACKER protocol)

Discard the first run after load (lazy module loading, cuBLAS heuristics: 12 s on the first slice pass); interleave
A/B in one session (renters on GPUs 0/3 change clocks, TRACKER #27); nsys + sqlite for kernel splits (no perf
counters without root); label estimates as *(est.)*.
