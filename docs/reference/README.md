# TRUSS reference: every file, what it does, how to change it

TRUSS is our own inference engine for trellis-quantized (PAW X3 / EXL3-style) MoE models on one RTX 3090. Target
model today: **Qwen3.8 Flash-Next PAW X3.1** (`qwen4exp` architecture, 48 layers, 512 experts). This folder is the
map of the code: for every source file it says what the file computes, its API (argument shapes and units), data
layouts, numerics, invariants, which test proves it, what can be tuned (with the measured number behind each
setting) and how to extend or replace it. **If you change a file, change its entry here in the same commit**
(docs/CODE.md rule).

Why this exists: in llama.cpp-derived code, finding out what a tensor layout is or why a kernel is shaped a certain
way means reading and experimenting. Here the answer is written down next to the evidence (TRACKER.md rows).

## Pages

| page | files |
|---|---|
| [formats-core.md](formats-core.md) | `src/formats/*` (GGUF reader, trellis expert tables), `src/core/*` (device tensors, scratch, error checks) |
| [kernels-trellis-moe.md](kernels-trellis-moe.md) | `src/kernels/trellis/*` (codecs, mma fragments, Hadamard), `src/kernels/moe/*` (routed experts: decode window, prefill), `src/encode/*` |
| [kernels-dense.md](kernels-dense.md) | `src/kernels/dense/*` (Q8_0 GEMM / GEMV / fp16-activation GEMM, embedding rows) |
| [kernels-mixers.md](kernels-mixers.md) | `src/kernels/dsa/*` (sparse attention prefill/decode), `src/kernels/gdn/*` (gated delta net) |
| [kernels-small.md](kernels-small.md) | `src/kernels/hc/*`, `src/kernels/ffn/*`, `src/kernels/ple/*`, `src/kernels/sampling/*` |
| [kernels-reference.md](kernels-reference.md) | `src/kernels/reference/*` (slow exact fp32 definitions every fast kernel is tested against) |
| [model-qwen4exp.md](model-qwen4exp.md) | `src/model/qwen4exp/*` (config, weight binding, PLE hashing, reference forward, the fast `Forward`) |
| [runtime-api-server.md](runtime-api-server.md) | `src/runtime/*` (expert residency and streaming), `include/truss/truss.h` + `src/api/*` (C API), `server/*` (OpenAI server) |
| [tests-tools.md](tests-tools.md) | `tests/*`, `tools/*`: what each checks, its gate, how to run it, which data it needs |

Other docs: `docs/CODE.md` (rules: where files go, extension points, kernel rules), `TRACKER.md` (every experiment
and measurement, numbered #1..; the "Do not repeat" list), `docs/06-truss.md` (design), `docs/04-architecture.md`
(layout and growth plan), `docs/01..05` (the pre-TRUSS evidence and plans; history).

## The whole engine in one picture

```
 GGUF file(s) ──gguf::File (mmap)──► qwen4exp::Config + qwen4exp::Weights (host views, every tensor bound once)
                                            │
                        qwen4exp::Forward (src/model/qwen4exp/forward.cu) builds, on one GPU:
                          • dense Q8_0 matrices repacked (dense::Q8Matrix), norms/F32/F16 tensors (DeviceTensors)
                          • runtime::ExpertStore: hot experts resident, cold experts pinned in host RAM
                          • per-layer state: DSA K/V + indexer caches, GDN recurrent state + conv rows, PLE history
                                            │
        truss C API (libtruss.so) ◄─────────┘        server/app.py (FastAPI, OpenAI API) ── ctypes ──► libtruss.so
```

### One chunk through `Forward::run(tokens, T)` (prefill chunk or decode step; T = rows)

| step | op (file) | shapes (Flash-Next) |
|---|---|---|
| embed | `dense::q8_rows` (kernels/dense) → `hc::expand` | token_embd Q8_0 [248320][2560] → res [T][4][2560] |
| per layer l = 0..47: | | |
| PLE (layer 1 only) | host `ple_rows` + `ple_gather` (model/qwen4exp/ple.cc) → 2 GEMMs → `ple::gate`, `ple::apply` | 16 rows × 160 from a 48 GiB int8 table |
| attention-side mix | `hc::norm` → GEMM down (10240→320) → `hc::silu` → GEMM up (320→10240) → `hc::collapse`; GEMM inject (10240→4) | mixed [T][2560] |
| mixer GDN (36 layers) | GEMMs qkv/z/alpha/beta → `gdn::prepare` → `gdn::delta_rule` → `gdn::output_norm` → GEMM out | state [48][128][128] fp32 |
| mixer DSA (12 layers) | GEMMs q/k/v/idx_q/idx_k → `dsa::rope_table`, `dsa::prepare_qkv`, `dsa::prepare_index` → `dsa::select` → `dsa::attention` → GEMM out | K/V cache fp16 [n_ctx][2][256] |
| combine | `hc::combine` | res += out · 2σ(inject/4) |
| FFN-side mix | as above (hc_ffn) | |
| router | fp32 SGEMM (cuBLAS) → `ffn::route` (softmax, top-10, renorm) | logits [T][512] |
| routed experts | `ExpertStore` acquire → `moe::prefill` (T > 8) or `moe::window` (T ≤ 8) → release | trellis K1..K4 experts 2560×640 |
| shared expert | GEMMs gate/up → `ffn::swiglu` → GEMM down; fp32 SGEMM gate → `ffn::shared_add` | |
| combine | `hc::combine` | |
| head (on request) | `Forward::head`: hc_head mix → output GEMM in vocab tiles | logits [n][248320] |

Dense GEMMs: `Forward::lin` picks `dense::q8_gemv` (rows ≤ 8), `dense::q8_gemm` (Q8_1 activations, default) or
`dense::q8_gemm_a16` (fp16 activations, option). Routed-expert residency: chunks of ≤ 32 rows fetch the cold experts
they route to (`ExpertStore::fetch`); longer chunks stream whole layers ahead of compute (`ExpertStore::prefetch`).

## Build, test, run

```bash
cmake -S . -B build -G Ninja && cmake --build build          # CUDA 12.6, sm_86; tests link llama-paw from ~/llama-paw
CUDA_VISIBLE_DEVICES=2 ./build/q8_gemm_test                    # every unit test: exit code is the verdict
S=~/ML_projects/flashnext/20260930_truss_cp2/data/slice8/flashnext-x3-slice8-00001-of-00002.gguf
D=~/ML_projects/flashnext/20260930_truss_cp2/data
CUDA_VISIBLE_DEVICES=2 ./build/qwen4exp_parity $S $D/llama_dump_v1          # block parity vs llama-paw (160 cases)
CUDA_VISIBLE_DEVICES=2 ./build/qwen4exp_forward $S short $D/llama_dump_v1   # engine chain vs fp32 reference
M=~/ML_projects/flashnext/20260918_ngram_q8/data/qwen38-flash-next-paw-x3-q8_0-00001-of-00002.gguf
CUDA_VISIBLE_DEVICES=2 ./build/tk-bench-prefill $M 4096 4096              # full-model prefill speed
CUDA_VISIBLE_DEVICES=2 python3 -m server.app --model $M \
    --tokenizer /data/www/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/models/Qwen3.8-27B-EXL3-3.5bpw/tokenizer.json --port 8090
```

The full list with gates and data: [tests-tools.md](tests-tools.md).

## Measured status (GPU 2, RTX 3090, 2026-09-30)

| what | number | evidence |
|---|---|---|
| full-model prefill, 4K / 8K / 16K / 32K prompt | 2,263 / 2,499 / 2,548 / 2,321 tok/s (PCIe-bound: ~24 GB of cold experts per chunk) | TRACKER #52 |
| full-model KL vs llama-paw Q8 logits (64 × 2048 tokens) | 0.0131, top-1 97.04% (llama vs itself: 0.0115, 97.06%) | TRACKER #53 |
| decode (serving, bring-up) | 13.6–15.5 tok/s | TRACKER #55 |
| Strata on the same box (bar) | PP 960–1,330 tok/s, TG 84–107 tok/s | TRACKER #1 |

## Glossary

| term | meaning |
|---|---|
| **trellis / X3 / mul1** | weight quantization where each weight is a codebook value of a 16-bit state that shifts K bits per weight (QTIP / exllamav3 EXL3). "mul1" is the codebook (`codec_mul1.cuh`), K = 1..4 bits per weight per expert |
| **tile** | 16×16 weights = 256 states = 8K uint32 words; laid out in tensor-core fragment order |
| **suh / svh** | per-expert fp16 sign·scale vectors applied before (input, `suh`) and after (output, `svh`) the Hadamard-rotated matmul: y = svh · H128(Wᵀ · H128(suh · x)) |
| **H128** | 128-point Walsh–Hadamard transform (`hadamard.cuh`), scaled by 1/√128 |
| **hc (hyper-connections)** | the residual is 4 parallel streams [T][4][2560]; each block reads a learned mix ("hc mix") and writes back with learned gates ("hc combine") |
| **GDN** | gated delta net: linear attention with a 128×128 state per value head (36 of 48 layers) |
| **DSA / QSA** | sparse attention: an indexer scores 4-token blocks, each query attends to its top 512 blocks + its tail (12 layers) |
| **PLE** | per-layer n-gram embedding: hashed rows of a 320M×160 int8 table mixed into the residual at layer 1 |
| **hot / cold expert** | resident in VRAM / kept in pinned host RAM and copied over PCIe when needed |
| **Q8_0 / Q8_1** | llama.cpp 8-bit blocks of 32 (weights with fp16 scale / activations quantized per 32-block) |
| **chunk** | the tokens of one `Forward::run` call: a prompt piece (≤ max_chunk) or a decode step |
| **slice** | an 8-layer cut of the model (tools/tk-parity/slice_gguf.py) that fits the GPU with llama-paw, for parity tests |

## Cookbook: common changes

| to... | do | where it is documented |
|---|---|---|
| add a trellis codec | new `src/kernels/trellis/codec_<name>.cuh` with the codec interface; a case in the rate dispatch; a row in `tools/tk-bench/ceiling.cu`; an encoder in `src/encode/` | kernels-trellis-moe.md |
| retune a kernel | change its `Tune<Shape>` / constants (each names the TRACKER row it came from); rerun its unit test (gate + timing) | the kernel's page |
| replace a dense GEMM | keep `dense::Q8Matrix` and the function contract; `Forward::lin` is the only caller | kernels-dense.md |
| add a model family | `src/model/<family>/` (config, weights, reference, forward), shape structs for the ops it reuses, explicit template instantiations | model-qwen4exp.md |
| change expert placement | `runtime::ExpertStore::plan` (the hot set) — nothing else reads placement | runtime-api-server.md |
| serve a new API field | `server/app.py` (`params`, the endpoint); engine calls stay `Engine.generate` | runtime-api-server.md |
