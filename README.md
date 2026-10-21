# TRUSS (trellis-kernel)

Our own inference engine for **trellis-quantized MoE models on one or two RTX 3090s** (Ampere, sm_86), served through
an OpenAI-compatible API. Two models run on it today:
**[PAW-763B-X3](https://huggingface.co/lackonendes/PAW-763B-X3)** (DeepSeek-V4.1-Flash-1017, 763B parameters, two RTX 3090s)
and **[PAW-125B-FLASH-NEXT-X3.1](https://huggingface.co/lackonendes/PAW-125B-FLASH-NEXT-X3.1)** (Qwen3.8 Flash-Next, 125B).

## PAW-763B-X3: DeepSeek-V4.1-Flash-1017 on two RTX 3090s (2026-10-21)

TRUSS now serves **PAW-763B-X3**, a 763B-parameter DeepSeek model (557B routed experts, 197B n-gram tables) on two 24 GB
cards plus CPU and NVMe, with the full 262,144-token context. The pack keeps 256 of the 384 experts per layer at 2-3
bit. Files and model card: **[lackonendes/PAW-763B-X3](https://huggingface.co/lackonendes/PAW-763B-X3)**
(215 GB). How to run it: **[docs/SERVING-DS41.md](docs/SERVING-DS41.md)**.

| two RTX 3090s, 128 GB RAM, NVMe | measured |
|---|---:|
| decode, bench (speculative, 3 drafts) | **29-30.5 tok/s** |
| decode, real agent session at 70-95K context | **20-23 tok/s** |
| prefill, 21K-token prompt | **1,100-1,300 tok/s** |
| KL against the original checkpoint (8 sequences; budget 0.0728) | **0.061-0.069** |
| recall of a passage copied from 21K / 60K / 100K / 200K tokens back | 99.6% / 100% / 100% / 100% |

Pending, not solved: (1) past about 60K tokens of context roughly 30% of agent turns loop in their reasoning; a loop
guard in the server ends the loop, the cause is not found and the test that isolates it (`tools/tk-bench/arms_loop.py`)
has not run yet; (2) decode and prefill are below the 70 / 2,000 tok/s goals, because 130-150 routed experts per pass are
not in VRAM and every prompt chunk re-streams the cold ones; (3) quality is measured by KL and recall only, no task
benchmarks yet. The plan and every measurement: [docs/PLAN-20261021-ds41-70-2000.md](docs/PLAN-20261021-ds41-70-2000.md).

## PAW-125B-FLASH-NEXT-X3.1

## Status (2026-10-13)

PAW-125B-FLASH-NEXT-X3.1 (Qwen3.8-Flash-Next, 125B MoE, 24,576 experts at 3.5 / 2.5 bit, Q8_0 dense, 47.7 GiB n-gram
table read from NVMe), full 262,144-token context, Threadripper 3960X:

| | two RTX 3090s (layer split) | one RTX 3090 (250 W) |
|---|---:|---:|
| decode, real agent sessions (temperature 0.8, thinking, tools, up to 150K context) | **115-116 tok/s** | - |
| decode, 150K-token context (greedy) | **128 tok/s** | 57-60 tok/s |
| decode, short prompt | not measured | peak 80+ tok/s |
| prefill, fresh 150K-token prompt | **about 4,500 tok/s** | 2,170-2,270 tok/s |
| memory | about 23.5 GiB per GPU, RAM about 25 GB | GPU 23.6 GiB, RAM 43 GB |

NVMe 97 GB in both setups. Two GPUs split the model by layers (layers 0-23 on one card; layers 24-47, the output head
and the MTP drafter on the other; prompt chunks pipelined across both, no NVLink): 55% of the routed experts fit on the
GPUs instead of about a third, which is where most of the 2x comes from. Decode varies by a few percent on a shared
machine. Model card: [PAW-125B-FLASH-NEXT-X3.1](https://huggingface.co/lackonendes/PAW-125B-FLASH-NEXT-X3.1).
Every number has a row in TRACKER.md (kept locally, internal-only, not in this repo) with the command that produced it.

Tried on the two-GPU decode on 2026-10-13 (measured on a replay of 16 real agent requests, served
sampling, run-to-run spread 1.2%):
- **Fused small kernels** (`TRUSS_HC_SILU`, `TRUSS_HC_COLLAPSE`, `TRUSS_ROUTE_PUBLISH`): adopted, bit-exact,
  +4.7% (122 -> 128 tok/s greedy at 150K).
- **Context lookup drafts** (branch `agent/draft-tokens`, `TRUSS_LOOKUP`, off by default): exact, but 114 tok/s vs
  115-116 without. MTP drafts already cover what the context would add.
- **Draft stop rule and draft count** (min-p 0.3-0.6, 5-7 drafts): the served 0.5 / 7 is on a plateau; other values
  lose 4-8%.
- **Early expert window, helper experts** (branches `agent/early-window`, `agent/helper-experts`): slower; the
  expert window is DRAM-bound (~580 GB/s) and the split adds ~70 us per layer.

## Start here

- **[docs/SERVING-DS41.md](docs/SERVING-DS41.md)** — run PAW-763B-X3 (DeepSeek-V4.1-Flash-1017): install, download, settings, benchmark.
- **[docs/reference/](docs/reference/README.md)** — every source file: what it computes, API and shapes, layouts,
  numerics, invariants, tests, tunables, how to change it. Start with its README (data flow of one forward pass).
- **[docs/HANDOVER-20261014-engine-lfm.md](docs/HANDOVER-20261014-engine-lfm.md)** — current state, what is running, next steps in order.
- **[docs/ENGINE.md](docs/ENGINE.md)** — the engine / blocks / model split, and how to add a model (a folder, one registry line).
- [docs/CODE.md](docs/CODE.md) — code rules: where files go, extension points, kernel and test rules, no file over 500 lines.
- **[docs/PLAN-20261003-x31-tg140-pp4000.md](docs/PLAN-20261003-x31-tg140-pp4000.md)** — current plan: X3.1 weights that stop the thinking loops (no K1 holes, rates from agent traffic), 1 GPU + 64 GB RAM, decode 140 / prefill 4,000 tok/s at 256K (budgets, phases A-D, gates, kill rules).
- [docs/PLAN-20261002-tg100-pp3000.md](docs/PLAN-20261002-tg100-pp3000.md) — previous plan (decode 100 / prefill 3,000; its §5 results are the new plan's starting point).
- TRACKER.md (kept locally, internal-only — not in this repo) — every experiment and measurement, checkpoints (CP0–CP9), the "Do not repeat" list.
- [docs/06-truss.md](docs/06-truss.md) — design; `docs/01..05`, `PLAN.md` — the evidence and plans that led here.

## Quickstart

PAW-763B-X3 (two 24 GB GPUs, 128 GB RAM): see [docs/SERVING-DS41.md](docs/SERVING-DS41.md). Serve **PAW-125B-FLASH-NEXT-X3.1** ([model on Hugging Face](https://huggingface.co/lackonendes/PAW-125B-FLASH-NEXT-X3.1))
on one or two 24 GB GPUs plus CPU: **[docs/SERVING-FLASHNEXT-X31.md](docs/SERVING-FLASHNEXT-X31.md)** (install, download,
the best settings, memory, expected speed). Short version:

```bash
cmake -S . -B build -G Ninja && cmake --build build
MODEL_DIR=/nvme/flashnext TOKENIZER=/nvme/flashnext/tokenizer.json GPUS=0,1 tools/serve/serve_flashnext_x31.sh   # two GPUs
MODEL_DIR=/nvme/flashnext TOKENIZER=/nvme/flashnext/tokenizer.json GPU=0 tools/serve/serve_flashnext_x31.sh      # one GPU
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

Tests and benchmarks: [docs/reference/tests-tools.md](docs/reference/tests-tools.md).

## Credits

TRUSS is an independent, from-scratch implementation (no shared code, no fork), but these projects shaped how it's
built. Thank you to everyone below; their work is what we measured against and learned from.

- **[Strata](https://github.com/Niko1221/Strata)** (Niko1221, MIT) is where the serving architecture comes
  from. Running a 125B MoE on one consumer GPU means most experts can't live in VRAM, and Strata's answer —
  keep the hot experts on the GPU, compute the rest on the CPU from RAM, hand off between them through a
  flag in pinned memory instead of a blocking copy, and recover the lost speed with speculative decoding —
  is the shape TRUSS's engine follows and the bar its benchmarks are measured against. The specific
  techniques (CPU-resident experts, the async doorbell handoff, adaptive/usage-ranked residency, an int8 KV
  layout, accepting speculative drafts row-by-row) were studied from Strata's engine and reimplemented here
  independently, not copied: see `docs/06-truss.md` and `docs/reference/` for what each piece does in TRUSS
  and how it differs. **[mmnga/Strata](https://github.com/mmnga/Strata)**'s DeepSeek port was our cross-check for the
  DeepSeek-V4.1 tensor names and metadata keys, and **[PolyStrata](https://github.com/VecSzn/PolyStrata)** (VecSzn)
  for the tensor-core attention pattern.
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** (turboderp, MIT) is where the weight codec
  comes from. Its EXL3 format (a QTIP-style trellis/Viterbi quantizer) is the basis for the X3 / X3.1
  codec that compresses this model's experts to 2-3 bits/weight on GPU; TRUSS's decode kernels and its
  encoder's refit/drift step port that design (`docs/reference/kernels-trellis-moe.md`,
  `docs/reference/formats-core.md`). The underlying method is **QTIP**
  ([arXiv 2406.11235](https://arxiv.org/abs/2406.11235)).
- **[DeepInfer](https://github.com/deepseek-ai/DeepInfer)** (DeepSeek, Apache-2.0): the reference for the
  DeepSeek-V4.1 architecture details (YaRN only on the compressed layers; its lazy hyper-connection post step is on our plan).
- **[tqLLM](https://github.com/turbo-tan/tqllm)** (turbo-tan, Apache-2.0): the trellis GEMV kernel family that
  reaches Q8 speed, and the retrieval-draft and trellis-to-int8 prefill ideas on our plan.
- **[TensorFold](https://github.com/ashhart/TensorFold)** (ashhart): exact DSpark speculation, the q4 draft head, and
  pipelined prompt reading.
- **[Maya](https://github.com/mw00/project-maya)** (mw00): large prompt chunks, experts borrowing VRAM during prompts,
  and the long-prompt skip that our tail prefill is compared with.
- **[sybil-solutions/dsv41-flash-offload](https://github.com/sybil-solutions/dsv41-flash-offload)** (MIT): staged-DMA
  prefill and FP16 GEMM for experts with many rows.
- **[dsx](https://github.com/lxe/dsx)** (lxe): deferring low-rank CPU experts by one layer.
- **[Xolotl](https://github.com/Indras-Mirror/Xolotl)** (Indras-Mirror): moving the draft model off the card during long
  prompts, and learning experts per conversation (ideas we studied).
- **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (DeepSeek, MIT): the
  model itself.

## Layout

```
include/truss/truss.h      C API (libtruss.so)
src/formats/               GGUF reader, trellis expert tables
src/core/                  device tensors, scratch, error checks
src/kernels/<op>/          fast kernels (trellis, moe, dense, dsa, gdn, hc, ffn, ple, sampling) + reference/
src/engine/                model-agnostic engine: Pipeline (sequences, slots, spec rounds, layer split), Stage, options
src/blocks/                reusable layer blocks (Linear, Head, Hc, Gdn, Dsa, Ple, Moe) with per-slot state
src/model/<family>/        THE ONLY MODEL CODE: config, weight binding, composition of blocks (qwen4exp today)
src/model/registry.cc      general.architecture -> family
src/runtime/               routed-expert runtime: residency, PCIe streaming, doorbell driver, CPU tier
src/api/                   C API implementation
src/encode/                pack-time trellis encoders
server/                    OpenAI-compatible server (Python, ctypes)
tests/unit, tests/layer    op tests; model blocks and chains vs llama-paw and the fp32 reference
tools/                     benchmarks, parity dumps and KL, pack inspection, codec lab
```
