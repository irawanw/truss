# TRUSS (trellis-kernel)

Our own inference engine for **trellis-quantized MoE models on one RTX 3090** (Ampere, sm_86). First model:
**Qwen3.8 Flash-Next PAW X3.1** (qwen4exp, 48 layers, 512 experts, 37 GiB of 2.63-bpw trellis experts + Q8_0 dense),
served on GPU 2 of this box through an OpenAI-compatible API. Next models: PAW X3.1 27B, PAW 35B.

## Status (2026-09-30, measured on GPU 2)

| | TRUSS | Strata (same box) |
|---|---|---|
| prefill, bench tool, 4K / 8K / 16K / 32K prompt | 2,263 / 2,499 / 2,548 / 2,321 tok/s | 960–1,330 |
| prefill, server + code-agent bench, ~4K / ~32K | 938–1,403 / 1,508–1,778 tok/s | 960–1,330 |
| decode (serving bring-up) | 13–15 tok/s | 84–107 |
| quality: full-model KL vs llama-paw Q8 logits | 0.0131 (= llama's own run-to-run floor 0.0115) | expert error 2× ours |

Every number has a row in TRACKER.md (kept locally, internal-only — not in this repo) with the command that produced it.

## Start here

- **[docs/reference/](docs/reference/README.md)** — every source file: what it computes, API and shapes, layouts,
  numerics, invariants, tests, tunables, how to change it. Start with its README (data flow of one forward pass).
- [docs/CODE.md](docs/CODE.md) — code rules: where files go, extension points, kernel and test rules.
- **[docs/PLAN-20261003-x31-tg140-pp4000.md](docs/PLAN-20261003-x31-tg140-pp4000.md)** — current plan: X3.1 weights that stop the thinking loops (no K1 holes, rates from agent traffic), 1 GPU + 64 GB RAM, decode 140 / prefill 4,000 tok/s at 256K (budgets, phases A-D, gates, kill rules).
- [docs/PLAN-20261002-tg100-pp3000.md](docs/PLAN-20261002-tg100-pp3000.md) — previous plan (decode 100 / prefill 3,000; its §5 results are the new plan's starting point).
- TRACKER.md (kept locally, internal-only — not in this repo) — every experiment and measurement, checkpoints (CP0–CP9), the "Do not repeat" list.
- [docs/06-truss.md](docs/06-truss.md) — design; `docs/01..05`, `PLAN.md` — the evidence and plans that led here.

## Quickstart

```bash
cmake -S . -B build -G Ninja && cmake --build build
M=~/ML_projects/flashnext/20260918_ngram_q8/data/qwen38-flash-next-paw-x3-q8_0-00001-of-00002.gguf
TOK=/data/www/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/models/Qwen3.8-27B-EXL3-3.5bpw/tokenizer.json
CUDA_VISIBLE_DEVICES=2 python3 -m server.app --model $M --tokenizer $TOK --port 8090 --n-ctx 65536 --chunk 8192
curl localhost:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64,"temperature":0}'
```

Tests and benchmarks: [docs/reference/tests-tools.md](docs/reference/tests-tools.md).

## Credits

TRUSS is an independent, from-scratch implementation (no shared code, no fork), but two projects shaped
how it's built:

- **[Strata](https://github.com/Niko1221/Strata)** (Niko1221, MIT) is where the serving architecture comes
  from. Running a 125B MoE on one consumer GPU means most experts can't live in VRAM, and Strata's answer —
  keep the hot experts on the GPU, compute the rest on the CPU from RAM, hand off between them through a
  flag in pinned memory instead of a blocking copy, and recover the lost speed with speculative decoding —
  is the shape TRUSS's engine follows and the bar its benchmarks are measured against. The specific
  techniques (CPU-resident experts, the async doorbell handoff, adaptive/usage-ranked residency, an int8 KV
  layout, accepting speculative drafts row-by-row) were studied from Strata's engine and reimplemented here
  independently, not copied: see `docs/06-truss.md` and `docs/reference/` for what each piece does in TRUSS
  and how it differs.
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** (turboderp, MIT) is where the weight codec
  comes from. Its EXL3 format (a QTIP-style trellis/Viterbi quantizer) is the basis for the X3 / X3.1
  codec that compresses this model's experts to ~2.6 bits/weight on GPU; TRUSS's decode kernels and its
  encoder's refit/drift step port that design (`docs/reference/kernels-trellis-moe.md`,
  `docs/reference/formats-core.md`).

## Layout

```
include/truss/truss.h      C API (libtruss.so)
src/formats/               GGUF reader, trellis expert tables
src/core/                  device tensors, scratch, error checks
src/kernels/<op>/          fast kernels (trellis, moe, dense, dsa, gdn, hc, ffn, ple, sampling) + reference/
src/model/qwen4exp/        config, weight binding, PLE hashing, reference forward, fast Forward
src/runtime/               expert residency and PCIe streaming
src/api/                   C API implementation
src/encode/                pack-time trellis encoders
server/                    OpenAI-compatible server (Python, ctypes)
tests/unit, tests/layer    op tests; model blocks and chains vs llama-paw and the fp32 reference
tools/                     benchmarks, parity dumps and KL, pack inspection, codec lab
```
