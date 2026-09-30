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

Every number has a row in [TRACKER.md](TRACKER.md) with the command that produced it.

## Start here

- **[docs/reference/](docs/reference/README.md)** — every source file: what it computes, API and shapes, layouts,
  numerics, invariants, tests, tunables, how to change it. Start with its README (data flow of one forward pass).
- [docs/CODE.md](docs/CODE.md) — code rules: where files go, extension points, kernel and test rules.
- [TRACKER.md](TRACKER.md) — every experiment and measurement, checkpoints (CP0–CP9), the "Do not repeat" list.
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
