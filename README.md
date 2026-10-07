# TRUSS (trellis-kernel)

Our own inference engine for **trellis-quantized MoE models on one RTX 3090** (Ampere, sm_86). First model:
**Qwen3.8 Flash-Next PAW X3.1** (qwen4exp, 48 layers, 512 experts), served through an OpenAI-compatible API.

## Status (2026-10-07)

PAW-125B-FLASH-NEXT-X3.1 (Qwen3.8-Flash-Next, 125B MoE, 24,576 experts at 3.5 / 2.5 bit, Q8_0 dense, 47.7 GiB n-gram
table read from NVMe) on one RTX 3090 at 250 W, Threadripper 3960X:

| | |
|---|---|
| decode, short prompt | peak 80+ tok/s |
| decode, 150K-token context | 57-60 tok/s |
| prefill, fresh 150K-token prompt | 2,170-2,270 tok/s |
| context | 262,144 tokens |
| memory | GPU 23.6 GiB, RAM 43 GB, NVMe 97 GB |

Decode varies by about 10% on a shared machine. Model card: [PAW-125B-FLASH-NEXT-X3.1](https://huggingface.co/lackonendes/PAW-125B-FLASH-NEXT-X3.1).
Every number has a row in TRACKER.md (kept locally, internal-only, not in this repo) with the command that produced it.

## Start here

- **[docs/reference/](docs/reference/README.md)** — every source file: what it computes, API and shapes, layouts,
  numerics, invariants, tests, tunables, how to change it. Start with its README (data flow of one forward pass).
- [docs/CODE.md](docs/CODE.md) — code rules: where files go, extension points, kernel and test rules.
- **[docs/PLAN-20261003-x31-tg140-pp4000.md](docs/PLAN-20261003-x31-tg140-pp4000.md)** — current plan: X3.1 weights that stop the thinking loops (no K1 holes, rates from agent traffic), 1 GPU + 64 GB RAM, decode 140 / prefill 4,000 tok/s at 256K (budgets, phases A-D, gates, kill rules).
- [docs/PLAN-20261002-tg100-pp3000.md](docs/PLAN-20261002-tg100-pp3000.md) — previous plan (decode 100 / prefill 3,000; its §5 results are the new plan's starting point).
- TRACKER.md (kept locally, internal-only — not in this repo) — every experiment and measurement, checkpoints (CP0–CP9), the "Do not repeat" list.
- [docs/06-truss.md](docs/06-truss.md) — design; `docs/01..05`, `PLAN.md` — the evidence and plans that led here.

## Quickstart

Serve **PAW-125B-FLASH-NEXT-X3.1** ([model on Hugging Face](https://huggingface.co/lackonendes/PAW-125B-FLASH-NEXT-X3.1))
on one 24 GB GPU plus CPU: **[docs/SERVING-FLASHNEXT-X31.md](docs/SERVING-FLASHNEXT-X31.md)** (install, download, the
best settings, memory, expected speed). Short version:

```bash
cmake -S . -B build -G Ninja && cmake --build build
MODEL_DIR=/nvme/flashnext TOKENIZER=/nvme/flashnext/tokenizer.json GPU=0 tools/serve/serve_flashnext_x31.sh
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
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
