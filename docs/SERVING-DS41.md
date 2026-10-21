# Serving PAW-763B-X3 with TRUSS

Model: [lackonendes/PAW-763B-X3](https://huggingface.co/lackonendes/PAW-763B-X3) (DeepSeek-V4.1-Flash-1017 with 256 of 384
experts per layer kept, X3 trellis codec). Two 24 GB GPUs plus CPU, 262,144-token context, OpenAI-compatible API on
port 8115. Measured speed, KL and the open issues are in the model card
([docs/hf/PAW-763B-X3.md](hf/PAW-763B-X3.md)); this page is how to run it.

## What you need

| | |
|---|---|
| GPU | **two** 24 GB NVIDIA cards, Ampere (sm_86), CUDA 12.x. Tested on 2 x RTX 3090. One GPU is not supported yet |
| CPU / RAM | AVX2, 16+ cores; **128 GB RAM**: about 82 GB of cold experts stay in pinned RAM (a kept copy, see below) |
| Disk | 215 GB on **NVMe**. The two n-gram tables (104 GB) are read row by row on every step |
| Software | Linux, CMake 3.24+, Ninja, Python 3.10+ (`fastapi uvicorn anyio numpy transformers torch huggingface_hub`) |

## Install

```sh
git clone https://github.com/irawanw/truss && cd truss
cmake -S . -B build-dsv41 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build-dsv41
pip install fastapi uvicorn anyio numpy transformers torch huggingface_hub

hf download lackonendes/PAW-763B-X3 --local-dir /nvme/ds41      # 215 GB
```

## Serve

```sh
export TRUSS_MODEL_DIR=/nvme/ds41            # the download above (flat layout of the repo)
export TRUSS_HOME=$HOME/ds41-logs            # logs and the kept-copy log
export CUDA_VISIBLE_DEVICES=0,1              # the two cards (the layer split uses both)
scripts/serve_ds41.sh                        # http://127.0.0.1:8115/v1/chat/completions, model name ds41-flash
curl localhost:8115/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"ds41-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

`scripts/serve_ds41.sh` sources `scripts/ds41_env.sh`, which holds the whole configuration; the benchmark
(`scripts/bench_ds41.sh`) sources the same file, so what is measured is what is served. The first load reads the 93 GB
of experts from disk; the first request after it is slower (page-locking the cold copy, about 20 s).

## The settings and why

| setting | value | why |
|---|---|---|
| `TRUSS_LAYER_SPLIT` | 20 | layers 0-19 on one card, 20-39 and the drafter on the other (2, 8, 14 or 20 are allowed) |
| `TRUSS_SKIP_WEIGHT` | 0.10 | drops the lowest-weight missed experts; 0.20 is faster (+10 tok/s) but KL 0.104 vs 0.069 |
| `--drafts` | 3 | the DSpark drafter; 2 and 4 were slower at this skip |
| `--n-ctx` / `--chunk` | 262144 / 8192 | 16384-token chunks are slower and need 7.5 GB more per card |
| `TRUSS_TAIL_PREFILL` | 134 | consumer layers read only the last 134 rows of a prompt chunk (+29% prefill, KL +0.001) |
| `TRUSS_FETCH_PROMPT` | 96 | prompts shorter than this run on the decode path instead of streaming every cold expert |
| `TRUSS_COLD_KEEP` | socket | a small keeper process holds the cold copy in RAM, so a server restart maps it instead of reading 80 GB. `0` turns it off; free its RAM with `tools/cold_keeper.py --drop` |

Sampling defaults the server applies when a request leaves a field unset: temperature 1.0, top-p 1.0, top-k off,
min-p 0.05. `reasoning_effort` is read from the request (`minimal` = low, `medium` = high, `xhigh` = max); `high` is
the default and the best choice here, `low` looped more. A loop guard in `server/generate.py` closes a repeating
reasoning stretch and ends a repeating answer; `TRUSS_LOOP_GUARD=0` turns it off.

## Benchmark

```sh
scripts/bench_ds41.sh --tasks chat,code,long,kl --kl-seqs 8 --repeat 3 --arm base --arm d4:drafts=4
```

One model load, many arms (`--arm name:option=value`), health checks (finite logits, drafter acceptance, recall, KL).
Details: [docs/reference/tests-tools.md](reference/tests-tools.md). The bench reads frozen prompt files from the
development tree (`data/bench_in`), which are not part of the download.

## Notes

- The pack and the codec are described in [docs/reference/model-deepseek41.md](reference/model-deepseek41.md).
- The plan toward 70 tok/s decode and 2,000 tok/s prefill, with every measurement:
  [docs/PLAN-20261021-ds41-70-2000.md](PLAN-20261021-ds41-70-2000.md).
- Flash-Next (the other model TRUSS serves): [docs/SERVING-FLASHNEXT-X31.md](SERVING-FLASHNEXT-X31.md).
