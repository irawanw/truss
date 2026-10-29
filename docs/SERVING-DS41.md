# Serving PAW-763B-X3.1 / PAW-763B-X3 with TRUSS

Models: [lackonendes/PAW-763B-X3.1](https://huggingface.co/lackonendes/PAW-763B-X3.1) (DeepSeek-V4.1-Flash-1017 with all
384 experts per layer, the 128 rarest per layer on the NVMe, and image input; card [docs/hf/PAW-763B-X3.1.md](hf/PAW-763B-X3.1.md))
and the older [lackonendes/PAW-763B-X3](https://huggingface.co/lackonendes/PAW-763B-X3) (256 of 384 experts kept, text
only; card [docs/hf/PAW-763B-X3.md](hf/PAW-763B-X3.md)). Two 24 GB GPUs plus CPU, 262,144-token context, OpenAI-compatible
API on port 8115; this page is how to run them. `scripts/ds41_env.sh` sees which one `TRUSS_MODEL_DIR` holds (X3.1 has
`far.u8`) and sets its measured configuration: X3.1 serves E1 by default (far experts not routed, no NVMe reads,
faster); `TRUSS_FAR_DROP=0` serves E2 (all 384 experts, KL 0.071 vs 0.203), and images are on when
`ds41-mmproj-f16.gguf` is there.

## What you need

| | |
|---|---|
| GPU | **two** 24 GB NVIDIA cards, Ampere (sm_86), CUDA 12.x. Tested on 2 x RTX 3090. One GPU is not supported yet |
| CPU / RAM | AVX2, 16+ cores; **128 GB RAM**: about 82 GB of cold experts stay in pinned RAM (a kept copy, see below) |
| Disk | 230 GB on **NVMe**. The two n-gram tables (111 GB) are read row by row on every step |
| Software | Linux, CMake 3.24+, Ninja, Python 3.10+ (`fastapi uvicorn anyio numpy transformers torch huggingface_hub`) |

## Install

```sh
git clone https://github.com/irawanw/truss && cd truss
cmake -S . -B build-dsv41 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build-dsv41
pip install fastapi uvicorn anyio numpy transformers torch huggingface_hub

hf download lackonendes/PAW-763B-X3.1 --local-dir /nvme/ds41    # 277 GB (PAW-763B-X3: 227 GB)
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
(`scripts/bench_ds41.sh`) sources the same file, so what is measured is what is served. The first load reads the 99 GB
of experts from disk; the first request after it is slower (page-locking the cold copy, about 20 s). A pack under test can replace the served files without editing it: `TRUSS_EXPERT_OVERLAY` (the overlay list), `TRUSS_DENSE` (the dense GGUF) and `TRUSS_USAGE` (the expert usage table); unset = the served files.

## The settings and why

| setting | value | why |
|---|---|---|
| `TRUSS_LAYER_SPLIT` | 20 | layers 0-19 on one card, 20-39 and the drafter on the other (2, 8, 14 or 20 are allowed) |
| `TRUSS_SKIP_WEIGHT` | 0.10 | drops the lowest-weight missed experts; 0.20 is faster (+10 tok/s) but KL 0.104 vs 0.069 |
| `--drafts` | 3 | the DSpark drafter; 2 and 4 were slower at this skip |
| `--n-ctx` / `--chunk` | 262144 / 8192 | 16384-token chunks are slower and need 7.5 GB more per card |
| `TRUSS_TAIL_PREFILL` | 134 | consumer layers read only the last 134 rows of a prompt chunk (+29% prefill, KL +0.001) |
| `TRUSS_FETCH_PROMPT` | 96 | prompts shorter than this run on the decode path instead of streaming every cold expert |
| `TRUSS_FAR` | unset | far set file (40 x 384 uint8, 1 = far): those experts have no RAM copy and are read from the NVMe on demand into pinned slots (pack E2; docs/PLAN-20261024-ds41-nvme-tier.md). Prompt chunks route around them. X3.1 = pack E2; with TRUSS_FAR_DROP=0 (E2) KL 0.071 vs 0.203 (E1), 20-24 vs ~36 tok/s decode (TRACKER 283-289) |
| `TRUSS_FAR_CACHE_GB` / `TRUSS_FAR_THREADS` / `TRUSS_FAR_WAIT_US` / `TRUSS_FAR_DROP` | 6 / 8 / 2000 / 0 | far tier: pinned slot GB (whole run, split by the stages' far share), NVMe reader threads, how long a decode layer waits for its far reads before skipping them, 1 = never route to the far set (the pack without it) |
| `TRUSS_FAR_PIECE_KB` | 1024 | far reads are cut into pieces of this size that all readers take (queue depth 8 for one expert: 10-25 per-miss wait 4.4 -> 3.0 ms); 0 = one reader per expert |
| `TRUSS_HINT_FAR_ONLY` | 0 | 1: the pre-gated hints (`TRUSS_HINT_K`) only start far reads. Measured 10-25 on E2: no help (the next layer's router does not predict the rare experts: misses unchanged, 18.5 vs 20.1 tok/s), so hints stay off |
| `TRUSS_MMPROJ` / `TRUSS_MMPROJ_DEVICE` | unset / 0 | image input: `ds41-mmproj-f16.gguf` (`tools/tk-pack/ds41_mmproj.py`, 0.97 GB: DeepSeek's vision tower, aligner and image router bias). The tower opens on that card before the model and takes ~1.5 GB of its expert budget. OpenAI `image_url` parts (data URI or http; local paths need `--allow-local-images`), any number per request; re-sent images are reused from the slot's cache. Details: [model-deepseek41.md "Vision"](reference/model-deepseek41.md) |
| `TRUSS_COLD_KEEP` | socket | a small keeper process holds the cold copy in RAM, so a server restart maps it instead of reading 80 GB. `0` turns it off; free its RAM with `tools/cold_keeper.py --drop` |

Sampling defaults the server applies when a request leaves a field unset: temperature 1.0, top-p 1.0, top-k off,
min-p 0.05. `reasoning_effort` is read from the request (`minimal` = low, `medium` = high, `xhigh` = max); `high` is
the default and the best choice here, `low` looped more. `TRUSS_REASONING_EFFORT` (low / high / max or 1-100) sets
the server's default for requests that name none (10-25: the E2 test server runs `max`). A loop guard in `server/generate.py` closes a repeating
reasoning stretch and ends a repeating answer; `TRUSS_LOOP_GUARD=0` turns it off.

Watching decode speed: the log prints every 3 s `decode N tokens, X tok/s (now Y)` (whole reply, last 3 s), and each
stage every 10 s of decode `decode layers A-B: ... misses cpu C pcie P, far routed ... waited W ..., wait T ms = S ms/s`
(S = ms per second the stage stalled on NVMe reads). `POST /debug/option {"name": "pcie_frac", "value": 0.35}` changes
a decode knob live (drafts up to the open value, skip_weight, pcie_frac, hint_k, hint_far_only, adapt_*); it is applied
before the next engine call and logged as `truss: option ...`.

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
