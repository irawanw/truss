# Serving PAW-125B-FLASH-NEXT-X3.1 with TRUSS (best settings)

Model: [lackonendes/PAW-125B-FLASH-NEXT-X3.1](https://huggingface.co/lackonendes/PAW-125B-FLASH-NEXT-X3.1)
(Qwen3.8-Flash-Next, 125B MoE). One 24 GB GPU plus CPU, 262,144-token context, OpenAI-compatible API.

## What you need

| | |
|---|---|
| GPU | one 24 GB NVIDIA card, Ampere (sm_86). Tested on an RTX 3090 only. CUDA 12.x |
| CPU | AVX2, 12+ cores (tested on a 24-core Threadripper 3960X) |
| RAM | 64 GB. The server uses about 43 GB; leave room for the OS and your other programs |
| Disk | 97 GB on **NVMe**. The 47.7 GiB n-gram table is read from disk on every step, so a spinning or USB disk is slow |
| Software | Linux, CMake 3.24+, Ninja, Python 3.10+ (`fastapi uvicorn anyio numpy transformers torch`) |

## Install

```sh
git clone https://github.com/irawanw/truss && cd truss
cmake -S . -B build -G Ninja && cmake --build build
pip install fastapi uvicorn anyio numpy transformers torch huggingface_hub

hf download lackonendes/PAW-125B-FLASH-NEXT-X3.1 flashnext-x31-00001-of-00002.gguf flashnext-x31-00002-of-00002.gguf flashnext-mtp-x3k3.gguf --local-dir /nvme/flashnext
hf download Qwen/Qwen3.8-Flash-Next tokenizer.json --local-dir /nvme/flashnext
```

## Run (best settings)

```sh
MODEL_DIR=/nvme/flashnext TOKENIZER=/nvme/flashnext/tokenizer.json GPU=0 PORT=8080 tools/serve/serve_flashnext_x31.sh
```

The script sets the environment and starts the server. `DRY=1` prints the command without starting it. Loading takes
about 2 minutes. The equivalent by hand:

```sh
export CUDA_VISIBLE_DEVICES=0
export TRUSS_CPU_TRELLIS=1 TRUSS_CPU_DYNAMIC=1 TRUSS_PCIE_FRAC=0.2 TRUSS_HINT_K=3
export TRUSS_CPU_THREADS=22 TRUSS_CPU_PIN=1          # your physical cores minus 2
export TRUSS_KV_INT8=1 TRUSS_KV_LEND=1 TRUSS_ADMIT_IDLE=64 TRUSS_FETCH_PROMPT=96 TRUSS_SPLIT_ROWS=2048

python3 -m server.app \
  --model /nvme/flashnext/flashnext-x31-00001-of-00002.gguf --tokenizer /nvme/flashnext/tokenizer.json \
  --n-ctx 262144 --chunk 8192 \
  --expert-usage data/usage_strata_rank.f32 \
  --mtp /nvme/flashnext/flashnext-mtp-x3k3.gguf --drafts 3 \
  --draft-vocab data/draft_vocab_en.bin --draft-min-p 0.5 \
  --host 127.0.0.1 --port 8080
```

Test it:

```sh
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

Endpoints: `/v1/chat/completions` and `/v1/completions` (streaming, tool calling, `temperature`, `top_p`, `top_k`,
`min_p`). Thinking is on by default; send `"chat_template_kwargs": {"enable_thinking": false}` to turn it off. For
thinking mode Qwen recommends `temperature 1.0, top_p 0.95, top_k 20`; for direct answers `temperature 0.7,
top_p 0.8, top_k 20`. Requests are handled one at a time.

## What each setting does

| setting | what it does |
|---|---|
| `--expert-usage data/usage_strata_rank.f32` | keeps the most-used experts on the GPU (ships in the repo). Without it decode is slower |
| `--mtp ... --drafts 3`, `--draft-vocab`, `--draft-min-p 0.5` | speculative decoding with the MTP head: about 2.7 tokens per step instead of 1 |
| `TRUSS_CPU_TRELLIS=1`, `TRUSS_CPU_DYNAMIC=1` | experts that are not on the GPU run on the CPU or are copied over PCIe, whichever finishes first |
| `TRUSS_CPU_THREADS`, `TRUSS_CPU_PIN=1` | CPU threads for those experts, pinned to cores. Use your physical core count minus 2 |
| `TRUSS_KV_INT8=1` | int8 KV cache: what makes 262,144 tokens fit in 24 GB |
| `TRUSS_KV_LEND=1` | lends unused KV memory to the expert cache while the context is short |
| `TRUSS_ADMIT_IDLE=64`, `TRUSS_PCIE_FRAC=0.2`, `TRUSS_HINT_K=3` | expert-cache and PCIe tuning for decode |
| `TRUSS_FETCH_PROMPT=96`, `TRUSS_SPLIT_ROWS=2048`, `--chunk 8192` | prompt (prefill) tuning: prompts of 96+ tokens use the streaming path, in 8,192-token chunks |

## Memory at 262,144 context

GPU about 23.6 GiB (dense weights 4.6, experts 13.5, KV and state 3.9, buffers). RAM about 43 GB. If the GPU is
shared with a desktop or another job, lower `NCTX` (the KV cache shrinks) or the start may fail with an out-of-memory error.

## Expected speed (RTX 3090, 250 W, Threadripper 3960X)

| | |
|---|---:|
| decode, short prompt | peak 80+ tok/s (best 87.5) |
| decode, 150K-token context | 57-60 tok/s |
| prefill, fresh 150K-token prompt | 2,170-2,270 tok/s |

Decode varies by about 10% on a busy machine. The CPU tier shares cores with whatever else is running, so close
other heavy jobs for the best numbers.

## Troubleshooting

- **Slow decode:** check `TRUSS_CPU_THREADS` is set, nothing else is saturating the CPU, and the model files are on NVMe.
- **Out of memory at start:** lower `--n-ctx`, or free the GPU.
- **`build/libtruss.so` not found:** build first, or point `TRUSS_LIB` at the library.
- **Greedy output differs slightly between runs:** expected; which experts run on the CPU or GPU depends on timing.
