# Serving PAW-125B-FLASH-NEXT-X3.1 with TRUSS (best settings)

Model: [lackonendes/PAW-125B-FLASH-NEXT-X3.1](https://huggingface.co/lackonendes/PAW-125B-FLASH-NEXT-X3.1)
(Qwen3.8-Flash-Next, 125B MoE). One or two 24 GB GPUs plus CPU, 262,144-token context, OpenAI-compatible API.

## What you need

| | |
|---|---|
| GPU | one 24 GB NVIDIA card, Ampere (sm_86), or two for about twice the speed. Tested on RTX 3090s. CUDA 12.x |
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

### Two GPUs (layer split, about twice as fast)

```sh
MODEL_DIR=/nvme/flashnext TOKENIZER=/nvme/flashnext/tokenizer.json GPUS=0,1 PORT=8080 tools/serve/serve_flashnext_x31.sh
```

`GPUS=<a>,<b>` splits the model by layers: the first card runs layers 24-47, the output head and the MTP drafter, the
second runs layers 0-23. No NVLink is needed; the cards hand one small activation buffer to each other per step.
On top of the one-GPU settings the script sets:

```sh
export CUDA_VISIBLE_DEVICES=0,1
export TRUSS_LAYER_SPLIT=24 TRUSS_LAYER_SPLIT_DEVICE=1   # layers 0-23 on the second card
export TRUSS_HINT_MIN_ROWS=8                              # next-layer expert prefetch hints on full draft windows only
export TRUSS_HC_SILU=1 TRUSS_HC_COLLAPSE=1 TRUSS_ROUTE_PUBLISH=1   # fused decode kernels (same output, fewer launches)
# and --drafts 7 instead of 3: a longer draft window costs little when two cards share the work
```

With two GPUs the CPU expert tier is optional: `TRUSS_CPU_TRELLIS=0` sends every expert that is not on a GPU over
PCIe instead. Decode is about 6% slower (8-token pass 36.2 ms instead of 34.0), but it frees the CPU cores and
makes greedy output reproducible run to run. The experts that do not fit on the GPUs still live in RAM either way.

Test it:

```sh
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

Endpoints: `/v1/chat/completions` and `/v1/completions` (streaming, tool calling, `temperature`, `top_p`, `top_k`,
`min_p`). Thinking is on by default; send `"chat_template_kwargs": {"enable_thinking": false}` to turn it off. For
thinking mode Qwen recommends `temperature 1.0, top_p 0.95, top_k 20`; for direct answers `temperature 0.7,
top_p 0.8, top_k 20`. Requests are handled one at a time.

### Images (vision)

The vision encoder ships as `flashnext-mmproj-f16.gguf` (0.84 GiB: the Qwen3.8-Flash-Next ViT in f16). Pass the script
`VIT=` and the server accepts images; put the ViT on a card of its own when you can:

```sh
hf download lackonendes/PAW-125B-FLASH-NEXT-X3.1 flashnext-mmproj-f16.gguf --local-dir /nvme/flashnext
MODEL_DIR=/nvme/flashnext TOKENIZER=/nvme/flashnext/tokenizer.json GPUS=0,1 PORT=8080 \
  VIT=/nvme/flashnext/flashnext-mmproj-f16.gguf VIT_GPU=1 tools/serve/serve_flashnext_x31.sh
```

`VIT_GPU` is an index into `CUDA_VISIBLE_DEVICES` (default: the last card). The ViT needs about 1.5 GiB. An image goes in
as an `image_url` part, a data URI or an http(s) URL (local paths need `IMAGES=1`), one image per request:

```sh
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d "{
  \"messages\":[{\"role\":\"user\",\"content\":[
    {\"type\":\"text\",\"text\":\"Describe this image.\"},
    {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$(base64 -w0 pic.png)\"}}]}],
  \"max_tokens\":256}"
```

The picture is resized to a 32-pixel patch grid, long side at most 1344 (up to 1,764 image tokens; 448x448 is 196), and
read before the text, so an image request never reuses a previous prefix. One RTX 3090: a 448x448 image plus a short
question prefills in about 2.0-2.4 s, a 1344x1344 one (1,831 prompt tokens) in 4.6 s. Thinking eats the first tokens,
so allow 200+. A
server started without `VIT=` answers image requests with a 400 and serves text as before. The equivalent flags are
`--mmproj <file> --mmproj-device <index> [--allow-local-images]`.

## What each setting does

| setting | what it does |
|---|---|
| `--expert-usage data/usage_strata_rank.f32` | keeps the most-used experts on the GPU (ships in the repo). Without it decode is slower |
| `--mtp ... --drafts 3`, `--draft-vocab`, `--draft-min-p 0.5` | speculative decoding with the MTP head: about 2.7 tokens per step instead of 1 |
| `TRUSS_CPU_TRELLIS=1`, `TRUSS_CPU_DYNAMIC=1` | experts that are not on the GPU run on the CPU or are copied over PCIe, whichever finishes first |
| `TRUSS_CPU_THREADS`, `TRUSS_CPU_PIN=1` | CPU threads for those experts, pinned to cores. Use your physical core count minus 2 |
| `TRUSS_ACT` (default `fp16`) | 16-bit activations in the dense layers: KL 0.040 against BF16 instead of 0.065 with the 8-bit path, at the same decode speed (a fused kernel reads the int8 weights once). `TRUSS_ACT=q8` selects the 8-bit path: prefill about 4% faster |
| `TRUSS_KV_INT8=1` | int8 KV cache: what makes 262,144 tokens fit in 24 GB |
| `TRUSS_KV_LEND=1` | lends unused KV memory to the expert cache while the context is short |
| `TRUSS_ADMIT_IDLE=64`, `TRUSS_PCIE_FRAC=0.2`, `TRUSS_HINT_K=3` | expert-cache and PCIe tuning for decode |
| `TRUSS_FETCH_PROMPT=96`, `TRUSS_SPLIT_ROWS=2048`, `--chunk 8192` | prompt (prefill) tuning: prompts of 96+ tokens use the streaming path, in 8,192-token chunks |
| `VIT=...`, `VIT_GPU` | image input: loads the mmproj ViT (0.84 GiB) on its own card; `IMAGES=1` allows local image paths |

## Memory at 262,144 context

One GPU: about 23.6 GiB (dense weights 4.6, experts 13.5, KV and state 3.9, buffers). Two GPUs: about 23.5 GiB on
each card, with 13,466 of the 24,576 routed experts resident on the GPUs. RAM: about 43 GB resident with one GPU,
about 25 GB with two (fewer experts live in RAM). If the GPU is
shared with a desktop or another job, lower `NCTX` (the KV cache shrinks) or the start may fail with an out-of-memory error.

## Expected speed (RTX 3090s, Threadripper 3960X)

| | two RTX 3090s | one RTX 3090 (250 W) |
|---|---:|---:|
| decode, short prompt | not measured | peak 80+ tok/s (best 87.5) |
| decode, 150K-token context (greedy) | 128 tok/s | 57-60 tok/s |
| decode, real agent sessions (temperature 0.8, thinking, tools, up to 150K context) | 115-116 tok/s | - |
| prefill, fresh 150K-token prompt | about 4,500 tok/s | 2,170-2,270 tok/s |

Decode varies by about 10% on a busy machine. The CPU tier shares cores with whatever else is running, so close
other heavy jobs for the best numbers.

## Troubleshooting

- **Slow decode:** check `TRUSS_CPU_THREADS` is set, nothing else is saturating the CPU, and the model files are on NVMe.
- **Out of memory at start:** lower `--n-ctx`, or free the GPU(s). With two GPUs both cards need about 23.5 GiB free.
- **`build/libtruss.so` not found:** build first, or point `TRUSS_LIB` at the library.
- **Greedy output differs slightly between runs:** expected; which experts run on the CPU or GPU depends on timing.
