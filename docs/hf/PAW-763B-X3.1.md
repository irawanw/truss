---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: gguf
pipeline_tag: image-text-to-text
tags:
- quantization
- trellis-codec
- moe
- paw
- gguf
- deepseek
- vision
---

# PAW-763B-X3.1

_If this model is useful to you, support the work by following [@nawariokra](https://x.com/nawariokra) on X._

**[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (763B parameters,
counting its n-gram tables) on two 24 GB GPUs plus CPU, with the full 262,144-token context, all 384 experts and image
input.** It runs on our own engine [TRUSS](https://github.com/irawanw/truss). Stock llama.cpp cannot read these files.

What changed from [PAW-763B-X3](https://huggingface.co/lackonendes/PAW-763B-X3):

- **All 384 routed experts per layer** (X3 kept 256). The 128 rarest experts of each layer are "far": they stay on the
  NVMe and are read on demand. Serving can use them (E2, best quality) or route around them (E1, faster).
- **Experts chosen and coded for reasoning**: which experts are hot and how many bits each gets were picked by their
  measured damage to the model's own long reasoning (not by how often they are used).
- **Image input**: DeepSeek's vision tower and aligner (`ds41-mmproj-f16.gguf`), any number of images per request.

| | |
|---|---|
| source model | DeepSeek-V4.1-Flash-1017: 40 MoE layers, 384 routed experts + 1 shared, compressed KV, 2 n-gram ("Engram") tables, DSpark drafter, vision tower. 763.2 B parameters: 557.2 B routed experts, 196.9 B n-gram tables, 8.6 B dense, 0.5 B vision |
| experts | all 384 per layer, trellis codec "X3": 256 per layer in VRAM / RAM at 1-4 bit (2.24 bit average, 2.93 weighted by traffic), 128 rare ones per layer on the NVMe at 2 bit |
| dense | Q8_0, plus an X3 trellis overlay for the attention and shared tensors |
| n-gram tables | 2 x 384 M rows, int4, 55 GB each, read row by row from NVMe |
| vision | DeepSeek's ViT (32 blocks) + 3x3 aligner, f16, and the image router bias of every layer |
| drafter | the model's own DSpark drafter, 5.8 GB |
| context | 262,144 tokens |
| hardware tested | 2 x RTX 3090 (24 GB), 128 GB RAM, NVMe |

## Files

| file | size | what |
|---|---:|---|
| `ds41-dense.gguf` | 9.2 GB | dense weights, Q8_0 (the model file the server opens) |
| `ds41-dense-x3.gguf` (+ `.layers.json`) | 2.3 GB | X3 overlay for attention / shared tensors |
| `ds41-experts.gguf` (+ `.layers.json`) | 148 GB | all 384 routed experts per layer, X3 |
| `far.u8` | 15 KB | which experts are far (40 x 384, 1 = far) |
| `usage.f32` | 60 KB | which experts are hot (the residency plan's starting point) |
| `ds41-engram-L1.gguf`, `ds41-engram-L14.gguf` | 55 GB each | n-gram tables (NVMe, not loaded into RAM) |
| `ds41-mmproj-f16.gguf` | 0.97 GB | vision tower, aligner, image router bias |
| `drafter/` | 5.8 GB | DSpark drafter: dense part and its experts |
| `tokenizer/` | 7 MB | tokenizer from the base model |

About 277 GB in all. Serving steps: [docs/SERVING-DS41.md](https://github.com/irawanw/truss/blob/main/docs/SERVING-DS41.md)
(`TRUSS_MODEL_DIR=<download> scripts/serve_ds41.sh`; E1 by default, `TRUSS_FAR_DROP=0` for E2). Needs about 100 GB of
free RAM.

## Measured (2026-10-25, 2 x RTX 3090, TRUSS `main`, no expert skipping)

| | E1 (far experts off) | E2 (all 384) |
|---|---|---|
| KL against the original checkpoint's logits (8 code sequences) | 0.203, top-1 88.6% | **0.071**, top-1 93.5% |
| decode, a real agent session | 36.5 tok/s (early in a session, one sample) | 20-24 tok/s at 150-190K context |
| prefill | about 960 tok/s (8.7K prompt) | about 1,050 tok/s (150K prompt) |
| far expert reads | none | 1 MiB pieces, 8 readers, 6 GB of pinned slots |

E2 needs no expert skipping to stay under our KL budget (0.0728); E1 is faster but its KL is about 3x higher. The
previous pack (X3, 256 experts, skip 0.10) measured 0.061-0.069.

Image input: the vision tower matches DeepSeek's reference (f32) at cosine 0.99999 (DeepSeek's own bf16 run: 0.9986-
0.9994) and takes 58-249 ms per image on a 3090; the two example images of DeepSeek's repo and a screenshot's text came
back right. Images are sent as OpenAI `image_url` parts; a re-sent image is reused from the server's cache.

KL is the engine's served path against the original FP4 checkpoint's logits (`tools/tk-bench/arms.py`,
`tools/tk-parity/ds41_kl.cu`; TRUSS `docs/reference/model-deepseek41.md`). Speed varies by a few percent from run to run
on a shared machine. Sampling defaults: temperature 1.0, top-p 1.0, min-p 0.05.

## Known issues (pending)

- **Decode is below our 40 tok/s goal in E2.** Each pass waits on the CPU tier (cold experts in RAM) and, in E2, on NVMe
  reads of far experts (about 10% of the time with 6 GB of slots).
- **Prompts route around the far experts** in both modes (only decode uses them in E2), so image and long prompts are
  read as E1 does.
- **Vision costs VRAM**: the tower takes about 1.5 GB of the first card's expert budget.
- **Quality evidence is KL, recall and small task checks only.** No public task benchmarks on this pack yet.
- Needs two Ampere 24 GB GPUs; one GPU and other cards are not supported yet.

## Credits

TRUSS and this pack stand on other people's open work. TRUSS is its own implementation, but these projects gave us the
ideas and the numbers to beat:

- **[Strata](https://github.com/Niko1221/Strata)** (Niko1221, MIT): the serving design: hot experts on the GPU, the rest
  on the CPU, a doorbell handoff in pinned memory, speculative decoding; and
  **[mmnga/Strata](https://github.com/mmnga/Strata)**, whose DeepSeek port we used to cross-check tensor names.
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** (turboderp, MIT) and the **QTIP** paper
  ([arXiv 2406.11235](https://arxiv.org/abs/2406.11235)): the trellis weight codec behind "X3".
- **[DeepInfer](https://github.com/deepseek-ai/DeepInfer)** (DeepSeek, Apache-2.0): the reference for the model's
  architecture, e.g. YaRN only on the compressed layers.
- **[tqLLM](https://github.com/turbo-tan/tqllm)** (turbo-tan, Apache-2.0): trellis GEMV kernel and retrieval-draft ideas.
- **[TensorFold](https://github.com/ashhart/TensorFold)** (ashhart): exact DSpark speculation and pipelined prompt reading.
- **[Maya](https://github.com/mw00/project-maya)** (mw00): large prompt chunks, experts borrowing VRAM during prompts,
  the long-prompt skip we compare our tail prefill with.
- **[sybil-solutions/dsv41-flash-offload](https://github.com/sybil-solutions/dsv41-flash-offload)** (MIT): staged-DMA
  prefill and FP16 GEMM for experts with many rows.
- **[dsx](https://github.com/lxe/dsx)** (lxe): deferring low-rank CPU experts by one layer, and reading the coldest
  experts from disk with O_DIRECT (our far tier).
- **[Xolotl](https://github.com/Indras-Mirror/Xolotl)** (Indras-Mirror): moving the draft model off the card during
  long prompts.
- **[PolyStrata](https://github.com/VecSzn/PolyStrata)** (VecSzn): tensor-core attention pattern.
- Base model: **DeepSeek-V4.1-Flash-1017** by DeepSeek (MIT), including its vision tower and reference image
  preprocessing. This pack is a derivative of it under the same license.
