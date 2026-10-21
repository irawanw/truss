---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: gguf
tags:
- quantization
- trellis-codec
- moe
- paw
- gguf
- deepseek
---

# PAW-763B-X3

_If this model is useful to you, support the work by following [@nawariokra](https://x.com/nawariokra) on X._

**[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (763B parameters,
counting its n-gram tables) on two 24 GB GPUs plus CPU, with the full 262,144-token context.** It runs on our own engine
[TRUSS](https://github.com/irawanw/truss). Stock llama.cpp cannot read these files.

| | |
|---|---|
| source model | DeepSeek-V4.1-Flash-1017: 40 MoE layers, 384 routed experts + 1 shared, compressed KV, 2 n-gram ("Engram") tables, DSpark drafter. 763.2 B parameters: 557.2 B routed experts, 196.9 B n-gram tables, 8.6 B dense, 0.5 B vision |
| what is in this pack | text only (no vision tower). **256 of the 384 routed experts per layer are kept** (the most used); the rest are dropped |
| experts | 73 hot experts per layer at 2.5 bit (3 bit in layers 25-35) and 183 cold experts at 2 bit, trellis codec "X3" |
| dense | Q8_0, plus an X3 trellis overlay for the attention and shared tensors |
| n-gram tables | 2 x 384 M rows, int4, 55 GB each, read row by row from NVMe |
| drafter | the model's own DSpark drafter (3 drafts per step), 5.8 GB |
| context | 262,144 tokens |
| hardware tested | 2 x RTX 3090 (24 GB), 128 GB RAM, NVMe |

## Files

| file | size | what |
|---|---:|---|
| `ds41-dense.gguf` | 9.1 GB | dense weights, Q8_0 (the model file the server opens) |
| `ds41-dense-x3.gguf` (+ `.layers.json`) | 2.3 GB | X3 overlay for attention / shared tensors |
| `ds41-experts.gguf` (+ `.layers.json`) | 99 GB | the 256 kept routed experts per layer, X3 |
| `ds41-engram-L1.gguf`, `ds41-engram-L14.gguf` | 55 GB each | n-gram tables (NVMe, not loaded into RAM) |
| `drafter/` | 5.8 GB | DSpark drafter: dense part and its experts |
| `tokenizer/` | 7 MB | tokenizer from the base model |
| `usage_calib256.f32` | 40 KB | which experts are hot (the residency plan's starting point) |

227 GB in all (211 GiB). Serving steps: [docs/SERVING-DS41.md](https://github.com/irawanw/truss/blob/main/docs/SERVING-DS41.md).

## Measured (2026-10-21, 2 x RTX 3090, TRUSS `main`)

| | |
|---|---|
| decode, bench (short prompts, speculative, 3 drafts) | **29-30.5 tok/s**, about 3 tokens per pass |
| decode, a real agent session at 70-95K context | **20-23 tok/s** |
| prefill, 21K-token prompt | **1,100-1,300 tok/s** (32K prompt: about 1,350) |
| per-turn cost in an agent session | about 6 s before the first token, even for a few hundred new tokens |
| KL against the original checkpoint's logits | **0.061-0.069** (8 sequences: 0.0687; our budget 0.0728); top-1 agreement 93.4-94.1% |
| recall (copying a passage from N tokens back) | 99.6% at 21K; 100% at 60K, 100K and 200K |
| faster, lower quality | expert skipping 0.20 instead of 0.10: 34-42 tok/s (older settings), KL 0.104-0.107 |

KL is the engine's served path against the original FP4 checkpoint's logits on 8 fixed sequences
(`tools/tk-bench/arms.py`, `tools/tk-parity/ds41_kl.cu`; see TRUSS `docs/reference/model-deepseek41.md`). The speed rows
vary by about 2% from run to run on a shared machine. Sampling defaults of the server for this model: temperature 1.0,
top-p 1.0, min-p 0.05.

## Known issues (pending)

- **Long-context loops.** Past about 60K tokens of context, roughly 30% of agent turns fall into a repeating
  stretch in the reasoning (every turn at 120K in our worst session). The server's loop guard (`server/generate.py`)
  detects exact and near repeats, closes the reasoning and carries on, so the turn finishes, but the cause is **not
  found**. It is not the sampling settings, and recall to 200K is 100%, so the engine's attention reads are fine. The
  test that switches off one approximation at a time (tail prefill, expert skipping, speculation) on real looping
  prompts is written (`tools/tk-bench/arms_loop.py`) and has **not run yet**. Low reasoning effort made it worse.
- **Decode is below the 70 tok/s goal.** About 130-150 routed experts per pass are not in VRAM and run on the CPU, or
  cross PCIe. More experts in VRAM is the lever; the refill we tried (pinned refill of the prompt buffers) did not
  reduce those misses.
- **Prefill is below the 2,000 tok/s goal**, and every prompt chunk streams all cold experts again (about 82 GB per 8K
  chunk at the tested split), which is also the fixed ~6 s per agent turn.
- **Quality evidence is KL and recall only.** We have not run task benchmarks (code, math, long-context) on this pack.
  Dropping a third of the experts and coding the rest at 2-3 bit costs quality that KL 0.06-0.07 describes only on
  average.
- **Greedy decoding with speculation differs from plain greedy** by design (the expert skip is applied per call).
- Needs two Ampere 24 GB GPUs; one GPU and other cards are not supported yet. Text only.

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
- **[dsx](https://github.com/lxe/dsx)** (lxe): deferring low-rank CPU experts by one layer.
- **[Xolotl](https://github.com/Indras-Mirror/Xolotl)** (Indras-Mirror): moving the draft model off the card during
  long prompts.
- **[PolyStrata](https://github.com/VecSzn/PolyStrata)** (VecSzn): tensor-core attention pattern.
- Base model: **DeepSeek-V4.1-Flash-1017** by DeepSeek (MIT). This pack is a derivative of it under the same license.
