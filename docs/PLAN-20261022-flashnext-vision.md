# PLAN 2026-10-22: image input (mmproj / vision) for Flash-Next on TRUSS

Request: a user of `lackonendes/PAW-125B-FLASH-NEXT-X3.1` asked for vision support. Base model
`Qwen/Qwen3.8-Flash-Next` (`Qwen4ExpForConditionalGeneration`). The pack is text-only today.
Executor: opencode with DeepSeek V4.1 Flash (free). Reviewer: the user, via the TRACKER and the gates below.

## Rules (read first, they are not optional)

1. Work only in `~/trellis-kernel` (branch `e17`). Do NOT touch `~/trellis-serve`, `pm2`, or any running server.
2. GPUs: use only GPU 0 and GPU 2, and only through `scripts/ds41_gpu_job.sh <cmd>` (watchdog). Never touch GPUs 1 and 3.
   Phases 0-3 need no GPU or only GPU 0. If a command needs a GPU and you are unsure, stop and write a note in the TRACKER.
3. No hardware changes, no installs outside a venv in `~/ML_projects/flashnext/20261022_vision/`, no `pm2`, no deleting files.
4. Do not touch `~/ML_projects/flashnext/20261004_selfopt/data/keys/A0_KEY.md` or `~/.local/share/opencode/opencode.db`.
5. Files under 500 lines each. Every new file gets a header comment and an entry in `docs/reference/model-qwen4exp.md`
   (API, layouts, invariants, tests, how to change). Docs go in the same commit as the code.
6. Vision code lives in its own folder `src/model/qwen4exp/vision/` and is gated by the request carrying an image.
   A request without an image must run the exact old code path.
7. Shared-code changes (anything outside `src/model/qwen4exp/` and `server/vision*`) need
   `scripts/golden_ds41.sh` to print 7/7 IDENTICAL against `golden_1019b` before you commit. Flash-Next text must stay identical.
8. Log every step in `TRACKER.md` (edit on disk only, never `git add` it): what you ran, the number you got, pass/fail.
9. Judge by numbers (max abs diff, cosine, KL). Never write "looks right". If a gate fails, stop, record it, do not work around it.
10. Commit per phase with trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Do not push. Do not upload to HF.
11. No sleep-polling. Wait on log lines with `tail -F | grep -m1` or the Monitor tool.
12. Explain results in simple words in the TRACKER.

## Model files (HDD exception, approved by the user 2026-10-22)

The Flash-Next pack is on the HDD only. For this vision work it may be loaded from there (slow first load is fine):
- text: `/mnt/hdd/ml/flashnext/20261003_x31/data/flashnext-x31-0000{1,2}-of-00002.gguf` (51 GB each; pass the first shard)
- MTP drafter: `/mnt/hdd/ml/flashnext/20260930_truss_tg/data/mtp_x3/flashnext-mtp-x3k3.gguf`
Read-only use. Do NOT move or copy them to the NVMe (it has under 25 GB free), do not use `storage_move.py` on them, and do not
write anything new to the HDD except under `/mnt/hdd/ml/flashnext/20261022_vision/` if `W` outgrows the NVMe.
Page cache will warm after the first load; do not rerun full loads without need (batch the checks in one server/test session).
This exception is for this plan only; the "no active files on the HDD" rule still applies to everything else.

## Facts (verified 2026-10-22)

- Vision config (base `config.json`): 27 blocks, hidden 1152, 16 heads, MLP 4304, GELU-tanh, patch 16, temporal patch 2,
  spatial merge 2, `num_position_embeddings` 2304, `out_hidden_size` 2560 (= text hidden), `deepstack_visual_indexes` = [] (no DeepStack).
- Weights (in the original safetensors, 333 tensors, all under `model.visual.`): `patch_embed.proj.{weight,bias}`, `pos_embed.weight`,
  `blocks.N.{norm1,norm2}.{weight,bias}`, `blocks.N.attn.{qkv,proj}.{weight,bias}`, `blocks.N.mlp.linear_fc{1,2}.{weight,bias}`,
  `merger.{norm,linear_fc1,linear_fc2}.{weight,bias}`.
- Tokens: `image_token_id`, `video_token_id`, `vision_start_token_id`, `vision_end_token_id` are in the base `config.json`.
- Text side: `src/model/qwen4exp/` (1,618 lines). `StageModel::embed(Stage&, const Chunk&, float* res)` is the hook that turns
  token ids into the residual rows (`res_width()` = `hc_dim`, hyper-connection streams). Layers use PLE (per-layer embedding table
  indexed by token id) and imrope (`rope_sections` in `config.h`).
- Reference code: the HF `modeling_qwen4_exp.py` and `preprocessor_config.json` in the base repo. They are the ground truth for
  the ViT, the patchify order, the pos-embed interpolation and the position ids. Download them into the work dir; do not guess.

## Work dir and files

`W=~/ML_projects/flashnext/20261022_vision` (create it). Keep downloads, fixtures and logs there, not in the repo.
The original weights are large; download only the shards that hold `model.visual.*` (use the index json to find them).

## Phase 0: fixtures (CPU, no GPU)

Goal: ground-truth tensors from HF transformers for one fixed test image (use a 448x448 crop of any image in `W/img/`).
1. venv in `W`, `pip install torch transformers safetensors pillow numpy` (CPU torch is fine).
2. Load only the vision tower from HF (`model.visual`), run it on the preprocessed image, save to `W/fix/`:
   `pixel_values.npy`, `grid_thw.npy`, `vit_out.npy` (after the merger, shape [N_img_tokens, 2560]), and per-stage taps:
   `patch_embed.npy`, `pos_embed_added.npy`, `block0_out.npy`, `block13_out.npy`, `block26_out.npy`.
3. Save the 3D position ids HF builds for a prompt with this image plus 20 text tokens: `position_ids.npy` ([3, T]).
4. Gate: files exist, shapes printed in the TRACKER, `vit_out` has N = (H/16/2) * (W/16/2) rows.

## Phase 1: mmproj file (CPU)

Goal: a single `flashnext-mmproj-f16.gguf` with the `model.visual.*` tensors in f16 (biases and norms f32).
1. Check first whether `~/llama-paw*/convert_hf_to_gguf.py` has `Qwen4Exp` (use `grep`). If yes, use `--mmproj`. If no, write
   `tools/tk-pack/flashnext_mmproj.py` (under 300 lines) that reads safetensors and writes GGUF with the repo's gguf writer
   conventions (look at `tools/tk-pack/ds41_dense.py` for the pattern). Tensor names: keep `v.`-prefixed short names and write
   the vision hyperparameters as GGUF metadata keys (`qwen4exp.vision.*`). Document the names in `docs/reference/model-qwen4exp.md`.
2. Reader test: load the file back, compare 5 random tensors to the safetensors source, max abs diff must be 0 (f32) or <= 1e-3 relative (f16).
3. Size should be about 1.0 GB. Record it.
4. Gate: reader test passes. Commit.

## Phase 2: ViT forward in plain PyTorch-free C++/CUDA, checked per stage (GPU 0 only)

Goal: `src/model/qwen4exp/vision/vit.{h,cu}` computing `vit_out` from `pixel_values`. Plain f16 GEMMs with f32 accumulate are enough;
use cuBLAS if the repo already links it, else the repo's dense GEMM block (see `docs/reference/kernels-dense.md`). No trellis codec for the ViT.
Build in this order and check each stage against the fixtures with a test `tools/tk-parity/vit_check.cu` (max abs diff and cosine in the TRACKER):
1. patch embed (Conv3d with kernel [2,16,16] as a GEMM over flattened patches; mind the HF patch ordering, it groups the 2x2 merge window).
2. pos embed: 2304 = 48x48 grid, bilinearly interpolated to the image grid exactly as HF does; add it.
3. one block: LayerNorm -> qkv -> 2D rotary on q and k (as HF) -> full (non-causal) attention inside one image -> proj -> residual -> LayerNorm -> fc1 -> GELU-tanh -> fc2 -> residual.
4. all 27 blocks. 5. merger: LayerNorm, group 2x2 patches into one row of 4*1152, fc1, GELU, fc2 -> 2560.
Gates: block0 cosine >= 0.9999; `vit_out` cosine >= 0.999 and max abs diff <= 0.05 relative to the fixture's max abs. If a stage misses, fix that stage before moving on.
Image sizes: support any multiple of 32 up to 1344 on the long side (clamp, then resize per `preprocessor_config.json`); attention per image only.
Commit.

## Phase 3: feed embeddings into the text model (GPU 0 and 2 via the wrapper)

1. Server/API (Python, `server/vision.py`, under 250 lines): accept OpenAI content parts `{"type":"image_url","image_url":{"url": data-URI or http(s)}}`
   (also local path only if the server flag `--allow-local-images` is set; default off). Fetch with a size cap (20 MB) and a 10 s timeout.
   Preprocess exactly as `preprocessor_config.json` (resize, rescale, normalize, patchify) and return `pixel_values`, `grid_thw`.
   Build the prompt: `vision_start`, then N copies of `image_token_id`, then `vision_end`, with N from the grid. Use the chat template in the base repo.
2. Engine: add an optional `image_embeds` input to the request path (a `[N, 2560]` f16 buffer plus the index of each placeholder row).
   In `StageModel::embed`, rows whose token is an image placeholder are replaced by the ViT rows, expanded into the hyper-connection streams
   the same way a normal token row is. No other code path changes.
3. PLE: the per-layer embedding table is indexed by token id. For image rows use the HF behaviour (read `modeling_qwen4_exp.py`:
   it is either a zero row or the pad id row). Do exactly what HF does and record which it is.
4. imrope positions: image tokens get (t, h, w) position ids from the grid; the text after the image continues from max(position)+1, as HF does.
   Pass `position_ids` [3, T] into the rope kernel only when an image is present; text-only keeps the old arguments.
5. Chunking: the prompt chunker splits at `chunk_align()`; an image's tokens may span chunks, so the embeds and position ids are indexed by absolute row.
   Test with an image that crosses a chunk boundary (set `--chunk` small in the test).
6. Gates (record each number):
   a. text-only golden: `scripts/golden_ds41.sh` 7/7 IDENTICAL, and a Flash-Next text-only request gives bit-identical logits to before (compare 64 tokens).
   b. image prompt (fixture image + "Describe this image."): our last-position logits vs HF transformers' logits on the same inputs.
      KL <= 0.1 (the pack is quantized, so not zero; the text-only KL of this pack is the reference, quote it). Top-1 agreement >= 85% over 64 positions.
   c. a short generation of 64 tokens describes the image in a way the user can read; paste it into the TRACKER (use 3 different images).
   d. prefill time of a 448x448 image and of a 1344x1344 image, and peak VRAM; record them.

## Phase 4: docs and release notes (no upload)

1. Update `docs/reference/model-qwen4exp.md` and `docs/SERVING-FLASHNEXT-X31.md`: flags, the mmproj file name, memory cost, limits, how to test.
2. Draft the HF model-card text and the `hf upload` command for `flashnext-mmproj-f16.gguf` into `docs/hf/` as a draft. Do NOT run it; the user decides.
3. Final TRACKER summary: what works, numbers from the gates, what does not (video is out of scope), known limits.

## Out of scope

Video input, multi-image batching across requests, DeepStack, quantizing the ViT, serving changes in `~/trellis-serve`, pushing to GitHub.

## Stop and ask the user when

- a gate fails twice after a fix attempt; - a step needs a GPU other than 0 or 2, or the server stopped; - the HF reference code contradicts this plan;
- a shared-code change breaks golden; - you need to delete anything or spend more than 5 GB of disk.

## Order and size (estimate)

Phase 0: 0.5 day. Phase 1: 0.5 day. Phase 2: 1.5-2 days (largest). Phase 3: 1-1.5 days. Phase 4: 0.5 day.
