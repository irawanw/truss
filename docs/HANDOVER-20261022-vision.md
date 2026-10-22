# HANDOVER 2026-10-22: Flash-Next image input (mmproj / vision)

Request: a user of `lackonendes/PAW-125B-FLASH-NEXT-X3.1` asked for TRUSS vision support. Plan: `docs/PLAN-20261022-flashnext-vision.md`
(read its Rules section first). Repo `~/trellis-kernel`, branch `e17`. Nothing is pushed, nothing uploaded to HF.

## State

| phase | status | commit |
|---|---|---|
| 0 fixtures (HF vision tower taps for a 448x448 test image) | done | scripts live outside the repo |
| 1 mmproj gguf (`flashnext-mmproj-f16.gguf`, 0.899 GB, 333 tensors) | done, read-back rel err <= 4e-7 | 686fc9b |
| 2 ViT forward on GPU (`src/model/qwen4exp/vision/vit.{h,cu}`) | done, `vit_check` PASS | d4da53c |
| 3 text-model integration + server | NOT started (code read only, see below) | - |
| 4 docs / HF card draft | not started | - |

Phase 2 numbers vs HF (448x448, 784 patches -> 196 tokens): patch+pos cos 1.000000, block0 1.000000, block13 0.999989,
block26 0.999932 (max diff 540 on absmax 9405: f16 GEMM inputs on outlier channels), vit_out cos 0.999947, max diff 0.0066 on absmax 0.34.
Speed: 551 ms for 784 patches (naive per-(query,head) attention kernel). Fine for correctness, too slow for production.

## Where things are

- Work dir (outside repo): `~/ML_projects/flashnext/20261022_vision/`
  `hf/` (config, tokenizer, chat template, preprocessor config, **shard 1 of the base model = all `model.visual.*`**), `fix/` (fixtures),
  `data/flashnext-mmproj-f16.gguf`, `scripts/p0_fixtures.py`, `venv/` (system site-packages + pillow), `img/test448.png` (synthetic).
- Repo files: `tools/tk-pack/flashnext_mmproj.py`, `server/vision_pre.py`, `src/model/qwen4exp/vision/vit.{h,cu}`,
  `tests/unit/vit_check.cu`, CMake targets `truss_vision` and `vit_check`, docs in `docs/reference/model-qwen4exp.md` ("Vision").
- Build/test: `cmake -S . -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build-vision --target vit_check`
  then `CUDA_VISIBLE_DEVICES=0 build-vision/vit_check <mmproj.gguf> <fix dir> [--time]` (exit 0 = PASS).
- Re-make fixtures: `~/ML_projects/flashnext/20261022_vision/venv/bin/python scripts/p0_fixtures.py` (CPU, ~1 min).
- TRACKER.md (disk only, not in git) has the log lines for P0-P2.
- Flash-Next model files are on the HDD only; the user allowed loading from there for this work (plan, "Model files"):
  `/mnt/hdd/ml/flashnext/20261003_x31/data/flashnext-x31-0000{1,2}-of-00002.gguf`, drafter `/mnt/hdd/ml/flashnext/20260930_truss_tg/data/mtp_x3/flashnext-mtp-x3k3.gguf`.

## Facts learned (all from transformers 5.16.1 `models/qwen4_exp/modeling_qwen4_exp.py`, the ground truth)

- ViT: 27 blocks, hidden 1152, 16 heads (head dim 72), MLP 4304 GELU-tanh, LayerNorm eps 1e-6, patch 16, temporal patch 2, merge 2,
  learned 48x48 pos-embed resampled bilinear **align_corners=True** with border clamp, 2D rotary (h,w; inv_freq = 10000^-(2i/36), `rotate_half` over all 72 dims),
  merger = LayerNorm(1152) -> view 4 consecutive tokens (4608) -> fc1 -> **GELU erf** -> fc2 -> 2560. Tokens are in 2x2 merge-block order, which is also the text-token order.
- HF `Qwen4ExpModel.forward`: image rows replace `inputs_embeds` at `image_token_id` positions (`masked_scatter`). **PLE and the n-gram tables still get the original
  `input_ids`** (`ple_input_ids = input_ids`), so image placeholders use the `image_token_id` row of the PLE table. Position ids come from `compute_3d_position_ids` (3D, [3,T]).
- The repo's GGUF reader allows at most 4 dims, so the conv kernel is stored flattened [1152,1536] (order C,T,ph,pw = the patchify order).
- Tokenizer/chat template for images: base repo `chat_template.jinja` (vision_start, N x image_token, vision_end).

## Phase 3: what is left (read the plan for gates)

1. **Embedding injection.** `StageModel::embed(Stage&, const Chunk&, float* res)` at `src/model/qwen4exp/model.cu:142` builds the residual rows (hyper-connection
   width `hc_dim`) from `Chunk::tokens` (host ids; token embedding is gathered from pinned host memory). Add an optional per-chunk image-row source (device f32 [rows][2560] plus the
   chunk-relative placeholder row indices) to `engine::Chunk` (`src/engine/model.h:25`), default null = old path. Rows that are image placeholders get the ViT row instead of the table row;
   everything after (PLE, n-gram via `c.tokens`) stays as is. Check how a normal token row is expanded into the hc streams and do the same.
2. **3D rope positions.** Today `dsa_[i]->forward(g, c.slot, mixed16, c.pos0, T, out, tentative)` takes `pos0` and uses 1D positions `pos0+i`. `rope_sections` (imrope) is in `config.h`
   (`qwen4exp.rope.dimension_sections`) but the engine only ever sees text, where t=h=w. For an image: t constant, h/w from the merged grid; text after the image continues at `max(position)+1`
   (`rope_deltas` in HF). Needs a per-row position-id array into the DSA rope kernel (only when an image is present) and into the MTP drafter. GDN layers do not use rope.
   Also decide the decode path: after an image, `pos` for new tokens = row index + rope_delta.
3. **Chunking.** `chunk_align()` = DsaShape::RATIO; an image can span chunks; index embeds and position ids by absolute prompt row. Test with a small `--chunk`.
4. **Server** (`server/`, new `server/vision.py` < 250 lines; `vision_pre.py` exists): accept `image_url` parts (data URI / http, 20 MB cap, 10 s timeout; local paths only behind a flag),
   expand to `vision_start + N*image_token + vision_end`, run the Vit (needs mmproj path env/flag and a device; it is ~1 GB f16 + buffers: put it on the card with spare VRAM), pass rows to the engine.
5. **Gates:** text-only unchanged (`scripts/golden_ds41.sh` 7/7 IDENTICAL + Flash-Next 64-token logits bit-identical), image prompt KL <= 0.1 and top-1 >= 85% vs HF on the same inputs,
   3 readable image descriptions, prefill time and VRAM for 448 and 1344 images. HF reference for the full model is heavy (125B pack vs 100+ GB bf16 base): the practical reference for
   gate (b) is **HF text+vision with the original weights is not runnable here**; use instead (i) ViT parity (done) and (ii) text-only parity of the injection path (feed the *text* embedding of a
   prompt through the new `image_rows` path and require bit-identical logits) plus (iii) the human check of descriptions. Decide this with the user before spending time.
6. **Speed lever (after correctness):** replace `k_attn` by a tiled/flash-style or cuBLAS strided-batched QK^T/PV; 7056 patches (1344x1344) would take many seconds today.
   `MAX_TOKENS` is 8192; `k_attn` keeps all scores in shared memory (N*4 bytes, 32 KB max).

## Gotchas

- **GPUs:** the vast.ai renter has `wildrig-multi` (pearlhash miner, root) on GPUs 0 and 2 (since ~08:00) and ollama on 1 and 3. `run_watched2.sh` / `ds41_gpu_job.sh` refuse to start
  ("GPU 0 busy ... renter has priority"). The user said for vision work: **"no need watchdog run it"**, GPU 0 only, short runs. This does not extend to GPU 1/3 or long jobs. Phase 3 full-model runs need
  both 0 and 2 for minutes: ask again before starting them. The DS41 server (pm2 `truss-ds41`) is stopped; do not start it without asking.
- The user said earlier "stop bandwidth test": the P2P bandwidth test is dropped; a compiled binary sits in the session scratchpad and can be ignored.
- Do NOT install into `~/.local/lib/python3.10` (live site-packages for TRUSS tools). An earlier `pip install torchvision` into the venv broke imports; it is removed. HF image
  processors need torchvision, so `server/vision_pre.py` reimplements them (checked by reading HF's `patchify`/`smart_resize`, not numerically).
- `/` has about 20 GB free (rule: keep 25). The work dir is ~1.9 GB (shard 1 + mmproj). No big downloads without asking; cron `~/bin/disk_guard.sh` runs every 15 min.
- Standing rules still apply: docs with code, files < 500 lines, TRACKER on disk only, commit trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`, never push/upload without the user.
- Unrelated untracked files in the repo (`build-dsv41/`, `docs/paw-125B-X3.1-TRUSS-90s.mp4`) are not ours; `build-vision/` is new (ignore or add to .gitignore).

## Open questions for the user

1. Is the real Phase 3 gate allowed to be "text path bit-identical + ViT parity + human-checked descriptions" (no full-model HF reference exists on this box)?
2. May Phase 3 use GPUs 0 and 2 together (minutes, miner running), or wait for the renter to leave?
3. Ship the mmproj as a new file in `lackonendes/PAW-125B-FLASH-NEXT-X3.1` once it works? (HF upload needs the user's OK.)
