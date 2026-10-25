# DeepSeek-V4.1-Flash-1017: `src/model/deepseek41/`, pack tools

Plan and phases: `docs/PLAN-20261019-dsv41.md`. Math reference: the vendored official `dsref/model.py`
(`~/ML_projects/dsv41flash/20261018_poc/scripts/dsref`); architecture cross-check: mmnga/Strata `deepseek`
`src/deepseek41/` (tensor names and metadata keys used here).

## Pack (two files)

| file | from | contents |
|---|---|---|
| `ds41-dense.gguf` (9.1 GB) | `tools/tk-pack/ds41_dense.py`: HTTP ranges of the HF shards (dense tensors only, 9.85 GB) | Q8_0: attention (q_a, q_b, kv, output_a, output_b), shared expert, indexer q_b / proj / attn_k, ratio-1 compressor, Engram wkv, embedding, head (Q8_0 vs bf16 rel 0.54%). F32: norms, sinks, router rows of the kept experts + bias, mHC fn / base / scale, ratio-2 compressor kv / gate, Engram q / k. Metadata `deepseek41.*` incl. `expert_keep`, source layers, candidate settings, Engram hash layout (uint64 multipliers / primes / offsets, token_map padded to the vocab), tokens |
| `ds41-experts.gguf` (~97 GB, HDD) | `tools/tk-pack/ds41_experts.py` from the X3.1 encodes | `blk.L.ffn_{gate,up,down}_exps.m3_{trellis,meta,suh,svh}`, 256 kept experts per layer renumbered in ascending original id; K codes 1 / 15 / 2 / 25 / 3 / 35 / 4 (served pack: 2 / 25 / 3) |

The engine reads the experts as an overlay (`TRUSS_EXPERT_OVERLAY`, `gguf::File::overlay`). The Engram tables
(2 x 384M rows) are their own files: `ds41-engram-L{1,14}.gguf` from `tools/tk-pack/ds41_engram.py`, int4 rows of
144 B (55.3 GB each; PoC TRACKER R8 for the error), read row by row by `blocks::Engram`; they belong on NVMe.

## Files

| file | what |
|---|---|
| `config.{h,cc}` | `Config::from_gguf`: dims, rope (YaRN), mHC, indexer, candidates, compress ratios, source lists (`kv_source_of`, `index_source_of`: the last source at or before a layer), SwiGLU clamp (one value; the kernel shape carries it), Engram layout. Throws on inconsistent metadata |
| `model.{h,cu}` | `Family` (shape checks vs `moe::DsV41`, Engram tables present, stage boundaries on attention sources) and `StageModel`: residual row = 4 x 5120 + the 24-float next-collapse coef; per layer [Engram] -> mHC mix / collapse -> attn_norm -> `AttnDs41` -> post -> mHC mix / collapse -> ffn_norm -> `Moe` (sqrt-softplus route, plain shared expert, clamp 10, x1.5) -> post -> coef into the row; head = collapse -> norm -> `Head`. `TRUSS_DS41_DUMP=<dir>` appends each layer's `res_in`, `res_eng` (after Engram), `attn_in`, `attn_out`, `moe_in`, `moe_out` as fp32 rows to `<dir>/<name>_L<nn>.f32` (debug, synchronous). `TRUSS_DUMP_DSPARK=<file>` appends one record per forward call of the stage owning layers 37-39: `int32 pos0, int32 T, int32 tokens[T], half hidden[T][3][5120]`, `hidden` the hc-copy mean of the input residual of layers 37, 38, 39 in that order (DSpark drafter input, E8; debug, synchronous). No CPU expert tier (`MoeLayer::cpu_ok = false`: the pool is Flash-Next-shaped) |
| `dspark.{h,cu}`, `dspark_attn.cu`, `dspark_draft.cu`, `dspark_weights.cc` | The DSpark drafter (`mtp.0-2`, dsref `DSparkBlock` / `forward_spec`) as an `engine::Drafter` on the last stage (which must own layers 37-39). **Its own `runtime::MoeExec`** (`moe::DSpark`: 128 experts, top-3, all resident, no ring / CPU tier), built in `StageModel::build` before the stage's store takes the VRAM left; `moe_layers()` stays the main layers. `capture`: the hc-copy mean of the input residual of layers 37/38/39 per row (device kernel) -> `main_proj` + `main_norm` -> each layer's kv latent (rms, base-theta rope, **no YaRN**, as dsref's window-only layers) into a 128-slot ring per slot (`seed`, on commit / accept). `draft`: one 5-row block [next, 4 x noise 128799] at positions pos .. pos+4, per layer hc collapse -> norm -> window attention (noncausal over ring + block, scale head_dim^-0.5) -> hc post -> MoE (sqrt-softplus, no VL bias) -> hc post; then the last layer's norm, the main head, the markov bias of the previous token per row and argmax (or `draft_sample`). Weights: `ds41-dspark-dense.gguf` (`tools/tk-pack/ds41_dspark.py`, 690 MB) as the drafter file + `TRUSS_DSPARK_EXPERT_OVERLAY=ds41-dspark-experts.gguf` (`dspark.L.ffn_*_exps.*`, K3, 5.1 GB). `TRUSS_DSPARK_DEBUG=<file>`: per draft call `pos, next, n, made, drafts[made]`, row 0's top 5 (id, logit). Measured E9 (TRACKER #205): 3.40 tokens per verify pass on code (Python reference E8: 3.89 teacher-forced); spec output = plain greedy until a router near-tie flips between the 6-row window and 1-row decode (`TRUSS_BENCH_ORACLE`) |
| `weights.{h,cc}` | `bind`: every tensor checked (ggml shapes from mmnga's `Model::validate`) and used once; compressor / index-key tensors on KV sources, index-query tensors on index sources, Engram on its layers |

## The tail prefill (E15 item 7) and the DSpark confidence head (E15 item 8)

`TRUSS_TAIL_PREFILL=<n>` (default 0 = off; read per chunk, so a bench arm can change it on a loaded model): on a **prompt** chunk the *consumer* layers (the ones that read the
candidate source's compressed KV, `compress_ratio > 0` and `kv_source_of(l) == candidate_source`, i.e. layers 21-39)
run on the **last n rows** of the chunk only (the reference's "SWA bounded replay", `n = 134 = 128 + 6`): the
residual, the collapse coef, the token ids, the positions and the row count shift to that suffix
(`src/model/deepseek41/model.cu` ✓), the attention's top-k is read at the shifted row (`row_off` ✓,
`src/kernels/ds41/sparse_attn*.cuh` ✓) and the sliding window is **clipped at the chunk's start** (`clip` ✓, the
reference's `max(s, i-127)` rule ✓ — the skipped rows' ring slots hold stale data ✓ and must not be read ✓). The
decode path is untouched ✓ (the mode applies to the prompt chunks only ✓). Measured (E15 item 7): the 32K prompt's
prefill 1032-1052 → **1348-1349 tok/s** ✓ (+29% ✓), the decode unchanged ✓, the decode-path KL 0.0678 → 0.0692
(budget 0.0728 ✓). The **exact (cone)** variant of the brief is not implemented ✓ (it needs a KV-only attention pass
for the skipped rows ✓).

`TRUSS_DSPARK_VARLEN=1` (default off) + `TRUSS_VARLEN_C0` / `TRUSS_VARLEN_C1` (the cost model
`cost(n) = c0 + c1 n` in ms ✓): the **confidence-scheduled verify** ✓. The drafter's confidence head
(`confidence_proj`, fp32 in the pack ✓) gives `p_i = sigmoid(proj([hidden_i before the final norm ; markov_emb_i]) / T)`
✓ (`TRUSS_DSPARK_CONF_T`, default 1.0 ✓ = the reference's ✓) — it is **only computed when the schedule or the
calibration log asks for it** ✓ (`TRUSS_DSPARK_CONF_LOG=<file>`: per draft call `pos, next, made, p_0..p_{n-1}` ✓;
`TRUSS_ACCEPT_LOG=<file>`: per round `pos, kept, verified prefix, drafts` ✓), so the shipped path pays nothing ✓.
The engine (`src/engine/pipeline_slots.cu` ✓) then verifies only the prefix that maximizes
`(1 + Σ cumprod(p)_1..n) / cost(n)` ✓. Measured (E15 item 8): the head is **well calibrated** ✓ (the mean p_i per
position 0.80 / 0.73 / 0.63 / 0.57 / 0.52 ✓, the predicted 3.365 tokens/pass vs the measured 3.37-3.61 ✓) but the
schedule is a **no-op** ✓ (the DSpark's drafts are good enough that the longest window always wins ✓) and the varlen
arm measured **+43 ms per pass in the draft phase** ✓ — unexplained ✓ (the `TRUSS_ACCEPT_LOG` file was not written
in those runs ✓) → the mode stays **off** ✓.

## Serving fixes (10-21, item 10)

Three bugs kept the served config (spec, 5 drafts, `TRUSS_HINT_K=0`, chunk 8192, `TRUSS_TAIL_PREFILL=134`) from
running; all three were found with `compute-sanitizer` / NaN probes on the real model (`$P/data/p10_dbg.sh`):

- **DSpark capture overflowed fp16.** `capture` stored the layer-37/38/39 hc-mean rows as fp16; on a short prompt
  some captured rows hold residual values above 65504 (30 of 322,560 values on a 21-token prompt; long prompts keep
  only later rows) → inf → NaN in the draft attention → the drafter's router picked garbage expert ids →
  `window_kernel<moe::DSpark>` read 1.5 GB before a valid allocation. Now `Slot::cap` is **fp32** and `seed`
  converts each row with its own scale into the fp16 range (`rows_to_half_scaled`, into the drafter's own
  `hid16_` buffer, not scratch): `main_proj` has no bias and `main_norm` (RMSNorm) follows it, so a per-row positive
  scale leaves the result unchanged up to eps.
- **Attention row pieces counted the piece offset twice.** `AttnDs41::forward` runs the attention in
  `engine::prefill_piece()` rows (1024) and passed both `a.idx = topk_ + r0 * k` and `a.row_off = row_off + r0`;
  the kernels index `idx[(t + row_off) * k]`, so every piece after the first read `r0` rows past its top-k (garbage
  key indices; a crash on any prompt longer than 1024 rows once the memory past the buffer was not zero, silently
  wrong keys otherwise). `a.idx` is now the chunk's top-k base. This came from merging the item-7 `row_off` onto the
  item-2 row pieces (10-21 06:45); the item-9 measurements predate it.
- **`ExpertStore::admit` wrote `adapt_hits` unconditionally** (`expert_store_ring.cu`), a vector sized only with
  `TRUSS_ADAPT_REUSE`: a segfault at the first adaptive admission. Now guarded with the reuse counters.

Checked after the fixes (`tk-bench-spec`, e14g env + drafter, chunk 2048): a 21-token prompt and two 2,048-token
prompts in one process run; spec 41.6 / 50.4 tok/s vs plain 26.1 / 30.7 on the long prompts (3.37 / 3.61 tokens per
pass); the decoded spec text is coherent and correct (it leaves plain greedy at near ties, e.g. "check status of" vs
"check the status of", and continues with different but equally valid code). Greedy spec != plain greedy is expected
(the CPU tier's skip rule is call-scoped, E15 item 9).

**Tested.** `tk-pack-inspect` binds the pack (`TRUSS_EXPERT_OVERLAY=<experts>`); blocks: `ds41_ops_test` (router,
mHC), `moe_window_test` (DsV41 shape), `ds41_attn_parity` (attention); whole layers: `ds41_stage_dump` (layers [0, N) on one GPU with the dumps above), checked
against dsref's mHC math on TRUSS's own sub-block outputs and against the PoC's `moe_in_L00.pt` capture (10-19: found
`HcSinkhorn::mix` passing base / scale swapped, KL 15 -> fixed; the kernel test called `sk_mix` directly).
**Layer split.** Sources hand buffers to consumers by pointer, so a stage boundary must fall at a source layer
(2, 8, 14 or 20 for this model); 20 splits the work roughly in half.

## Vision (image input, 10-25)

Ground truth: DeepSeek's own `inference/{vision.py,image_processor.py,model.py,engram.py}` in the HF repo. Work dir,
fixtures and plan: `~/ML_projects/dsv41flash/20261025_vision/` (`PLAN.md`; TRACKER 288-289).

| piece | where | what |
|---|---|---|
| mmproj | `tools/tk-pack/ds41_mmproj.py` -> `ds41-mmproj-f16.gguf` (0.971 GB, 309 tensors) | arch `deepseek41-mmproj`: `v.*` the ViT (32 blocks x 1024, 16 heads, SwiGLU 2816, RMSNorm, linear patch embed 14x14, no position embedding), `a.*` the aligner (3x3 unfold 9216 -> 5120, GELU erf, 5120), `span.{start,newline,end}` the learned span vectors, `blk.L.bias_vl` (40 x 384, original expert ids) and `mtp.I.bias_vl` the image router biases. Reads shards 1-2 locally, the per-layer biases by HTTP range |
| tower | `src/model/deepseek41/vision/vit.{h,cu}` (`truss::deepseek41::Vit`, in `truss_vision`) | patches -> span rows `START + (lw rows + NEWLINE) x lh + END` (lh, lw = ceil(patch grid / 3)). 2D RoPE (16 freqs of the row, 16 of the column, rotate-half over 64 dims). Attention = cuBLAS strided-batched GEMMs over query chunks (~100 MB of scores). Workspace reserved at open (`reserve`), ~0.5 GB at the largest image (9,216 patches, span <= 1024) on top of the 0.97 GB weights |
| C API | `src/api/truss_vision.cu` | dispatch by `general.architecture`; `truss_vision_rows(gh, gw)` = span rows (DS41) or gh*gw/4 (Flash-Next). The span goes to `truss_eval_img_s` as `nh = 1, nw = span`: 1-D positions, the pipeline's rope offset is then 0 and DS41 ignores pos3 anyway |
| text model | `model.cu` (`Family` reads `TRUSS_MMPROJ`) | span rows' embeddings = the tower's rows (`embed`); every span row carries the placeholder id 129264, and the engine keys on it: the router re-routes those rows with `bias_vl` (renumbered through `expert_keep`; `blocks::Moe` vl runs -> `MoeExec::Call::vl`, far mask applied as to `bias`), Engram treats it as DEAD (n-grams reaching back to it pad from there on; span rows get no lookup: `engram_gate` row list). Text without the placeholder is bit-identical |
| server | `server/vision_ds41.py`, `server/vision.py` `Tower`, `server/app.py`, `server/engine.py` | the reference's resize plan + grey pad + bf16 rounding in numpy; the encoder's one `<｜deepseek_image｜>` expands to the span. `--mmproj` opens the tower **before** the model (its card's expert budget shrinks by ~1.5 GB) and exports `TRUSS_MMPROJ`. `scripts/serve_ds41.sh`: `TRUSS_MMPROJ` / `TRUSS_MMPROJ_DEVICE` |

Any number of images per request: `server/generate.py` feeds the prompt as one `eval_img` call per span not yet in
the slot (`nh = 1`: the engine allows a 1-D image call at any position), records the spans per slot
(`Slot.images` / `ck_images`: row, length, sha1 of the image bytes) and stops reuse / checkpoint restore at a cached
span that is not the prompt's image there (`vision_ds41.reusable`); the tower runs only for spans being read and keeps
the last 16 spans by digest (an agent re-sends its screenshots every turn). Limits: prompt chunks route around the
far experts for image rows as for text (E2); the DSpark drafter routes image rows with its text bias (drafts only,
verify is exact). Host test: `tests/server/ds41_image_reuse_test.py`. Tests: `build-vision/ds41_vit_check <mmproj> <fix dir>
[--time]` (fixtures `scripts/v1_fixtures.py`): corn (33x47 patches) rows cos 0.999993 vs bf16-reference floor
0.999358, carrots (51x74) 0.999985 vs 0.998601, span layout exact; 58 / 249 ms on a 3090. End to end:
`scripts/v3_describe.py` (the reference's corn / carrots example).
