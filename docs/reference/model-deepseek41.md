# DeepSeek-V4.1-Flash-1017: `src/model/deepseek41/`, pack tools

Plan and phases: `docs/PLAN-20261019-dsv41.md`. Math reference: the vendored official `dsref/model.py`
(`~/ML_projects/dsv41flash/20261018_poc/scripts/dsref`); architecture cross-check: mmnga/Strata `deepseek`
`src/deepseek41/` (tensor names and metadata keys used here).

## Pack (two files)

| file | from | contents |
|---|---|---|
| `ds41-dense.gguf` (9.1 GB) | `tools/tk-pack/ds41_dense.py`: HTTP ranges of the HF shards (dense tensors only, 9.85 GB) | Q8_0: attention (q_a, q_b, kv, output_a, output_b), shared expert, indexer q_b / proj / attn_k, ratio-1 compressor, Engram wkv, embedding, head (Q8_0 vs bf16 rel 0.54%). F32: norms, sinks, router rows of the kept experts + bias, mHC fn / base / scale, ratio-2 compressor kv / gate, Engram q / k. Metadata `deepseek41.*` incl. `expert_keep`, source layers, candidate settings, Engram hash layout (uint64 multipliers / primes / offsets, token_map padded to the vocab), tokens |
| `ds41-experts.gguf` (~97 GB, HDD) | `tools/tk-pack/ds41_experts.py` from the X3.1 encodes | `blk.L.ffn_{gate,up,down}_exps.m3_{trellis,meta,suh,svh}`, 256 kept experts per layer renumbered in ascending original id; K codes 2 / 25 / 3 |

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

**Tested.** `tk-pack-inspect` binds the pack (`TRUSS_EXPERT_OVERLAY=<experts>`); blocks: `ds41_ops_test` (router,
mHC), `moe_window_test` (DsV41 shape), `ds41_attn_parity` (attention); whole layers: `ds41_stage_dump` (layers [0, N) on one GPU with the dumps above), checked
against dsref's mHC math on TRUSS's own sub-block outputs and against the PoC's `moe_in_L00.pt` capture (10-19: found
`HcSinkhorn::mix` passing base / scale swapped, KL 15 -> fixed; the kernel test called `sk_mix` directly).
**Layer split.** Sources hand buffers to consumers by pointer, so a stage boundary must fall at a source layer
(2, 8, 14 or 20 for this model); 20 splits the work roughly in half.
