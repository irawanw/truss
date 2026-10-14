# TRUSS engine layout: engine, blocks, models

TRUSS will serve many models, several with architectures not seen yet (Qwen4 27B dense, Qwen4 Flash-Next MoE,
Nemotron 70B Mamba-2 hybrid, LFM2 MoE, ...). Adding one must be small, local and boring. This page is the contract
that makes it so; `CODE.md` has the per-file rules.

**Why this exists (2026-10-14).** Until v1.6 the whole engine lived in `src/model/qwen4exp/forward.cu` (3,550 lines):
sequence slots, speculative rounds, the layer split, expert residency and the CPU tier were all members of the
Flash-Next model class. A second model would have meant copying it. The split below moves every model-agnostic piece
out of the model folder, and every reusable layer piece into a block, with Flash-Next's output unchanged bit for bit
(`tests/golden/`).

## What TRUSS is for (user, 2026-10-14)

**A trellis-quant engine for Ampere consumer GPUs: RTX 3060 (12 GB), 3080 (10/12 GB), 3090 (24 GB), sm_86 only.**
Every model it serves is trellis-coded (X3 / X3.1 and successors), dense and MoE alike: Flash-Next, PAW-27B-X3.1
(dense), LFM2.5-24B-A2B, Qwen4 27B / Flash-Next, Nemotron 70B (Mamba-2 hybrid), ... Consequences:

- **Trellis is the main weight format** of every projection (`blocks/linear`), not only of the experts; Q8_0 / f32
  remain for the few small tensors that are not trellis-coded (router, norms, embeddings where the pack keeps them).
- **Weights larger than VRAM are the normal case**, dense too (27B on a 3060, 70B on a 3090): the residency runtime
  (today `runtime::ExpertStore`) places and streams any trellis matrix, an expert being one kind of matrix.
- **Kernels are tuned per card, not per model**: one Tune<> per (shape class, card), picked at load from the device.

## Three layers

```
include/truss/truss.h          C API (model-agnostic, unchanged): what server/ binds to
src/api/                       C API on top of engine::Pipeline; picks the model by general.architecture
src/engine/                    MODEL-AGNOSTIC runtime
  pipeline.*                   the sequence calls: prompt / round / verify / accept / draft / head / slots / checkpoint
  stage.*                      one device's layer range: stream, buffers, scratch, hand-off to the next stage
  stage_lock.h                 the per-stage lock for concurrent slots (hand-off ticket order)
  state.*                      StateSet: every per-sequence buffer a block owns, with slot / window / checkpoint rules
  options.*                    EngineOptions + apply_env (one place for every TRUSS_* knob)
  faults.* profile.*           fault injection / bounded waits; section timing
src/blocks/                    REUSABLE layer pieces; each owns its weights' upload and its state
  linear.*                     every projection, by the TENSOR's format: trellis (X3 / X3.1 m3_* quads; dense
                               models like PAW-27B-X3.1 are all trellis), Q8_0, f32. Blocks call lin(); the weight
                               format is never named in a block or a model
  attn_gqa.* attn_dsa.*        attention mixers
  gdn.* shortconv.* (mamba2.*) recurrent mixers
  moe.*                        router + routed experts (through runtime/moe_exec) + optional shared expert
  ffn_dense.*                  SwiGLU FFN
  hc.* ple.* norm.* head.*     hyper-connections, per-layer embeddings, norms, output head
src/runtime/                   expert residency: ExpertStore, doorbell driver, CPU tier, admission (MoE models)
src/kernels/                   CUDA kernels, templated on a shape struct (unchanged rules, CODE.md)
src/model/<family>/            THE ONLY MODEL-SPECIFIC CODE
  config.*                     hyperparameters from GGUF metadata
  model.*                      which block each layer uses, the residual stream, embed / head; drafter if any
src/model/registry.cc          general.architecture -> factory (one line per family)
```

## The Model interface (`src/engine/model.h`)

A model family implements one class for one stage (a contiguous layer range on one device):

| call | what the family does |
|---|---|
| `res_width()` | floats per residual row (Flash-Next: hc x d_model; LFM: d_model) |
| `build(Stage &, StateSet &)` | create its blocks for layers [lb, le); blocks upload weights and register state |
| `embed(Ctx &)` | tokens -> residual rows (first stage only) |
| `layer(int l, Ctx &)` | one layer on the residual rows: calls its blocks |
| `head(Ctx &, rows, n, logits, draft)` | residual rows -> logits (last stage only) |
| `drafter()` | optional speculative drafter (Flash-Next's MTP block); null: no drafts |
| `scratch_per_token()` | peak temporaries per row, from its blocks |

`Ctx` carries the chunk (tokens, pos0, T, tentative, stream/fetch mode) and the stage's services (linear, scratch,
MoE executor). A family file is composition only: no kernels, no residency, no slots, no locks.

## State rules (`src/engine/state.h`)

Every per-sequence buffer is registered with the stage's `StateSet` by the block that owns it, with its kind:

| kind | per slot | in a verify window | checkpoint |
|---|---|---|---|
| `Positional` (K/V, indexer blocks) | sub-range of one n_ctx array | rows past the committed position: nothing to undo | none needed |
| `Recurrent` (GDN / Mamba state, conv rows, PLE history) | own buffer | block's `snapshot()` before, `accept(n)` after | copied to pinned host |

`accept(n)` stays the block's job (only the block knows how to replay n accepted rows), but when and for which slot it
runs is the engine's. Slots, checkpoints, rollback and the layer split then work for every model without code in
the model folder.

## Adding a model (the measure of this design)

1. `src/model/<family>/config.cc` (metadata), `model.cc` (layer composition), one line in `registry.cc`.
2. New block only for a new layer kind (e.g. `mamba2`), with its unit test vs a reference; new kernel shape = one
   struct + an explicit instantiation (CODE.md).
3. `tests/layer/<family>_parity.cu` vs llama-paw or HF activations; `tests/golden/` unchanged for the other models.

Nothing in `engine/`, `runtime/`, `api/` or `server/` changes for a new model. If it has to, the abstraction is
wrong: fix it first, in its own commit, gated by the golden test.
