# Model family qwen4exp (Qwen3.8 Flash-Next): src/model/qwen4exp/

A model family directory holds what is specific to one architecture: hyperparameters, how tensors are named and
shaped, host-side preprocessing (PLE hashing), the slow reference forward (the math definition, checked against
llama-paw), and the fast `Forward` that chains the kernels. Kernels themselves stay generic in `src/kernels/`.

## Flash-Next numbers (from the GGUF; `Config`)

| | value |
|---|---|
| layers | 48: 36 GDN + 12 DSA (`mixer[l]`), PLE at layer 1 (`ple_layers`) |
| d_model / vocab | 2560 / 248,320 (248,077 real tokens + padding) |
| hyper-connections | hc = 4 streams, low rank 320 |
| MoE | 512 experts, top-10, expert FFN 640, shared expert FFN 640, router F32 |
| GDN | 16 key heads, 48 value heads, head 128, conv kernel 4 |
| DSA | 24 q heads, 2 KV heads, head 256, rope 64 dims base 1e7, indexer 4 × 128, top 2048 cells = 512 blocks of 4 |
| PLE | n-gram 3, 8 heads per n-gram → 16 rows of 160 per token from a 320M-row int8 table (48 GiB, host) |
| rms eps | from `attention.layer_norm_rms_epsilon` |

---

## `config.h`, `config.cc` — hyperparameters

`Config::from_gguf(file)` reads `qwen4exp.*` keys (and `n_vocab` from `token_embd.weight`), throws on a missing or
inconsistent key (e.g. `ssm.inner_size != state_size · time_step_rank`, PLE head count ≠ (n-gram − 1)·heads per
n-gram, multipliers length ≠ n-gram). Helpers: `key_dim()`, `value_dim()`, `conv_dim()`, `hc_dim()`,
`ple_heads()`, `is_ple(layer)`. PLE fields: `ple_head_dim, ple_conv, ple_ngram, ple_heads_per_ngram,
ple_head_offsets, ple_head_vocab, ple_multipliers, ple_eos` (**`ple.eos_token_id` = 248044, not the tokenizer's EOS
248046**).

## `weights.h`, `weights.cc` — binding every tensor

`qwen4exp::bind(file, config)` → `Weights`: typed host pointers (`T = const gguf::Tensor *`) for every tensor, per
layer `Layer { mixer, hc_attn, hc_ffn (HyperConnection: norm, down, up, inject), gdn (qkv, gate(z), conv1d, dt_bias,
a, beta, alpha, norm, out), dsa (q, k, v, out, q_norm, k_norm, idx_q, idx_k, idx_q_norm, idx_k_norm), ple (key, value,
norm_key, norm_query, norm_conv, conv1d), moe (router, gate/up/down ExpertTables, shexp_gate_inp, shexp_gate,
shexp_up, shexp_down) }`, plus `token_embd`, `output`, `hc_head` (`output_hc_*`, no inject), `ple_table`,
`ple_scale`. **Every shape is checked against the config and every tensor in the file must be used exactly once**,
so a converter change fails at load instead of computing garbage. Names follow llama-paw
`src/models/qwen4exp.cpp` (`blk.N.attn_qkv.weight`, `blk.N.ffn_gate_exps.m3_trellis`, `per_layer_token_embd.q8`, ...).
The MTP layer (blk.48 in MTP packs) is not bound yet (CP5).

**MTP block.** `bind_mtp(file, config)` → `Mtp { Layer layer (DSA mixer, routed experts, no PLE); eh_proj [2d → d],
enorm [d], hnorm [hc_dim] }` from a separate file (`flashnext_truss_mtp_pack.py`: blk.48 dense tensors from the Q8_0
MTP pack, its 512 experts re-encoded to X3 K3 by `flashnext_truss_mtp_encode.py` with exllamav3 and an identity
Hessian, weight NMSE 0.017). It uses the main model's token_embd, output and hc_head (llama-paw `graph_mtp`). The
per-layer binding is one function (`bind_layer`) shared by `bind` and `bind_mtp`.

PLE hashing and the row reader moved to `src/blocks/` (`ple_hash`, `ple_reader`, `ple`): see
[engine-blocks.md](engine-blocks.md). `Config::ple_dims()` is this family's view of them.

## `reference.h`, `reference.cu` — the math, block by block

fp32 blocks on the reference ops (kernels-reference.md), one sequence from position 0, temporaries from a
`Scratch`; `Ctx { config, DeviceTensors, scratch, stream, Numerics }`. Blocks: `hc_mix`, `hc_combine`, `gdn`,
`dsa`, `ple`, `route`, `routed`, `shared`, `ffn`. Optional `*Trace` structs expose intermediates for tests
(`GdnTrace`, `DsaTrace` incl. q/k/v, gate, idx_q/idx_k, selection mask; `PleTrace`). Each block's comment gives the
llama-paw function it follows. Checked by `qwen4exp_parity` (160/160, block by block on llama's own inputs).

## `model.h`, `model.cu` — the family as engine composition (docs/ENGINE.md)

`Family` binds the file (`Config::from_gguf`, `bind`), checks the expert and DSA shapes against `moe::FlashNext` /
`dsa::FlashNext`, and binds the MTP drafter file when given (`DrafterFiles`). `StageModel` is one stage:

- `add_tensors`: the stage's layers' tensors (+ token embedding on the first stage and the drafter's, the head and its
  hc mix on the last), into `blocks::Linear`; the untied token embedding stays in pinned host memory (`host_rows`).
- `build`: `Hc`, a `PleTable` when the stage has PLE layers, per layer `Gdn` or `Dsa<FlashNext>` and `Ple`, the
  `Head` on the last stage, the `MtpDrafter`. `bind_experts`: per layer a `Moe` (softmax top-10 router `ffn::route`,
  shared expert, the next layer's router as the pre-gating hint), store layer `l - lb`.
- `layers`: per layer [PLE] -> hc mix (the previous layer's combine folded into its norm when allowed) -> mixer -> hc
  mix (combine #1 folded) -> MoE -> combine (or fold into the next layer). Section events 0, 1, 2, 6.
- `head`: hc_head mix -> `Head::logits`. `scratch_bytes`: per-row peak of the widest block (+ the MTP join when the
  family has a drafter) + DSA select / attention workspaces.
- `table_stats`: the PLE host gather counters (`tk-bench-spec` prints them).

## `mtp.h`, `mtp.cu`, `mtp_draft.cu` — the MTP drafter

`MtpDrafter` (engine::Drafter + Stateful) on the last stage: its own `Dsa<FlashNext>` and `Moe` (store layer
n_store - 1, not profiled, no CPU tier), the stage's `Hc` and `Head`. Per slot: pending hidden row, deferred commit
(the accepted rows; merged with the first draft row into one pass, `TRUSS_MTP_MERGE`), the "ahead" row, parked rows
for slot switches. `on_commit` runs the block over every committed row from the previous position's hidden state;
`draft` chains single-row steps (argmax or a sample of q), restores the indexer's open block after, stops below
`DrafterFiles::min_p`. `TRUSS_MTP_CALIB=file` appends the block's FFN inputs of true-path commits.

## Vision (image input) - in progress, plan: `docs/PLAN-20261022-flashnext-vision.md`

Base model has a 27-block ViT (hidden 1152, 16 heads, MLP 4304, GELU-tanh, patch 16, temporal patch 2, 2x2 merge, 2D rotary,
learned 48x48 pos-embed bilinearly resampled, no DeepStack) whose merger outputs 2560 = text hidden. The X3.1 pack is text-only.

- `tools/tk-pack/flashnext_mmproj.py` writes `flashnext-mmproj-f16.gguf` (0.9 GB, arch `qwen4exp-mmproj`, 333 tensors named
  `v.<HF name without model.visual.>`; matrices + conv kernel + pos_embed F16, biases/norms F32; hyperparameters in
  `qwen4exp.vision.*`). Verified: 9 tensors read back equal to the safetensors (rel err <= 4e-7).
  Run: `python3 tools/tk-pack/flashnext_mmproj.py --model-dir <HF dir> --out flashnext-mmproj-f16.gguf`.
- `server/vision_pre.py`: image -> `pixel_values [N, 1536]` + `grid_thw`. Same patch order as HF `Qwen2VLImageProcessor`
  (merge-window grouped); sizes multiples of 32, long side capped at 1344. N/4 image tokens reach the text model.
- HF behaviour to copy (transformers `modeling_qwen4_exp.py`, `Qwen4ExpModel.forward`): image rows replace the embeddings at
  `image_token_id` positions only; PLE and the n-gram tables still receive the original token ids (so image rows use the
  `image_token_id` PLE row); position ids are 3D (`compute_3d_position_ids`).
- Fixtures (ViT taps from HF on a 448x448 test image) live outside the repo in `ML_projects/flashnext/20261022_vision/fix/`.
