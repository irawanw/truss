# Model family lfm2moe (LiquidAI LFM2 / LFM2.5 MoE): src/model/lfm2moe/

The second family on the engine split (docs/ENGINE.md), added 2026-10-14 with **no edits to engine/, api/ or
server/**: three files here, one registry line, new blocks for the new layer kinds.

## LFM2.5-24B-A2B numbers (GGUF metadata)

40 layers: 30 short-conv (L 3), 10 GQA attention (layers 2, 6, ..., 38; 32 q / 8 kv heads, head dim 64, q/k RMSNorm,
NEOX rope base 1e7); d_model 2048; layers 0-1 dense SwiGLU (11,776), layers 2-39 MoE (64 experts, top 4, d_ff 1536,
sigmoid + expert-bias router, renormalized, no shared expert); vocab 128,000, head tied to token_embd; RMSNorm eps 1e-5.

## Files

| file | content |
|---|---|
| `config.h/.cc` | `Config::from_gguf` (llama.cpp `lfm2moe.*` keys; attention where `attention.head_count_kv[l] > 0`; checks `expert_gating_func` 2) |
| `weights.h/.cc` | binding (`formats::Binder`): `attn_norm` = operator norm, `token_embd_norm` = final norm; experts as trellis tables; `exp_probs_b.bias` the router's selection bias |
| `model.h/.cu` | `Family` (shape checks vs `moe::Lfm2`), `StageModel`: per layer `res += op(op_norm(res))` (ShortConv or AttnGqa), `res += ffn(ffn_norm(res))` (DenseFfn or Moe with `route_sigmoid_bias`); embed = Q8 rows; head = final norm + `Head` on token_embd. Store layers are the MoE layers only (`sl_`) |

## Blocks and kernels it added (reusable)

`blocks/norm.h` (RmsNorm), `blocks/ffn_dense.h` (DenseFfn), `blocks/shortconv.*` (ShortConv, stateful: conv rows,
window rollback by `spec::tail_rows`), `blocks/attn_gqa.*` (AttnGqa, positional K/V); kernels `norm/rms_norm`,
`conv/short_conv`, `attn/gqa` (v1 CUDA-core tiles + split cells), `ffn/route_sigmoid`; `moe::Lfm2` shape.

## Tests

`tests/layer/lfm2moe_parity.py`: TRUSS through the C API vs HF's Lfm2Moe layers on the same weights (Q8_0
dequantized, trellis reconstructed by exllamav3). 4-layer K2 slice, prompts 120 / 128 / 200 + 6-8 decode steps:
mean KL 2.3e-4 / 5.3e-4 / 5.4e-4 (gate 2e-3). Kernel tests: `gqa_attn_test`, `lfm_small_ops_test`.
