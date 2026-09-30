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

## `ple.h`, `ple.cc` — PLE rows on the host

- `ple_rows(config, tokens, T, rows [T][16])`: for n = 2..3 the window (token, n−1 predecessors) is hashed,
  `mixed = xor_j window[j]·multiplier[j]` (uint64, wrapping); head h of that n-gram reads row
  `mixed % vocab[h] + offset[h]`. A predecessor before the sequence start, or at/before a `ple_eos` in the window,
  reads as `ple_eos`; the token's own `ple_eos` does not cut its window (llama-paw `llm_graph_input_ple::set_input`).
- `ple_gather(config, weights, rows, T, emb [T][16·160])`: int8 row × its fp16 scale, heads outermost per token.

Bit-exact vs llama-paw's `ple_embd` (parity test). `Forward` keeps the last n−1 tokens (`tail`) so chunks hash as one
sequence.

## `reference.h`, `reference.cu` — the math, block by block

fp32 blocks on the reference ops (kernels-reference.md), one sequence from position 0, temporaries from a
`Scratch`; `Ctx { config, DeviceTensors, scratch, stream, Numerics }`. Blocks: `hc_mix`, `hc_combine`, `gdn`,
`dsa`, `ple`, `route`, `routed`, `shared`, `ffn`. Optional `*Trace` structs expose intermediates for tests
(`GdnTrace`, `DsaTrace` incl. q/k/v, gate, idx_q/idx_k, selection mask; `PleTrace`). Each block's comment gives the
llama-paw function it follows. Checked by `qwen4exp_parity` (160/160, block by block on llama's own inputs).

## `forward.h`, `forward.cu` — the fast engine for one sequence

**API.**

```
Forward(config, weights, n_ctx, max_chunk, Options{ expert_budget = 0 (= all free memory − 768 MiB),
                                             act = Q8_1, expert_usage = {} (ExpertStore::plan),
                                             ring_bytes = 4 GiB (decode FIFO; prompt chunks borrow it) })
run(tokens, T, hook = nullptr)   // append T tokens (a prompt chunk or a decode step); hook(layer, res, T) per layer
head(first, n, logits)           // logits [n][vocab] (device fp32) for rows of the last chunk; Q8_1: one
                                 // q8_gemv/q8_gemm over the vocab; FP16: q8_gemm_a16 in vocab tiles
reset()                          // new sequence
verify(tokens, T); accept(n)     // speculative window (Options::spec_rows): T rows tentative, keep the first n
draft(next, n, out)              // MTP drafts (Options::mtp) for the tokens after `next`
position(), hot_experts() (all layers), cold_bytes(), stream(), experts() (the ExpertStore: stats)
profile_routes(on), route_counts()   // routing profile [layer][expert] (the usage file; tk-profile)
```

`config` and `weights` must outlive it (weights point into the file mapping). Single stream; not thread-safe.

**What it owns (`Forward::Impl`).**

| member | content |
|---|---|
| `dev` | `DeviceTensors` for norms, F32/F16 tensors, conv weights (not Q8_0 matrices, not experts) |
| `q8` | map tensor → `dense::Q8Matrix`, all Q8_0 matrices repacked at load (one staging buffer). The token embedding (untied) is repacked through temporaries into pinned mapped host memory: the gather reads one 2.7 KB row per token over PCIe and its 0.68 GB of VRAM goes to the expert cache |
| `experts` | `runtime::ExpertStore` (hot/cold routed experts; runtime-api-server.md) |
| `st[l]` | per-layer state: DSA `k`, `v` fp16 [n_ctx][2][256], `idx_k` [n_ctx/4][128], `idx_partial` [3][128]; GDN `state` [48][128][128] fp32, `conv` [3][10240]; PLE `ple_hist` [9][10240] |
| `small`, `big` (`Buffers`) | per-chunk scratch (`scratch_bytes`: per-token peak of the widest block + DSA select and split-attention workspaces), `moe::prefill` workspace, residual [rows][4][2560]. `small` is resident, sized for 32 rows (decode, verify windows); `big` (max_chunk rows) lives in the ExpertStore ring's spare region and exists only during prompt chunks. `use()` selects one per `run`; the residual persists after `run` for `head` |
| `w16` | `q8_gemm_a16`'s dequantized-weight scratch (largest non-head matrix) |
| `window_ws`, `ids_host`, `route_counts` | `moe::window` workspace; pinned routing buffer for the expert fetch; routing profile while on |

**Per chunk.** See the table in [README.md](README.md). Mode choices inside:
- `lin`: Q8_1 → `q8_gemv` (≤ 8 rows) / `q8_gemm`; FP16 → `q8_gemm_a16`. `lin32`: cuBLAS SGEMM for F32 weights.
  An input read by several projections is quantized once (`quant()` → `Act`, in the caller's scratch scope): GDN
  qkv/gate/alpha/beta, DSA q/k/v/indexer, hc down + inject, shared gate + up, PLE key + value.
- MTP drafts may score only token ids < `Options::draft_vocab` (the head reads that share of the output matrix).
- FFN residency: `fetch_mode(T)` = T ≤ `FETCH_ROWS` (32): read routing back (with layer l+1's router applied to this
  layer's input: pre-gating predicts 72% of its experts; each row's top `hint_k` go to `ExpertStore::prefetch_hint`),
  `ExpertStore::fetch` the routed cold experts, then `moe::window` (T ≤ 8) or `moe::prefill`; otherwise whole layers stream (`prefetch(0), prefetch(1)`
  at chunk start, `prefetch(l + 2)` after layer l's experts).
- Chunks may start at any position (DSA partial blocks, GDN conv rows, PLE history and window are carried).

**Speculative decoding (Options::spec_rows, Options::mtp).**
- `verify(tokens, T ≤ spec_rows)` runs the window like a chunk but tentative: each GDN layer's delta rule writes to the
  layer's second state buffer; GDN conv rows, the DSA indexer's open block and PLE history are snapshotted, and the
  window's raw rows (qkv, alpha, beta; indexer keys; PLE normed rows) saved. `head()` gives every row's logits.
- `accept(n)`: all rows → swap the GDN buffers; fewer → restore the snapshots and redo only the small recurrent parts
  for rows 0..n−1 (`gdn::prepare` + `delta_rule` in place, `dsa::carry_partial`, `spec::tail_rows`). K/V and
  indexer-block caches are position-indexed and are rewritten before any query reads them. The PLE token window
  advances by n.
- MTP: every committed chunk (run or accept) also runs the MTP block at its positions, each from the previous
  position's target hidden state (`pending_h` for the first; position 0 skipped), so the MTP cache matches the true
  states. `draft(next, n)` chains single-row MTP steps from `pending_h`: embed token → join → eh_proj → DSA layer →
  experts (store layer 48) → head → device argmax → next step; its cache rows and indexer block are tentative (the
  block is snapshotted and restored; the rows are rewritten by the next commit).
- Without an MTP usage profile the MTP layer's experts rank above every layer's, so all 512 stay resident.
- Measured (TRACKER #58): greedy output identical to plain decode; 2.91 tokens per pass with 3 drafts; 38 tok/s
  (plain 32): each 4-row pass fetches ~363 experts (640 MB, ~47 ms of PCIe) — the miss bytes are the wall.

**Tunables.** `FETCH_ROWS = 32` (a 67-token prompt spent 2.1 s streaming ~24 GB before this); the expert budget
margin `768 MiB`; `max_chunk` (constructor; larger chunks amortize the per-layer expert stream: 4K 2,263 tok/s,
8K+ ~2,500).

**Tested by.** `qwen4exp_forward` (short: chain vs fp32 reference and llama at every layer; stream: streamed ==
resident bit-exact; decode: token steps vs one chunk; long: chunked vs one chunk and mixer inputs vs llama at 3.7K
tokens) on the slice; `tk-parity-kl` on the full model (KL vs llama-paw logits); `tk-bench-prefill` for speed.

**Change it.** A new block or model: write its reference block first and parity-test it, then its fast kernels and
unit tests, then call them here and extend `scratch_bytes`. Keep the math identical to `reference.cu` — the tests
compare the chains layer by layer.
