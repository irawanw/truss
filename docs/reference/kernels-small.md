# Small ops: kernels/hc/, kernels/ffn/, kernels/ple/, kernels/sampling/

Memory-bound elementwise and per-row kernels around the GEMMs. Together ~12% of prefill time (nsys, TRACKER #51).
Rules they all follow: fp32 math, fixed reduction order (block or warp trees, no float atomics), one stream,
`THREADS = 256`; row reductions use one block per row (`hc`, `ple`) or one warp per row (`gdn`, `dsa`).

---

## `src/kernels/hc/hc_prefill.{cuh,cu}` — hyper-connections

The residual is 4 streams, `res [T][4][2560]` fp32. A block reads a mix of the streams and writes back gated.

```
hc mix:     xn[c] = RMSNorm(res[c]) ⊙ γ[c]                  (per stream c, γ [4][2560]; the converter folded 1 + w)
            gate  = W_up · silu(W_down · xn / 4)               (low rank 320; the two GEMMs run in Forward)
            mixed = (1/4) Σ_c xn[c] ⊙ sigmoid(gate[c])
            inject = W_inject · xn                             ([T][4]; skipped for the head mix)
hc combine: res[c] += out ⊙ 2 sigmoid(inject[c] / 4)
```

| function | contract |
|---|---|
| `expand(emb [T][d], T, hc, d, res)` | every stream = the embedding (start of the residual) |
| `norm(res, gamma, T, hc, d, eps, xn16, rstd)` | xn16 fp16 [T][hc][d] (GEMM input), rstd [T][hc] (for collapse) |
| `silu(lo, n, 1/hc, lo16)` | lo16 = silu(lo / hc) in fp16 |
| `collapse(res, rstd, gamma, gate, T, hc, d, mixed, mixed16)` | recomputes xn from res·rstd·γ in fp32; mixed16 nullable |
| `combine(res, out, inject, T, hc, d)` | in place |

## `src/kernels/ffn/ffn_ops.{cuh,cu}` — router and shared-expert elementwise

| function | contract |
|---|---|
| `route(logits [T][E], T, E, k, ids [T][k], wts [T][k])` | softmax, top-k (ties: lower id), weights renormalized over the k with sum clamped ≥ 2^-14; ids in descending probability (the reference's `partial_sort`); one warp per token, E ≤ 1024 (16 or 32 per lane) |
| `swiglu(g, u, n, mid16)` | silu(g)·u → fp16 (down projection input) |
| `shared_add(routed, y, gate [T], T, d, out)` | out = routed + y·sigmoid(gate) |
| `count(ids, n, counts)` | counts[ids[i]] += 1 (float atomics; `Forward::profile_routes`) |

The router logits are fp32 (cuBLAS SGEMM, `Forward::lin32`): top-10 of 512 has near-ties, and rounding the router
would flip them.

## `src/kernels/ple/ple_prefill.{cuh,cu}` — per-layer n-gram embedding block (layer 1)

```
key   = RMSNorm_c(W_key · emb) ⊙ γ_k,  query = RMSNorm_c(res) ⊙ γ_q            (per stream c)
s_c   = key_c · query_c / √2560 ;  gate_c = sigmoid(sign(s_c) √max(|s_c|, 1e-6))
gated_c = value ⊙ gate_c            (value = W_value · emb, shared by the streams)
normed  = RMSNorm_c(gated) ⊙ γ_conv
conv    = silu(Σ_k w[c][k] · normed[t − (K−1−k)·dil])      K = 4, dilation = n-gram size 3
res   += gated + conv
```

| function | contract |
|---|---|
| `gate(key, res, norm_key, norm_query, T, hc, d, eps, gate [T][hc])` | the three sums (Σk², Σr², Σ k γ_k r γ_q) in one block reduction |
| `apply(value, gate, norm_conv, conv_w fp16 [hc·d][K], hist, T, hc, d, K, dil, eps, normed, res)` | `hist` [(K−1)·dil][hc·d] carries the previous normed rows (zeros at start), updated; `normed` is scratch |

The hashing of token n-grams to table rows and the gather from the 48 GiB table run on the host
(`model/qwen4exp/ple.cc`, model-qwen4exp.md).

## `src/kernels/sampling/argmax.{cuh,cu}`

`sampling::argmax(x, n, out, stream)`: one 1024-thread block, ties → lowest index. Used by `truss_eval_argmax` so
greedy decoding copies 4 bytes instead of 1 MB of logits per token.

`sampling::argmax_prob(x, n, map, out, prob, stream)`: the same over a subset of logits, plus the maximum's softmax
probability: `out = map ? map[argmax] : argmax`, `prob = exp(x_max - logsumexp(x))`. The MTP draft head scores the
`draft_vocab` rows and stops drafting once `prob < Options::draft_min_p`.

## Tested by

No unit test per small op: each is checked inside the layer chain. `qwen4exp_parity` checks the reference blocks
they implement against llama-paw; `qwen4exp_forward short/long` checks the engine chain (these kernels + the fast
GEMMs) against the fp32 reference chain at every layer (2.6–3.4e-4 at 12 tokens). A bug here shows up there as a
jump at the layer or block that uses the op.

---

## `src/kernels/spec/rollback.{cuh,cu}` — verify-window rollback

`tail_rows(old [H][w], H, rows [n][w], n, w, out)`: out = the last H rows of concat(old, rows). Carried row histories
(GDN conv rows, PLE conv history) after accepting n of a window's rows, from the pre-window snapshot and the saved
window rows (`Forward::accept`). `out` must not alias the inputs.

## `src/kernels/mtp/mtp_ops.{cuh,cu}` — MTP block input

`join(emb [T][d], enorm [d], hn16 [T][hc][d], T, hc, d, eps, out16 [T][hc][2d])`: per stream `[RMSNorm(emb)·enorm |
hn]`, the eh_proj input (llama-paw `graph_mtp`: `concat(e_norm repeated over streams, h_norm)`). `hn16` is the previous
hidden state normed per stream and scaled by hnorm (`hc::norm`). One block per token.
