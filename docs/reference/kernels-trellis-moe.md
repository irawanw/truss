# Trellis codecs and routed experts: kernels/trellis/, kernels/moe/, encode/

Routed experts are ~40% of Flash-Next's matmul work and ~93% of its bytes (37 GiB of 2.63-bpw trellis). Everything
on this page is about turning trellis words into matmul results fast and exactly.

## The math of one routed expert (all files below implement exactly this)

For token row x (fp32, 2560), expert e with projections gate, up (2560→640) and down (640→2560):

```
A_p   = H128(x ⊙ suh_p)                                   p = gate, up     (suh: fp16 per-input sign·scale)
C_p   = A_p · W_p                                         W_p decoded from trellis tiles
mid   = H128(silu(H128(C_g) ⊙ svh_g) ⊙ H128(C_u) ⊙ svh_u ⊙ suh_d)
y_e   = H128(mid · W_d) ⊙ svh_d
out   = Σ_{s = 1..10, routing order} w_s · y_{e_s}          w_s: renormalized router weight
```

H128 = 128-point Hadamard over consecutive blocks of 128 values, scaled 1/√128. Rounding points (both kernels, and
the same as llama-paw's PAW X3 op): activations are fp16 into the tensor cores; accumulation fp32 (prefill) or fp16
per two k-slices then fp32 (window).

---

## `src/kernels/trellis/mma.cuh` — tensor-core fragments

**What.** Wrappers around `mma.sync.m16n8k16` and `ldmatrix`, with the fragment layout written down.

| helper | use |
|---|---|
| `FragB` | a decoded 16×16 weight tile as two B fragments (f0 = columns 0..7, f1 = 8..15); what a codec's `tile()` returns |
| `mma_w(f0, f1, act_lo, act_hi, FragCh &)` | weights as the A operand, 8 activation rows as B, fp16 accumulate (window kernel) |
| `mma_w_f32(..., float (&c)[4])` | same with fp32 accumulate (prefill kernel; GA102 runs fp32-accumulate at the fp16 rate for this shape) |
| `mma_f32(a[4], b0, b1, c[4])` | generic m16n8k16 fp16→fp32 with ldmatrix-shaped fragments (DSA kernels) |
| `ldsm_x4`, `ldsm_x4_t` | `ldmatrix.x4` (and `.trans`) from shared memory; lane l addresses row l%8 of matrix l/8 |

Fragment order (used by every codec): in a tile, weight j = 8·lane + i sits at (n, k) per the m16n8k16 A-fragment
layout; `ref::trellis_dequant` spells out the mapping. Exllamav3's `tensor_core_perm` is the same map.

## `src/kernels/trellis/hadamard.cuh`

`had4x32(h0..h3, lane)`: in-warp 128-point Hadamard, 4 values per lane in natural order, 5 xor-shuffle stages,
scaled 1/√128. Used by both MoE kernels for H128.

## `src/kernels/trellis/codec_mul1.cuh` — the production codec (PAW X3 / EXL3)

**Format.** 16-bit trellis state per weight, shifted K bits per weight (K = 1..4 here); tile = 256 states =
8K uint32 words. Weight = fp16 fma(bytesum(state·0x83DCD12D) + 0x6400 as fp16, 1/147.7, −10.39)
(`codebook2`, two states per `dp4a`).

**Codec interface** (every codec header provides it; kernels take the codec as a template, never call codebooks
directly): `template <int K> struct Mul1 { NAME; TILE_WORDS = 8K; TILES_PER_VEC (4/2/1/1 for K1..4); VEC_WORDS;
Mul1(lane); loads(lane); tile(w, sub, FragB & f0, FragB & f1) }`. A warp-wide load: lane l reads word l if
`l < VEC_WORDS`; `tile()` then shuffles the neighbouring words into place and extracts 8 states per lane
(`states8_k1..k4`), decoding them to this lane's fragment. Warp-synchronous.

**Speed.** Decode ceiling 2.1–2.7 T weights/s from registers (`tk-bench-ceiling`, TRACKER #17): K2 71%, K3
84–91%, K4 >100% of the 936 GB/s stream rate, so K2/K3 experts are decode-bound at decode sizes.

**Tested by.** `moe_window_test` (vs llama-paw's PAW X3 op), `ref::trellis_dequant` in the parity tests.

## `src/kernels/trellis/codec_v2one.cuh` — faster codec (not in the shipped pack)

Two weights per 16-bit state: `h = ((state·A + B) & 0x8fff8fff) ^ 0x3b603b60`, the two fp16 halves are the
weights (one IMAD + one LOP3 per two weights). Rates K = 1, 1.5, 2, 2.5 (template parameter **S = 2K** bits per
step, since K is fractional). Tile = 128 steps = 4S words, MSB-first, tail-biting. `v2one_codebook_bits` and
`v2one_pack_tile` / `v2one_state` are host+device so encoder, tests and kernel share one definition. Decodes
1.5–1.7× faster than mul1 (3.84 T weights/s at K2, memory-bound), costs +1.5/+10/+15% expert error at
K1.5/2/2.5 (TRACKER #36–#39). Paused: it helps decode only; prefill is not decode-bound.

## `src/encode/v2one_encode.h`, `v2one_encode.cu` — pack-time encoder

Tail-biting Viterbi for v2one tiles (exllamav3's two-pass seam method): pass 1 runs the ring rotated by half with a
free start and reads the state overlap at the seam, pass 2 fixes the seam. Costs keep only the min over the
predecessor-relevant low bits: 2^(16−S) floats per step (shared memory for S ≥ 3, a global slice for S = 2).
`v2one_encode(S, tiles [n][256] fp32 in codebook units, n, q out, states out [n][128], ws, ws_bytes, stream)`;
`v2one_workspace_bytes_per_tile(S)`. Tested by `codec_v2one_test` (bit-exact ring, MSE within 10% of the lab).

---

## `src/kernels/moe/moe_weights.cuh` — shapes and weight views

`moe::FlashNext { D_MODEL = 2560, D_FF = 640, TOPK = 10 }` (a new model = a new struct + explicit instantiations in
the op .cu files). `ProjView { trellis (uint16 base), meta ((K, int32 offset from base) per expert), suh
[n_expert][in], svh [n_expert][out], shift }`; `Weights { ProjView proj[3] (gate, up, down); n_expert }`.
**The kernels address expert e as `trellis + (meta[2e+1] << shift)` words** (int64 math). `shift` = 0 for GGUF
tables (word offsets, what the tests pass); `ExpertStore` uses 4 (32-byte units, ±64 GiB) so one arena holds every
tier and all three projections share a base.

## `src/kernels/moe/moe_window.cuh`, `moe_window.cu` — routed experts for ≤ 8 rows (decode / verify window)

**API.** `workspace_bytes<Shape>()`, `workspace_init<Shape>(ws, stream)` (once after allocation: zeroes queue
counters; each launch leaves them zero again), `window<Shape>(W, x [n][2560] fp32, ids [n][10] int, wts [n][10] fp32,
n ≤ MAX_ROWS = 8, out [n][2560] fp32, ws, stream, trace = nullptr)`. Device-only, CUDA-graph capturable.

**Design (v7a, TRACKER #22).** One persistent launch. Every block builds the routing table in shared memory, then
pulls items from a global queue: H (slot, proj) → GU (slot, proj, column group) → D (slot, column group). Dependencies
are counters, not grid barriers; the block finishing a slot's last GU item runs `mid`, the block finishing a column
group's last D item runs `combine` (routing order, deterministic). Waiting warps issue their weight loads first.
Deadlock-free without co-residency (items only wait on earlier queue items).

**Tunables.** `Tune<FlashNext>`: `BLOCKS_PER_SM = 2` (3 was 13–17% slower, #28); `GateUp = Geom<8,1,2,8>`,
`Down = Geom<4,2,2,8>` (WK k-split warps × WG column groups, PF k-slices in flight, WNT tiles per warp; cp3 tune log
v6); `Plan::GU_KSPLIT = 1` (2 was 5% slower, #24).

**Speed.** 4 rows: K2 ~118–123 µs, K3 ~131, mixed ~130 per layer (#22). GU/D items run at the decode ceiling; losses
are the prologue (~7 µs), start waits (~7 µs) and tail (~10–15 µs) (#26). `tools/tk-bench/moe_trace.cu` prints the
timeline from the kernel's own trace.

**Tested by.** `moe_window_test` (vs llama-paw's unfused PAW X3 chain, random weights, all K) and the parity test
(`moe_window kernel-l` lines vs llama and vs the fp32 reference).

## `src/kernels/moe/moe_prefill.cuh`, `moe_prefill.cu` — routed experts for a prompt chunk

**API.** `prefill_workspace_bytes<Shape>(max_tokens)`, `prefill<Shape>(W, x, ids, wts, n_tokens, out, ws,
max_tokens, stream)`; same math and shapes as `window`, any n_tokens ≤ max_tokens.

**Kernels in order.** `route_count/route_scan/route_place` (stable counting sort of (token, choice) pairs by expert,
one warp per 2048 pairs with `__match_any_sync`) → `prep` (A_gate, A_up = H128(x·suh) in token order) → `gate_up`
(item = expert × 64 rows × 128 D_FF columns; 8 warps = 2 row halves × (proj × column half); decodes each tile once
and applies it to 4 × 8 rows, fp32 accumulate; the mid step is the epilogue) → `down` (item = expert × 64 rows × 256
columns; epilogue writes fp16 C_d = w·H128·svh) → `combine` (per token, sum in routing order). Deterministic.

**Tunables** (constants at the top of the .cu): `BM = 64` rows per item, `KC = 64` (cp.async double-buffered
activation chunks), `AS = KC + 8` (bank-conflict-free stride), `WNT = 4` tiles per warp, `RG = 8` row groups.

**Speed.** 41.7 TFLOPS at 8192 tokens (19.3 ms/layer, ~113 µs/token over 48 layers); 2.0 / 17.8 / 37.6 TFLOPS at
64 / 512 / 2048 tokens (TRACKER #42). **Row results do not depend on the other rows of the chunk** (bit-identical,
`moe_prefill_test` prefix check), so chunked prefill equals one-shot here.

**Tested by.** `moe_prefill_test` (vs `moe::window` 8 rows at a time: rel 3.95e-3, which is the window's fp16
accumulation; worst token ≤ 2e-2; prefix invariance; timing).

**Change it.** A new shape: `Shape` struct + instantiations. Faster: the GEMM loop is tensor-bound at large T;
`prep` and `combine` are separate kernels (~12% of the op) and could be fused (TRACKER #42 notes).
