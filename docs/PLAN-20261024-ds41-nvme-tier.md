# PLAN 10-24: a fourth expert tier on the NVMe (pack E2)

User 10-24: "lets jump to E2", "try start engine change for nvme". TRACKER #261 / #262.

## 1. Why

Pack E2 = pack E1 (each layer's 256 experts the API's reasoning routes to most, 101.6 GB, VRAM + pinned RAM) plus the
other 128 experts of every layer at K2 (5,120 experts x 8.85 MB = **45.3 GB**), which do not fit RAM (125 GB box, the
cold copy already pins ~82 GB). Estimated output error 0.123 (E1) -> 0.097 (E2). E0 (= E1's experts, older rates)
measured gap +0.105 vs A's +0.212 (#261): the rare experts are the next lever, and no layer is pruned.

## 2. Load per token (decode)

- 3.2% of the reasoning routings (P1 train windows) go to the 128 rare experts: 240 routings/token x 0.032 =
  **~7.7 far experts/token = 68 MB/token before any cache**.
- NVMe: PS0 3.2 GB/s (PS1 ~1 GB/s, `nvme-ps1` memory), and the Engram tables (111 GB) are already read row by row
  from the same drive on every step (small rows, ~50/token).
- 40 tok/s x 68 MB = 2.7 GB/s uncached = the drive's ceiling. **Reuse decides it**: rare experts are bursty (a
  topic routes to the same few); the slot cache must hold the recent ones. To measure from P1's ids before writing
  more than the reader: hit rate vs cache size on the 64 x 4096 reasoning windows (data/p1 err_LNN ids).

**F1 result** (`20261019_truss_port/scripts/f1_far_sim.py`, P1 routing over all 64 reasoning windows = 262K
tokens, E1's far set): **7.96 far routings/token (70 MB/token uncached)**. Misses/token with an LRU slot cache
(cold start per 4,096-token window; a chained 262K session is within 1%):

| cache | 0.5 GB | 1 | 2 | 4 | **6** | 8 | 12 | 20 |
|---|---|---|---|---|---|---|---|---|
| misses/token | 6.08 | 5.37 | 4.52 | 3.54 | **2.94** | 2.51 | 1.89 | 1.20 |
| NVMe GB/s at 40 tok/s | 2.15 | 1.90 | 1.60 | 1.25 | **1.04** | 0.89 | 0.67 | 0.43 |

LFU with decay (halve every 1,000 tokens) is worse on the 16 gap windows (6 GB: 5.10 vs LRU 3.07 misses/token):
**LRU** (as the ring: FIFO = LRU within 3%, TRACKER #57). 6 GB -> ~1 GB/s at 40 tok/s, a third of PS0's 3.2 GB/s
(but 100% of PS1's ~1 GB/s: PS0 is required). ~3 misses/token x ~3 ms read: they must overlap (8 reader threads,
hint prefetch) or be skipped.

## 3. What others do (pulled 10-24)

| project | cold experts on disk | how |
|---|---|---|
| dsx (lxe, 2x3090, 128 GB, same model) | ~3,500 coldest stay in the GGUF, read on demand | O_DIRECT `read_at` into a 4 KiB-aligned bounce buffer: **5.9 GB/s vs 3.9-4.6 through the mmap** on their drive, no page cache to reclaim. DESIGN.md rule 1: "No experts on disk" (7x a RAM read) - disk is the last resort |
| Maya v1.1.26 (DS41 Maya-S) | `--mmap-experts auto`: the experts RAM cannot hold are read from the SSD | mmap + `posix_fadvise(WILLNEED)` for the layer's routed blobs, issued before the pool needs them; CPU lane computes them. Notes: a random-access hint measured 14x slower ("retention is the whole game"). Own `DirectFile` (O_DIRECT preads from a 16-thread pool, 4 KiB contract) only for the n-gram table |
| sybil dsv41-flash-offload | none (256 GiB RAM) | Engram tables mmapped on NVMe |
| Strata (Niko1221, mmnga) | GGUF experts via mmap fallback when RAM is short ("much slower") | page cache |
| TensorFold, tqLLM, Xolotl, DeepInfer | nothing on expert NVMe tiers | |

Takeaways: O_DIRECT, not mmap (our RAM is full: page cache would fight the pinned cold copy); expert byte ranges
4 KiB-aligned in the far file so a read lands **directly in a pinned slot** (no bounce copy, unlike dsx); read
ahead of need (Maya's fadvise point = our `prefetch_hint`).

## 4. Design

New `runtime/expert_far.{h,cc}` (host only, no CUDA in its core, unit-testable). The reads reuse `gguf::read_direct`
(formats/gguf_io.cc: O_DIRECT through a 4 MiB bounce buffer, already used by the cold-copy fill), so the far file is
an ordinary mapped GGUF overlay and needs no alignment of its own:

- **FarFile**: one GGUF (`ds41-experts-far.gguf`, tk-pack writes it: the far experts' trellis bytes, each expert's
  gate|up|down contiguous and its start 4 KiB-aligned). Opened `O_DIRECT`; falls back to buffered `pread` (tmpfs,
  tests) with a counter.
- **FarCache**: N pinned slots of the largest far expert (TRUSS_FAR_CACHE_GB, default 6 -> ~680 K2 slots), one
  `cudaHostRegister`ed region. Map (layer, expert) -> slot. Eviction: LRU (F1: beats LFU with decay), never a slot being read or copied (pin count).
- **Reader pool**: TRUSS_FAR_THREADS (8) threads doing `pread` into slots; requests: `want(l, e, prio)` with
  priority demand > hint. Completion = an atomic state per slot (EMPTY / READING / READY).
- **ExpertStore** changes: a cold expert is "far" when it has no host copy (`Layer::far[e]`); `host_part()` returns
  the slot pointer when READY, nullptr otherwise. `fetch()` for a far expert: READY -> copy from its slot like a
  host expert; else per policy (below). `prefetch_hint(l, ids)` also calls `want()` for far ids of layer l + 1
  (and l + 2: one NVMe read ~3 ms for 8.85 MB at 3 GB/s = longer than a layer's ~0.6 ms).
- **Router**: E2's dense keeps all 384 router rows (unlike E0 / E1, where dropped experts leave the router).

### Miss policy (decode), TRUSS_FAR_MISS

- `wait`: block until the read lands (~3 ms each, overlapped across the token's misses): ~30-35 tok/s estimated.
- `skip` (default to try first): a far expert not READY when its layer runs is dropped for this token and the
  other routed weights renormalised (the same code path as `skip_weight`, moe_serve.cu); its read continues, so the
  next token has it. 40+ tok/s; quality between E1 and E2. Counted: far routed / hit / skipped / waited per pass.

### Prompt chunks (the open question)

The stream path copies **every** cold expert of a layer per chunk. With far experts that would be 1.1 GB/layer =
45 GB per chunk from the NVMe (~15 s per 8K chunk at 3 GB/s: prefill ~2,200 -> ~500 tok/s). Options:
1. **skip far experts in prompt chunks** (route them out, renormalise): prefill speed unchanged; the prompt's KV is
   built without them (as E1), decode uses them. Cheap; quality loss only on prompt positions.
2. read only the chunk's routed far experts, cached ones first: for 8K-token chunks almost all are routed (8K x 6 x
   3.2% = ~1,570 routings over 128 experts/layer), so ~= 45 GB anyway; only short chunks gain.
3. compute far experts on the CPU from their slots (the CPU tier reads `host_part`): the CPU tier's rate limits it.
Proposal: 1 for chunks >= TRUSS_FETCH_PROMPT tokens, 2 below; measure the gap with both (the gap task's 3,840-token
prompt runs on the stream path, so its number includes the prompt choice).
**Decided 10-24 (user: "yes OK make fast but accurate"):** option 1, done more accurately than a zeroed term: in a
chunk the far experts' router bias is -1e30, so the router picks the next experts (exactly E1's routing on prompt
tokens). Decode: wait up to TRUSS_FAR_WAIT_US (2 ms) for all of a layer's far reads together, then skip the late ones.

## 5. Steps

| # | step | GPU | done when |
|---|---|---|---|
| F1 | hit-rate sim of the slot cache on P1 reasoning routing | no | DONE 10-24 (sec. 2) |
| F2 | `expert_far.{h,cc}` (FarCache: slots, LRU, reader threads, demand/hint queues; reads via gguf::read_direct) + `tests/unit/expert_far_test.cc` | no | DONE 10-24: ok (plain + ASan/UBSan, 4 runs); found + fixed: a hint evicted a demand read's slot before its waiter took it (`fresh` flag) |
| F3 | one 384-expert file (no separate far GGUF: the far experts stay unread in the mapped pack): `20261019_truss_port/scripts/s9_e2_inputs.py` (keep all, kmap E1 + K2, far.u8, usage.f32) + `s9_e2_run.sh` (dense 384 rows, ds41_experts.py over enc_s8 + old roots, then bench E1 = TRUSS_FAR_DROP=1 and E2) | no | written 10-24; inputs dry-run OK (146.9 GB, far 45.3 GB, 3.2% traffic); runs after the S8 pass |
| F4 | ExpertStore far experts (plan / slots / fetch / hints / CPU tier exclusion), decode wait-then-skip (TRUSS_FAR_WAIT_US), prompt chunks route around (bias mask, user OK 10-24 "make fast but accurate"), TRUSS_FAR_DROP, far_report | build after the S8 pass | code written 10-24, every changed file compiles (nvcc/g++, no warnings); engine build + Flash-Next golden 7/7 + DS41 host tests pending |
| F5 | bench E2 (gap + plan + tok/s) wait vs skip | yes | TRACKER row |

Rules: no engine build or GPU test while the S8 stream / encode pass runs (#262); docs/ENGINE.md + SERVING-DS41.md
updated with F4; gated (no far file = today's engine exactly).
