# formats/ and core/: reading model files, device memory, errors

Everything here is host-side plumbing with no model knowledge. Files never copy tensor data on the host: the GGUF
is memory-mapped and every consumer reads through pointers into the mapping.

---

## `src/formats/gguf.h`, `gguf.cc` — GGUF v3 reader

**What.** Opens a single-file or split (`...-0000N-of-0000M.gguf`) model, maps every shard read-only, parses header,
metadata and tensor infos with bounds checks, and exposes one catalog.

**API.**

| call | returns / does |
|---|---|
| `gguf::File::open(path)` | `std::unique_ptr<File>`; `path` may be any shard, all shards are opened. Throws `std::runtime_error("gguf: <path>: <why>")` on anything malformed |
| `tensors()` | `std::vector<Tensor>` in file order across shards |
| `find(name)` / `at(name)` | `const Tensor *` (nullptr if absent) / `const Tensor &` (throws) |
| `get(key)` | `const Value *` or nullptr; `Value` = `std::variant<int64_t, double, bool, std::string, vector<int64_t>, vector<double>, vector<bool>, vector<string>>` |
| `get_int / get_float / get_string / get_ints / get_floats / get_bools(key)` | typed access; throws if absent or of another kind (`get_float` accepts ints) |
| `shard_paths()` | the shard files in order |
| `type_info(Type)` | `{name, block, bytes}` per ggml type, nullptr if unknown |

**`Tensor` fields.** `name`, `type` (`gguf::Type`, ggml numbering, e.g. `Q8_0 = 8`, `I16 = 25`), `shape` (**ggml
order: `shape[0]` is the contiguous dimension**, so a matrix with `out` rows of `in` weights has `shape = {in,
out}`), `shard`, `file_offset` (absolute in its shard), `bytes`, `data` (`const std::byte *` into the mapping),
`elements()`.

**Invariants / checks.** Magic and version 3; every tensor type known (`type_info` non-null); data offsets aligned
to `general.alignment` (default 32) and inside the shard; for split files `split.no` matches the file name and
`split.tensors.count` equals the tensors found; integer metadata widened to int64 (uint64 values above INT64_MAX are
rejected); metadata comes from shard 0.

**Tested by.** `tests/unit/gguf_reader_check.py` (every tensor's name, type, shape, shard, offset, bytes vs
llama-paw's gguf-py, through `tk-pack-inspect list`), TRACKER #29.

**Change it.** A new ggml type: add it to `enum class Type` and the `type_info` table. Nothing else parses GGUF.

---

## `src/formats/trellis_table.h`, `trellis_table.cc` — routed-expert trellis tables

**What.** One routed projection (gate, up or down) of all experts of a layer, stored by the PAW X3 converter as four
tensors `<prefix>.m3_trellis` (I16, all experts' tiles concatenated), `.m3_meta` (I32 `[n_expert][2]` = K, word
offset), `.m3_suh` (F16 `[n_expert][in]`), `.m3_svh` (F16 `[n_expert][out]`).

**API.** `formats::read_expert_table(file, prefix)` → `ExpertTable { n_expert, in, out, k[e], offset[e] (uint16
words), trellis, meta, suh, svh (tensor pointers), words(e) = in·out·K/16 }`.

**Checks (throws naming the projection).** Types as above; meta `[n_expert][2]`; suh/svh shapes agree; in, out
multiples of 16; `1 ≤ K ≤ 8`; offsets contiguous and in expert order; total words = the trellis tensor.

**Layout of one expert's words.** 16×16 tiles, **k-tile-major**: tile (kt, nt) starts at uint32 word
`(kt · out/16 + nt) · 8K`; inside a tile, weight j = 8·lane + i in tensor-core fragment order; the exact bit
definition is `ref::trellis_dequant` (kernels-reference.md).

**Used by.** `qwen4exp::bind` (weights.cc), `runtime::ExpertStore` (copies expert byte ranges by `offset`/`words`),
`tk-pack-inspect`.

---

## `src/core/cuda_check.h`, `cublas_check.h` — error checks

`TRUSS_CUDA(call)` and `TRUSS_CUBLAS(call)` throw `std::runtime_error` with the error string, file, line and the
call text. Every host CUDA / cuBLAS call in `src/` goes through one of them (CODE.md: fail loudly).

---

## `src/core/device_tensors.h`, `device_tensors.cc` — model tensors on the GPU

**What.** Uploads a list of `gguf::Tensor`s into **one** `cudaMalloc` (each 256-byte aligned) with synchronous copies
from the mapping, and maps the host `Tensor *` to its device copy, so ops look weights up with the host binding
(`qwen4exp::Weights`) and there is no second model structure on the device.

**API.** `DeviceTensors(std::vector<const gguf::Tensor *>)` (duplicates and nullptrs ignored);
`operator()(const Tensor *)` → `const DTensor &` (throws if not uploaded); `bytes()`.
`DTensor { data (device), type, ne[4] (ggml order), elements(), as<T>() }`.

**Used by.** `Forward` (norms, F32/F16 tensors, conv weights; *not* Q8_0 matrices, which are repacked, nor experts,
which live in the `ExpertStore`), the reference forward and parity tests (everything).

---

## `src/core/scratch.h` — bump allocator for temporaries

**What.** One device buffer; `alloc<T>(count)` hands out 256-byte-aligned pieces; `reset()` frees all;
`mark()` / `release(mark)` free everything allocated after a mark. Throws `"Scratch: out of space"` when full (sizes
are computed up front: `Forward::Impl::scratch_bytes`), never grows.

**Why it is safe to reuse memory right after release.** All users run on one stream: a kernel that reuses freed
space is ordered after the kernels that used it.

**Change it.** If a new op needs more temporaries, add its per-token bytes to `Forward::Impl::scratch_bytes`; an
under-estimate shows up as the exception above (it happened once: the embedding buffer was missing, TRACKER #52).
