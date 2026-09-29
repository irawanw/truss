#!/usr/bin/env python3
"""Cut a qwen4exp split GGUF down to its first N layers, for layer parity on one GPU (CP2): llama-paw's X3 MoE op
is CUDA-only, so the reference must fit the GPU. Layers are sequential, so layer l of the slice computes exactly
what layer l of the full model does; only the final head sees a different input.

Writes a new first shard (<out_stem>-00001-of-0000M.gguf) with block_count = N, the per-layer metadata arrays
cut to N, and split.tensors.count updated. The later shards (the PLE table, 52 GB) are unchanged, so they are
hard-linked beside it instead of copied (same filesystem required).

usage: slice_gguf.py <first shard of the source> <out_stem> <N> [llama-paw dir]"""
import os
import re
import sys

src, out_stem, n_keep = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, os.path.join(sys.argv[4] if len(sys.argv) > 4 else os.path.expanduser("~/llama-paw"), "gguf-py"))
import gguf  # noqa: E402

split = re.fullmatch(r"(.*)-00001-of-(\d{5})\.gguf", src)
if not split:
    sys.exit("source must be the first shard of a split GGUF")
n_split = int(split[2])
r = gguf.GGUFReader(src)
arch = r.fields["general.architecture"].contents()
if arch != "qwen4exp":
    sys.exit(f"architecture {arch}: only qwen4exp is supported")
n_layer = r.fields[f"{arch}.block_count"].contents()
if not 0 < n_keep <= n_layer:
    sys.exit(f"N must be in 1..{n_layer}")

per_layer = {f"{arch}.attention.recurrent_layers", f"{arch}.attention.compress_ratios"}
keep = [t for t in r.tensors if not (m := re.match(r"blk\.(\d+)\.", t.name)) or int(m[1]) < n_keep]
dropped = len(r.tensors) - len(keep)
total = r.fields["split.tensors.count"].contents()

out = f"{out_stem}-00001-of-{n_split:05d}.gguf"
w = gguf.GGUFWriter(out, arch)
for f in r.fields.values():
    if f.name == "general.architecture" or f.name.startswith("GGUF."):
        continue
    val = f.contents()
    if f.name == f"{arch}.block_count":
        val = n_keep
    elif f.name in per_layer:
        val = val[:n_keep]
    elif f.name == "split.tensors.count":
        val = total - dropped
    sub = f.types[-1] if f.types[0] == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, val, f.types[0], sub_type=sub)
for t in keep:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for t in keep:
    w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.close()

for i in range(2, n_split + 1):
    link = f"{out_stem}-{i:05d}-of-{n_split:05d}.gguf"
    if not os.path.exists(link):
        os.link(f"{split[1]}-{i:05d}-of-{n_split:05d}.gguf", link)
print(f"{out}: {n_keep} of {n_layer} layers, {len(keep)} tensors in shard 1 ({dropped} dropped), "
      f"{n_split - 1} later shard(s) hard-linked")
