#!/usr/bin/env python3
"""Check truss::gguf against an independent reader (llama-paw's gguf-py): every tensor's name, type, shape, shard,
absolute data offset and byte count must match. Exit code is the verdict.

usage: gguf_reader_check.py <tk-pack-inspect binary> <model.gguf | any shard> [llama-paw dir]"""
import os
import re
import subprocess
import sys

inspect, model = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/llama-paw"), "gguf-py"))
import gguf  # noqa: E402

ours = [l.split("\t") for l in subprocess.run([inspect, model, "list"], check=True, capture_output=True,
                                              text=True).stdout.splitlines()]
split = re.fullmatch(r"(.*)-\d{5}-of-(\d{5})\.gguf", model)
paths = [f"{split[1]}-{i + 1:05d}-of-{split[2]}.gguf" for i in range(int(split[2]))] if split else [model]
ref = []
for s, p in enumerate(paths):
    for t in gguf.GGUFReader(p).tensors:
        ref.append([t.name, t.tensor_type.name, ",".join(str(int(d)) for d in t.shape), str(s), str(t.data_offset),
                    str(t.n_bytes)])

bad = [(a, b) for a, b in zip(ours, ref) if a != b]
if len(ours) != len(ref) or bad:
    print(f"FAIL: {len(ours)} vs {len(ref)} tensors, {len(bad)} differ; first: {bad[:2]}")
    sys.exit(1)
print(f"PASS: {len(ours)} tensors in {len(paths)} shards match gguf-py (name, type, shape, shard, offset, bytes)")
