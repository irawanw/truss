#!/usr/bin/env python3
"""Reference data for tests/layer/ds41_engram_parity (blocks::Engram):
  1. a sparse full-size int4 table GGUF of one Engram layer (tools/tk-pack/ds41_engram.py layout) holding only the
     rows the PoC captured for its corpus (20261018_poc data/engram/engram_L<L>.pt); holes read as zeros
  2. tokens of corpus sequence --seq, the reference hash ids for them (engram_hash.pt, from dsref NgramHashState),
     a random residual h [T][4][5120] and the dsref Engram output on it, with the table's int4 rows (dequantized)
     and the dense pack's wkv (Q8_0, dequantized) / q / k, so parity measures TRUSS's arithmetic

usage: PYTHONPATH=<poc>/scripts/dsref:~/llama-paw/gguf-py python ds41_engram_ref.py --dense ds41-dense.gguf \
         --poc <poc dir> --layer 1 --seq 0 --tokens 1024 --out <dir>
"""
import argparse
import importlib.util
import os

import numpy as np
import torch

import gguf

ap = argparse.ArgumentParser()
ap.add_argument("--dense", required=True)
ap.add_argument("--poc", required=True)
ap.add_argument("--layer", type=int, default=1)
ap.add_argument("--seq", type=int, default=0)
ap.add_argument("--tokens", type=int, default=1024)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("eg", os.path.join(here, "..", "tk-pack", "ds41_engram.py"))
eg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg)

li = a.layer
cap = torch.load(f"{a.poc}/data/engram/engram_L{li}.pt", mmap=True)
r = gguf.GGUFReader(a.dense)


def arr(key):
    f = r.fields[key]
    return [int(f.parts[j][0]) for j in f.data]


lidx = arr("deepseek41.engram.layer_ids").index(li)
rows_total = arr("deepseek41.engram.table_rows")[lidx]
# 1. sparse table
path = f"{a.out}/engram-L{li}-sparse.gguf"
hdr, start = eg.gguf_header(f"blk.{li}.engram_embd.q4", rows_total)
with open(path, "wb") as f:
    f.write(hdr)
    f.truncate(start + rows_total * eg.ROW_OUT)
    ids = cap["rows"].numpy()
    q4 = eg.convert(cap["weight"].numpy(), cap["scale"].numpy())
    # contiguous runs of row ids as one write each
    order = np.argsort(ids)
    ids, q4 = ids[order], q4[order]
    breaks = np.nonzero(np.diff(ids) != 1)[0] + 1
    for s, e in zip(np.r_[0, breaks], np.r_[breaks, len(ids)]):
        f.seek(start + int(ids[s]) * eg.ROW_OUT)
        f.write(q4[s:e].tobytes())
print(f"{path}: {rows_total:,} rows, {len(ids):,} written", flush=True)

# 2. tokens, reference ids, reference output
d = torch.load(f"{a.poc}/data/ids_code_ds_1024.pt")
toks = torch.cat([d["calib"], d["heldout"]])[a.seq, : a.tokens].int()
hashes = torch.load(f"{a.poc}/data/engram/engram_hash.pt", mmap=True)["hash"]
ref_ids = hashes[a.seq, : a.tokens, lidx].int()
tens = {t.name: t for t in r.tensors}


def get(name):
    t = tens[name]
    x = gguf.quants.dequantize(t.data, t.tensor_type) if t.tensor_type != gguf.GGMLQuantizationType.F32 else \
        np.asarray(t.data)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(*reversed([int(v) for v in t.shape]))


wkv = get(f"blk.{li}.engram_wkv.weight").cuda()
qk = (get(f"blk.{li}.engram_q.weight") * get(f"blk.{li}.engram_k.weight")).cuda()
pos = np.searchsorted(cap["rows"].numpy(), ref_ids.numpy().reshape(-1))
assert (cap["rows"].numpy()[pos] == ref_ids.numpy().reshape(-1)).all(), "hash rows missing from the capture"
q4 = torch.from_numpy(eg.convert(cap["weight"].numpy()[pos], cap["scale"].numpy()[pos]))
sc = q4[:, :16].contiguous().view(torch.float16).float()
nib = torch.empty(len(q4), 256, dtype=torch.int32)
nib[:, 0::2], nib[:, 1::2] = (q4[:, 16:] & 15).int(), (q4[:, 16:] >> 4).int()
emb = ((nib - 8).float().view(-1, 8, 32) * sc[:, :, None]).view(a.tokens, -1).half().float().cuda()
T, hc, dim = a.tokens, 4, 5120
g = torch.Generator().manual_seed(1)
h = torch.randn(T, hc, dim, generator=g).cuda()
kv = emb @ wkv.T
key, value = kv.split([hc * dim, dim], dim=-1)
key = key.unflatten(-1, (hc, dim))
eps = 1e-20
rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
dot = (h * qk * key).sum(-1) * rstd * dim ** -0.5
gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
out = h + gate.unsqueeze(-1) * value.unsqueeze(-2)
toks.numpy().astype(np.int32).tofile(f"{a.out}/tokens.bin")
ref_ids.numpy().astype(np.int32).tofile(f"{a.out}/ids.bin")
h.cpu().numpy().astype(np.float32).tofile(f"{a.out}/h.bin")
out.cpu().numpy().astype(np.float32).tofile(f"{a.out}/out.bin")
with open(f"{a.out}/meta.txt", "w") as f:
    f.write(f"layer {li}\ntokens {T}\n")
print(f"wrote {a.out}: gate mean {gate.mean():.3f}", flush=True)
