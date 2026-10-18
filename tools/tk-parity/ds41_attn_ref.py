#!/usr/bin/env python3
"""Reference activations for TRUSS's DeepSeek-V4.1 attention (blocks/attn_ds41), from the vendored official
dsref/model.py Attention on the SAME weights TRUSS reads: the dense pack's Q8_0 / F32 tensors, dequantized to fp32.

The reference's storage simulations (fp8 act_quant of the window KV and of o before wo_a, fp4 of the indexer q / k
and the compressed KV) are switched off with --exact (default), so parity measures TRUSS's arithmetic only; --model
keeps them (what the real model computes; KL covers that gap end to end).

Runs a chain of attention layers in model order with the reference's shared state (compressed KV, index keys,
top-k, candidates from the source layers), on random inputs N(0,1) * attn_norm: a prefill of --prefill tokens, then
--decode single-token steps. Saves per layer and step: x, out, and the top-k indices of index-source layers.

usage: PYTHONPATH=<poc>/scripts/dsref:~/llama-paw/gguf-py python ds41_attn_ref.py --dense ds41-dense.gguf \
         --layers 0,1,2,3,8,20,21,24 --prefill 3000 --decode 4 --out <dir>
"""
import argparse
import sys

import numpy as np
import torch

import gguf

ap = argparse.ArgumentParser()
ap.add_argument("--dense", required=True)
ap.add_argument("--layers", default="0,1,2,3,8,20,21,24")
ap.add_argument("--prefill", type=int, default=3000)
ap.add_argument("--decode", type=int, default=4)
ap.add_argument("--model", action="store_true", help="keep the fp8 / fp4 storage simulations")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True, help="output dir: meta.json + s<step>_L<layer>_{x,out,topk}.bin (raw fp32 / int32)")
a = ap.parse_args()

import load as L  # noqa: E402  (registers the kernel stand-ins)
M = L.M
if not a.model:
    M.act_quant = lambda x, *args, **kw: x          # only ever called in place by Attention
    M.fp4_act_quant = lambda x, *args, **kw: x
M.default_dtype = torch.float32
torch.set_default_dtype(torch.float32)
torch.set_default_device("cuda")      # the reference builds its index tensors on the default device

layers = [int(v) for v in a.layers.split(",")]
assert layers == sorted(layers), "layers run in model order (shared state flows down)"
T0, ND = a.prefill, a.decode
args = L.make_args(1, T0 + ND + 8)
M.default_dtype = torch.float32
r = gguf.GGUFReader(a.dense)
tens = {t.name: t for t in r.tensors}


def get(name):
    t = tens[name]
    x = gguf.quants.dequantize(t.data, t.tensor_type) if t.tensor_type != gguf.GGMLQuantizationType.F32 else \
        np.asarray(t.data)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(
        *reversed([int(d) for d in t.shape])).cuda()


NAMES = {"attn_sink": "attn_sinks.weight", "wq_a.weight": "attn_q_a.weight", "q_norm.weight": "attn_q_a_norm.weight",
         "wq_b.weight": "attn_q_b.weight", "wkv.weight": "attn_kv.weight", "kv_norm.weight": "attn_kv_a_norm.weight",
         "wo_a.weight": "attn_output_a.weight", "wo_b.weight": "attn_output_b.weight",
         "compressor.wkv.weight": "attn_compressor_kv.weight", "compressor.wgate.weight": "attn_compressor_gate.weight",
         "compressor.norm.weight": "attn_compressor_norm.weight", "indexer.wk.weight": "indexer.attn_k.weight",
         "indexer.k_norm.weight": "indexer.k_norm.weight", "indexer.wq_b.weight": "indexer.attn_q_b.weight",
         "indexer.weights_proj.weight": "indexer.proj.weight"}
mods = {}
with torch.device("cuda"), torch.no_grad():
    for l in layers:
        m = M.Attention(l, args)
        for n, p in m.named_parameters():
            if n.endswith(".scale") or n == "scale":
                continue
            src = get(f"blk.{l}." + NAMES[n])
            assert src.shape == p.shape, (l, n, src.shape, p.shape)
            p.data = src.to(p.device)   # fp32 even where the module asked for bf16
        mods[l] = m
        print(f"L{l}: ratio {m.compress_ratio} kv_source {m.is_kv_source} index_source {m.is_index_source}",
              flush=True)

g = torch.Generator().manual_seed(a.seed)
norms = {l: get(f"blk.{l}.attn_norm.weight") for l in layers}
rec = {"layers": layers, "prefill": T0, "decode": ND, "exact": not a.model, "steps": []}
with torch.no_grad():
    for step in range(1 + ND):
        start, T = (0, T0) if step == 0 else (T0 + step - 1, 1)
        out = {}
        for l in layers:
            x = (torch.randn(1, T, args.dim, generator=g, device="cpu") * norms[l].cpu()).cuda()
            o = mods[l](x, start)
            out[l] = {"x": x[0].cpu(), "out": o[0].float().cpu()}
            if mods[l].is_index_source and M.shared_attn.topk_idxs is not None:
                out[l]["topk"] = M.shared_attn.topk_idxs[0].cpu()
        rec["steps"].append({"start": start, "T": T, "layers": out})
        print(f"step {step}: start {start} T {T} done", flush=True)
import json, os  # noqa: E401,E402
os.makedirs(a.out, exist_ok=True)
meta = {"layers": layers, "exact": rec["exact"], "steps": [{"start": st["start"], "T": st["T"]} for st in rec["steps"]]}
for si, st in enumerate(rec["steps"]):
    for l, d in st["layers"].items():
        for k, v in d.items():
            v.contiguous().numpy().astype(np.int32 if k == "topk" else np.float32).tofile(f"{a.out}/s{si}_L{l}_{k}.bin")
json.dump(meta, open(f"{a.out}/meta.json", "w"))
with open(f"{a.out}/meta.txt", "w") as f:            # what tests/layer/ds41_attn_parity reads
    f.write("layers " + " ".join(map(str, layers)) + "\n")
    for st in meta["steps"]:
        f.write(f"step {st['start']} {st['T']}\n")
print(f"wrote {a.out}")
sys.exit(0)
