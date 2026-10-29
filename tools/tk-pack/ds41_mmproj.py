#!/usr/bin/env python3
"""Writes ds41-mmproj-f16.gguf: the image part of deepseek-ai/DeepSeek-V4.1-Flash-1017, unquantized.

  python3 ds41_mmproj.py --model-dir <HF dir with config.json + model.safetensors.index.json> --out ds41-mmproj-f16.gguf

Tensors are read from the local shards when present, else by HTTP range reads of the Hugging Face repo (--repo), so
shards 1-2 (ViT, aligner, span vectors) are the only download needed; the router biases live in every layer's shard.

Layout: arch "deepseek41-mmproj"; HF names kept with these prefixes, so the engine and the reference line up:
  v.X              vision.X (ViT): matrices F16, biases / RMSNorm weights F32
  a.X              aligner.X: w1 [5120,9216], w2 [5120,5120] F16, biases F32
  span.start / span.newline / span.end   image_start / image_newline / image_end, F32 [5120]
  blk.L.bias_vl    layers.L.ffn.gate.bias_vl, F32 [384]: the router bias of image-span rows (replaces gate.bias)
  mtp.I.bias_vl    mtp.I.ffn.gate.bias_vl (the DSpark drafter layers)
Metadata `deepseek41.vision.*` carries the hyperparameters (docs/reference/model-deepseek41.md, "Vision").
"""
import argparse
import json
import os
import struct

import gguf
import numpy as np
import requests
import torch
from safetensors import safe_open

ARCH = "deepseek41-mmproj"


class Reader:
    """Tensor by name: local shard, else HTTP range read of the repo (header cached per shard)."""

    def __init__(self, model_dir, repo):
        self.dir, self.repo, self.heads = model_dir, repo, {}
        self.map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]

    def _range(self, url, a, b):
        r = requests.get(url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=120)
        r.raise_for_status()
        assert len(r.content) == b - a, (url, a, b, len(r.content))
        return r.content

    def get(self, name):
        shard = self.map[name]
        path = os.path.join(self.dir, shard)
        if os.path.exists(path):
            with safe_open(path, "pt") as f:
                return f.get_tensor(name).float().numpy()
        from huggingface_hub import hf_hub_url
        url = hf_hub_url(self.repo, shard)
        if shard not in self.heads:
            n = struct.unpack("<Q", self._range(url, 0, 8))[0]
            self.heads[shard] = (8 + n, json.loads(self._range(url, 8, 8 + n)))
        start, h = self.heads[shard]
        t = h[name]
        a, b = t["data_offsets"]
        raw = self._range(url, start + a, start + b)
        dt = {"F32": torch.float32, "BF16": torch.bfloat16}[t["dtype"]]
        return torch.frombuffer(bytearray(raw), dtype=dt).reshape(t["shape"]).float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    v, tc = cfg["vision_config"], cfg["text_config"]
    rd = Reader(a.model_dir, a.repo)
    w = gguf.GGUFWriter(a.out + ".tmp", ARCH)
    p = "deepseek41.vision."
    for k in ("num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size", "patch_size",
              "downsample_ratio", "max_image_tokens", "min_pixels"):
        w.add_uint32(p + k, int(v[k]))
    w.add_float32(p + "rope_theta", float(v["rope_theta"]))
    w.add_uint32(p + "image_token_id", int(cfg["image_token_id"]))
    w.add_uint32(p + "out_hidden_size", int(tc["hidden_size"]))
    w.add_array(p + "image_mean", [0.5, 0.5, 0.5])
    w.add_array(p + "image_std", [0.5, 0.5, 0.5])
    n = 0
    for k in sorted(k for k in rd.map if k.startswith(("vision.", "aligner."))):
        t = rd.get(k)
        name = ("v." + k[7:]) if k.startswith("vision.") else ("a." + k[8:])
        w.add_tensor(name, t.astype(np.float16 if t.ndim >= 2 else np.float32))
        n += 1
    for k in ("start", "newline", "end"):
        w.add_tensor("span." + k, rd.get("image_" + k).astype(np.float32)); n += 1
    for l in range(int(tc["num_hidden_layers"])):
        w.add_tensor(f"blk.{l}.bias_vl", rd.get(f"layers.{l}.ffn.gate.bias_vl").astype(np.float32)); n += 1
    for k in sorted(k for k in rd.map if k.startswith("mtp.") and k.endswith(".ffn.gate.bias_vl")):
        w.add_tensor(f"mtp.{k.split('.')[1]}.bias_vl", rd.get(k).astype(np.float32)); n += 1
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    os.replace(a.out + ".tmp", a.out)
    print(f"{n} tensors -> {a.out} ({os.path.getsize(a.out) / 1e9:.3f} GB)")


if __name__ == "__main__":
    main()
