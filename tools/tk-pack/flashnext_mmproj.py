#!/usr/bin/env python3
"""Writes flashnext-mmproj-f16.gguf: the vision tower of Qwen/Qwen3.8-Flash-Next (the 0.5B ViT + merger), unquantized.

  python3 flashnext_mmproj.py --model-dir <HF dir with config.json + shard holding model.visual.*> --out flashnext-mmproj-f16.gguf

Layout: arch "qwen4exp-mmproj"; every tensor `model.visual.X` is stored as `v.X` (HF names kept, so the engine and
the HF reference line up). Matrices, the conv kernel [1152,3,2,16,16] and pos_embed [2304,1152] are F16; biases and
LayerNorm weights are F32. Metadata `qwen4exp.vision.*` carries the hyperparameters (see docs/reference/model-qwen4exp.md).
"""
import argparse
import json
import os

import gguf
import numpy as np
from safetensors import safe_open

ARCH = "qwen4exp-mmproj"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    v = cfg["vision_config"]
    idx = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))["weight_map"]
    names = sorted(k for k in idx if k.startswith("model.visual."))
    shards = sorted({idx[k] for k in names})
    w = gguf.GGUFWriter(a.out + ".tmp", ARCH)
    p = "qwen4exp.vision."
    for k in ("depth", "hidden_size", "intermediate_size", "num_heads", "patch_size", "temporal_patch_size",
              "spatial_merge_size", "num_position_embeddings", "out_hidden_size", "in_channels"):
        w.add_uint32(p + k, int(v[k]))
    w.add_string(p + "hidden_act", v["hidden_act"])
    w.add_array(p + "image_mean", [0.5, 0.5, 0.5])
    w.add_array(p + "image_std", [0.5, 0.5, 0.5])
    for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        w.add_uint32(p + k, int(cfg[k]))
    n = 0
    for sh in shards:
        with safe_open(os.path.join(a.model_dir, sh), "pt") as f:
            for k in names:
                if idx[k] != sh:
                    continue
                t = f.get_tensor(k).float().numpy()
                is_mat = k.endswith("weight") and t.ndim >= 2        # pos_embed, qkv/proj/fc, conv kernel
                w.add_tensor("v." + k[len("model.visual."):], t.astype(np.float16 if is_mat else np.float32))
                n += 1
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    os.replace(a.out + ".tmp", a.out)
    print(f"{n} tensors -> {a.out} ({os.path.getsize(a.out) / 1e9:.3f} GB)")


if __name__ == "__main__":
    main()
