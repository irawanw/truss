#!/usr/bin/env python3
"""Uploads a DS41 pack (what scripts/ds41_env.sh loads from TRUSS_MODEL_DIR) to a Hugging Face model repo, one file at
a time. Resumable: a file whose size already matches the repo's is skipped, so a rerun after a dropped connection only
sends what is missing. The token comes from HF_TOKEN (ML_projects/.env).
  HF_TOKEN=... python3 tools/tk-pack/ds41_hf_upload.py [--set x3|x3.1] [--repo ...] [--pack DIR] [--readme FILE]
--set x3: PAW-763B-X3 (256 experts per layer). --set x3.1: PAW-763B-X3.1 = pack E2 (all 384 experts, far.u8, its usage
table) + the X3 dense overlay, Engram tables, drafter and tokenizer + the vision mmproj (TRACKER 283-289).
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# 10-25 (huggingface_hub 1.33.0, hf_xet 1.7.0): the xet uploader hung twice on the first large file (0 B/s, every
# hf-xet thread in a futex wait, its HTTPS socket in CLOSE-WAIT); plain LFS upload runs at 20-30 MB/s
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi   # noqa: E402

T = os.path.expanduser("~/ML_projects/dsv41flash/20261019_truss_port/data")
V = os.path.expanduser("~/ML_projects/dsv41flash/20261025_vision/data")
# (local path under --pack, path in the repo); the small ones first so the repo is readable early
FILES = [
    ("tokenizer/tokenizer.json", "tokenizer/tokenizer.json"),
    ("tokenizer/tokenizer_config.json", "tokenizer/tokenizer_config.json"),
    ("drafter/ds41-dspark-dense.gguf", "drafter/ds41-dspark-dense.gguf"),
    ("drafter/ds41-dspark-experts.gguf", "drafter/ds41-dspark-experts.gguf"),
    ("drafter/ds41-dspark-experts.gguf.layers.json", "drafter/ds41-dspark-experts.gguf.layers.json"),
    ("bench_in/usage_calib256.f32", "usage_calib256.f32"),
    ("pack/ds41-dense.gguf", "ds41-dense.gguf"),
    ("pack/ds41-dense-x3.gguf", "ds41-dense-x3.gguf"),
    ("pack/ds41-dense-x3.gguf.layers.json", "ds41-dense-x3.gguf.layers.json"),
    ("pack/ds41-experts.gguf.layers.json", "ds41-experts.gguf.layers.json"),
    ("pack/ds41-experts.gguf", "ds41-experts.gguf"),
    ("pack/ds41-engram-L1.gguf", "ds41-engram-L1.gguf"),
    ("pack/ds41-engram-L14.gguf", "ds41-engram-L14.gguf"),
]
FILES_X31 = [
    ("tokenizer/tokenizer.json", "tokenizer/tokenizer.json"),
    ("tokenizer/tokenizer_config.json", "tokenizer/tokenizer_config.json"),
    ("pack_e2/far.u8", "far.u8"),
    ("pack_e2/usage.f32", "usage.f32"),
    ("pack_e2/ds41-experts.gguf.layers.json", "ds41-experts.gguf.layers.json"),
    ("pack/ds41-dense-x3.gguf.layers.json", "ds41-dense-x3.gguf.layers.json"),
    (os.path.join(V, "ds41-mmproj-f16.gguf"), "ds41-mmproj-f16.gguf"),
    ("drafter/ds41-dspark-dense.gguf", "drafter/ds41-dspark-dense.gguf"),
    ("drafter/ds41-dspark-experts.gguf", "drafter/ds41-dspark-experts.gguf"),
    ("drafter/ds41-dspark-experts.gguf.layers.json", "drafter/ds41-dspark-experts.gguf.layers.json"),
    ("pack_e2/ds41-dense.gguf", "ds41-dense.gguf"),
    ("pack/ds41-dense-x3.gguf", "ds41-dense-x3.gguf"),
    ("pack_e2/ds41-experts.gguf", "ds41-experts.gguf"),
    ("pack/ds41-engram-L1.gguf", "ds41-engram-L1.gguf"),
    ("pack/ds41-engram-L14.gguf", "ds41-engram-L14.gguf"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["x3", "x3.1"], default="x3")
    ap.add_argument("--repo", help="default: lackonendes/PAW-763B-X3 or -X3.1 by --set")
    ap.add_argument("--pack", default=T)
    ap.add_argument("--readme", help="model card, uploaded as README.md")
    a = ap.parse_args()
    a.repo = a.repo or ("lackonendes/PAW-763B-X3.1" if a.set == "x3.1" else "lackonendes/PAW-763B-X3")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    have = {f.path: f.size for f in api.list_repo_tree(a.repo, recursive=True) if hasattr(f, "size")}
    readme = [(os.path.abspath(a.readme), "README.md")] if a.readme else []
    todo = list(FILES_X31 if a.set == "x3.1" else FILES) + readme
    for src, _ in todo:   # every file before the first byte goes up
        path = src if os.path.isabs(src) else os.path.join(a.pack, src)
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path}")
    for src, dst in todo:
        path = src if os.path.isabs(src) else os.path.join(a.pack, src)
        size = os.path.getsize(path)
        if have.get(dst) == size and dst != "README.md":
            print(f"skip {dst} ({size / 1e9:.2f} GB already there)", flush=True)
            continue
        t = time.time()
        api.upload_file(path_or_fileobj=path, path_in_repo=dst, repo_id=a.repo)
        dt = time.time() - t
        print(f"done {dst}: {size / 1e9:.2f} GB in {dt:.0f} s ({size / 1e6 / max(dt, 1e-3):.0f} MB/s)", flush=True)


if __name__ == "__main__":
    main()
