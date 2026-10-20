#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash-1017 dense matrices -> TRUSS X3.1 overlay over ds41_dense.py's file (the m3 naming E13's
binder reads).

Per encoded matrix the overlay holds the four group tensors and a shadow of the base's Q8 tensor:
  <name>.m3_trellis  I16 [tiles][words]   the trellis words (k-tile major / row-tile minor)
  <name>.m3_suh      F16 [in]             the input-side scales
  <name>.m3_svh      F16 [out]            the output-side scales
  <name>.m3_meta     I32 [2][1]           (the rate code, 0) - required for the half-integer rates
  <name>             I16 [1]              the shadow: the reader's overlay replaces same-named tensors, so this
                                           drops the base's Q8 tensor from the merged view (the binder takes the
                                           group; the shadow carries no weights)
The rate codes are src/formats/trellis_k.h's (35 = K3.5).

The file is created once with its full header and length; each layer is written at its offset and read back,
and recorded in <out>.layers.json (re-running skips the recorded layers).

usage: ds41_dense_x3.py --enc <enc dir> --out <file> [--layers 0-39] [--matrices a,b,c]
"""
import argparse
import json
import os
import struct
import time

import numpy as np

ALIGN = 32
GGML_F16, GGML_I16, GGML_I32 = 1, 25, 26
KCODE = {2.0: 2, 2.5: 25, 3.0: 3, 3.5: 35, 4.0: 4}
# the matrices we encode: the name suffix -> the encode dir's npz name
DEFAULT_MATRICES = ["attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight", "attn_output_b.weight",
                    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"]


def gguf_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def read_meta(enc, L, m):
    """the encode dir's manifest entry: the rate, the shapes and the word count"""
    man = json.load(open(os.path.join(enc, f"L{L:02d}", "manifest.json")))
    e = man["matrices"][f"blk.{L}.{m}"]
    npz = np.load(os.path.join(enc, f"L{L:02d}", m + ".npz"))
    trellis = npz["trellis"]
    k = float(e["K"])
    return e, trellis, npz, KCODE[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True, help="the encode dir: L<nn>/<name>.npz + manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--matrices", default="", help="comma list of the name suffixes (default all)")
    a = ap.parse_args()
    lo, hi = a.layers.split("-")
    layers = list(range(int(lo), int(hi) + 1))
    mats = a.matrices.split(",") if a.matrices else DEFAULT_MATRICES

    # the header: the tensors with their exact shapes (read from the encodes)
    tensors = []
    for L in layers:
        for m in mats:
            e, trellis, npz, kc = read_meta(a.enc, L, m)
            b = f"blk.{L}.{m}"
            n_tiles = trellis.shape[0] * trellis.shape[1]
            words = trellis.shape[2]
            # the X3.1 convention: the ggml shape is [words, tiles] (the words are the fastest dim); the flat
            # order is the tile order (k-tile major / row-tile minor), the same bytes as the PAW pack wrote
            tensors.append((b + ".m3_trellis", GGML_I16, [words, n_tiles], n_tiles * words * 2))
            tensors.append((b + ".m3_suh", GGML_F16, [trellis.shape[0] * 16], trellis.shape[0] * 16 * 2))
            tensors.append((b + ".m3_svh", GGML_F16, [trellis.shape[1] * 16], trellis.shape[1] * 16 * 2))
            tensors.append((b + ".m3_meta", GGML_I32, [2, 1], 8))
            tensors.append((b, GGML_I16, [1], 2))       # the shadow of the base's Q8 tensor
    kv = [("general.architecture", "deepseek41"), ("general.name", "deepseek-v4.1-flash-1017 X3.1 dense (overlay)"),
          ("x31.dense", f"{len(mats)} matrices x {len(layers)} layers, K3.5")]
    h = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, v in kv:
        h += gguf_str(k) + struct.pack("<I", 8) + gguf_str(v)
    offs, off = [], 0
    for name, ty, shape, nbytes in tensors:
        h += gguf_str(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
        h += struct.pack("<IQ", ty, off)
        offs.append(off)
        off += (nbytes + ALIGN - 1) // ALIGN * ALIGN
    start = (len(h) + ALIGN - 1) // ALIGN * ALIGN
    if not os.path.exists(a.out):
        with open(a.out + ".tmp", "wb") as f:
            f.write(h)
            f.truncate(start + off)
        os.replace(a.out + ".tmp", a.out)
        print(f"created {a.out}: {len(tensors)} tensors, {(start + off) / 1e9:.2f} GB", flush=True)
    else:
        with open(a.out, "rb") as f:
            if f.read(len(h)) != h:
                raise SystemExit(f"{a.out}: the header differs from this encode set; refusing to write into it")
    where = {t[0]: (start + o, t[3]) for t, o in zip(tensors, offs)}
    rec_path = a.out + ".layers.json"
    rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
    with open(a.out, "r+b") as f:
        for L in layers:
            if str(L) in rec:
                continue
            t0 = time.time()
            for m in mats:
                e, trellis, npz, kc = read_meta(a.enc, L, m)
                b = f"blk.{L}.{m}"
                words = trellis.shape[2]
                flat = np.ascontiguousarray(trellis.reshape(-1)).astype("<u2", copy=False).tobytes()
                assert len(flat) == where[b + ".m3_trellis"][1], (b, len(flat))
                f.seek(where[b + ".m3_trellis"][0])
                f.write(flat)
                f.seek(where[b + ".m3_suh"][0])
                f.write(npz["suh"].astype("<f2").tobytes())
                f.seek(where[b + ".m3_svh"][0])
                f.write(npz["svh"].astype("<f2").tobytes())
                f.seek(where[b + ".m3_meta"][0])
                f.write(struct.pack("<ii", kc, 0))
                f.seek(where[b][0])
                f.write(struct.pack("<h", 0))          # the shadow
            f.flush()
            # read back: the trellis of the last matrix and the small tensors
            e, trellis, npz, kc = read_meta(a.enc, L, mats[-1])
            b = f"blk.{L}.{mats[-1]}"
            flat = np.ascontiguousarray(trellis.reshape(-1)).astype("<u2", copy=False).tobytes()
            f.seek(where[b + ".m3_trellis"][0])
            if f.read(len(flat)) != flat:
                raise SystemExit(f"L{L}: trellis read-back differs")
            for nm, data in ((".m3_suh", npz["suh"].astype("<f2").tobytes()),
                             (".m3_svh", npz["svh"].astype("<f2").tobytes())):
                f.seek(where[b + nm][0])
                if f.read(len(data)) != data:
                    raise SystemExit(f"L{L} {nm}: read-back differs")
            os.fsync(f.fileno())
            rec[str(L)] = {"time": time.strftime("%m-%d %T"), "s": round(time.time() - t0)}
            json.dump(rec, open(rec_path + ".tmp", "w"), indent=1)
            os.replace(rec_path + ".tmp", rec_path)
            print(f"L{L}: written and verified in {time.time() - t0:.0f}s", flush=True)
    print("DS41-DENSE-X3-DONE", flush=True)


if __name__ == "__main__":
    main()
