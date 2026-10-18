#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash-1017 routed experts (PAW X3.1 encodes) -> TRUSS expert overlay GGUF over ds41_dense.py's file
(gguf::File::overlay: the engine reads blk.<L>.ffn_<p>_exps.* from here). Same four tensors per layer and projection
as the Flash-Next overlay (flashnext/20261003_x31/scripts/b7_overlay.py): m3_trellis (I16, experts concatenated),
m3_meta (I32 [n_expert][2]: K code, uint16 word offset; K codes 2 / 25 / 3 = K2 / K2.5 / K3, src/formats/trellis_k.h),
m3_suh / m3_svh (F16 [n_expert][in | out]).

Experts are the pack's kept set renumbered: new id i of layer L = the i-th kept original id (ascending), the order
ds41_dense.py uses for the router rows and `deepseek41.expert_keep`. Each kept expert must be in exactly one
<root>/L<nn>/K*/ encode dir (a3_encode.py output: expert_ids.npy, trellis_<p>.npy [n][in/16][out/16][16K],
suh_<p>.npy, svh_<p>.npy, DONE), at the K the kmap gives it.

The file is created once with its full header and length; each layer is written at its offset and read back, and
recorded in <out>.layers.json (re-running skips recorded layers).

usage: ds41_experts.py --roots <hdd enc_pack256>,<nvme enc_pack256> --keep keep256.pt --kmap kmap256.npy --out <file>
"""
import argparse
import json
import os
import struct
import time

import numpy as np
import torch

ALIGN = 32
GGML_F16, GGML_I16, GGML_I32 = 1, 25, 26
KCODE = {2.0: 2, 2.5: 25, 3.0: 3, 3.5: 35, 4.0: 4}
D, F = 5120, 2304
PROJ = (("gate", D, F), ("up", D, F), ("down", F, D))


def gguf_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def layout(kmap, kept):
    tensors, metas = [], {}
    for L, ids in enumerate(kept):
        n = len(ids)
        for p, fin, fout in PROJ:
            nt = (fin // 16) * (fout // 16)
            meta = np.zeros((n, 2), "<i4")
            off = 0
            for i, e in enumerate(ids):
                k = float(kmap[L, e])
                meta[i] = (KCODE[k], off)
                off += int(round(16 * k)) * nt
            metas[(L, p)] = meta
            b = f"blk.{L}.ffn_{p}_exps."
            tensors += [(b + "m3_trellis", GGML_I16, [off], off * 2), (b + "m3_meta", GGML_I32, [2, n], 8 * n),
                        (b + "m3_suh", GGML_F16, [fin, n], fin * n * 2), (b + "m3_svh", GGML_F16, [fout, n], fout * n * 2)]
    return tensors, metas


def header(tensors, note):
    kv = [("general.architecture", "deepseek41"), ("general.name", "deepseek-v4.1-flash-1017 X3.1 experts (overlay)"),
          ("x31.kmap", note)]
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
    return h + b"\0" * (start - len(h)), start, offs, off


def find_layer(roots, L):
    for r in roots:
        d = os.path.join(r, f"L{L:02d}")
        if os.path.isdir(d):
            return d
    raise SystemExit(f"L{L}: no encode dir under {roots}")


def pack_layer(f, L, ids, where, metas, kmap, roots):
    enc = find_layer(roots, L)
    groups, owner = {}, {}
    for kd in sorted(os.listdir(enc)):
        if not (kd.startswith("K") and os.path.isdir(os.path.join(enc, kd))):
            continue
        d = os.path.join(enc, kd)
        if not os.path.exists(os.path.join(d, "DONE")):
            raise SystemExit(f"L{L}: {kd} not DONE")
        k = float(kd[1:])
        groups[k] = d
        for pos, e in enumerate(np.load(os.path.join(d, "expert_ids.npy")).tolist()):
            if e in owner:
                raise SystemExit(f"L{L}: expert {e} in two encode dirs")
            owner[e] = (k, pos)
    for e in ids:
        if e not in owner or owner[e][0] != float(kmap[L, e]):
            raise SystemExit(f"L{L}: kept expert {e} missing or at a K other than the kmap's ({kmap[L, e]})")
    for p, fin, fout in PROJ:
        b = f"blk.{L}.ffn_{p}_exps."
        # whole files, read sequentially (mmap + per-expert reads alternate between the K files: HDD seeks, ~22 MB/s)
        tr = {k: np.load(f"{d}/trellis_{p}.npy") for k, d in groups.items()}
        su = {k: np.load(f"{d}/suh_{p}.npy", mmap_mode="r") for k, d in groups.items()}
        sv = {k: np.load(f"{d}/svh_{p}.npy", mmap_mode="r") for k, d in groups.items()}
        o, n = where[b + "m3_trellis"]
        f.seek(o)
        written = 0
        for e in ids:
            k, pos = owner[e]
            chunk = np.ascontiguousarray(tr[k][pos]).astype("<u2", copy=False).tobytes()
            f.write(chunk)
            written += len(chunk)
        if written != n:
            raise SystemExit(f"L{L} {p}: {written} trellis bytes, expected {n}")
        suh = np.stack([su[owner[e][0]][owner[e][1]] for e in ids]).astype("<f2")
        svh = np.stack([sv[owner[e][0]][owner[e][1]] for e in ids]).astype("<f2")
        for name, data in (("m3_meta", metas[(L, p)].tobytes()), ("m3_suh", suh.tobytes()), ("m3_svh", svh.tobytes())):
            o2, n2 = where[b + name]
            if len(data) != n2:
                raise SystemExit(f"L{L} {p} {name}: {len(data)} bytes, expected {n2}")
            f.seek(o2)
            f.write(data)
        f.flush()
        # read back: the last expert's tiles and the small tensors
        k, pos = owner[ids[-1]]
        last = np.ascontiguousarray(tr[k][pos]).astype("<u2", copy=False).tobytes()
        f.seek(o + n - len(last))
        if f.read(len(last)) != last:
            raise SystemExit(f"L{L} {p}: trellis read-back differs")
        for name, data in (("m3_suh", suh.tobytes()), ("m3_svh", svh.tobytes())):
            o2, n2 = where[b + name]
            f.seek(o2)
            if f.read(n2) != data:
                raise SystemExit(f"L{L} {p} {name}: read-back differs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True, help="comma list of enc_pack256 roots holding L<nn>/K*/")
    ap.add_argument("--keep", required=True)
    ap.add_argument("--kmap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=0, help="first N layers only (test); 0 = all")
    a = ap.parse_args()
    roots = a.roots.split(",")
    keep = torch.load(a.keep)["keep"]
    kmap = np.load(a.kmap)
    n_layer = a.layers or keep.shape[0]
    kept = [np.nonzero(keep[l].numpy())[0].tolist() for l in range(n_layer)]
    for L, ids in enumerate(kept):
        if any(float(kmap[L, e]) == 0 for e in ids):
            raise SystemExit(f"L{L}: a kept expert has K 0 in the kmap")
    tensors, metas = layout(kmap, kept)
    note = f"{os.path.basename(a.kmap)} over {os.path.basename(a.keep)}"
    hdr, start, offs, total = header(tensors, note)
    if not os.path.exists(a.out):
        with open(a.out + ".tmp", "wb") as f:
            f.write(hdr)
            f.truncate(start + total)
        os.replace(a.out + ".tmp", a.out)
        print(f"created {a.out}: {len(tensors)} tensors, {(start + total) / 1e9:.2f} GB", flush=True)
    with open(a.out, "rb") as f:
        if f.read(len(hdr)) != hdr:
            raise SystemExit(f"{a.out}: header differs from this keep / kmap; refusing to write into it")
    where = {t[0]: (start + o, t[3]) for t, o in zip(tensors, offs)}
    rec_path = a.out + ".layers.json"
    rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
    with open(a.out, "r+b") as f:
        for L in range(n_layer):
            if str(L) in rec:
                continue
            t0 = time.time()
            pack_layer(f, L, kept[L], where, metas, kmap, roots)
            os.fsync(f.fileno())
            rec[str(L)] = {"time": time.strftime("%m-%d %T"), "s": round(time.time() - t0)}
            json.dump(rec, open(rec_path + ".tmp", "w"), indent=1)
            os.replace(rec_path + ".tmp", rec_path)
            print(f"L{L}: written and verified in {time.time() - t0:.0f}s", flush=True)
    print("DS41-EXPERTS-DONE", flush=True)


if __name__ == "__main__":
    main()
