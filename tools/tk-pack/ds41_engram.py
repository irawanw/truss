#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash-1017 Engram tables (2 x ~384M rows of 256 fp8 e4m3 + an e8m0 scale per 32; 101.5 GB each) ->
int4 rows for TRUSS: one GGUF per Engram layer with the single tensor `blk.<L>.engram_embd.q4` (I8 [144][rows]).

Row layout (144 B): 8 fp16 scales (one per 32 values) then 128 bytes of nibbles, value i in byte i / 2 (low nibble
first), stored as q + 8 with q = round(x / s) in [-7, 7], s = amax / 7 of the group (fp16). PoC A12 (TRACKER R8):
value-output rel err 0.085-0.096 through wkv on held-out code tokens.

The shard is streamed over HTTP in --streams parallel ranges of whole rows (~32 MB/s per stream from HF); each
range converts and pwrites its rows; finished ranges are recorded in <out>.done.json, so a re-run resumes.

usage: ds41_engram.py --model-dir ~/models_nvme/dsv41flash_1017 --layer 1 --out <dir>/ds41-engram-L1.gguf
"""
import argparse
import json
import os
import struct
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

BASE = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/"
E4M3 = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float().numpy()
ROW_IN, SC_IN, ROW_OUT, ALIGN = 256, 8, 144, 32
GGML_I8 = 24


def http(shard, lo, hi):
    r = urllib.request.Request(BASE + shard, headers={"Range": f"bytes={lo}-{hi - 1}"})
    return urllib.request.urlopen(r, timeout=600)


def header(shard):
    n = struct.unpack("<Q", http(shard, 0, 8).read())[0]
    return 8 + n, json.loads(http(shard, 8, 8 + n).read())


def convert(w, s):
    """w uint8 [n][256] e4m3, s uint8 [n][8] e8m0 -> [n][144] bytes"""
    x = E4M3[w].reshape(-1, 8, 32) * np.exp2(s.astype(np.float32) - 127)[:, :, None]
    sc = (np.abs(x).max(-1) / 7).astype(np.float16)
    q = np.rint(x / np.maximum(sc.astype(np.float32), 1e-30)[:, :, None]).clip(-7, 7).astype(np.int8) + 8
    q = q.reshape(-1, 256).astype(np.uint8)
    out = np.empty((len(w), ROW_OUT), np.uint8)
    out[:, :16] = sc.view(np.uint8).reshape(-1, 16)
    out[:, 16:] = q[:, 0::2] | (q[:, 1::2] << 4)
    return out


def gguf_header(name, rows):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [("general.architecture", "deepseek41"), ("general.name", "deepseek-v4.1-flash-1017 engram int4 rows")]
    h = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kv))
    for k, v in kv:
        h += s(k) + struct.pack("<I", 8) + s(v)
    h += s(name) + struct.pack("<I", 2) + struct.pack("<QQ", ROW_OUT, rows) + struct.pack("<IQ", GGML_I8, 0)
    start = (len(h) + ALIGN - 1) // ALIGN * ALIGN
    return h + b"\0" * (start - len(h)), start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--range-rows", type=int, default=1 << 19, help="rows per resumable range (512K rows = 138 MB in)")
    a = ap.parse_args()
    wmap = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))["weight_map"]
    wn, sn = f"layers.{a.layer}.engram.embed.weight", f"layers.{a.layer}.engram.embed.scale"
    shard = wmap[wn]
    assert wmap[sn] == shard
    base, hd = header(shard)
    rows = hd[wn]["shape"][0]
    assert hd[wn]["shape"][1] == ROW_IN and hd[sn]["shape"] == [rows, SC_IN], (hd[wn], hd[sn])
    w0, s0 = base + hd[wn]["data_offsets"][0], base + hd[sn]["data_offsets"][0]
    hdr, start = gguf_header(f"blk.{a.layer}.engram_embd.q4", rows)
    if not os.path.exists(a.out):
        with open(a.out + ".tmp", "wb") as f:
            f.write(hdr)
            f.truncate(start + rows * ROW_OUT)
        os.replace(a.out + ".tmp", a.out)
    done_path = a.out + ".done.json"
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    ranges = [(r, min(rows, r + a.range_rows)) for r in range(0, rows, a.range_rows)]
    todo = [r for r in ranges if r[0] not in done]
    print(f"L{a.layer}: {rows:,} rows, {len(ranges)} ranges, {len(todo)} to do, {a.streams} streams", flush=True)
    lock, t0, moved = threading.Lock(), time.time(), [0]
    fd = os.open(a.out, os.O_WRONLY)

    def work(rg):
        r0, r1 = rg
        n = r1 - r0
        for attempt in range(8):
            try:
                w = np.frombuffer(http(shard, w0 + r0 * ROW_IN, w0 + r1 * ROW_IN).read(), np.uint8).reshape(n, ROW_IN)
                s = np.frombuffer(http(shard, s0 + r0 * SC_IN, s0 + r1 * SC_IN).read(), np.uint8).reshape(n, SC_IN)
                break
            except Exception as e:  # noqa: BLE001 - HF drops long streams; retry the range
                print(f"range {r0}: {e} (try {attempt + 1})", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            raise SystemExit(f"range {r0} failed")
        os.pwrite(fd, convert(w, s).tobytes(), start + r0 * ROW_OUT)
        with lock:
            done.add(r0)
            json.dump(sorted(done), open(done_path + ".tmp", "w"))
            os.replace(done_path + ".tmp", done_path)
            moved[0] += n * (ROW_IN + SC_IN)
            el = time.time() - t0
            print(f"L{a.layer}: {len(done)}/{len(ranges)} ranges, {moved[0] / el / 1e6:.0f} MB/s", flush=True)

    with ThreadPoolExecutor(a.streams) as ex:
        list(ex.map(work, todo))
    os.fsync(fd)
    os.close(fd)
    print(f"ENGRAM-L{a.layer}-DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
