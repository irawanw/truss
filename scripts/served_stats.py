#!/usr/bin/env python3
"""served_stats.py - the served KPI (Plan v3 Phase 0.2): parse the brain request log.

The KPI is the SERVED brain log, not the greedy bench. Lines (engine stdout, captured by --log or pm2):
  prompt P tokens = R reused + N read in T ms (X tok/s), G generated in U ms (Y tok/s), drafts accepted A of D
  truss serve: decode G tokens, Y tok/s, drafts accepted A of D        (older/decode-only format, no prompt)
A `request:` line precedes each serve line and carries the sampling params (temperature = served sampling).
Log lines carry NO timestamps; per-day = per-file (file mtime is the window's end).

Reports per file:
  decode tok/s p10/p50/p90 for requests with G >= --min-gen (default 200), split by context band (P+G):
    <=32K, 32-120K, >120K; drafts accepted/draft p10/p50/p90; prefill tok/s on FRESH prompts >= --min-fresh
    (default 30000: P >= min-fresh AND reused == 0; rate = the line's own prefill tok/s = P / T ms).

usage: served_stats.py [logfile ...] [--min-gen 200] [--min-fresh 30000] [--json]
"""
import argparse
import datetime
import json
import os
import re
import sys

SERVE = re.compile(
    r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
DECODE_ONLY = re.compile(r"truss serve: decode (\d+) tokens, ([\d.]+) tok/s, drafts accepted (\d+) of (\d+)")
REQUEST = re.compile(r"request: temperature ([\d.]+)")


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * q / 100.0
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def parse(path):
    """-> list of request dicts; temp attached from the last preceding `request:` line."""
    reqs = []
    last_temp = None
    with open(path, errors="replace") as f:
        for line in f:
            if "truss serve" not in line:
                m = REQUEST.search(line)
                if m:
                    last_temp = float(m.group(1))
                continue
            m = SERVE.search(line)
            if m:
                p, r, n, t_ms, x_tps, g, u_ms, y_tps, a, d = m.groups()
                reqs.append(dict(prompt=int(p), reused=int(r), read=int(n), prefill_ms=int(t_ms),
                                 prefill_tps=float(x_tps), gen=int(g), decode_ms=int(u_ms),
                                 decode_tps=float(y_tps), accepted=int(a), drafted=int(d),
                                 temp=last_temp))
                continue
            m = DECODE_ONLY.search(line)
            if m:
                g, y_tps, a, d = m.groups()
                reqs.append(dict(prompt=None, reused=None, read=None, prefill_ms=None,
                                 prefill_tps=None, gen=int(g), decode_ms=None,
                                 decode_tps=float(y_tps), accepted=int(a), drafted=int(d),
                                 temp=last_temp))
    return reqs


def band(ctx):
    return "<=32K" if ctx <= 32768 else ("32-120K" if ctx <= 131072 else ">120K")


def report(path, reqs, min_gen, min_fresh, out=print):
    day = datetime.date.fromtimestamp(os.path.getmtime(path))
    dec = [r["decode_tps"] for r in reqs if r["gen"] >= min_gen]
    dec_s = [r["decode_tps"] for r in reqs if r["gen"] >= min_gen and r["temp"] and r["temp"] > 0]
    acc = [r["accepted"] / r["drafted"] for r in reqs if r["gen"] >= min_gen and r["drafted"] > 0]
    fresh = [r["prefill_tps"] for r in reqs
             if r["prompt"] and r["prompt"] >= min_fresh and r["reused"] == 0]
    bands = {}
    for r in reqs:
        if r["gen"] >= min_gen and r["prompt"] is not None:
            bands.setdefault(band(r["prompt"] + r["gen"]), []).append(r["decode_tps"])
    out(f"== {path}  (day {day}, {len(reqs)} serve lines) ==")
    def row(name, xs):
        out(f"  {name:38s} {pct(xs,10):7.1f} {pct(xs,50):7.1f} {pct(xs,90):7.1f}   n={len(xs)}")
    out("  %-38s %7s %7s %7s" % ("", "p10", "p50", "p90"))
    row(f"decode tok/s (gen>={min_gen})", dec)
    row("decode tok/s, sampled (T>0)", dec_s)
    for b in ("<=32K", "32-120K", ">120K"):
        if b in bands:
            row(f"decode tok/s, context {b}", bands[b])
    out(f"  drafts accepted/draft                        {pct(acc,10):7.2f} {pct(acc,50):7.2f} {pct(acc,90):7.2f}   n={len(acc)}")
    row(f"prefill tok/s, fresh >= {min_fresh}", fresh)
    return dict(dec=dec, acc=acc, fresh=fresh, bands=bands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", default=[
        "/home/green-gpu/ML_projects/flashnext/20261004_selfopt/data/logs/brain-requests.log"])
    ap.add_argument("--min-gen", type=int, default=200)
    ap.add_argument("--min-fresh", type=int, default=30000)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    allr = {}
    for path in a.logs:
        reqs = parse(path)
        if a.json:
            allr[path] = report(path, reqs, a.min_gen, a.min_fresh, out=lambda *x: None)
        else:
            report(path, reqs, a.min_gen, a.min_fresh)
    if a.json:
        slim = {p: {k: [round(v, 2) for v in vals] if isinstance(vals, list)
                    else {kk: [round(x, 1) for x in vv] for kk, vv in vals.items()}
                    for k, vals in d.items() if vals}
                for p, d in allr.items()}
        print(json.dumps(slim))
    return 0


if __name__ == "__main__":
    sys.exit(main())
