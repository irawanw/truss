#!/usr/bin/env python3
"""Admission-policy replay (lead, 2026-10-07): CPU-computed experts per pass vs ring admission policy, on a TRUSS routing
trace (TRUSS_ROUTE_TRACE format, loader from 20261002_truss_d0/scripts/cache_replay.py), real per-expert bytes.

VRAM budget B = static hot set (ranked by --rank, the served usage_strata_rank.f32) + ring (share swept). Per layer,
ring/hot hits are free; of the misses, round(frac * misses) go over PCIe on demand into the ring (LRU insert), the rest
are computed on the CPU. After each pass, up to --cap of that pass's CPU experts may be ADMITTED into the ring (the idle
link). Policies:
  none     no admission (engine with TRUSS_ADMIT_IDLE=0)
  lru      admit the most-used candidates, ring evicts LRU (today's engine: ADMIT_IDLE caps it, victims are LRU)
  tinylfu  admit a candidate only if its decayed frequency > the would-be LRU victim's (Einziger et al. 2017, TinyLFU)
  lfu      ring evicts the lowest decayed frequency; admit only if the candidate beats that minimum
Frequencies: per (layer, expert) routed count, decayed by --decay per pass. Output: one TSV row per policy x cap."""
import argparse
import heapq
from collections import OrderedDict

import numpy as np

exec(open("/home/green-gpu/ML_projects/flashnext/20261002_truss_d0/scripts/cache_replay.py").read().split("def main():")[0])


def hot_set(score, nb, budget):
    order = np.argsort(-score, axis=None, kind="stable")
    H = np.zeros(score.size, bool)
    c = np.cumsum(nb.ravel()[order])
    H[order[c <= budget]] = True
    return H.reshape(score.shape)


def replay(P, H, nb, ring, frac, policy, cap, decay, wipe=0):
    q, used = OrderedDict(), 0.0          # ring: key -> None, LRU order (oldest first)
    f = {}                                # decayed frequency, lazily: key -> (value, pass stamp)
    t = 0
    cpu = dem = adm = 0

    def freq(k):
        v = f.get(k)
        return 0.0 if v is None else v[0] * decay ** (t - v[1])

    def bump(k):
        f[k] = (freq(k) + 1.0, t)

    def evict_one():
        nonlocal used
        if policy == "lfu":
            k = min(q, key=freq)
            del q[k]
        else:
            k, _ = q.popitem(last=False)
        used -= nb[k]

    def insert(k):
        nonlocal used
        if k in q:
            q.move_to_end(k); return
        q[k] = None; used += nb[k]
        while used > ring and len(q) > 1:
            evict_one()

    def victim_freq(need):
        """frequency of what an insert of `need` bytes would evict (max over the victims)"""
        if used + need <= ring: return -1.0
        if policy == "lfu":
            return min(freq(k) for k in q)
        free, worst = ring - used, -1.0
        for k in q:   # oldest first
            worst = max(worst, freq(k)); free += nb[k]
            if free >= need: break
        return worst

    for p in P:
        if p is None: continue
        T, layers = p
        t += 1
        if wipe and t % wipe == 0:   # a served request's prompt chunk: the stream area takes the whole ring
            q.clear(); used = 0.0
        cand = []
        for l, ids in layers:
            if l >= H.shape[0]: continue
            miss = []
            for e in sorted(ids):
                k = (l, e)
                bump(k)
                if H[l, e]: continue
                if k in q: q.move_to_end(k); continue
                miss.append(k)
            m = int(round(frac * len(miss)))
            for k in miss[len(miss) - m:]:
                dem += 1; insert(k)
            cpu += len(miss) - m
            cand += miss[:len(miss) - m]
        if policy == "none" or not cand: continue
        cand.sort(key=freq, reverse=True)
        n = 0
        for k in cand:
            if n >= cap: break
            if k in q: continue
            if policy in ("tinylfu", "lfu") and freq(k) <= victim_freq(nb[k]): continue
            insert(k); adm += 1; n += 1
    n = sum(1 for p in P if p)
    return cpu / n, dem / n, adm / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--rank", required=True)
    ap.add_argument("--budget", type=float, default=14.5, help="VRAM expert bytes, GB (1e9); 13.53 GiB = 14.5 GB")
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--ring-shares", default="0.2,0.3,0.4,0.5")
    ap.add_argument("--caps", default="61,120")
    ap.add_argument("--decay", type=float, default=0.98)
    ap.add_argument("--policies", default="none,lru,tinylfu,lfu")
    ap.add_argument("--limit", type=int, default=0, help="first N passes only (speed)")
    ap.add_argument("--wipe", default="0", help="empty the ring every N passes (a prompt chunk per request); list")
    a = ap.parse_args()
    L, E, K, nb0, hot, recs = load(a.trace)
    P = [p for p in passes(recs) if p]
    if a.limit: P = P[:a.limit]
    Lm = 48
    nb = {}
    nbm = nb0[:Lm].astype(np.float64)
    for l in range(Lm):
        for e in range(E): nb[(l, e)] = nbm[l, e]
    u = np.fromfile(a.rank, np.float32).reshape(-1, E)[:Lm].astype(np.float64)
    B = a.budget * 1e9
    print(f"# passes {len(P)}  budget {a.budget} GB  frac {a.frac}  decay {a.decay}")
    print("wipe\tpolicy\tcap\tring_share\tcpu/pass\tdemand/pass\tadmit/pass\tcpu_vs_none")
    base = {}
    for rs in map(float, a.ring_shares.split(",")):
        H = hot_set(u, nbm, B * (1 - rs))
        for pol in a.policies.split(","):
            for w in map(int, a.wipe.split(",")):
                for cap in ([0] if pol == "none" else map(int, a.caps.split(","))):
                    c, d, ad = replay(P, H, nb, B * rs, a.frac, pol, cap, a.decay, w)
                    if pol == "none" and w == 0: base[rs] = c
                    print(f"{w}\t{pol}\t{cap}\t{rs}\t{c:.1f}\t{d:.1f}\t{ad:.1f}\t{c / base[rs] - 1:+.1%}", flush=True)


if __name__ == "__main__":
    main()
