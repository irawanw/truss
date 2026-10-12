#!/usr/bin/env python3
"""Leapfrog decode simulator: split a verify window into groups that run staggered through the layers.

Idea (2026-10-12): in a verify pass the GPU idles while the CPU tier computes the cold experts of each layer
(X3.1 clean row: ~11.7 ms/pass of spin, decode_pipeline.py simulate_x). A later token's mixer at layer l (attention
KV, GDN state) needs only the earlier tokens' MIXER at l, which is ready before their MoE. So the window's rows can
be split into groups A (earlier tokens) and B (later): B's mixer(l) runs on the GPU while the CPU computes A's
experts(l). Every row still sees the same inputs in the same order (rows are computed independently, which is what
makes spec == plain greedy today), so the output does not change.

Model: list scheduling of per-layer ops on four serial resources, with dependencies:
  MIX(g,l)   GPU   mixer + hc + PLE + glue of group g        deps JOIN(g,l-1), MIX(g-1,l) (causal KV/state)
  SRV(g,l)   DRV   driver split + plan + copy issue           deps MIX(g,l) + poll
  DEM(g,l)   LINK  demand copies of g's GPU-missed experts    deps SRV(g,l)
  SPEC(g,l)  LINK  hint + admission copies (behind demand)    deps DEM(g,l)
  CPU(g,l)   POOL  one pool call: fixed + per expert          deps SRV(g,l)
  WIN(g,l)   GPU   resident scan + GPU expert gemv            deps MIX(g,l), SRV(g,l)+poll, DEM(g,l)
  JOIN(g,l)  GPU   combine the CPU rows                       deps WIN(g,l), CPU(g,l)
The CPU wait is an event (a stream waits on a mapped flag), not a GPU kernel spinning, so the GPU can run the other
group meanwhile; that is an engine change the leapfrog needs (today's window kernel spins on the CPU flag).

Constants: decode_pipeline.PX (X3.1 clean row 1007_064447: wall 35.85 ms, 2.61 tokens/pass). Unknowns swept:
  f_mix   cost of a group's MIX relative to the full window's (decode kernels are mostly weight/launch bound: ~0.7-1)
  pool_a  fixed cost of one pool call (wake + phases + join), ms; per expert b fitted so a + b*3.68 = 0.514
  share   a group's fraction of the window's distinct cold experts and GPU gemv work (routing overlap: 0.5-0.6)
usage: leapfrog.py
"""
import heapq
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_pipeline import PX, ROW_X  # noqa: E402


def simulate(groups=1, f_mix=1.0, pool_a=0.15, share=None, p=PX):
    L = p["n_layer"]
    share = (1.0 / groups if share is None else share) if groups > 1 else 1.0
    fmix = 1.0 if groups == 1 else f_mix
    E = 3.68                                            # distinct CPU experts per full-window call
    pool_b = (p["pool_wall_ms"] - pool_a) / E
    link = p["link_bytes_ms"]
    d = dict(
        MIX=p["tinies_ms"] * fmix,
        SRV=p["driver_api_ms"],
        DEM=p["demand"] * share * p["expert_bytes"] / link,
        SPEC=(p["prefetch"] + p["admission"]) * share * p["expert_bytes"] / link,
        CPU=pool_a + pool_b * E * share,
        WIN=(p["resident_ms"] + p["gemv_ms"]) * share,
        JOIN=0.01,
    )
    res = dict(MIX="gpu", WIN="gpu", JOIN="gpu", SRV="drv", DEM="link", SPEC="link", CPU="pool")
    poll = p["poll_ms"]

    def deps(op):
        k, g, l = op
        if k == "MIX":
            r = []
            if l > 0: r.append((("JOIN", g, l - 1), 0.0))
            if g > 0: r.append((("MIX", g - 1, l), 0.0))
            return r
        if k == "SRV": return [(("MIX", g, l), poll)]
        if k == "DEM": return [(("SRV", g, l), 0.0)]
        if k == "SPEC": return [(("DEM", g, l), 0.0)]
        if k == "CPU": return [(("SRV", g, l), 0.0)]
        if k == "WIN": return [(("MIX", g, l), 0.0), (("SRV", g, l), poll), (("DEM", g, l), 0.0)]
        if k == "JOIN": return [(("WIN", g, l), 0.0), (("CPU", g, l), 0.0)]

    ops = [(k, g, l) for l in range(L) for g in range(groups) for k in res]
    end, free = {}, dict(gpu=0.0, drv=0.0, link=0.0, pool=0.0)
    busy = dict(gpu=0.0, drv=0.0, link=0.0, pool=0.0)
    pending = set(ops)
    prio = {op: (op[2], op[1], i) for i, op in enumerate(ops)}   # earlier layer, then earlier group
    while pending:
        best = None
        for op in pending:
            ds = deps(op)
            if any(o not in end for o, _ in ds): continue
            ready = max([end[o] + lag for o, lag in ds], default=0.0)
            start = max(ready, free[res[op[0]]])
            key = (start, prio[op])
            if best is None or key < best[0]: best = (key, op, start)
        (_, op, start) = best
        r = res[op[0]]
        end[op] = start + d[op[0]]
        free[r] = end[op]
        busy[r] += d[op[0]]
        pending.discard(op)
    verify = max(end[("JOIN", g, L - 1)] for g in range(groups))
    wall = verify + p["host_gap_ms"]
    return wall, verify, busy


def main():
    tpp = PX["tokens_per_pass"]
    w1, v1, b1 = simulate(1)
    print(f"baseline (1 group): wall {w1:.2f} ms (measured {ROW_X['wall']:.2f}) -> {1000 * tpp / w1:.1f} tok/s "
          f"(measured {ROW_X['tg']}); GPU busy {b1['gpu']:.1f}, pool busy {b1['pool']:.1f}, verify {v1:.2f}")
    print("\nleapfrog, 2 groups: tok/s (change vs baseline)")
    print(f"{'f_mix':>6s} {'share':>6s} | " + " ".join(f"pool_a={a:<5.2f}" for a in (0.05, 0.15, 0.25, 0.35)))
    for f in (0.6, 0.75, 0.9, 1.0):
        for sh in (0.5, 0.6):
            cells = []
            for a in (0.05, 0.15, 0.25, 0.35):
                w, v, b = simulate(2, f, a, sh)
                tg = 1000 * tpp / w
                cells.append(f"{tg:5.1f} ({100 * (w1 - w) / w:+4.0f}%)")
            print(f"{f:6.2f} {sh:6.2f} | " + " ".join(cells))
    print("\n3 groups (share 0.4), pool_a 0.15:")
    for f in (0.6, 0.75, 0.9, 1.0):
        w, v, b = simulate(3, f, 0.15, 0.4)
        print(f"  f_mix {f:.2f}: {1000 * tpp / w:5.1f} tok/s ({100 * (w1 - w) / w:+.0f}%), GPU busy {b['gpu']:.1f}, "
              f"pool busy {b['pool']:.1f} ms/pass")


if __name__ == "__main__":
    main()
