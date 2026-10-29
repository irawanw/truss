"""The arms bench's gap task (tools/tk-bench/arms.py --tasks gap): how far the pack is from the ORIGINAL model on the
API's own reasoning. User 10-24: "the mission is deep thinking like the original model at xx bpw that fits our box".

Reference: P1 (20261019_truss_port/scripts/p1_stream.py) ran the full FP4 model over --gap-ids (64 windows x 4096 of
the API's reasoning turns, p1_windows.py) and saved --gap-head {nll, ent} fp16 [64, 4096]: nll[w, t] = -log p(ids[w,
t+1] | ids[w, :t+1]), ent[w, t] the entropy there. A position counts when ids[w, t+1] is an API reply token
(kind >= 0), as P1's own "ppl of the API tokens".
Per window (the first --gap-windows, spread evenly over the 64) the first 4096 - --gap-tokens tokens (or --gap-at) are read as a
prompt and checkpointed; per arm the last --gap-tokens positions are teacher-forced one decode step each (the decode
path the server uses), so each scored position sees 3.8K+ tokens of context. Printed per arm: the pack's mean NLL,
the original's on the same positions, the gap, and the gap on fork positions (top 20% original entropy), plus top-1
(the pack's argmax = the API token). A gap near 0 = thinks like the original on these tokens.
"""
import os
import time

import numpy as np
import torch

P1 = os.path.expanduser("~/ML_projects/dsv41flash/20261019_truss_port/data/p1")


def add_args(ap):
    ap.add_argument("--gap-ids", default=os.path.join(P1, "ids_4096.pt"))
    ap.add_argument("--gap-head", default=os.path.join(P1, "run", "head.pt"))
    ap.add_argument("--gap-windows", type=int, default=16)
    ap.add_argument("--gap-tokens", type=int, default=256, help="teacher-forced positions at the end of each window")
    ap.add_argument("--gap-at", type=int, default=0, help="first scored position (0: the last --gap-tokens of the window); "
                    "e.g. 512 scores positions 512-767: the same tokens' gap at short context (10-24: is it attention?)")


def gap_task(b, a):
    d, h = torch.load(a.gap_ids), torch.load(a.gap_head)
    ids, kind = d["calib"].numpy().astype(np.int64), d["kind"].numpy()
    ref_nll, ref_ent = h["nll"].float().numpy(), h["ent"].float().numpy()
    n, T = ids.shape
    wins = np.linspace(0, n - 1, min(a.gap_windows, n)).round().astype(int)
    P = a.gap_at or T - a.gap_tokens
    E = P + a.gap_tokens
    assert 0 < P and E <= T
    print(f"task gap: {len(wins)} windows x {a.gap_tokens} positions from {P} (context {P}-{E}) (reference: FP4, "
          f"{a.gap_head})", flush=True)
    rows = {arm["name"]: [] for arm in b.arms}   # (pack nll, ref nll, ref ent, top-1 hit) per scored position
    for w in wins:
        seq = ids[w]
        t0 = time.time()
        b.m.reset()
        first = b.m.eval([int(x) for x in seq[:P]])   # logits at position P-1
        b.m.checkpoint()
        for arm in b.arms:
            b.apply(arm)
            b.m.restore()
            lg = first
            for t in range(P - 1, E - 1):
                if t > P - 1:
                    lg = b.m.eval([int(seq[t])])
                if kind[w, t + 1] < 0 or not np.isfinite(ref_nll[w, t]):
                    continue
                x = lg.astype(np.float64)
                x -= x.max()
                lp = x - np.log(np.exp(x).sum())
                rows[arm["name"]].append((-lp[seq[t + 1]], ref_nll[w, t], ref_ent[w, t], int(np.argmax(x) == seq[t + 1])))
        print(f"  window {w}: {time.time() - t0:.0f} s", flush=True)
    for arm in b.arms:
        r = np.array(rows[arm["name"]])
        if not len(r):   # 10-24: --gap-at 512 scored nothing (each window's first 1024 tokens are the prompt)
            print(f"  {arm['name']:>14}: NO API reply tokens in positions {P}-{E}: nothing scored", flush=True)
            continue
        fork = r[:, 2] >= np.quantile(r[:, 2], 0.8)
        res = dict(n=len(r), nll=float(r[:, 0].mean()), ref=float(r[:, 1].mean()), gap=float((r[:, 0] - r[:, 1]).mean()),
                   fork_gap=float((r[fork, 0] - r[fork, 1]).mean()), top1=float(r[:, 3].mean()))
        b.res[arm["name"]]["tasks"]["gap"] = res
        print(f"  {arm['name']:>14}: nll {res['nll']:.4f} vs original {res['ref']:.4f}  gap {res['gap']:+.4f}  "
              f"fork gap {res['fork_gap']:+.4f}  top-1 {100 * res['top1']:.1f}%  ({res['n']} positions)", flush=True)
