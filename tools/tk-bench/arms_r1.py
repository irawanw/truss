"""The arms bench's r1 task (tools/tk-bench/arms.py --tasks r1): how well each arm follows deep reasoning, teacher-forced
on reference turns of the full-precision model (docs: 20261019_truss_port/docs/PLAN-20261023-expert-score.md, M1).

Why: the mean KL on held-out code reads fine while the served pack thinks shallowly and loops. Quantization damage
sits at high-entropy "fork" tokens (arXiv 2606.00206: per-position KL tracks entropy, Spearman 0.92) and loops come
from a pull to repeat (2512.12895), so the task measures those directly instead of a mean.

Input: --r1-dir (20261019_truss_port/scripts/r1_build.py output: <id>.npz prompt / target / kind, index.json), the
turns of split --r1-split. Per turn the prompt is read once (again per arm for env arms) and checkpointed; per arm the
first --r1-tokens target tokens are teacher-forced one decode step each, the rest of the thinking is read as a chunk,
and the --r1-close tokens before the reference's </think> are teacher-forced again (complete turns only).

Per scored position: ce (-log p of the reference token), ent (entropy), top1 (argmax = reference), p_end (p of
</think>, thinking positions before the close window only), p_mark (p mass on reflection markers such as " Wait",
" But", " Alternatively"), p_copy (p of the token an induction copy would emit: the token after the most recent earlier
occurrence of the last 4 tokens, where that token is not the reference's; the pull to repeat). Raw arrays go to
--r1-out/<arm>.npz; the table: means, ce on fork positions (top 20% entropy under the first arm, the same positions
for every arm) and p_end / p_copy relative to the first arm.
"""
import glob
import json
import os
import time

import numpy as np

MARKERS = (" Wait", "Wait", " But", " Alternatively", " Hmm", " Actually", " However", " Maybe", " maybe", " Let",
           " verify", " check", " reconsider")


def add_args(ap):
    ap.add_argument("--r1-dir", default=os.path.expanduser("~/ML_projects/dsv41flash/20261019_truss_port/data/r1"))
    ap.add_argument("--r1-split", default="heldout")
    ap.add_argument("--r1-max-ctx", type=int, default=131072, help="skip turns whose prompt + reply is longer")
    ap.add_argument("--r1-turns", type=int, default=0, help="first N turns of the split (0 = all)")
    ap.add_argument("--r1-tokens", type=int, default=384, help="teacher-forced target tokens per turn from the start")
    ap.add_argument("--r1-close", type=int, default=64, help="teacher-forced tokens before the reference </think>")
    ap.add_argument("--r1-out", help="raw per-position arrays (default: no files)")


def logsm(x):
    x = x.astype(np.float64)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def copy_cand(seq, table):
    """the token after the most recent earlier occurrence of seq's last 4 tokens (-1: none); updates the table"""
    if len(seq) < 5:
        return -1
    table[tuple(seq[-5:-1])] = seq[-1]
    return table.get(tuple(seq[-4:]), -1)


def r1_task(b, a):
    index = [r for r in json.load(open(os.path.join(a.r1_dir, "index.json")))   # the split, turns that fit n_ctx
             if r["split"] == a.r1_split and r["ctx"] + r["target"] < min(a.n_ctx, a.r1_max_ctx)]
    index.sort(key=lambda r: r["n"])
    if a.r1_turns:
        index = index[:a.r1_turns]
    tok = b.tok.tok
    end = tok.convert_tokens_to_ids("</think>")
    marks = sorted({i[0] for i in (tok.encode(w, add_special_tokens=False) for w in MARKERS) if len(i) == 1})
    raw = {arm["name"]: {k: [] for k in ("ce", "ent", "top1", "p_end", "p_mark", "p_copy", "kind")} for arm in b.arms}
    print(f"task r1: {len(index)} turns ({a.r1_split}), {a.r1_tokens} + {a.r1_close} scored tokens per turn, "
          f"{len(marks)} marker tokens", flush=True)
    for r in index:
        d = np.load(os.path.join(a.r1_dir, r["id"] + ".npz"))
        prompt, target, kind = [int(x) for x in d["prompt"]], [int(x) for x in d["target"]], d["kind"]
        n_think = int((kind == 0).sum())
        head = list(range(min(a.r1_tokens, len(target))))
        close = list(range(max(len(head), n_think - a.r1_close), n_think)) if r.get("complete") and end in target else []

        def read():
            b.m.reset()
            t = time.time()
            b.m.eval(prompt[:-1])
            b.m.checkpoint()
            return time.time() - t
        if not b.envs:
            read()
        t0 = time.time()
        for arm in b.arms:
            b.apply(arm)
            if b.envs:
                read()
            b.m.restore()
            lg = b.m.eval([prompt[-1]]).copy()
            R, seq, table = raw[arm["name"]], list(prompt), {}
            for j in range(max(4, len(prompt) - 32768), len(prompt)):   # the copy table over the prompt's last 32K
                table[tuple(prompt[j - 4:j])] = prompt[j]
            pos = 0
            for i in head + close:
                if i > pos:   # read the unscored stretch as one chunk
                    lg = b.m.eval(target[pos:i]).copy()
                    for t in target[pos:i]:
                        seq.append(t)
                        copy_cand(seq, table)
                    pos = i
                ls = logsm(lg)
                p = np.exp(ls)
                ref = target[i]
                cand = table.get(tuple(seq[-4:]), -1)
                R["ce"].append(-ls[ref])
                R["ent"].append(float(-(p * ls).sum()))
                R["top1"].append(int(np.argmax(lg) == ref))
                R["p_end"].append(float(p[end]) if kind[i] == 0 and i < n_think - a.r1_close else np.nan)
                R["p_mark"].append(float(p[marks].sum()) if kind[i] == 0 else np.nan)
                R["p_copy"].append(float(p[cand]) if cand >= 0 and cand != ref else np.nan)
                R["kind"].append(int(kind[i]))
                lg = b.m.eval([ref]).copy()
                seq.append(ref)
                copy_cand(seq, table)
                pos = i + 1
        print(f"  turn {r['id']} ctx {len(prompt)}: {len(head) + len(close)} scored x {len(b.arms)} arms, "
              f"{time.time() - t0:.0f} s", flush=True)
    first = next(iter(raw))
    ent0 = np.array(raw[first]["ent"])
    fork = ent0 >= np.percentile(ent0, 80) if len(ent0) else ent0.astype(bool)
    print("\n== r1 (teacher-forced on the reference turns; fork = top 20% entropy under the first arm)")
    for name, R in raw.items():
        A = {k: np.array(v, np.float64) for k, v in R.items()}
        th = A["kind"] == 0
        S = dict(ce=float(A["ce"].mean()), ce_think=float(A["ce"][th].mean()), ce_fork=float(A["ce"][fork].mean()),
                 top1=100 * float(A["top1"].mean()), p_end=float(np.nanmean(A["p_end"])),
                 p_mark=float(np.nanmean(A["p_mark"])), p_copy=float(np.nanmean(A["p_copy"])), n=int(len(A["ce"])))
        b.res[name]["tasks"]["r1"] = S
        print(f"{name:>14}: ce {S['ce']:.4f} think {S['ce_think']:.4f} fork {S['ce_fork']:.4f}  top-1 {S['top1']:.1f}%  "
              f"p(</think>) {S['p_end']:.2e}  p(markers) {S['p_mark']:.4f}  p(copy) {S['p_copy']:.4f}  ({S['n']} positions)",
              flush=True)
        if a.r1_out:
            os.makedirs(a.r1_out, exist_ok=True)
            np.savez(os.path.join(a.r1_out, f"{name}.npz"), **A)
