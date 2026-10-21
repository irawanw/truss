"""The arms bench's loop task (tools/tk-bench/arms.py --tasks loop): how often the model loops in its reasoning on real
long agent prompts, per arm. 10-21 the DS41 agent looped in ~30% of turns past 60K context (every turn at 120K) under
every server-side sampling setting, while copy/recall stayed 100% to 200K: this task ablates the engine's
approximations (tail prefill, expert skip, speculation, decode-path prompt pieces) on the prompts that looped.

Prompts: the server's saved requests (TRUSS_REQ_LOG_DIR, .npz with the prompt ids), copied into --loop-dir. Per prompt
the prompt is read once (again per arm for env arms) and checkpointed; per arm, --loop-k samples of up to --loop-tokens
tokens with the served sampling (temperature 1.0, top_p 1.0, min_p 0.05: server/app.py), each stopped at </think>,
the end of the turn, or a loop. Loop = the server's loop guard (server/generate.py: exact copies or low variety).
An arm `spec=0` samples one token per step instead of speculative rounds.
"""
import glob
import json
import os
import time

import numpy as np

from server.generate import LOOP_LONG_PERIOD, LOOP_VARIETY, LOOP_VARIETY_WINDOW, loop_period, variety
from server.truss_ctypes import Sampling


def looped(out):
    if loop_period(out) or loop_period(out, min_p=LOOP_LONG_PERIOD, copies=2):
        return True
    return len(out) >= LOOP_VARIETY_WINDOW and variety(out) < LOOP_VARIETY


def sample(m, last, sp, spec, n_max, stops, close):
    """one continuation; returns (tokens, looped?)"""
    nxt, out = m.eval_sample([last], sp), []
    check = 64
    while len(out) < n_max:
        if spec:
            emitted, nxt = m.spec_step_sampled(nxt, sp)
        else:
            emitted, nxt = [nxt], m.eval_sample([nxt], sp)
        out += emitted
        if any(t in stops or t == close for t in emitted):
            return out, False
        if len(out) >= check:
            check = len(out) + 64
            if looped(out):
                return out, True
    return out, False


def loop_task(b, a):
    files = sorted(glob.glob(os.path.join(a.loop_dir, "*.npz")))
    if not files:
        raise SystemExit(f"loop task: no .npz prompts in {a.loop_dir} (copy them from TRUSS_REQ_LOG_DIR)")
    close = b.tok.token_id("</think>") if b.tok else -1
    for f in files:
        ids = [int(x) for x in np.load(f, allow_pickle=False)["prompt"]]
        name = os.path.basename(f)
        print(f"task loop {name}: {len(ids)} prompt tokens, {a.loop_k} samples x <= {a.loop_tokens} tokens", flush=True)

        def read():
            b.m.reset()
            t = time.time()
            b.m.eval(ids[:-1])
            b.m.checkpoint()
            return time.time() - t
        if not b.envs:
            print(f"  read {read():.1f} s", flush=True)
        for arm in b.arms:
            b.apply(arm)
            if b.envs:
                print(f"  {arm['name']}: read {read():.1f} s", flush=True)
            spec = arm.get("spec", True)
            n_loop, lens, t = 0, [], time.time()
            for k in range(a.loop_k):
                b.m.restore()
                sp = Sampling(1.0, 1.0, 0, 0.05, (hash(name) + 7919 * k) % (1 << 63))
                out, lp = sample(b.m, ids[-1], sp, spec, a.loop_tokens, b.stop, close)
                n_loop += lp
                lens.append(len(out))
            rec = b.res[arm["name"]]["tasks"].setdefault("loop", dict(loops=0, samples=0))
            rec["loops"] += n_loop
            rec["samples"] += a.loop_k
            print(f"  {arm['name']:>14}: looped {n_loop} of {a.loop_k} (mean {np.mean(lens):.0f} tokens per sample, "
                  f"{time.time() - t:.0f} s)", flush=True)
    for name, R in b.res.items():
        r = R["tasks"].get("loop")
        if r:
            print(f"loop rate {name}: {r['loops']} of {r['samples']} ({100 * r['loops'] / max(1, r['samples']):.0f}%)",
                  flush=True)
