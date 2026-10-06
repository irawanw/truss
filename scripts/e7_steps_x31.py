#!/usr/bin/env python3
"""Plan v3 0.3 (port of 20261003_x31/scripts/e7_steps.py): the served agent-step pattern on X3.1.

The 150K eval session cut at its last N tool results, sent in order to the server on --port: each request =
the previous one + the session's own assistant reply + the next tool result, so the server reuses the prefix
and reads only the new tokens - the served pattern (fresh prefill only at step 0). Sampling = the server's
served defaults (T 0.8, top_p 0.95, top_k 40, min_p 0.05) - body0 carries none; seed fixed at 1.
Per-step prefill/decode/acceptance come from the server log (parse with served_stats.py or the runner).

usage: e7_steps_x31.py [--port 8193] [--steps 8] [--max-new 384]
"""
import argparse, glob, json, os, time, urllib.request
EXP = os.path.expanduser("~/ML_projects/flashnext/20261003_x31")
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8193)
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--max-new", type=int, default=384)
a = ap.parse_args()
body0 = json.load(open(glob.glob(f"{EXP}/data/replay/eval_clean_150k_*.json")[0]))
m = body0["messages"]
assert m[-1]["role"] == "tool"
for i in range(a.steps):
    cut = len(m) - 2 * (a.steps - 1 - i)
    body = dict(body0, messages=m[:cut], stream=False, seed=1, max_tokens=a.max_new)
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    res = json.load(urllib.request.urlopen(req, timeout=7200))
    print(json.dumps(dict(step=i, messages=cut, usage=res.get("usage"), s=round(time.time() - t0, 1))), flush=True)
