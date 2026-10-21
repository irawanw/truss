#!/usr/bin/env python3
"""tk-bench arms: one model load, many arms, on the served path (the C API the server drives, its env and args).

Why: tk-bench-spec / -decode load the model per run (~5 min with the old load, a 30 min matrix), and they ran other
shapes than serving (64K context, 2K prompts, no chat prompt): the drafter's NaN ids on a short chat prompt and the
attention offset on a multi-chunk prompt were found by the server, not the bench (TRACKER #219). This bench opens the
model once with the server's arguments (scripts/ds41_env.sh) and runs every arm on the same loaded model; an arm
changes runtime knobs only (truss_set_option: drafts, skip_weight, tail_skip, pcie_frac, hint_k, adapt_*), or env
knobs the engine reads per call (TRUSS_TAIL_PREFILL, TRUSS_FETCH_PROMPT: then each arm reads its prompts again).

Tasks (each prompt is read once and checkpointed; every arm restores it):
  chat   a short chat-encoded request (decode-path prompt chunks, TRUSS_FETCH_PROMPT)
  code   data/bench_in/prompt_code.i32 (2,048 tokens of code, the earlier benches' prompt 0)
  long   a ~20K-token chat request (three 8K prompt chunks)
  recall the long task's files as raw text (~21K, three chunks), then "app.py again": its first --recall tokens
         teacher-forced. They copy the prompt's first chunk, so a correct engine predicts nearly all of them (top-1);
         broken attention to earlier chunks cannot (the 10-21 attention offset bug for pieces after the first: spec
         and plain run on the same state there, so only a reference like this one sees it). Health: < --recall-min
  kl     held-out sequences: prompt P tokens, then teacher-forced decode steps, KL(FP4 reference || engine) per step
Per arm and prompt task: greedy speculative decode of --tokens tokens (tok/s, tokens/pass, experts per pass), then
the CHECK: the same tokens teacher-forced through the plain decode path (plain tok/s comes free) - every spec token
should be the plain path's argmax: gaps up to --margin (2) logits are near ties, larger ones are counted as
disagreements (the served CPU-tier skip is call-scoped, so a 6-row verify drops other experts than a 1-row step:
gaps of ~2.4 seen 10-21), and a gap over --fail-margin (6) is a broken rollback or verify state (the 10-21 bug: 13.9).
Health (exit 1 on any): a non-finite logit, a token outside the vocabulary, tokens/pass under --min-tpp (a broken
drafter accepts nothing), a gap over --fail-margin, KL over --kl-max (0.30: breakage, not a KL gate - one sequence
of the served config reads 0.08-0.11 by noise alone; the KL gate is --kl-seqs 8 against the 0.0728 budget).
Noise (A/A, two identical arms, 10-21): spec tok/s within ~2%, 2-sequence KL within ~0.012.
Speed repeats interleave the arms, in a rotated order per repeat; the table shows the mean and the spread.

usage (GPUs 0 + 2, under the watchdog; the wrapper stops and restarts the pm2 server):
  scripts/bench_ds41.sh [--quick] --arm base --arm skip_weight=0.12 --arm d3:drafts=3 --arm TRUSS_TAIL_PREFILL=0
An arm is [name:]k=v[,k=v...] or "base" (the served config). Knobs an arm does not name are at the served value.
"""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from server.truss_ctypes import Model  # noqa: E402

# runtime knobs: the served value comes from the TRUSS_* env twin, else runtime::MoeOptions's default
KNOB_DEFAULT = dict(skip_weight=0.0, tail_skip=0, pcie_frac=-1.0, hint_k=4, hint_min_rows=0, adapt_every=0,
                    adapt_swaps=96, adapt_min=2.0, adapt_decay=0.7)
LONG_ASK = "Explain, in 8 bullet points, how a request flows through these files from HTTP to the engine and back."
CHAT_ASK = ("Write a Python module implementing a binary min-heap class with push, pop, peek and heapify, "
            "with docstrings, followed by five unit tests using unittest.")


def parse_arm(s):
    name, _, body = s.rpartition(":") if ":" in s else ("", "", s)
    knobs, env = {}, {}
    if body != "base":
        for kv in filter(None, body.split(",")):
            k, v = kv.split("=", 1)
            if k.startswith("TRUSS_"):
                env[k] = v
            elif k == "drafts" or k in KNOB_DEFAULT:
                knobs[k] = float(v)
            else:
                raise SystemExit(f"arm {s!r}: {k} is not a runtime knob (load-time options need their own run)")
    return dict(name=name or body, knobs=knobs, env=env)


def served_value(k, drafts):
    if k == "drafts":
        return drafts
    e = os.environ.get("TRUSS_" + k.upper())
    return float(e) if e not in (None, "") else KNOB_DEFAULT[k]


class Bench:
    def __init__(self, a, model, tok):
        self.a, self.m, self.tok = a, model, tok
        self.arms = [parse_arm(s) for s in (a.arm or ["base"])]
        self.knobs = sorted({k for arm in self.arms for k in arm["knobs"]})
        self.envs = sorted({k for arm in self.arms for k in arm["env"]})
        self.base_env = {k: os.environ.get(k) for k in self.envs}
        self.drafts_open = a.drafts   # the served value (the model may be opened with more: --drafts-open)
        self.stop = {i for i in (tok.token_id("<｜end▁of▁sentence｜>"),) if i is not None} if tok else set()
        self.res = {arm["name"]: dict(arm=arm, tasks={}, health=[]) for arm in self.arms}

    # ---- arms
    def apply(self, arm):
        for k in self.knobs:
            self.m.set_option(k, arm["knobs"].get(k, served_value(k, self.drafts_open)))
        for k in self.envs:
            v = arm["env"].get(k, self.base_env[k])
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def bad(self, arm, msg):
        self.res[arm["name"]]["health"].append(msg)
        print(f"  HEALTH {arm['name']}: {msg}", flush=True)

    def finite(self, arm, lg, where):
        if not np.isfinite(lg).all():
            self.bad(arm, f"non-finite logits at {where} ({int((~np.isfinite(lg)).sum())} values)")
            return False
        return True

    # ---- prompt tasks
    def prefill(self, ids):
        self.m.reset()
        t = time.time()
        self.m.eval(ids[:-1])
        dt = time.time() - t
        self.m.checkpoint()
        return (len(ids) - 1) / dt, dt

    def warm(self, ids):
        """a discarded decode after a prompt read: the ring refills with decode experts after a prompt's stream, and
        without this the first arm after the read paid it (10-21: 215 vs ~70-115 ms/pass)"""
        if self.a.warm > 0:
            self.m.restore()
            nxt = self.m.eval_argmax(ids[-1:])
            n = 0
            while n < self.a.warm:
                emitted, nxt = self.m.spec_step(nxt)
                n += len(emitted)

    def spec(self, arm, ids, where):
        """restore, feed the prompt's last token, spec-decode: (tokens, seconds, passes, counts delta)"""
        self.m.restore()
        lg = self.m.eval(ids[-1:])
        self.finite(arm, lg, f"{where} prompt end")
        nxt = int(np.argmax(lg))
        out, passes = [], 0
        c0 = self.m.expert_counts()
        t = time.time()
        while len(out) < self.a.tokens and nxt not in self.stop:
            emitted, nxt = self.m.spec_step(nxt)
            passes += 1
            out += emitted
            if any(x in self.stop for x in emitted):
                out = out[:next(i for i, x in enumerate(out) if x in self.stop)]
                break
        dt = time.time() - t
        c1 = self.m.expert_counts()
        bad = [x for x in out if not 0 <= x < self.m.n_vocab]
        if bad:
            self.bad(arm, f"{where}: {len(bad)} tokens outside the vocabulary (first {bad[0]})")
        return out[:self.a.tokens], dt, passes, [b - a for a, b in zip(c0, c1)]

    def check(self, arm, ids, out, where):
        """teacher-force the spec tokens through the plain path: near ties, real disagreements, NLL, plain tok/s"""
        self.m.restore()
        lg = self.m.eval(ids[-1:])
        ties = big = 0
        worst, nll = (0.0, -1), 0.0
        t = time.time()
        for j, tk in enumerate(out):
            if not self.finite(arm, lg, f"{where} check step {j}"):
                break
            top = float(lg.max())
            gap = top - float(lg[tk])
            if gap > 0:
                ties += gap <= self.a.margin
                big += gap > self.a.margin
                worst = max(worst, (gap, j))
            x = lg.astype(np.float64)
            nll += float(np.log(np.exp(x - x.max()).sum()) + x.max() - x[tk])
            if j + 1 < len(out):
                lg = self.m.eval([tk])
        dt = time.time() - t
        if worst[0] > self.a.fail_margin:
            self.bad(arm, f"{where}: a spec token {worst[0]:.2f} logits below the plain argmax at output {worst[1]} "
                          f"(> {self.a.fail_margin}: a broken rollback / verify state)")
        return dict(ties=ties, big=big, worst=round(worst[0], 3), nll=nll / max(1, len(out)),
                    plain_tps=max(0, len(out) - 1) / max(dt, 1e-9))

    def prompt_task(self, name, ids):
        print(f"task {name}: {len(ids)} prompt tokens", flush=True)
        per_arm = bool(self.envs)
        if not per_arm:
            pp, pdt = self.prefill(ids)
            print(f"  prefill {pp:.0f} tok/s ({pdt:.1f} s)", flush=True)
            self.warm(ids)
        for r in range(self.a.repeat):
            k = r % len(self.arms)
            for arm in self.arms[k:] + self.arms[:k]:
                self.apply(arm)
                if per_arm:
                    pp, pdt = self.prefill(ids)
                    self.warm(ids)
                out, dt, passes, cnt = self.spec(arm, ids, name)
                rec = self.res[arm["name"]]["tasks"].setdefault(name, dict(runs=[], prefill=[]))
                rec["prefill"].append(pp)
                rec["runs"].append(dict(tokens=len(out), s=dt, passes=passes, counts=cnt))
                tpp = len(out) / max(1, passes)
                line = (f"  {arm['name']:>14} r{r}: spec {len(out) / max(dt, 1e-9):5.1f} tok/s, {tpp:.2f} tokens/pass,"
                        f" {1e3 * dt / max(1, passes):.1f} ms/pass")
                if passes and len(out) >= 16 and tpp < self.a.min_tpp:
                    self.bad(arm, f"{name}: {tpp:.2f} tokens/pass < {self.a.min_tpp} (drafter accepts nothing?)")
                if r == 0 and not self.a.no_check:
                    ck = self.check(arm, ids, out, name)
                    rec["check"] = ck
                    line += (f"; check: plain {ck['plain_tps']:.1f} tok/s, near ties {ck['ties']}, "
                             f"disagree {ck['big']}, nll {ck['nll']:.3f}")
                    if self.tok:
                        rec["text"] = self.tok.tok.decode(out[:200])
                print(line, flush=True)

    # ---- recall task
    def recall_task(self, ids, n_target, name="recall"):
        P = len(ids) - n_target
        print(f"task {name}: {P} prompt tokens + {n_target} copied tokens teacher-forced", flush=True)
        self.m.reset()
        if name.endswith("i"):   # incremental, as the server builds an agent's context turn by turn (see read_turns)
            self.read_turns(ids[:P - 1])
        else:
            self.m.eval(ids[:P - 1])
        self.m.checkpoint()
        for arm in self.arms:
            self.apply(arm)
            if self.envs:
                self.m.reset()
                self.m.eval(ids[:P - 1])
                self.m.checkpoint()
            self.m.restore()
            lg, hit, nll = self.m.eval(ids[P - 1:P]), 0, 0.0
            for j, t in enumerate(ids[P:]):
                if not self.finite(arm, lg, f"recall step {j}"):
                    break
                x = lg.astype(np.float64)
                hit += int(np.argmax(x) == t)
                nll += float(np.log(np.exp(x - x.max()).sum()) + x.max() - x[t])
                if j + 1 < n_target:
                    lg = self.m.eval([t])
            top1 = hit / n_target
            self.res[arm["name"]]["tasks"][name] = dict(top1=top1, nll=nll / n_target)
            print(f"  {arm['name']:>14}: recall top-1 {100 * top1:.1f}%, nll {nll / n_target:.3f}", flush=True)
            if top1 < self.a.recall_min:
                self.bad(arm, f"{name} top-1 {100 * top1:.1f}% < {100 * self.a.recall_min:.0f}% (attention to earlier "
                              f"prompt chunks broken?)")

    def read_turns(self, ids, first=40000, sizes=(1500, 81, 985, 60, 2600, 300)):
        """the first `first` tokens in one read, the rest in agent-turn sized pieces, each one as the server does it:
        checkpoint, a read past the checkpoint, restore, the same piece again (the client re-sent the turn), on - so
        the restore path, decode-path pieces (<= TRUSS_FETCH_PROMPT) and tail-prefill pieces all run at long context"""
        self.m.eval(ids[:first])
        i, k = first, 0
        while i < len(ids):
            n = min(sizes[k % len(sizes)], len(ids) - i)
            self.m.checkpoint()
            self.m.eval(ids[i:i + n])
            self.m.restore()
            self.m.eval(ids[i:i + n])
            i, k = i + n, k + 1
        print(f"  read in {k + 1} turns (checkpoint / read / restore / read again each)", flush=True)

    # ---- KL task
    def kl_task(self):
        a, V = self.a, self.m.n_vocab
        ids = np.fromfile(a.kl_ids, dtype=np.int32).reshape(-1, a.kl_ctx)
        ref = np.memmap(a.kl_ref, dtype=np.uint16, mode="r").reshape(ids.shape[0], a.kl_ctx, V)
        P = a.kl_ctx - a.kl_steps
        n_arms = len(self.arms)
        if a.kl_seqs % n_arms:   # the rotation below is balanced only over a multiple of the arm count
            a.kl_seqs += n_arms - a.kl_seqs % n_arms
        print(f"task kl: {a.kl_seqs} sequences, prompt {P} + {a.kl_steps} decode steps", flush=True)
        for s in range(a.kl_seqs):
            seq = [int(x) for x in ids[s]]
            self.m.reset()
            self.m.eval(seq[:P])
            self.m.checkpoint()
            rows = (ref[s, P:].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            lr = rows - rows.max(1, keepdims=True)
            lr -= np.log(np.exp(lr).sum(1, keepdims=True))
            self.m.restore()   # warm: as warm(), the first arm after the read had ~0.015 more KL (10-21)
            for t in range(P, min(a.kl_ctx, P + a.warm)):
                self.m.eval_argmax([seq[t]])
            k = s % len(self.arms)   # and a rotated order per sequence
            for arm in self.arms[k:] + self.arms[:k]:
                self.apply(arm)
                if self.envs:   # an env arm: its own prompt read
                    self.m.reset()
                    self.m.eval(seq[:P])
                    self.m.checkpoint()
                self.m.restore()
                kl, agree = [], 0
                for i, t in enumerate(range(P, a.kl_ctx)):
                    lg = self.m.eval([seq[t]])
                    if not self.finite(arm, lg, f"kl seq {s} step {i}"):
                        break
                    x = lg.astype(np.float64)
                    lq = x - x.max()
                    lq -= np.log(np.exp(lq).sum())
                    kl.append(float((np.exp(lr[i]) * (lr[i] - lq)).sum()))
                    agree += int(np.argmax(x) == np.argmax(lr[i]))
                rec = self.res[arm["name"]]["tasks"].setdefault("kl", dict(kl=[], agree=0, n=0))
                rec["kl"] += kl
                rec["agree"] += agree
                rec["n"] += len(kl)
                print(f"  {arm['name']:>14} seq {s}: KL {np.mean(kl):.4f}, top-1 {100 * agree / max(1, len(kl)):.1f}%",
                      flush=True)
        for arm in self.arms:
            rec = self.res[arm["name"]]["tasks"].get("kl")
            if rec and rec["n"] and np.mean(rec["kl"]) > a.kl_max:
                self.bad(arm, f"KL {np.mean(rec['kl']):.4f} > {a.kl_max}")

    # ---- report
    def table(self):
        print("\n== arms (spec tok/s: mean of the repeats over the prompt tasks [min-max]; per pass: cold experts "
              "routed / fetched / MB / CPU misses)")
        base_kl = None
        for name, R in self.res.items():
            T = {k: v for k, v in R["tasks"].items() if k != "kl" and not k.startswith("recall")}
            reps = []
            for r in range(self.a.repeat):
                tok = sum(v["runs"][r]["tokens"] for v in T.values() if len(v["runs"]) > r)
                sec = sum(v["runs"][r]["s"] for v in T.values() if len(v["runs"]) > r)
                reps.append(tok / max(sec, 1e-9))
            runs = [x for v in T.values() for x in v["runs"]]
            n_tok, n_pass = sum(x["tokens"] for x in runs), max(1, sum(x["passes"] for x in runs))
            cnt = np.sum([x["counts"] for x in runs], axis=0) / n_pass if runs else np.zeros(6)
            checks = [v["check"] for v in T.values() if "check" in v]
            ptok = sum(v["runs"][0]["tokens"] for v in T.values() if "check" in v)
            plain = (ptok / sum(v["runs"][0]["tokens"] / max(c["plain_tps"], 1e-9) for v, c in
                                ((v, v["check"]) for v in T.values() if "check" in v))) if checks else 0
            R["summary"] = dict(spec_tps=float(np.mean(reps)) if reps else 0, spread=[min(reps, default=0), max(reps, default=0)],
                                tokens_per_pass=n_tok / n_pass, plain_tps=plain,
                                per_pass=dict(routed=cnt[0], fetched=cnt[1], mb=cnt[2] / 1e6, cpu=cnt[5]),
                                prefill={k: float(np.mean(v["prefill"])) for k, v in T.items()},
                                ties=sum(c["ties"] for c in checks), disagree=sum(c["big"] for c in checks))
            S = R["summary"]
            kl = R["tasks"].get("kl")
            if kl and kl["n"]:
                S["kl"], S["top1"] = float(np.mean(kl["kl"])), 100 * kl["agree"] / kl["n"]
                base_kl = S["kl"] if base_kl is None else base_kl
            if not runs:   # only recall / kl ran
                print(f"{name:>14}:" + "".join(f"  {k} {100 * v['top1']:.1f}%" for k, v in R["tasks"].items() if k.startswith("recall"))
                      + (f"  KL {S['kl']:.4f} ({S['kl'] - base_kl:+.4f}) top-1 {S['top1']:.1f}%" if "kl" in S else "")
                      + ("  HEALTH FAIL" if R["health"] else "  ok"))
                continue
            print(f"{name:>14}: spec {S['spec_tps']:5.1f} [{S['spread'][0]:.1f}-{S['spread'][1]:.1f}]  plain "
                  f"{S['plain_tps']:5.1f}  {S['tokens_per_pass']:.2f} tok/pass  per pass {cnt[0]:.0f}/{cnt[1]:.1f}/"
                  f"{cnt[2] / 1e6:.0f}/{cnt[5]:.1f}  ties {S['ties']} disagree {S['disagree']}"
                  + "".join(f"  {k} {100 * v['top1']:.1f}%" for k, v in R["tasks"].items() if k.startswith("recall"))
                  + (f"  KL {S['kl']:.4f} ({S['kl'] - base_kl:+.4f}) top-1 {S['top1']:.1f}%" if "kl" in S else "")
                  + ("  " + " ".join(f"PP[{k}] {v:.0f}" for k, v in S["prefill"].items()))
                  + ("  HEALTH FAIL" if R["health"] else "  ok"))
        first = next(iter(self.res.values()))
        txt = first["tasks"].get("chat", {}).get("text")
        if txt:
            print(f"\nchat reply ({first['arm']['name']}), first 200 tokens:\n{txt}\n")


def chat_prompts(a, tok):
    """chat / long prompt ids: built once from the tokenizer and frozen in --prompts (the long one quotes repo files,
    which change; the frozen ids keep the bench comparable)"""
    from server.ds41 import Ds41Chat
    out = {}
    for name in ("chat", "long"):
        path = os.path.join(a.prompts, f"arms_{name}.i32")
        if not os.path.exists(path):
            if name == "chat":
                msg = CHAT_ASK
            else:
                repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
                files = LONG_FILES
                # no DeepSeek special-token text in a user message (the encoder refuses it): "<｜" -> "<|"
                msg = "".join(f"File {f}:\n```\n{open(os.path.join(repo, f)).read()}```\n\n" for f in files)
                msg = msg.replace("<｜", "<|") + LONG_ASK
            prompt, _ = Ds41Chat(tok.tok).prompt([{"role": "user", "content": msg}], template_kwargs={"enable_thinking": False})
            np.asarray(tok.encode(prompt), dtype=np.int32).tofile(path)
            print(f"wrote {path}", flush=True)
        out[name] = [int(x) for x in np.fromfile(path, dtype=np.int32)]
    return out


LONG_FILES = ["server/app.py", "server/engine.py", "server/generate.py", "server/engine_debug.py", "server/seq.py",
              "server/ds41.py", "server/truss_ctypes.py"]   # ~21K tokens: 3 chunks


def recall_prompt(a, tok, ctx=0):
    """raw text: every LONG_FILES file in a code block, then the first one again, cut after a.recall tokens; each
    piece encoded on its own so the repeat is the first copy's exact ids. ctx > 0 (tasks recall60 / recall90): more
    distinct repo files (src/, sorted) between them until the prompt reaches ~ctx K tokens, so the copy source sits
    that far back (long-context attention: 10-21 the agent looped only past ~60K). Frozen in --prompts like the others."""
    path = os.path.join(a.prompts, f"arms_recall{ctx or ''}.i32")
    if not os.path.exists(path):
        repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        names = list(LONG_FILES)
        if ctx:
            names += sorted(os.path.relpath(os.path.join(d, f), repo) for d, _, fs in os.walk(os.path.join(repo, "src"))
                            for f in fs if f.endswith((".cu", ".cc", ".h", ".cuh")))
        ids = [tok.token_id("<｜begin▁of▁sentence｜>")]
        first = None
        for f in names:
            body = tok.encode(open(os.path.join(repo, f)).read().replace("<｜", "<|"))
            first = body if first is None else first
            ids += tok.encode(f"File {f}:\n```\n") + body + tok.encode("```\n\n")
            if len(ids) >= ctx * 1000 - a.recall - 64 and (ctx or f == names[len(LONG_FILES) - 1]):
                break
        ids += tok.encode(f"File {LONG_FILES[0]} again, unchanged:\n```\n") + first[:a.recall]
        np.asarray(ids, dtype=np.int32).tofile(path)
        print(f"wrote {path} ({len(ids)} tokens)", flush=True)
    return [int(x) for x in np.fromfile(path, dtype=np.int32)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n-ctx", type=int, default=131072)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--expert-usage")
    ap.add_argument("--mtp")
    ap.add_argument("--drafts", type=int, default=5)
    ap.add_argument("--draft-min-p", type=float, default=0.0)
    ap.add_argument("--drafts-open", type=int, default=0, help="open with this many drafts (buffers) so arms can go "
                    "above the served --drafts; the served value stays the arms' default")
    ap.add_argument("--arm", action="append", help="[name:]k=v[,k=v...] or base; repeat for more arms")
    ap.add_argument("--tasks", default="chat,code,long,recall,kl")
    ap.add_argument("--recall", type=int, default=256, help="copied tokens teacher-forced in the recall task")
    ap.add_argument("--recall-min", type=float, default=0.90)
    ap.add_argument("--tokens", type=int, default=128, help="spec tokens per prompt task")
    ap.add_argument("--repeat", type=int, default=2, help="speed repeats per arm (the check runs on the first)")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--warm", type=int, default=32, help="discarded spec tokens after each prompt read")
    ap.add_argument("--margin", type=float, default=2.0, help="logits: a larger gap counts as a disagreement")
    ap.add_argument("--fail-margin", type=float, default=6.0, help="logits: a larger gap fails health")
    ap.add_argument("--min-tpp", type=float, default=1.5)
    ap.add_argument("--prompts", default=os.path.expanduser("~/ML_projects/dsv41flash/20261019_truss_port/data/bench_in"))
    ap.add_argument("--kl-ids", default=os.path.expanduser("~/ML_projects/dsv41flash/20261019_truss_port/data/kl_ref/heldout32.i32"))
    ap.add_argument("--kl-ref", default=os.path.expanduser("~/ML_projects/dsv41flash/20261019_truss_port/data/kl_ref/ref_fp4.bf16"))
    ap.add_argument("--kl-ctx", type=int, default=1024)
    ap.add_argument("--kl-steps", type=int, default=256)
    ap.add_argument("--kl-seqs", type=int, default=2, help="2: a smoke check (+-0.01); 8: the ds41_kl gate's set; "
                    "rounded up to a multiple of the arm count (the order rotates per sequence)")
    ap.add_argument("--kl-max", type=float, default=0.30, help="health: gross breakage only (see above)")
    ap.add_argument("--quick", action="store_true", help="1 repeat, 64 tokens, 1 KL sequence")
    ap.add_argument("--json", help="write the results here")
    a = ap.parse_args()
    if a.quick:
        a.repeat, a.tokens, a.kl_seqs = 1, 64, 1
    tasks = a.tasks.split(",")
    t0 = time.time()
    holder = {}
    th = threading.Thread(target=lambda: holder.setdefault("chat", __import__("server.chat").chat.Chat(a.tokenizer, None)),
                          daemon=True)
    th.start()
    m = Model(a.model, a.n_ctx, a.chunk, a.expert_usage, mtp=a.mtp, drafts=max(a.drafts, a.drafts_open),
              draft_min_p=a.draft_min_p)
    if m.drafts != a.drafts:
        m.set_option("drafts", a.drafts)
    th.join()
    tok = holder.get("chat")
    print(f"loaded in {time.time() - t0:.1f} s", flush=True)
    m.wait_ready()   # the page-locking of the host expert copy runs ~20 s after the load: measure after it
    print(f"ready (page-locked) at {time.time() - t0:.1f} s", flush=True)
    b = Bench(a, m, tok)
    prompts = chat_prompts(a, tok) if {"chat", "long"} & set(tasks) else {}
    if "code" in tasks:
        prompts["code"] = [int(x) for x in np.fromfile(os.path.join(a.prompts, "prompt_code.i32"), dtype=np.int32)]
    for t in tasks:
        if t == "kl":
            b.kl_task()
        elif t.startswith("recall"):   # recall (~21K), recall60, recall90: the copy source ~N K tokens back; a
            # trailing i (recall90i): the context read turn by turn as the server builds an agent's (read_turns)
            b.recall_task(recall_prompt(a, tok, int(t[6:].rstrip("i") or 0)), a.recall, t)
        elif t in prompts:
            b.prompt_task(t, prompts[t])
        else:
            raise SystemExit(f"unknown task {t}")
    b.apply(b.arms[0])
    b.table()
    fails = {n: R["health"] for n, R in b.res.items() if R["health"]}
    print(f"wall {time.time() - t0:.0f} s (load included); health: " + ("FAIL " + json.dumps(fails) if fails else "PASS"))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({n: dict(arm=R["arm"], summary=R.get("summary"), health=R["health"],
                               tasks={k: {kk: vv for kk, vv in v.items() if kk != "kl"} for k, v in R["tasks"].items()})
                       for n, R in b.res.items()}, f, indent=1, default=float)
    m.close()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
