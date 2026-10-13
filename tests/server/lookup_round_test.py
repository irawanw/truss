"""Host-only test (no GPU) of the server's lookup ROUND: which proposer a round uses, what drafts it hands the
engine, and what the request's telemetry counts. `server/lookup.py`'s own rules are tested by lookup_test.py; this
covers the wiring in Engine.generate:

  - the proposer is asked BEFORE the round with the engine's own sequence (`cached`) plus the token that round
    verifies (`nxt`), and what it returns is exactly what reaches `truss_spec_step_lookup_sampled` — checked against
    an independent brute-force scan;
  - a round it stays silent on runs the MTP chain instead, unchanged;
  - `drafts accepted A of D` counts the rows actually OFFERED, not `--drafts`: a 2-row lookup window is 2 rows, and
    counting it as 3 silently inflates the acceptance rate the served dashboard reads.

  python3 tests/server/lookup_round_test.py
"""
import os
import re
import sys
import threading
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.pop("TRUSS_REQ_LOG_DIR", None)   # a test must never write the served request log
from server import app            # noqa: E402
from lookup_test import reference  # noqa: E402  (the slow scan is the oracle for both files)

M, N, DRAFTS = 4, 5, 3


class FakeModel:
    """MTP rounds emit nxt + 2 drafts; lookup rounds emit nxt alone and record what they were handed."""

    def __init__(self, engine):
        self.e, self.n_ctx, self.drafts, self.seq = engine, 10000, DRAFTS, []
        self.rounds, self.i = [], 0

    @property
    def position(self):
        return len(self.seq)

    def reset(self):
        self.seq = []

    def checkpoint(self):
        pass

    def restore(self):
        return -1

    def eval(self, tokens):
        self.seq += list(tokens)

    def eval_sample(self, tokens, sp):
        self.seq += list(tokens)
        self.i += 1
        return 500 + self.i

    def spec_step_sampled(self, nxt, sp):
        self.i += 1
        self.rounds.append({"kind": "mtp", "offered": self.drafts})
        self.seq += [nxt, 700, 701]        # a fixed tail, so a later round's (cached + nxt) repeats a 4-gram
        return [nxt, 700, 701], 702

    def spec_step_lookup_sampled(self, nxt, draft, nd, sp):
        self.i += 1
        # the engine must hand us the sequence as it stood before the round, and the proposer's own proposal
        offered = min(nd, self.drafts)          # the engine clamps to the window; the C entrypoint reports it back
        # the oracle: the m-gram must end on `nxt` (this round's win[0]), and only decided tokens can be proposed
        self.rounds.append({"kind": "lookup", "got": list(draft),
                            "want": reference(self.e.cached, M, N, extra=nxt),
                            "seq": list(self.e.cached[-2 * M:]) + [nxt], "offered": offered})
        self.seq += [nxt]
        return [nxt], 702, offered


def main():
    e = app.Engine.__new__(app.Engine)
    e.chat = types.SimpleNamespace(tok=types.SimpleNamespace(decode=lambda ids, **k: "".join(f"<{i}>" for i in ids)))
    e.stop_ids, e.log_tag, e.log_path, e.dump_dir, e.cached, e.lock = set(), "test", None, None, [], threading.Lock()
    e.ck_tokens, e.im_start, e.think_close = [], -1, None   # -1: no prompt-end checkpoint in this test
    e.lookup = app.SuffixLookup(M, N)
    e.model = FakeModel(e)
    lines = []
    e.log = lines.append

    prompt = [1, 2, 1, 2, 1, 2]   # a period-2 repeat, so the proposer has something to find early on
    # 14 tokens is six rounds: three MTP (each emits a window) before the fixed tail repeats a 4-gram, then the
    # proposer takes over on alternate rounds. Fewer tokens and the request ends before the lookup can ever fire.
    out = [t for t, _, _ in e.generate(prompt, 14, 0.8, 0.95, 40, 7, [], 0.05) if t is not None]

    kinds = [r["kind"] for r in e.model.rounds]
    bad = [r for r in e.model.rounds if r["kind"] == "lookup" and r["got"] != r["want"]]
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= cond
        print(f"{name}: {detail}  {'PASS' if cond else 'FAIL'}")

    check("both proposers ran", "lookup" in kinds and "mtp" in kinds, f"rounds {kinds}")
    check("drafts are the proposer's own", not bad,
          f"{len(bad)} of {kinds.count('lookup')} lookup rounds differ from the scan"
          + (f": got {bad[0]['got']} want {bad[0]['want']} tail {bad[0]['seq']}" if bad else ""))
    # the last round emits a whole window but the request ends mid-window: the engine keeps what was not streamed
    check("output is what the rounds emitted",
          len(out) == 14 and e.cached[:len(prompt)] == prompt and e.cached[len(prompt):len(prompt) + len(out)] == out
          and len(e.cached) >= len(prompt) + len(out),
          f"{len(out)} tokens streamed, engine holds {len(e.cached)}")

    summary = next((l for l in lines if "drafts accepted" in l), "")
    offered = sum(r["offered"] for r in e.model.rounds)
    accepted = sum(1 for r in e.model.rounds if r["kind"] == "mtp") * 2
    logged = re.search(r"drafts accepted (\d+) of (\d+)", summary)
    check("telemetry counts offered rows", logged and logged.groups() == (str(accepted), str(offered)),
          f"log says {summary.rsplit('drafts accepted', 1)[-1].strip()!r}, rounds offered {offered}, accepted "
          f"{accepted}")
    check("a short window was offered", any(r["offered"] < DRAFTS for r in e.model.rounds),
          f"offered {[r['offered'] for r in e.model.rounds]} with --drafts {DRAFTS}")
    check("the lookup clause is logged", "lookup supplied" in summary, f"{summary[-90:]!r}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
