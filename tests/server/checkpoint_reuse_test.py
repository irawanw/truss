"""Host-only test of the server's prompt-end checkpoint (no GPU): Engine.generate on a fake model with the engine's
checkpoint semantics (one slot: checkpoint() saves the sequence, restore() returns to it). Checks, over a scripted
agent loop: (1) a prompt that extends the whole engine sequence reads only its tail (plain reuse, as before);
(2) a prompt that extends the last prompt but not its reply (the reply re-sent without its reasoning) restores the
checkpoint and reads only the rest; (3) an unrelated prompt resets; (4) the fake's sequence always equals the prompt
+ the generated tokens, i.e. no stale tokens survive a restore.

  python3 tests/server/checkpoint_reuse_test.py
"""
import os
import sys
import threading

IM = 99999   # <|im_start|>
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from server import app  # noqa: E402


class FakeModel:
    def __init__(self):
        self.n_ctx, self.drafts, self.seq, self.ck, self.fed, self.restores, self.resets = 10000, 0, [], None, 0, 0, 0

    @property
    def position(self):
        return len(self.seq)

    def reset(self):
        self.seq, self.resets = [], self.resets + 1

    def checkpoint(self):
        self.ck = list(self.seq)

    def restore(self):
        if self.ck is None:
            return -1
        self.seq, self.restores = list(self.ck), self.restores + 1
        return len(self.seq)

    def eval(self, tokens):
        self.seq += list(tokens)
        self.fed += len(tokens)

    def eval_argmax(self, tokens):
        self.seq += list(tokens)
        self.fed += len(tokens)
        return 5000 + len(self.seq)


def request(e, prompt, n_gen=5):
    m = e.model
    fed0, r0, z0 = m.fed, m.restores, m.resets
    out = [t for t, _, _ in e.generate(prompt, n_gen, 0.0, 1.0, 0, None, []) if t is not None]
    return out, m.fed - fed0, m.restores - r0, m.resets - z0


def main():
    e = app.Engine.__new__(app.Engine)
    e.model = FakeModel()
    e.chat = types.SimpleNamespace(tok=types.SimpleNamespace(decode=lambda ids, **k: "".join(f"<{i}>" for i in ids)))
    e.stop_ids, e.log_tag, e.log_path, e.dump_dir, e.cached, e.lock = set(), "test", None, None, [], threading.Lock()
    e.ck_tokens, e.keep_reasoning, e.log, e.im_start = [], 0, (lambda line: None), IM
    S = [IM, 7, 8]   # the generation prompt's "<|im_start|>assistant\n": the checkpoint goes before it
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= cond
        print(f"{name}: {detail}  {'PASS' if cond else 'FAIL'}")

    p1 = list(range(1, 1001)) + S
    g1, fed, rs, zs = request(e, p1)
    check("first prompt", fed == 1003 + 4 and zs == 1 and e.ck_tokens == p1[:1000], f"fed {fed}, resets {zs}, "
          f"checkpoint at {len(e.ck_tokens)}")   # 4 decode steps feed tokens; the checkpoint before <|im_start|>
    # the client re-sends the reply with its reasoning (the generated tokens) and a tool result: plain reuse (the
    # last generated token was never fed, so it is read with the tail)
    p2 = p1 + g1 + list(range(2001, 2051)) + S
    g2, fed, rs, zs = request(e, p2)
    check("reply re-sent as generated", fed == 1 + 50 + 3 + 4 and rs == 0 and zs == 0, f"fed {fed}, restores {rs}, resets {zs}")
    # the client re-sends the reply WITHOUT its reasoning: the generation prompt's last token now tokenizes differently
    # too (\n vs \n\n), so only the part before the last <|im_start|> is shared: restore there
    ck = len(e.ck_tokens)
    p3 = p2[:ck] + [IM, 7, 8001] + [9001, 9002] + list(range(3001, 3031)) + S
    g3, fed, rs, zs = request(e, p3)
    check("reply re-sent stripped", fed == len(p3) - ck + 4 and rs == 1 and zs == 0,
          f"fed {fed} (prompt tail {len(p3) - ck}), restores {rs}, resets {zs}")
    check("sequence after restore", e.model.seq == p3 + g3[:-1], f"{len(e.model.seq)} tokens, prompt + generated")
    # an unrelated prompt: reset
    g4, fed, rs, zs = request(e, list(range(7001, 7101)) + S)
    check("unrelated prompt", fed == 103 + 4 and zs == 1 and rs == 0, f"fed {fed}, restores {rs}, resets {zs}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
