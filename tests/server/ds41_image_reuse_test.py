"""Host-only test (no GPU) of DS41 image spans through Engine.generate (server/vision_ds41.py spans): the spans not
in the slot go in with the read that holds them (one eval_imgs call for any number, rows offset in that call), cached
spans are reused only when the prompt has the same image at the same row, the prompt-end checkpoint still works with
images before it, and the tower runs once per image (span cache by digest).

  python3 tests/server/ds41_image_reuse_test.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
from server.vision_ds41 import Span  # noqa: E402
from slot_serving_test import IM, FakeSlots, engine  # noqa: E402

PAD = 129264


class FakeDs41(FakeSlots):
    """Image calls record (kind, tokens, [(span rows, row0)]) and append to the slot's sequence."""

    def imgs(self, kind, tokens, spans):
        self.img_calls.append((kind, len(tokens), [(r.shape[0], a) for r, a in spans]))
        for r, a in spans:
            assert all(t == PAD for t in tokens[a:a + r.shape[0]]), "the span rows carry the placeholder"
        self.seq[self.selected] += list(tokens)
        self.rec("eval_imgs_" + kind, len(tokens))

    def eval_imgs(self, tokens, spans):
        self.imgs("read", tokens, spans)

    def eval_imgs_argmax(self, tokens, spans):
        self.imgs("argmax", tokens, spans)
        return 5000 + len(self.seq[self.selected])


class FakeTower:
    def __init__(self):
        self.runs, self.cache = [], {}

    def span_rows(self, image, digest):
        if digest not in self.cache:
            self.runs.append(digest)
            self.cache[digest] = np.zeros((image, 4), np.float32)   # the fake's "image" is its span length
        return self.cache[digest]


def build(parts):
    """parts: ints n (n text ids) or ('img', digest, n): the prompt ids and its spans, then the generation prompt."""
    ids, spans, k = [], [], 0
    for p in parts:
        if isinstance(p, tuple):
            spans.append(Span(len(ids), p[2], p[1], p[2]))
            ids += [PAD] * p[2]
        else:
            ids += [1000 + k + i for i in range(p)]
            k += p
    return ids + [IM, 7, 8], spans


ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{name}: {detail}  {'PASS' if cond else 'FAIL'}")


def gen(e, ids, spans, n=3):
    return [t for t, _, _ in e.generate(ids, n, 0.0, 0.95, 40, 7, [], 0.05, spans=spans) if t is not None]


m = FakeDs41(slots=1)
m.img_calls = []
e = engine(m)
e.vision = FakeTower()

# turn 1: text, image A (12 rows), text, image B (9 rows), text
p1, s1 = build([20, ("img", "A", 12), 30, ("img", "B", 9), 40])
g1 = gen(e, p1, s1)
reads = [c for c in m.img_calls]
check("turn 1: both images in one call", [c[0] for c in reads] == ["read"] and len(reads[0][2]) == 2, str(reads))
check("turn 1: rows and offsets", reads[0][2] == [(12, 20), (9, 62)], str(reads))
check("turn 1: sequence", m.seq[0] == p1 + g1[:-1], f"{len(m.seq[0])} tokens")
check("turn 1: tower once per image", e.vision.runs == ["A", "B"], str(e.vision.runs))
check("turn 1: slot spans", e.slots[0].images == [(20, 12, "A"), (62, 9, "B")], str(e.slots[0].images))

# turn 2: the same history + reply + a new image C: the history is reused, only C and the tail go in
m.img_calls = []
p2 = p1 + g1 + [1500 + i for i in range(5)]
c0 = len(p2)
p2 += [PAD] * 7 + [1600, IM, 7, 8]
s2 = s1 + [Span(c0, 7, "C", 7)]
g2 = gen(e, p2, s2)
check("turn 2: only the new image", len(m.img_calls) == 1 and [r for r, _ in m.img_calls[0][2]] == [7], str(m.img_calls))
check("turn 2: no reset", not [c for c in m.calls if c[1] == "reset" and c[0] >= 0][1:], "")
check("turn 2: sequence", m.seq[0] == p2 + g2[:-1], f"{len(m.seq[0])} tokens")
check("turn 2: tower ran only for C", e.vision.runs == ["A", "B", "C"], str(e.vision.runs))

# turn 3: same token ids but image B replaced by B2 (same size): the cached span no longer matches
m.img_calls = []
p3 = list(p1)
s3 = [s1[0], Span(s1[1].row0, 9, "B2", 9)]
gen(e, p3, s3)
check("turn 3: B2 read, no stale B", any(9 in [r for r, _ in c[2]] for c in m.img_calls) and (62, 9, "B") not in e.slots[0].images,
      f"{m.img_calls} spans {e.slots[0].images}")
check("turn 3: sequence restarted or restored before B", m.seq[0][:len(p3)] == p3, "")

# a span that ends the prompt's last call is the sampling step
m2 = FakeDs41(slots=1)
m2.img_calls = []
e2 = engine(m2)
e2.vision = FakeTower()
ids = [1, 2, 3] + [PAD] * 5
sp = [Span(3, 5, "Z", 5)]
gen(e2, ids, sp)
check("span at the end: argmax call", [c[0] for c in m2.img_calls] == ["argmax"] and m2.img_calls[0][2] == [(5, 3)],
      str(m2.img_calls))

print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
