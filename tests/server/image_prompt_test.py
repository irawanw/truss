"""Host-only test (no GPU) of the image path through Engine.generate: the whole prompt is one eval_img call, the
prompt's checkpoint (and with it any prefix reuse) is skipped, and a second image request starts its sequence over.

  python3 tests/server/image_prompt_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
from server.vision import ImageIn  # noqa: E402
from slot_serving_test import FakeSlots, engine, prompt, run  # noqa: E402


class FakeImg(FakeSlots):
    """Records the image calls' arguments; the sequence and slots behave as in the text fake."""

    def eval_img_argmax(self, tokens, embeds, row0, nh, nw):
        self.img_calls = getattr(self, "img_calls", []) + [("argmax", len(tokens), embeds.shape, row0, nh, nw)]
        self.seq[self.selected] += list(tokens)
        self.rec("eval_img_argmax", len(tokens))
        return 5000 + len(self.seq[self.selected])

    def eval_img_sample(self, tokens, embeds, row0, nh, nw, sp):
        self.img_calls = getattr(self, "img_calls", []) + [("sample", len(tokens), embeds.shape, row0, nh, nw)]
        self.seq[self.selected] += list(tokens)
        self.rec("eval_img_sample", len(tokens))
        return 1000 + len(self.seq[self.selected])


ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{name}: {detail}  {'PASS' if cond else 'FAIL'}")


m = FakeImg()
e = engine(m)
p = prompt(1, 50)
emb = np.zeros((6, 8), np.float32)
img = ImageIn(emb, 10, 3, 2)
out = [t for t, _, _ in e.generate(p, 4, 0.0, 0.95, 40, 7, [], 0.05, image=img) if t is not None]
img_calls = getattr(m, "img_calls", [])
check("one image call", len(img_calls) == 1, str(img_calls))
check("image call args", img_calls[0] == ("argmax", len(p), emb.shape, 10, 3, 2), str(img_calls[0]))
check("no plain prompt eval", not [c for c in m.calls if c[1] == "eval"], str(m.calls))
check("no checkpoint", not [c for c in m.calls if c[1] == "checkpoint"], str(m.calls))
check("sequence", m.seq[0] == p + out[:-1], f"{len(m.seq[0])} tokens")
check("token count", len(out) == 4, str(out))

before = len(m.calls)
p2 = prompt(2, 40)
out2 = [t for t, _, _ in e.generate(p2, 2, 0.0, 0.95, 40, 7, [], 0.05, image=img) if t is not None]
img2 = getattr(m, "img_calls", [])
check("second image: one call", len(img2) == 2 and img2[1][1] == len(p2), str(img2))
check("second image: reset before", m.calls[before][1] == "reset", str(m.calls[before:before + 2]))
check("second image: no restore", not [c for c in m.calls[before:] if c[1] == "restore"], str(m.calls[before:]))

m2 = FakeImg()
e2 = engine(m2)
p3 = prompt(3, 30)
out3 = [t for t, _, _ in e2.generate(p3, 2, 0.7, 0.95, 40, 7, [], 0.05, image=img) if t is not None]
check("sampled goes to eval_img_sample", [c[0] for c in m2.img_calls] == ["sample"], str(m2.img_calls))

print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
