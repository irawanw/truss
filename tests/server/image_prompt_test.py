"""Host-only test (no GPU) of the image path through Engine.generate: an image prompt is read from position 0 as the
image prefix (eval_img), the prompt-end checkpoint, then the last turn as text; the next turn of the same conversation
(the same picture at the same place) reuses the sequence or the checkpoint instead of reading everything again; a
different picture with the same ids, or a picture past the reusable part, starts the sequence over; a picture that
runs into the last turn is one call with no checkpoint.

  python3 tests/server/image_prompt_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
from server.generate import CK_BACK  # noqa: E402
from server.vision import ImageIn  # noqa: E402
from slot_serving_test import IM, FakeSlots, engine, prompt  # noqa: E402


class FakeImg(FakeSlots):
    """Records the image calls' arguments; the sequence and slots behave as in the text fake."""

    def img_call(self, kind, tokens, embeds, row0, nh, nw):
        self.img_calls = getattr(self, "img_calls", []) + [(kind, len(tokens), embeds.shape, row0, nh, nw)]
        self.seq[self.selected] += list(tokens)
        self.rec("eval_img" if kind == "read" else "eval_img_" + kind, len(tokens))

    def eval_img(self, tokens, embeds, row0, nh, nw):
        self.img_call("read", tokens, embeds, row0, nh, nw)
        return np.zeros(4, np.float32)

    def eval_img_argmax(self, tokens, embeds, row0, nh, nw):
        self.img_call("argmax", tokens, embeds, row0, nh, nw)
        return 5000 + len(self.seq[self.selected])

    def eval_img_sample(self, tokens, embeds, row0, nh, nw, sp):
        self.img_call("sample", tokens, embeds, row0, nh, nw)
        return 1000 + len(self.seq[self.selected])


ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{name}: {detail}  {'PASS' if cond else 'FAIL'}")


def gen(e, p, img, n=4, temperature=0.0):
    m = e.model
    c0, i0 = len(m.calls), len(getattr(m, "img_calls", []))
    out = [t for t, _, _ in e.generate(p, n, temperature, 0.95, 40, 7, [], 0.05, image=img) if t is not None]
    return out, m.calls[c0:], getattr(m, "img_calls", [])[i0:]


def names(calls):
    return [c[1] for c in calls]


def fed(calls):
    return sum(c[2] for c in calls if c[1] in ("eval", "eval_argmax", "eval_sample") or c[1].startswith("eval_img"))


emb = np.zeros((6, 8), np.float32)
img = ImageIn(emb, 10, 3, 2, key="pic-a")

# 1. the first turn: the image prefix up to the checkpoint, the checkpoint, the last turn as text
m = FakeImg(slots=1)
e = engine(m)
p = prompt(1, 50)   # 50 ids + [IM, 7, 8]: the checkpoint goes CK_BACK before the IM
cut = 50 - CK_BACK
out, calls, ic = gen(e, p, img)
check("first turn: image prefix call", ic == [("read", cut, emb.shape, 10, 3, 2)], str(ic))
check("first turn: then checkpoint, then text", names(calls)[:4] == ["reset", "eval_img", "checkpoint", "eval_argmax"],
      str(names(calls)[:4]))
check("first turn: checkpoint holds the image", e.slots[0].ck_images == [img.span] and len(e.slots[0].ck_tokens) == cut,
      str((e.slots[0].ck_images, len(e.slots[0].ck_tokens))))
check("first turn: sequence", m.seq[0] == p + out[:-1], f"{len(m.seq[0])} tokens")

# 2. the next turn continues the whole sequence, the same picture re-sent: only the new tokens are read
held = list(m.seq[0])
p2 = held + prompt(2, 20)
out2, calls2, ic2 = gen(e, p2, ImageIn(emb, 10, 3, 2, key="pic-a"))
check("continued: no image call", ic2 == [], str(ic2))
check("continued: no reset or restore", not {"reset", "restore"} & set(names(calls2)), str(names(calls2)))
check("continued: reads only the new tokens", fed(calls2) - len(out2) + 1 == len(p2) - len(held),
      f"fed {fed(calls2)}, new {len(p2) - len(held)}")
check("continued: sequence", m.seq[0] == p2 + out2[:-1], f"{len(m.seq[0])} tokens")

# 3. the reply re-rendered differently: back to the checkpoint (which holds the image), no image call
ck = list(e.slots[0].ck_tokens)
p3 = ck + prompt(3, 10)
out3, calls3, ic3 = gen(e, p3, ImageIn(emb, 10, 3, 2, key="pic-a"))
check("re-rendered: restore", "restore" in names(calls3) and "reset" not in names(calls3), str(names(calls3)))
check("re-rendered: no image call", ic3 == [], str(ic3))
check("re-rendered: sequence", m.seq[0] == p3 + out3[:-1], f"{len(m.seq[0])} tokens")

# 4. the same ids with another picture: nothing is reused
p4 = list(m.seq[0]) + prompt(4, 10)
out4, calls4, ic4 = gen(e, p4, ImageIn(emb, 10, 3, 2, key="pic-b"))
check("other picture: reset", names(calls4)[0] == "reset" and "restore" not in names(calls4), str(names(calls4)[:3]))
check("other picture: image read again", len(ic4) == 1 and ic4[0][3] == 10, str(ic4))
check("other picture: sequence", m.seq[0] == p4 + out4[:-1], f"{len(m.seq[0])} tokens")

# 5. a text conversation, then a picture past what it holds: read from position 0 (the engine's rule)
m5 = FakeImg(slots=1)
e5 = engine(m5)
q = prompt(5, 30)
out5, _, _ = gen(e5, q, None)
held5 = list(m5.seq[0])
img5 = ImageIn(emb, len(held5) + 2, 3, 2, key="pic-c")
q2 = held5 + [1, 2] + [77] * 6 + prompt(6, 10)
out6, calls6, ic6 = gen(e5, q2, img5)
check("image past the reusable part: reset", names(calls6)[0] == "reset", str(names(calls6)[:3]))
check("image past the reusable part: read from 0", ic6 and ic6[0][1] + fed([c for c in calls6 if not c[1].startswith(
    "eval_img")]) - len(out6) + 1 == len(q2), str(ic6))
check("image past the reusable part: sequence", m5.seq[0] == q2 + out6[:-1], f"{len(m5.seq[0])} tokens")
# a text-only prompt never reuses a sequence that holds a picture
q3 = list(m5.seq[0]) + prompt(7, 5)
_, calls7, _ = gen(e5, q3, None)
check("text prompt on an image sequence: reset", names(calls7)[0] == "reset", str(names(calls7)[:3]))

# 6. a picture that runs into the last turn: one call over the whole prompt, no checkpoint
m8 = FakeImg(slots=1)
e8 = engine(m8)
p8 = prompt(8, 50)
out8, calls8, ic8 = gen(e8, p8, ImageIn(emb, 48, 3, 2, key="pic-d"))
check("image in the last turn: one call", ic8 == [("argmax", len(p8), emb.shape, 48, 3, 2)], str(ic8))
check("image in the last turn: no checkpoint", "checkpoint" not in names(calls8), str(names(calls8)))

# 7. sampled: the image prefix, then the sampled text tail
m9 = FakeImg(slots=1)
e9 = engine(m9)
p9 = prompt(9, 30)
_, calls9, ic9 = gen(e9, p9, ImageIn(emb, 4, 3, 2, key="pic-e"), temperature=0.7)
check("sampled: image prefix then eval_sample", [c[0] for c in ic9] == ["read"] and "eval_sample" in names(calls9),
      str(names(calls9)))

check("calls stayed on their slot", IM and all(c[0] == 0 for mm in (m, m5, m8, m9) for c in mm.calls), "")
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
