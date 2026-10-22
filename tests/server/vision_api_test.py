"""Host test (small GPU use) of the vision API contract on the 8-layer slice: the ViT on one device, the text model on
another, both called from the SAME thread (the server's thread pool does exactly that). A leaked cudaSetDevice in
truss_vision_forward used to fail the next model call with "CUDA invalid resource handle" - this catches it in
seconds instead of after a full 125B load.

  CUDA_VISIBLE_DEVICES=2,0 python3 tests/server/vision_api_test.py
    device 0 = the first listed GPU (the text model), device 1 = the second (the ViT)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from server import vision, vision_pre  # noqa: E402
from server.chat import Chat  # noqa: E402
from server.truss_ctypes import Model, Vision  # noqa: E402

SLICE = os.environ.get("SLICE", "/mnt/hdd/ml/flashnext/20260930_truss_cp2/data/slice8/"
                                "flashnext-x3-slice8-00001-of-00002.gguf")
MMPROJ = os.environ.get("MMPROJ", "/home/green-gpu/ML_projects/flashnext/20261022_vision/data/"
                                  "flashnext-mmproj-f16.gguf")
TOKENIZER = os.environ.get("TOKENIZER", "/home/green-gpu/ML_projects/flashnext/20261022_vision/hf/tokenizer.json")
IMAGE = os.environ.get("IMAGE", "/home/green-gpu/ML_projects/flashnext/20261022_vision/img/test448.png")
VIT_DEV = int(os.environ.get("VIT_DEV", "1"))
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{name}: {detail}  {'PASS' if cond else 'FAIL'}")


m = Model(SLICE, 4096, 4096)
vit = Vision(MMPROJ, VIT_DEV)
px, (_, gh, gw) = vision_pre.preprocess(vision.load(IMAGE, allow_local=True))
nh, nw = gh // 2, gw // 2

# 1. the ViT leaves the caller's device alone (the bug: Vision.forward then a model call in one thread)
before = torch.cuda.current_device()
emb = vit.forward(px, gh, gw)
check("device restored after ViT", torch.cuda.current_device() == before,
      f"device {before} -> {torch.cuda.current_device()} (ViT ran on {VIT_DEV})")

# 2. the text model works right after the ViT on the same thread
nxt = m.eval_argmax([1, 2, 3, 4])
check("model call after ViT", isinstance(nxt, int) and 0 <= nxt < m.n_vocab, f"token {nxt}")

# 3. the image call: rows replace the pad tokens, and decode continues
ids = [10] + [248056] * (nh * nw) + [11, 12, 13]
m.reset_s(0)
out = m.eval_img(ids, emb, 1, nh, nw)
check("eval_img logits", out.shape == (m.n_vocab,) and np.isfinite(out).all(), f"shape {out.shape}")
check("position after image", m.position_s(0) == len(ids), f"position {m.position_s(0)} of {len(ids)}")
nxt = m.eval_argmax([14])
check("decode after image", isinstance(nxt, int), f"token {nxt}")

# 4. an expanding prompt: text around one <|image_pad|>, expanded by vision.expand (the real tokenizer ids)
chat = Chat(TOKENIZER, None)
pre = chat.encode("<|im_start|>user\nWhat text is in this image?\n")
post = chat.encode("<|im_end|>\n<|im_start|>assistant\n")
ids = pre + [248056] + post
m.reset_s(0)
ids, img = vision.expand(chat, ids, emb, nh, nw)
out = m.eval_img(ids, img.embeds, img.row0, img.nh, img.nw)
check("expand + eval_img", np.isfinite(out).all() and img.row0 == len(pre),
      f"{len(ids)} tokens, row0 {img.row0} (+{len(pre)} pre), grid {nh}x{nw}")

print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
