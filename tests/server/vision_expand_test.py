"""Host-only test (no GPU) of server/vision.py: the image part's URL, the <|image_pad|> expansion and its guards.

  python3 tests/server/vision_expand_test.py
"""
import base64
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from server import vision  # noqa: E402

PAD = 248056
ok = True


class Chat:
    def token_id(self, text):
        return PAD if text == "<|image_pad|>" else None


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{name}: {detail}  {'PASS' if cond else 'FAIL'}")


def refuse(fn, why):
    try:
        fn()
    except ValueError as e:
        return "" if why in str(e) else f"wrong message: {e}"
    except Exception as e:   # noqa: BLE001
        return f"{type(e).__name__} instead of ValueError: {e}"
    return "accepted"


msgs = [{"role": "user", "content": [
    {"type": "text", "text": "what is this?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
check("url/data-uri", vision.image_url(msgs) == "data:image/png;base64,AAAA")
check("url/none", vision.image_url([{"role": "user", "content": "hi"}]) is None)
check("url/legacy image", vision.image_url([{"role": "user", "content": [{"type": "image", "image": "x.png"}]}])
      == "x.png")
check("url/plain url key", vision.image_url([{"role": "user", "content": [{"url": "http://h/i.png"}]}])
      == "http://h/i.png")

ids = [1, 2, PAD, 3]
emb = np.zeros((6, 8), np.float32)          # 3x2 grid = 6 rows
out, img = vision.expand(Chat(), ids, emb, 3, 2)
check("expand/ids", out == [1, 2] + [PAD] * 6 + [3], str(out))
check("expand/row0", img.row0 == 2, str(img.row0))
check("expand/grid", (img.nh, img.nw) == (3, 2))
check("expand/embeds", img.embeds is emb)
check("expand/none", refuse(lambda: vision.expand(Chat(), [1, 2], emb, 3, 2), "one image per request") == "")
check("expand/two", refuse(lambda: vision.expand(Chat(), [PAD, PAD], emb, 3, 2), "one image per request") == "")
check("expand/rows", refuse(lambda: vision.expand(Chat(), ids, emb[:5], 3, 2), "ViT rows") == "")

buf = io.BytesIO()
Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, "PNG")
uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
im = vision.load(uri)
check("load/data-uri", im.size == (4, 4) and im.getpixel((0, 0)) == (10, 20, 30), str(im.size))
check("load/local refused", refuse(lambda: vision.load("/etc/hostname"), "allow-local-images") == "")
check("load/not base64", refuse(lambda: vision.load("data:image/png,abc"), "base64") == "")
import tempfile  # noqa: E402
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
    f.write(buf.getvalue())
try:
    check("load/local allowed", vision.load(f.name, allow_local=True).size == (4, 4))
finally:
    os.unlink(f.name)

print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
