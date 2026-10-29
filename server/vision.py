"""Image input for Flash-Next: fetch an OpenAI `image_url` part, preprocess it (server/vision_pre.py), run the ViT
(truss_vision, host pixel_values -> host f32 text rows) and expand the template's single <|image_pad|> into the image
token grid. One image per request (multi-image is out of scope). A prompt may reuse a slot's sequence (or checkpoint)
only where that sequence holds the same image (ImageIn.span: place, rows and pixel hash) at the same place.
"""
from __future__ import annotations

import base64
import io
import urllib.request
import hashlib
from dataclasses import dataclass

import numpy as np

from server.truss_ctypes import Vision

try:   # Pillow only for image input: a text-only server (DS41's interpreter has no PIL) still imports this module
    from PIL import Image
    from server import vision_pre
except ImportError:
    Image = vision_pre = None

MAX_BYTES = 20 << 20          # 20 MB fetch cap
TIMEOUT = 10.0                # seconds
MAX_PIXELS = 40_000_000       # decoded image cap (a decompression-bomb guard on top of the byte cap)
IMAGE_PARTS = ("image", "image_url", "input_image")


@dataclass
class ImageIn:
    embeds: np.ndarray        # f32 [nh * nw][2560], ViT output in text-token order
    row0: int                 # first prompt row replaced
    nh: int
    nw: int
    key: str = ""             # the image's identity: a hash of its decoded pixels (pixels_key); "" = of the embeds

    def __post_init__(self):
        if not self.key:
            self.key = hashlib.blake2b(np.ascontiguousarray(self.embeds).tobytes(), digest_size=16).hexdigest()

    @property
    def span(self) -> tuple[int, int, str]:
        """(first row, rows, identity): what a slot's sequence must hold at the same place to be reused"""
        return self.row0, self.nh * self.nw, self.key


def pixels_key(image) -> str:
    """The identity of a decoded image (the RGB pixels and size): the same picture re-sent every turn of a
    conversation gets the same key, whatever the ViT's run-to-run float noise"""
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{image.width}x{image.height}".encode())
    h.update(image.tobytes())
    return h.hexdigest()


def image_url(messages):
    """The first image part's URL (or local path) in the messages, or None."""
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = str(part.get("type") or "")
            if kind not in IMAGE_PARTS and not (kind == "" and ("image_url" in part or "image" in part or "url" in part)):
                continue
            v = part.get("image_url", part.get("image", part.get("url")))
            if isinstance(v, dict):
                v = v.get("url")
            if isinstance(v, str) and v:
                return v
    return None


class Tower:
    """The mmproj ViT on one device; host image -> host f32 rows [nh*nw][width]."""

    def __init__(self, mmproj: str, device: int = 0):
        self.vit = Vision(mmproj, device)

    def rows(self, image) -> tuple[np.ndarray, int, int]:
        px, (_, gh, gw) = vision_pre.preprocess(image)
        return self.vit.forward(px, gh, gw), gh // 2, gw // 2

    def close(self):
        self.vit.close()


def load(url: str, allow_local: bool = False) -> Image.Image:
    """Decode a data URI, an http(s) URL, or (only with allow_local) a local path."""
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        if ";base64" not in header:
            raise ValueError("image data URI is not base64")
        data = base64.b64decode(payload, validate=True)
    elif url.startswith(("http://", "https://")):
        req = urllib.request.Request(url, headers={"User-Agent": "truss"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            n = int(r.headers.get("Content-Length") or 0)
            if n > MAX_BYTES:
                raise ValueError(f"image is {n} bytes (limit {MAX_BYTES})")
            data = r.read(MAX_BYTES + 1)
    elif allow_local:
        with open(url, "rb") as f:
            data = f.read(MAX_BYTES + 1)
    else:
        raise ValueError("local image paths need --allow-local-images")
    if len(data) > MAX_BYTES:
        raise ValueError(f"image over the {MAX_BYTES} byte limit")
    if Image is None:
        raise ValueError("image input needs Pillow in the server's interpreter")
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.width * img.height > MAX_PIXELS:
        raise ValueError(f"image is {img.width}x{img.height} (limit {MAX_PIXELS} pixels)")
    return img.convert("RGB")


def expand(chat, ids, embeds: np.ndarray, nh: int, nw: int, key: str = "") -> tuple[list[int], ImageIn]:
    """Replace the template's one <|image_pad|> with nh*nw copies and return the rows' placement."""
    pad = chat.token_id("<|image_pad|>")
    if pad is None:
        raise ValueError("the tokenizer has no <|image_pad|>")
    at = [i for i, t in enumerate(ids) if t == pad]
    if len(at) != 1:
        raise ValueError(f"one image per request (the prompt has {len(at)} <|image_pad|> tokens)")
    n = nh * nw
    if embeds.shape[0] != n:
        raise ValueError(f"ViT rows {embeds.shape[0]} != image grid {nh}x{nw}")
    row0 = at[0]
    return ids[:row0] + [pad] * n + ids[row0 + 1:], ImageIn(embeds, row0, nh, nw, key)
