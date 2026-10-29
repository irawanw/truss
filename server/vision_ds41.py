"""Image input for DS41 (DeepSeek-V4.1-Flash-1017): DeepSeek's image_processor.load_image in numpy (the reference,
huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/image_processor.py, uses torch) and the span
expansion of each <｜deepseek_image｜> the encoder emits (server/encoding.py emits it for an image part).

The span the tower returns (truss_vision_forward on ds41-mmproj-f16.gguf) is START + (n_llm_w IMAGE + NEWLINE) x
n_llm_h + END; every span position keeps the placeholder id in the prompt (the engine keys the image router bias and
the Engram mask on it), and its rows replace those positions' embeddings. Any number of images per prompt: the span
length follows from the image size alone (plan), so a span whose image is already in the slot's cache (same digest at
the same row, server/generate.py) is neither recomputed nor read again.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = ImageOps = None

PLACEHOLDER = "<｜deepseek_image｜>"
PATCH, RATIO, MAX_TOKENS, MIN_PIXELS = 14, 3, 1024, 295936   # config.json vision_config (max_wh_ratio null)


def num_tokens(lh: int, lw: int) -> int:
    return lh * (lw + 1) + 2


def llm_grid(h: int, w: int) -> tuple[int, int]:
    return math.ceil((h // PATCH) / RATIO), math.ceil((w // PATCH) / RATIO)


def _resize_ratio(height, width):
    """The reference's solve_resize_ratio: the largest aspect-keeping size whose span fits MAX_TOKENS."""
    r = height / width
    max_w = math.sqrt((MAX_TOKENS - 2) / r + 0.25) - 0.5
    max_h = max_w * r
    cell = PATCH * RATIO
    if max_w < 1.0:
        return (MAX_TOKENS - 2) // 2 * cell, cell
    if max_h < 1.0:
        return cell, (MAX_TOKENS - 3) * cell
    beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
    return math.floor(height * beta / PATCH) * PATCH, math.floor(width * beta / PATCH) * PATCH


def plan(width: int, height: int) -> tuple[int, int]:
    """Target pixel size (best_height, best_width) of an image: the reference's plan_image_grid + safe_resize."""
    if 0 < width * height < MIN_PIXELS:
        ratio = (MIN_PIXELS / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    bw, bh = math.ceil(width / PATCH) * PATCH, math.ceil(height / PATCH) * PATCH
    if num_tokens(*llm_grid(bh, bw)) > MAX_TOKENS:
        bh, bw = _resize_ratio(height, width)
        assert num_tokens(*llm_grid(bh, bw)) <= MAX_TOKENS
    return bh, bw


def preprocess(image) -> tuple[np.ndarray, int, int]:
    """RGB image -> (pixel_values f32 [gh*gw][3*14*14], gh, gw): pad to the plan with grey 127 keeping the aspect
    (ImageOps.pad), (x/255 - 0.5)/0.5 rounded to bf16 as the reference feeds the ViT, patches row-major (c, ph, pw)."""
    if Image is None:
        raise ValueError("image input needs Pillow in the server's interpreter")
    bh, bw = plan(image.width, image.height)
    img = ImageOps.pad(image.convert("RGB"), (bw, bh), color=(127, 127, 127))
    x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    x = (x - 0.5) / 0.5
    x = (x.view(np.uint32) + 0x7FFF + ((x.view(np.uint32) >> 16) & 1)) & 0xFFFF0000   # round to bf16 (nearest even)
    x = x.view(np.float32)
    gh, gw = bh // PATCH, bw // PATCH
    px = x.reshape(3, gh, PATCH, gw, PATCH).transpose(1, 3, 0, 2, 4).reshape(gh * gw, 3 * PATCH * PATCH)
    return np.ascontiguousarray(px), gh, gw


@dataclass
class Span:
    """One image of a prompt: prompt rows [row0, row0 + n), the image (rows from Tower.span_rows when fed)."""
    row0: int
    n: int
    digest: str
    image: object

    def key(self):
        return (self.row0, self.n, self.digest)


def span_len(image) -> int:
    bh, bw = plan(image.width, image.height)
    return num_tokens(*llm_grid(bh, bw))


def expand(chat, ids, images) -> tuple[list[int], list[Span]]:
    """images: [(bytes, PIL image)] in prompt order. Each of the encoder's placeholders becomes its span (span_len
    copies of the placeholder id)."""
    pad = chat.token_id(PLACEHOLDER)
    if pad is None:
        raise ValueError(f"the tokenizer has no {PLACEHOLDER}")
    at = [i for i, t in enumerate(ids) if t == pad]
    if len(at) != len(images):
        raise ValueError(f"the prompt has {len(at)} image placeholders for {len(images)} images")
    out, spans, prev = [], [], 0
    for i, (data, img) in zip(at, images):
        out += ids[prev:i]
        n = span_len(img)
        spans.append(Span(len(out), n, hashlib.sha1(data).hexdigest(), img))
        out += [pad] * n
        prev = i + 1
    return out + ids[prev:], spans


def reusable(L: int, cached, spans) -> int:
    """The longest prompt prefix <= L a slot may keep: every cached span (row0, n, digest) that starts before it must
    be the prompt's own image at that row, and it may not end inside one of the prompt's spans."""
    want = {s.key() for s in spans}
    for r in cached:
        if r[0] < L and tuple(r) not in want:
            L = min(L, r[0])
    for s in spans:
        if s.row0 < L < s.row0 + s.n:
            L = s.row0
    return L
