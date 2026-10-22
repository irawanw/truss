"""Image preprocessing for Flash-Next vision (Qwen3-VL style): resize to a multiple of 32, normalize, patchify.

Pure numpy + PIL (no torch). Mirrors preprocessor_config.json of Qwen/Qwen3.8-Flash-Next:
patch 16, temporal patch 2, merge 2, mean/std 0.5, pixel budget 65536..16777216 (we also cap the long side).
preprocess(img) -> (pixel_values [N, 3*2*16*16] f32, grid_thw (1, H/16, W/16)); the model sees N/4 image tokens.
Patch order is the HF one: rows grouped by 2x2 merge window (row-major inside the window), windows row-major.
"""
import math
import numpy as np
from PIL import Image

PATCH, TEMPORAL, MERGE = 16, 2, 2
FACTOR = PATCH * MERGE            # sizes are multiples of 32
MIN_PIXELS, MAX_PIXELS = 65536, 16777216
MAX_SIDE = 1344                   # our cap on the long side (token budget: (1344/32)^2 = 1764 tokens)


def smart_resize(h: int, w: int) -> tuple[int, int]:
    s = min(1.0, MAX_SIDE / max(h, w))
    h, w = max(FACTOR, round(h * s / FACTOR) * FACTOR), max(FACTOR, round(w * s / FACTOR) * FACTOR)
    if h * w > MAX_PIXELS:
        b = math.sqrt(h * w / MAX_PIXELS); h, w = math.floor(h / b / FACTOR) * FACTOR, math.floor(w / b / FACTOR) * FACTOR
    elif h * w < MIN_PIXELS:
        b = math.sqrt(MIN_PIXELS / (h * w)); h, w = math.ceil(h * b / FACTOR) * FACTOR, math.ceil(w * b / FACTOR) * FACTOR
    return h, w


def preprocess(img: Image.Image) -> tuple[np.ndarray, tuple[int, int, int]]:
    img = img.convert("RGB")
    h, w = smart_resize(img.height, img.width)
    if (h, w) != (img.height, img.width):
        img = img.resize((w, h), Image.BICUBIC)
    a = (np.asarray(img, np.float32) / 255.0 - 0.5) / 0.5                       # [H, W, 3]
    a = np.repeat(a.transpose(2, 0, 1)[None], TEMPORAL, axis=0)                  # [T=2, 3, H, W]
    gh, gw = h // PATCH, w // PATCH
    a = a.reshape(1, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    a = a.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)                                   # grid_t, bh, bw, mh, mw, C, T, ph, pw
    return np.ascontiguousarray(a.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH)), (1, gh, gw)
