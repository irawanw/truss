"""ctypes binding of the TRUSS C API (include/truss/truss.h). One Model = one sequence on one GPU.

ctypes releases the GIL during each call, so a long prefill does not block the server's event loop threads.
"""
import ctypes
import os

import numpy as np

_DEFAULT_LIB = os.environ.get("TRUSS_LIB") or os.path.join(os.path.dirname(__file__), "..", "build", "libtruss.so")


class TrussError(RuntimeError):
    pass


class Params(ctypes.Structure):   # truss_params
    _fields_ = [("n_ctx", ctypes.c_int), ("max_chunk", ctypes.c_int), ("expert_usage", ctypes.c_char_p),
                ("mtp", ctypes.c_char_p), ("drafts", ctypes.c_int), ("cpu_dir", ctypes.c_char_p),
                ("cpu_share", ctypes.c_float), ("cpu_threads", ctypes.c_int), ("draft_vocab", ctypes.c_char_p),
                ("draft_min_p", ctypes.c_float)]


PENALTY_CAP = 4096   # TRUSS_PENALTY_CAP


class Sampling(ctypes.Structure):   # truss_sampling; penalties off unless set (history: set_history)
    _fields_ = [("temperature", ctypes.c_float), ("top_p", ctypes.c_float), ("top_k", ctypes.c_int),
                ("min_p", ctypes.c_float), ("seed", ctypes.c_uint64),
                ("repetition_penalty", ctypes.c_float), ("frequency_penalty", ctypes.c_float),
                ("presence_penalty", ctypes.c_float), ("penalty_last_n", ctypes.c_int),
                ("history", ctypes.POINTER(ctypes.c_int32)), ("n_history", ctypes.c_int),
                ("ban", ctypes.POINTER(ctypes.c_int32)), ("n_ban", ctypes.c_int), ("ban_lift", ctypes.c_int32),
                ("word_repeat", ctypes.c_float), ("word_lift", ctypes.c_int32),
                ("filler", ctypes.c_float), ("filler_window", ctypes.c_int)]

    def __init__(self, temperature, top_p, top_k, min_p, seed, repetition_penalty=1.0, frequency_penalty=0.0,
                 presence_penalty=0.0, penalty_last_n=64):
        super().__init__(temperature, top_p, top_k, min_p, seed, repetition_penalty, frequency_penalty,
                         presence_penalty, penalty_last_n, None, 0, None, 0, -1, 0.0, -1, 0.0, 0)
        self._hist = None
        self._ban = None

    @property
    def penalized(self) -> bool:
        return self.repetition_penalty != 1.0 or self.frequency_penalty != 0.0 or self.presence_penalty != 0.0

    def set_ban(self, tokens, lift: int = -1):
        """Tokens no sampled row may produce until `lift` has been fed (-1: every row); empty: off (truss.h)."""
        tokens = list(tokens)[:8]
        self._ban = np.ascontiguousarray(tokens, dtype=np.int32) if tokens else None
        self.ban = self._ban.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if tokens else None
        self.n_ban, self.ban_lift = len(tokens), lift

    def set_word_repeat(self, pen: float, lift: int = -1):
        """Logits taken off a word that repeats the previous token's word, until `lift` is fed; 0: off (truss.h)."""
        self.word_repeat, self.word_lift = float(pen), lift

    def set_filler(self, pen: float, window: int = 64):
        """Filler recency: logits off a filler word per use in the last `window` tokens (history + fed); 0: off."""
        self.filler, self.filler_window = float(pen), int(window)

    def set_history(self, tokens):
        """The tokens the penalties count (the last PENALTY_CAP are kept); the array lives until the next call."""
        self._hist = np.ascontiguousarray(tokens[-PENALTY_CAP:], dtype=np.int32)
        self.history = self._hist.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self.n_history = len(self._hist)


class Vision:
    """The vision tower of an mmproj gguf (Flash-Next or DS41, truss.h) on one device: host pixel_values -> host f32 rows."""

    def __init__(self, mmproj_path: str, device: int = 0, lib_path: str = _DEFAULT_LIB):
        lib = ctypes.CDLL(os.path.abspath(lib_path))
        p, c_int, f32p = ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
        lib.truss_vision_open.restype, lib.truss_vision_open.argtypes = p, [ctypes.c_char_p, c_int]
        lib.truss_vision_close.restype, lib.truss_vision_close.argtypes = None, [p]
        lib.truss_vision_width.restype, lib.truss_vision_width.argtypes = c_int, [p]
        lib.truss_vision_rows.restype, lib.truss_vision_rows.argtypes = c_int, [p, c_int, c_int]
        lib.truss_vision_forward.restype = c_int
        lib.truss_vision_forward.argtypes = [p, f32p, c_int, c_int, f32p]
        lib.truss_last_error.restype, lib.truss_last_error.argtypes = ctypes.c_char_p, []
        self._lib = lib
        self._v = lib.truss_vision_open(mmproj_path.encode(), device)
        if not self._v:
            raise TrussError(lib.truss_last_error().decode(errors="replace"))
        self.width = lib.truss_vision_width(self._v)

    def forward(self, pixel_values, gh: int, gw: int) -> np.ndarray:
        px = np.ascontiguousarray(pixel_values, dtype=np.float32)
        out = np.empty((max(self._lib.truss_vision_rows(self._v, gh, gw), 0), self.width), dtype=np.float32)
        rc = self._lib.truss_vision_forward(self._v, px.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), gh, gw,
                                            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if rc < 0:
            raise TrussError(self._lib.truss_last_error().decode(errors="replace"))
        return out[:rc]

    def close(self):
        if self._v:
            self._lib.truss_vision_close(self._v)
            self._v = None


class Model:
    """mtp: MTP draft GGUF (enables spec_step); cpu_dir: CPU expert tier (needs expert_usage)."""

    def __init__(self, gguf_path: str, n_ctx: int, max_chunk: int, expert_usage: str | None = None,
                 mtp: str | None = None, drafts: int = 3, cpu_dir: str | None = None, cpu_share: float = 0.5,
                 cpu_threads: int = 12, draft_vocab: str | None = None, draft_min_p: float = 0.5,
                 lib_path: str = _DEFAULT_LIB):
        lib = ctypes.CDLL(os.path.abspath(lib_path))
        c_int, c_i64, p = ctypes.c_int, ctypes.c_int64, ctypes.c_void_p
        i32p, f32p = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)
        sig = {
            "truss_open": (p, [ctypes.c_char_p, c_int, c_int, ctypes.c_char_p]),
            "truss_close": (None, [p]),
            "truss_last_error": (ctypes.c_char_p, []),
            "truss_n_vocab": (c_int, [p]),
            "truss_n_ctx": (c_int, [p]),
            "truss_position": (c_int, [p]),
            "truss_progress": (None, [p, ctypes.POINTER(c_int), ctypes.POINTER(c_int)]),
            "truss_meta_string": (ctypes.c_char_p, [p, ctypes.c_char_p]),
            "truss_meta_int": (c_i64, [p, ctypes.c_char_p, c_i64]),
            "truss_reset": (c_int, [p]),
            "truss_eval": (c_int, [p, i32p, c_int, f32p]),
            "truss_eval_argmax": (c_int, [p, i32p, c_int, i32p]),
            "truss_default_params": (Params, []),
            "truss_open_params": (p, [ctypes.c_char_p, ctypes.POINTER(Params)]),
            "truss_spec_step": (c_int, [p, ctypes.c_int32, i32p, ctypes.POINTER(c_int), i32p]),
            "truss_drafts": (c_int, [p]),
            "truss_slots": (c_int, [p]),
            "truss_select": (c_int, [p, c_int]),
            "truss_position_s": (c_int, [p, c_int]),
            "truss_lock_ms": (None, [p, ctypes.POINTER(ctypes.c_double), c_int]),
            "truss_progress_s": (None, [p, c_int, ctypes.POINTER(c_int), ctypes.POINTER(c_int)]),
            "truss_reset_s": (c_int, [p, c_int]),
            "truss_eval_s": (c_int, [p, c_int, i32p, c_int, f32p]),
            "truss_eval_step_s": (c_int, [p, c_int, i32p, c_int, c_int, f32p]),
            "truss_eval_argmax_s": (c_int, [p, c_int, i32p, c_int, i32p]),
            "truss_spec_step_s": (c_int, [p, c_int, ctypes.c_int32, i32p, ctypes.POINTER(c_int), i32p]),
            "truss_checkpoint_s": (c_int, [p, c_int]),
            "truss_restore_s": (c_int, [p, c_int]),
            "truss_eval_sample_s": (c_int, [p, c_int, i32p, c_int, ctypes.POINTER(Sampling), i32p]),
            "truss_spec_step_sampled_s": (c_int, [p, c_int, ctypes.c_int32, ctypes.POINTER(Sampling), i32p,
                                                  ctypes.POINTER(c_int), i32p]),
            "truss_checkpoint": (c_int, [p]),
            "truss_restore": (c_int, [p]),
            "truss_eval_sample": (c_int, [p, i32p, c_int, ctypes.POINTER(Sampling), i32p]),
            "truss_spec_step_sampled": (c_int, [p, ctypes.c_int32, ctypes.POINTER(Sampling), i32p,
                                                ctypes.POINTER(c_int), i32p]),
        }
        for name, (res, args) in sig.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
        # The track-U trace hooks bind when present: a library without them still serves, and the two wrappers
        # below no-op. Binding them eagerly turns a stale .so into "the served server will not start".
        for name, (res, args) in {
            "truss_round_trace_dump": (None, []),
            "truss_round_trace_mark": (None, [c_int]),
            "truss_set_option": (c_int, [p, ctypes.c_char_p, ctypes.c_double]),
            "truss_set_word_classes": (c_int, [p, ctypes.POINTER(ctypes.c_int32), c_int]),
            "truss_set_filler_seqs": (c_int, [p, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), c_int]),
            "truss_expert_counts": (None, [p, ctypes.POINTER(ctypes.c_longlong)]),
            "truss_wait_ready": (c_int, [p]),
            "truss_eval_img_s": (c_int, [p, c_int, i32p, c_int, f32p, c_int, c_int, c_int, c_int, c_int, f32p]),
            "truss_eval_img_argmax_s": (c_int, [p, c_int, i32p, c_int, f32p, c_int, c_int, c_int, c_int, c_int, i32p]),
            "truss_eval_img_sample_s": (c_int, [p, c_int, i32p, c_int, f32p, c_int, c_int, c_int, c_int, c_int,
                                                ctypes.POINTER(Sampling), i32p]),
            "truss_eval_imgs_s": (c_int, [p, c_int, i32p, c_int, f32p, c_int, c_int, i32p, i32p, f32p]),
            "truss_eval_imgs_argmax_s": (c_int, [p, c_int, i32p, c_int, f32p, c_int, c_int, i32p, i32p, i32p]),
            "truss_eval_imgs_sample_s": (c_int, [p, c_int, i32p, c_int, f32p, c_int, c_int, i32p, i32p,
                                                 ctypes.POINTER(Sampling), i32p]),
            "truss_vision_open": (p, [ctypes.c_char_p, c_int]),
            "truss_vision_close": (None, [p]),
            "truss_vision_width": (c_int, [p]),
            "truss_vision_forward": (c_int, [p, f32p, c_int, c_int, f32p]),
        }.items():
            fn = getattr(lib, name, None)
            if fn is not None:
                fn.restype, fn.argtypes = res, args
        self._lib = lib
        prm = lib.truss_default_params()
        enc = lambda v: v.encode() if v else None   # noqa: E731
        prm.n_ctx, prm.max_chunk, prm.expert_usage = n_ctx, max_chunk, enc(expert_usage)
        prm.mtp, prm.drafts, prm.cpu_dir = enc(mtp), drafts, enc(cpu_dir)
        prm.cpu_share, prm.cpu_threads = cpu_share, cpu_threads
        prm.draft_vocab, prm.draft_min_p = enc(draft_vocab), draft_min_p
        self._m = lib.truss_open_params(gguf_path.encode(), ctypes.byref(prm))
        if not self._m:
            raise TrussError(self._error())
        self.n_vocab = lib.truss_n_vocab(self._m)
        self.n_ctx = lib.truss_n_ctx(self._m)
        self._logits = np.empty(self.n_vocab, dtype=np.float32)
        self.drafts = lib.truss_drafts(self._m)
        self.slots = lib.truss_slots(self._m)   # TRUSS_SLOTS; n_ctx above is per slot
        self._emitted = (ctypes.c_int32 * 8)()
        self._cur = 0
        # vision bindings are optional (a text-only build still serves); eval_img* raise if they are missing
        self.vision_ok = all(hasattr(lib, k) for k in
                             ("truss_eval_img_s", "truss_eval_img_argmax_s", "truss_eval_img_sample_s",
                              "truss_vision_open", "truss_vision_forward"))

    def select(self, slot: int):
        """Make sequence slot `slot` current (TRUSS_SLOTS); every other call acts on it."""
        self._check(self._lib.truss_select(self._m, slot))
        self._cur = slot

    def _error(self) -> str:
        return self._lib.truss_last_error().decode(errors="replace")

    def _check(self, rc: int):
        if rc < 0:
            raise TrussError(self._error())

    def close(self):
        if self._m:
            self._lib.truss_close(self._m)
            self._m = None

    @property
    def position(self) -> int:
        return self._lib.truss_position(self._m)

    def progress(self):
        """(tokens run, tokens asked) of the eval running on another thread; updated per prompt chunk."""
        d, t = ctypes.c_int(0), ctypes.c_int(0)
        self._lib.truss_progress(self._m, ctypes.byref(d), ctypes.byref(t))
        return d.value, t.value

    def meta_string(self, key: str):
        v = self._lib.truss_meta_string(self._m, key.encode())
        return v.decode() if v is not None else None

    def meta_int(self, key: str, default: int = -1) -> int:
        return self._lib.truss_meta_int(self._m, key.encode(), default)

    def reset(self):
        self._check(self._lib.truss_reset(self._m))

    def checkpoint(self):
        """Save the sequence's recurrent state at the current position (one slot)."""
        self._check(self._lib.truss_checkpoint(self._m))

    def restore(self) -> int:
        """Return the sequence to the checkpoint; its position, or -1 if none was taken."""
        r = self._lib.truss_restore(self._m)
        if r < -1:
            raise TrussError(self._error())
        return r

    def eval(self, tokens) -> np.ndarray:
        """Append tokens; returns the last token's next-token logits (a view reused by the next call)."""
        arr = np.ascontiguousarray(tokens, dtype=np.int32)
        self._check(self._lib.truss_eval(self._m, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(arr),
                                         self._logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
        return self._logits

    def _img_args(self, embeds):
        e = np.ascontiguousarray(embeds, dtype=np.float32)
        if e.ndim != 2:
            raise ValueError("image embeds must be [rows, width]")
        return e, e.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), e.shape[1]

    def _eval_img_s(self, libfn, slot, tokens, embeds, img_row0, nh, nw, *extra):
        arr = np.ascontiguousarray(tokens, dtype=np.int32)
        e = np.ascontiguousarray(embeds, dtype=np.float32)
        if e.ndim != 2:
            raise ValueError("image embeds must be [rows, width]")
        return arr, e, libfn(self._m, slot, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(arr),
                             e.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), e.shape[0], e.shape[1], img_row0, nh, nw,
                             *extra)

    def _eval_imgs_s(self, libfn, slot, tokens, spans, *extra):
        """spans: [(rows f32 [n][width], row0 in this call)], in order (DS41 1-D images, truss_eval_imgs_s)."""
        arr = np.ascontiguousarray(tokens, dtype=np.int32)
        e = np.ascontiguousarray(np.concatenate([r for r, _ in spans]), dtype=np.float32)
        r0 = np.array([a for _, a in spans], dtype=np.int32)
        ln = np.array([r.shape[0] for r, _ in spans], dtype=np.int32)
        i32 = ctypes.POINTER(ctypes.c_int32)
        rc = libfn(self._m, slot, arr.ctypes.data_as(i32), len(arr), e.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   e.shape[1], len(spans), r0.ctypes.data_as(i32), ln.ctypes.data_as(i32), *extra)
        self._check(rc)
        return arr, e, r0, ln

    def eval_imgs(self, tokens, spans):
        self._eval_imgs_s(self._lib.truss_eval_imgs_s, self._cur, tokens, spans,
                          self._logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def eval_imgs_s(self, slot: int, tokens, spans):
        self._eval_imgs_s(self._lib.truss_eval_imgs_s, slot, tokens, spans,
                          self._logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def eval_imgs_argmax(self, tokens, spans) -> int:
        out = ctypes.c_int32()
        self._eval_imgs_s(self._lib.truss_eval_imgs_argmax_s, self._cur, tokens, spans, ctypes.byref(out))
        return out.value

    def eval_imgs_argmax_s(self, slot: int, tokens, spans) -> int:
        out = ctypes.c_int32()
        self._eval_imgs_s(self._lib.truss_eval_imgs_argmax_s, slot, tokens, spans, ctypes.byref(out))
        return out.value

    def eval_imgs_sample(self, tokens, spans, sp: Sampling) -> int:
        out = ctypes.c_int32()
        self._eval_imgs_s(self._lib.truss_eval_imgs_sample_s, self._cur, tokens, spans, ctypes.byref(sp), ctypes.byref(out))
        return out.value

    def eval_imgs_sample_s(self, slot: int, tokens, spans, sp: Sampling) -> int:
        out = ctypes.c_int32()
        self._eval_imgs_s(self._lib.truss_eval_imgs_sample_s, slot, tokens, spans, ctypes.byref(sp), ctypes.byref(out))
        return out.value

    def eval_img(self, tokens, embeds, img_row0: int, nh: int, nw: int) -> np.ndarray:
        """Append a prompt containing one image; logits of the last token."""
        lp = self._logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        arr, e, rc = self._eval_img_s(self._lib.truss_eval_img_s, self._cur, tokens, embeds, img_row0, nh, nw, lp)
        self._check(rc)
        return self._logits

    def eval_img_s(self, slot: int, tokens, embeds, img_row0: int, nh: int, nw: int) -> np.ndarray:
        lp = self._logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        arr, e, rc = self._eval_img_s(self._lib.truss_eval_img_s, slot, tokens, embeds, img_row0, nh, nw, lp)
        self._check(rc)
        return self._logits

    def eval_img_argmax(self, tokens, embeds, img_row0: int, nh: int, nw: int) -> int:
        out = ctypes.c_int32()
        arr, e, rc = self._eval_img_s(self._lib.truss_eval_img_argmax_s, self._cur, tokens, embeds, img_row0, nh, nw,
                                      ctypes.byref(out))
        self._check(rc)
        return out.value

    def eval_img_argmax_s(self, slot: int, tokens, embeds, img_row0: int, nh: int, nw: int) -> int:
        out = ctypes.c_int32()
        arr, e, rc = self._eval_img_s(self._lib.truss_eval_img_argmax_s, slot, tokens, embeds, img_row0, nh, nw,
                                      ctypes.byref(out))
        self._check(rc)
        return out.value

    def eval_img_sample(self, tokens, embeds, img_row0: int, nh: int, nw: int, sp: Sampling) -> int:
        out = ctypes.c_int32()
        arr, e, rc = self._eval_img_s(self._lib.truss_eval_img_sample_s, self._cur, tokens, embeds, img_row0, nh, nw,
                                      ctypes.byref(sp), ctypes.byref(out))
        self._check(rc)
        return out.value

    def eval_img_sample_s(self, slot: int, tokens, embeds, img_row0: int, nh: int, nw: int, sp: Sampling) -> int:
        out = ctypes.c_int32()
        arr, e, rc = self._eval_img_s(self._lib.truss_eval_img_sample_s, slot, tokens, embeds, img_row0, nh, nw,
                                      ctypes.byref(sp), ctypes.byref(out))
        self._check(rc)
        return out.value

    def eval_argmax(self, tokens) -> int:
        """Append tokens; returns the greedy next token."""
        arr = np.ascontiguousarray(tokens, dtype=np.int32)
        out = ctypes.c_int32()
        self._check(self._lib.truss_eval_argmax(self._m, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(arr),
                                                ctypes.byref(out)))
        return out.value

    def spec_step(self, nxt: int):
        """One greedy speculative round after token `nxt`: returns (tokens appended, starting with nxt; the next
        greedy token). Requires the model opened with mtp."""
        n, new = ctypes.c_int(), ctypes.c_int32()
        self._check(self._lib.truss_spec_step(self._m, nxt, self._emitted, ctypes.byref(n), ctypes.byref(new)))
        return list(self._emitted[:n.value]), new.value

    def eval_sample(self, tokens, sp: Sampling) -> int:
        """Append tokens; returns a sample (on the device) of the next token."""
        arr = np.ascontiguousarray(tokens, dtype=np.int32)
        out = ctypes.c_int32()
        self._check(self._lib.truss_eval_sample(self._m, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(arr),
                                                ctypes.byref(sp), ctypes.byref(out)))
        return out.value

    def spec_step_sampled(self, nxt: int, sp: Sampling):
        """spec_step for sampled decoding: drafts accepted while each verify row's sample equals the draft."""
        n, new = ctypes.c_int(), ctypes.c_int32()
        self._check(self._lib.truss_spec_step_sampled(self._m, nxt, ctypes.byref(sp), self._emitted, ctypes.byref(n),
                                                      ctypes.byref(new)))
        return list(self._emitted[:n.value]), new.value

    # ---- slot-explicit calls (TRUSS_SLOTS): calls on different slots may run on different threads at once (the
    # engine overlaps them on the two GPUs); calls on one slot must not overlap. Buffers are per call.

    def position_s(self, slot: int) -> int:
        return self._lib.truss_position_s(self._m, slot)

    def progress_s(self, slot: int):
        d, t = ctypes.c_int(0), ctypes.c_int(0)
        self._lib.truss_progress_s(self._m, slot, ctypes.byref(d), ctypes.byref(t))
        return d.value, t.value

    def reset_s(self, slot: int):
        self._check(self._lib.truss_reset_s(self._m, slot))

    def checkpoint_s(self, slot: int):
        self._check(self._lib.truss_checkpoint_s(self._m, slot))

    def restore_s(self, slot: int) -> int:
        r = self._lib.truss_restore_s(self._m, slot)
        if r < -1:
            raise TrussError(self._error())
        return r

    def eval_s(self, slot: int, tokens) -> np.ndarray:
        arr = np.ascontiguousarray(tokens, dtype=np.int32)
        out = np.empty(self.n_vocab, dtype=np.float32)
        self._check(self._lib.truss_eval_s(self._m, slot, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(arr),
                                           out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
        return out

    def eval_step_s(self, slot: int, tokens, step: int, logits: bool = False):
        """eval_s in prompt chunks of at most `step` tokens (lets another slot's rounds through more often)."""
        arr = np.ascontiguousarray(tokens, dtype=np.int32)
        out = np.empty(self.n_vocab, dtype=np.float32) if logits else None
        self._check(self._lib.truss_eval_step_s(self._m, slot, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                                len(arr), step,
                                                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if logits else None))
        return out

    def eval_argmax_s(self, slot: int, tokens) -> int:
        arr = np.ascontiguousarray(tokens, dtype=np.int32)
        out = ctypes.c_int32()
        self._check(self._lib.truss_eval_argmax_s(self._m, slot, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                                  len(arr), ctypes.byref(out)))
        return out.value

    def eval_sample_s(self, slot: int, tokens, sp: Sampling) -> int:
        arr = np.ascontiguousarray(tokens, dtype=np.int32)
        out = ctypes.c_int32()
        self._check(self._lib.truss_eval_sample_s(self._m, slot, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                                  len(arr), ctypes.byref(sp), ctypes.byref(out)))
        return out.value

    def spec_step_s(self, slot: int, nxt: int):
        n, new, em = ctypes.c_int(), ctypes.c_int32(), (ctypes.c_int32 * 8)()
        self._check(self._lib.truss_spec_step_s(self._m, slot, nxt, em, ctypes.byref(n), ctypes.byref(new)))
        return list(em[:n.value]), new.value

    def spec_step_sampled_s(self, slot: int, nxt: int, sp: Sampling):
        n, new, em = ctypes.c_int(), ctypes.c_int32(), (ctypes.c_int32 * 8)()
        self._check(self._lib.truss_spec_step_sampled_s(self._m, slot, nxt, ctypes.byref(sp), em, ctypes.byref(n),
                                                        ctypes.byref(new)))
        return list(em[:n.value]), new.value

    def lock_ms(self, reset: bool = False):
        """[front held, front wait, last held, last wait] ms since the last reset (engine stage locks)."""
        out = (ctypes.c_double * 4)()
        self._lib.truss_lock_ms(self._m, out, int(reset))
        return list(out)

    def set_word_classes(self, cls):
        """Every token's word class for Sampling.word_repeat (truss_set_word_classes; server/words.py builds it)."""
        a = np.ascontiguousarray(cls, dtype=np.int32)
        self._check(self._lib.truss_set_word_classes(self._m, a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(a)))

    def set_filler_seqs(self, seqs):
        """The filler words' token sequences (server/words.py filler_seqs) for Sampling.set_filler."""
        flat = np.ascontiguousarray([t for q in seqs for t in q], dtype=np.int32)
        lens = np.ascontiguousarray([len(q) for q in seqs], dtype=np.int32)
        i32p = ctypes.POINTER(ctypes.c_int32)
        self._check(self._lib.truss_set_filler_seqs(self._m, flat.ctypes.data_as(i32p), lens.ctypes.data_as(i32p), len(seqs)))

    def set_option(self, name: str, value: float):
        """A runtime knob between calls (truss_set_option: drafts and the decode-tier knobs; tools/tk-bench/arms.py)."""
        self._check(self._lib.truss_set_option(self._m, name.encode(), float(value)))
        if name == "drafts":
            self.drafts = int(value)

    def wait_ready(self):
        """Blocks until the load's background page-locking is done (truss_wait_ready)."""
        self._check(self._lib.truss_wait_ready(self._m))

    def expert_counts(self):
        """Cumulative [cold experts routed, fetched on demand, their bytes, prefetched, their bytes, CPU-tier misses]."""
        out = (ctypes.c_longlong * 6)()
        self._lib.truss_expert_counts(self._m, out)
        return list(out)

    def trace_mark(self, tag: int):
        """TRUSS_ROUND_TRACE: a phase boundary in the engine's interval log (nothing when the variable is unset)."""
        if getattr(self._lib, "truss_round_trace_mark", None) is not None:
            self._lib.truss_round_trace_mark(tag)

    def trace_dump(self):
        """TRUSS_ROUND_TRACE: flush the buffered intervals to the file (scripts/u_trace_report.py reads it)."""
        if getattr(self._lib, "truss_round_trace_dump", None) is not None:
            self._lib.truss_round_trace_dump()
