"""ctypes binding of the TRUSS C API (include/truss/truss.h). One Model = one sequence on one GPU.

ctypes releases the GIL during each call, so a long prefill does not block the server's event loop threads.
"""
import ctypes
import os

import numpy as np

_DEFAULT_LIB = os.path.join(os.path.dirname(__file__), "..", "build", "libtruss.so")


class TrussError(RuntimeError):
    pass


class Params(ctypes.Structure):   # truss_params
    _fields_ = [("n_ctx", ctypes.c_int), ("max_chunk", ctypes.c_int), ("expert_usage", ctypes.c_char_p),
                ("mtp", ctypes.c_char_p), ("drafts", ctypes.c_int), ("cpu_dir", ctypes.c_char_p),
                ("cpu_share", ctypes.c_float), ("cpu_threads", ctypes.c_int), ("draft_vocab", ctypes.c_char_p),
                ("draft_min_p", ctypes.c_float)]


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
            "truss_meta_string": (ctypes.c_char_p, [p, ctypes.c_char_p]),
            "truss_meta_int": (c_i64, [p, ctypes.c_char_p, c_i64]),
            "truss_reset": (c_int, [p]),
            "truss_eval": (c_int, [p, i32p, c_int, f32p]),
            "truss_eval_argmax": (c_int, [p, i32p, c_int, i32p]),
            "truss_default_params": (Params, []),
            "truss_open_params": (p, [ctypes.c_char_p, ctypes.POINTER(Params)]),
            "truss_spec_step": (c_int, [p, ctypes.c_int32, i32p, ctypes.POINTER(c_int), i32p]),
            "truss_drafts": (c_int, [p]),
        }
        for name, (res, args) in sig.items():
            fn = getattr(lib, name)
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
        self._emitted = (ctypes.c_int32 * 8)()

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

    def meta_string(self, key: str):
        v = self._lib.truss_meta_string(self._m, key.encode())
        return v.decode() if v is not None else None

    def meta_int(self, key: str, default: int = -1) -> int:
        return self._lib.truss_meta_int(self._m, key.encode(), default)

    def reset(self):
        self._check(self._lib.truss_reset(self._m))

    def eval(self, tokens) -> np.ndarray:
        """Append tokens; returns the last token's next-token logits (a view reused by the next call)."""
        arr = np.ascontiguousarray(tokens, dtype=np.int32)
        self._check(self._lib.truss_eval(self._m, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(arr),
                                         self._logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
        return self._logits

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
