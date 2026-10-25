"""Sequence bookkeeping for the server: prompt / cached-sequence prefix matching, raw token ids, the fair gate
that orders engine calls between requests, and the per-slot record.
"""
import threading
import time
from collections import deque
import numpy as np


def prefix_len(prompt, seq):
    """len(seq) when seq is a proper prefix of prompt (the request still has to fit its own reply on top of it),
    else 0. This is what makes a slot's cached prefix reusable by the next request."""
    n = len(seq)
    return n if 0 < n < len(prompt) and prompt[:n] == seq else 0


def lcp(prompt, seq):
    """How many leading tokens a slot's cache shares with this prompt, whether or not it can be reused. Cheap in
    the common case: when seq is a proper prefix of prompt the answer is already known from the reuse test, so only
    a genuine miss pays the scan, and a miss is a read of seconds. A1 logs this per slot - the size of the miss (a
    126K conversation whose first difference sits at 60K) is what A2's checkpoints have to be sized against."""
    n = prefix_len(prompt, seq)
    if n:
        return n
    m = min(len(prompt), len(seq))
    for i in range(m):
        if prompt[i] != seq[i]:
            return i
    return m


def raw_ids(raw, n_vocab):
    """The token ids of a client-supplied `prompt` field, validated (review F1). That field moved the trust boundary
    from the tokenizer to the client and nothing downstream checks the values: rows_kernel takes the embedding row
    at (size_t) id with no bound, so one id >= n_vocab reads off the end of the table and a negative one becomes an
    address of ~1.8e19 - an illegal address, which a CUDA context does not survive, and both slots go down with it
    until a restart. So: a flat list of ints in [0, n_vocab), never empty (an empty prompt leaves the engine nothing
    to sample and its error reaches the client as a 500). numpy does the range test because a Python loop over a
    150K-token prompt costs ~50 ms; the dtype test is what rejects floats, strings, bools and None."""
    if not raw:
        raise ValueError("prompt: the token id list must not be empty")
    a = np.asarray(raw)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"prompt: token ids must be a flat list of ints, got {a.dtype.name} of shape {a.shape}")
    if a.min() < 0 or a.max() >= n_vocab:
        raise ValueError(f"prompt: token ids must be in [0, {n_vocab}), got {int(a.min())}..{int(a.max())}")
    return a.tolist()


class FairGate:
    """A counting semaphore that serves waiters strictly in arrival order. threading.Lock/Semaphore make no fairness
    promise: the thread that just released can re-acquire ahead of the other one indefinitely, which with two
    streams means one of them freezes behind the other's prefill. Tickets go in a deque and only the head may take
    a permit. Not re-entrant (a request must never hold the engine lock while asking for it again)."""

    def __init__(self, permits: int, name: str = "gate", thread_bound: bool = True, timeout=None):
        # timeout (seconds): how long a waiter may queue before it is an error rather than a hung request. A request
        # stuck behind an engine call that never returns holds its worker thread and its client's connection open
        # forever (review H3); the engine's own waits are bounded the same way (TRUSS_SPIN_TIMEOUT_S).
        self.timeout = timeout
        self.cv, self.permits, self.queue, self.name = threading.Condition(), permits, deque(), name
        self.holders = set()   # thread ids holding a permit, so a re-acquire raises instead of deadlocking
        # thread_bound False: a permit belongs to a request, not a thread (the slot gate: a streaming request's
        # generator steps run on whichever anyio worker thread is free, so it may take its slot on one thread and
        # give it back on another, and one worker may serve two requests' steps). No holder bookkeeping then.
        self.thread_bound = thread_bound

    def acquire(self, timeout=None):
        me, tid = object(), threading.get_ident()
        limit = self.timeout if timeout is None else timeout
        until = None if limit is None else time.monotonic() + limit
        with self.cv:
            if self.thread_bound and tid in self.holders:
                raise RuntimeError(f"{self.name}: re-entrant acquire would deadlock; every engine call goes "
                                   f"through Engine.call, which locks for one call only")
            self.queue.append(me)
            try:
                while not (self.permits and self.queue[0] is me):
                    left = None if until is None else until - time.monotonic()
                    if left is not None and left <= 0:
                        raise TimeoutError(f"{self.name}: no permit within {limit:.0f} s")
                    self.cv.wait(left)
            except BaseException:   # a cancelled wait must not leave its ticket at the head of the queue
                if me in self.queue:
                    self.queue.remove(me)
                    self.cv.notify_all()
                raise
            self.permits -= 1
            self.queue.popleft()
            if self.thread_bound:
                self.holders.add(tid)
            return me

    def release(self):
        with self.cv:
            tid = threading.get_ident()
            if self.thread_bound:
                if tid not in self.holders:
                    raise RuntimeError(f"{self.name}: release without a permit")
                self.holders.discard(tid)
            self.permits += 1
            self.cv.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc):
        self.release()
        return False


class Slot:
    """One sequence in the engine, as the server sees it: the tokens it holds, where its checkpoint is, whether a
    request owns it, and when it was last used (the tie-break when a prompt matches neither). `pos` is the engine's
    own position, recorded by Engine.call after every call, so it is never read off the wrong slot. `images` /
    `ck_images`: the DS41 image spans (row0, n, digest) in `cached` / `ck_tokens`, whose ids alone (all the
    placeholder) do not say which image the engine holds (server/vision_ds41.py reusable)."""
    __slots__ = ("idx", "cached", "ck_tokens", "busy", "last_used", "taken", "pos", "images", "ck_images")

    def __init__(self, idx: int):
        self.idx, self.cached, self.ck_tokens = idx, [], []
        self.images, self.ck_images = [], []
        self.busy, self.last_used, self.taken, self.pos = False, 0.0, 0.0, 0
