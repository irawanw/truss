"""Host-side suffix (prompt-lookup) drafts for the speculative loop.

A deterministic proposer: find where the current tail already occurred in this request's own sequence and propose
what followed there. No model, no device, no synchronisation — it reads the host-side token list the server already
owns (`Engine.cached`) and hands the ids to `truss_spec_step_lookup`, whose verify/accept rule is the proposer-agnostic
one, so a wrong proposal costs a round and changes no output.

The rule is the one measured offline (`PROJ/scripts/lookup_replay.py` walks this class over real generations): the
m-gram that ENDS on the token being verified selects candidate positions, the candidate whose match extends furthest
BACK wins (that length is known before proposing, so it is a legal confidence signal), and the proposal is what
followed that occurrence — at most `n` tokens, never past the last decided token.

Hash collisions are defeated by comparing the m tokens explicitly before trusting a candidate: a collision would
otherwise propose garbage and lose the round.
"""
from __future__ import annotations

BASE = 1000003
MASK = (1 << 64) - 1
MAX_BACK = 64          # how far back a match may be extended when scoring candidates


class SuffixLookup:
    """Index of the m-grams of one sequence, extended as the sequence grows. Not thread-safe: the engine's
    generation lock already serialises the loop that calls it."""

    def __init__(self, m: int = 8, n: int = 7, keep: int = 8, max_back: int = MAX_BACK):
        assert 1 <= m <= 32 and 1 <= n <= 7, "m is the match length, n the drafts offered (the window is 8 rows)"
        self.m, self.n, self.keep, self.max_back = m, n, keep, max_back
        self.pow_m1 = pow(BASE, m - 1, 1 << 64)
        self.tbl: dict[int, list[int]] = {}   # m-gram hash -> the `keep` most recent start positions
        self.upto = 0                         # positions [0, upto) are indexed
        self.h = 0                            # hash of seq[upto : upto + m]
        self.mark = -1                        # seq[upto - 1]: cheap guard against a truncated/rewritten sequence
        self.built_ms = 0.0                   # index-build time, for the log line

    def reset(self):
        self.tbl.clear()
        self.upto, self.h, self.mark = 0, 0, -1

    def _extend(self, seq):
        """Index every m-gram that starts before the current tail. Incremental: a round adds its emitted tokens."""
        import time
        t0 = time.perf_counter()
        if self.upto > len(seq) or (self.upto and seq[self.upto - 1] != self.mark):
            self.reset()                      # the sequence was truncated (reset / checkpoint restore)
        end = len(seq) - self.m               # a start at `end` IS the tail: it can never be a match
        if end <= self.upto:
            return
        m, tbl, keep = self.m, self.tbl, self.keep
        h = self.h
        if self.upto == 0:
            for j in range(m):
                h = (h * BASE + seq[j]) & MASK
        for p in range(self.upto, end):
            lst = tbl.get(h)
            if lst is None:
                tbl[h] = [p]
            else:
                lst.append(p)
                if len(lst) > keep:
                    del lst[:len(lst) - keep]
            h = ((h - seq[p] * self.pow_m1) * BASE + seq[p + m]) & MASK
        self.h, self.upto, self.mark = h, end, seq[end - 1]
        self.built_ms += 1000.0 * (time.perf_counter() - t0)

    def propose(self, seq, banned=(), extra=None) -> list[int]:
        """Up to `n` ids that followed the best earlier occurrence of this sequence's tail, or [] if there is none.

        `extra` is the token the round has already decided and is about to verify (`win[0]`): the engine knows
        `seq` + `extra`, and a draft is a guess for a position AFTER `extra`, so the m-gram being looked up must END
        on `extra`. Keying on `seq` alone - the obvious mistake - proposes every draft one position early, so a
        round that looked like a 7-token match accepts nothing (measured: 0.6 tokens per fired round).

        `banned` is the set of ids the sampler currently refuses (the end-of-turn tokens while a reasoning block is
        open): a row can never draw them, so a proposal containing one is truncated there — a round that starts with
        a banned draft is a round thrown away.
        """
        self._extend(seq)
        m, n = self.m, self.n
        L = len(seq) + (1 if extra is not None else 0)     # virtual length, counting the token under verification
        limit = len(seq)                                   # followers we may propose: decided tokens only
        if L < m + 1 or self.upto == 0:
            return []
        # the tail is the last m DECIDED tokens; with `extra` that is the last m-1 of seq plus extra, and
        # seq[L - m:] already means that because L counts extra
        tail = list(seq[L - m:]) + ([extra] if extra is not None else [])
        h = 0
        for t in tail:                                     # the key is the tail's own hash, computed here
            h = (h * BASE + t) & MASK                      # (self.h tracks the index, not the tail)
        cands = self.tbl.get(h)
        if not cands:
            return []
        best_p, best_back, best_room = -1, 0, 0
        for p in reversed(cands):
            room = min(n, limit - (p + m))        # decided followers only; an occurrence overlapping the tail has
            if room <= 0:                         # room 0 by construction, so the tail can never match itself
                continue
            if any(seq[p + j] != tail[j] for j in range(m)):
                continue                      # hash collision: this is not the same m-gram
            back = 0
            while (back < self.max_back and p - back - 1 >= 0 and seq[p - back - 1] == seq[L - m - 1 - back]):
                back += 1                       # context before the occurrence vs context before the tail
            if (back, room) > (best_back, best_room):
                best_p, best_back, best_room = p, back, room
        if best_p < 0:
            return []
        out = list(seq[best_p + m: best_p + m + best_room])
        if banned:
            for i, t in enumerate(out):
                if t in banned:
                    return out[:i]
        return out
