"""Host-only test of the suffix proposer (no GPU): server/lookup.py must propose the continuation of the best earlier
occurrence of the sequence's tail, and nothing else. A wrong proposal is not a correctness bug (the engine's verify
still gates it), but it throws away a round, so the rules that decide firing and length are worth pinning:

  - an occurrence may not be the tail itself, and must have something after it to copy;
  - the candidate whose match extends furthest BACK wins — that length is the only confidence signal a proposer has
    before verify, and it is what the offline study (`PROJ/scripts/sim_lookup.py --mlen`) measured;
  - a proposal is capped by n and by what actually follows the occurrence;
  - a banned id cuts the proposal there: a verify row can never draw a banned token, so drafting one throws the round;
  - the index is incremental: extending it one token at a time must agree with a brute-force scan of every prefix,
    and a truncated sequence (engine reset / checkpoint restore) must not propose from dropped tokens.

  python3 tests/server/lookup_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from server.lookup import SuffixLookup  # noqa: E402


def reference(seq, m, n, max_back=64, banned=(), extra=None):
    """The same rule computed the slow obvious way: scan every start position of the prefix. `extra` is the token
    under verification (the round's win[0]): the m-gram must end on it, and only decided tokens can be proposed."""
    L = len(seq) + (1 if extra is not None else 0)
    limit = len(seq)
    if L < m + 1:
        return []
    tail = seq[L - m:] + ([extra] if extra is not None else [])
    best_p, best_back, best_room = -1, 0, 0
    for p in range(0, limit - m):
        if seq[p:p + m] != tail:
            continue
        back = 0
        while back < max_back and p - back - 1 >= 0 and seq[p - back - 1] == seq[L - m - 1 - back]:
            back += 1
        room = min(n, limit - (p + m))
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


def test_silence():
    lk = SuffixLookup(m=4, n=3)
    assert lk.propose([1, 2, 3, 4, 5, 6, 7, 8]) == []          # the tail never occurred: nothing to copy
    lk = SuffixLookup(m=4, n=3)
    assert lk.propose([9, 1, 2]) == []                         # shorter than m + 1: no match is even possible
    lk = SuffixLookup(m=4, n=3)
    seq = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3]                   # the tail occurs only as a prefix of itself
    assert lk.propose(seq) == []


def test_copy_the_continuation():
    lk = SuffixLookup(m=4, n=3)
    seq = [1, 2, 3, 4, 5, 1, 2, 3, 4]
    assert lk.propose(seq) == [5, 1, 2]                        # last time 1,2,3,4 came 5; then a new block began
    lk = SuffixLookup(m=4, n=3)
    seq = [1, 2, 3, 4, 7, 8, 9, 1, 2, 3, 4, 7, 8]
    assert lk.propose(seq) == [9, 1, 2]                        # n=3 rows from the occurrence at 2
    seq.append(9)
    assert lk.propose(seq) == [1, 2, 3]                        # the longer occurrence [4,7,8,9] now wins


def test_caps():
    seq = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]
    assert SuffixLookup(m=4, n=2).propose(seq) == [5, 6]                       # capped by n
    assert SuffixLookup(m=4, n=7).propose(seq) == [5, 6, 7, 8, 1, 2, 3]        # n rows, when 7 follow the occurrence
    short = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
    assert SuffixLookup(m=4, n=7).propose(short) == [5, 6, 1, 2, 3, 4]         # only 6 follow: never invent a seventh


def test_longest_context_wins():
    # the tail 9,0,1,2 occurred twice: once cold, once after the same 6,6,6 that precede it now
    seq = [9, 0, 1, 2, 7, 7, 7, 6, 6, 6, 9, 0, 1, 2, 6, 6, 6, 9, 0, 1, 2]
    got = SuffixLookup(m=4, n=3).propose(seq)
    assert got == [6, 6, 6], got
    assert got == reference(seq, 4, 3)


def test_banned_truncates():
    seq = [1, 2, 3, 4, 7, 8, 9, 1, 2, 3, 4]
    assert SuffixLookup(m=4, n=3).propose(seq, banned={9}) == [7, 8]
    seq = [1, 2, 3, 4, 9, 8, 9, 1, 2, 3, 4]
    assert SuffixLookup(m=4, n=3).propose(seq, banned={9}) == []               # it would open with a refused token


def test_incremental_matches_brute_force():
    """The rolling hash, the per-key position window and the incremental extend are all off-by-one candidates: every
    prefix of a sequence full of planted repeats must agree with the slow scan."""
    m, n, rng, seq = 4, 3, 12345, []
    for _ in range(4000):
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        seq.append((rng >> 7) % 29)                        # a small alphabet: repeats happen constantly
    lk = SuffixLookup(m=m, n=n, keep=1000)                 # keep large: this tests the rule, not the eviction
    bad, checked = 0, 0
    for L in range(1, len(seq) + 1):
        pre = seq[:L]
        for extra in [None] + ([seq[L]] if L < len(seq) else []):    # the round verifies seq[L], or nothing does
            got, want = lk.propose(pre, extra=extra), reference(pre, m, n, extra=extra)
            checked += 1
            if got != want:
                bad += 1
                if bad <= 3:
                    print(f"  prefix {L} extra {extra}: got {got} want {want} "
                          f"tail {seq[max(0, L - m + (1 if extra is not None else 0)):L]} + {extra}")
    assert bad == 0, f"{bad} of {checked} prefixes disagree"


def test_truncated_sequence_is_dropped():
    lk = SuffixLookup(m=4, n=3)
    lk.propose([1, 2, 3, 4, 7, 8, 9, 1, 2, 3, 4])          # indexes this sequence
    assert lk.propose([5, 6, 7, 8, 1, 2]) == []            # shorter: the engine reset or restored, index must follow
    assert lk.propose([5, 6, 7, 8, 1, 2, 5, 6, 7, 8]) == [1, 2, 5]   # rebuilt over the new sequence only


def test_extra_is_the_token_under_verification():
    """The round verifies `extra` and the drafts are the positions AFTER it, so the m-gram being looked up ends on
    `extra`. Keying it on `seq` alone proposes one position early - the bug the first GPU run caught (a fired round
    worth 0.6 tokens instead of a match), and invisible to a test that never passes `extra`."""
    seq = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4]          # 1,2,3,4 occurred once, followed by 5,6,7
    lk = SuffixLookup(m=4, n=3)
    assert lk.propose(seq, extra=5) == [6, 7, 1]     # tail 2,3,4,5 matches the earlier 2,3,4,5 -> 6,7, then 1
    assert lk.propose(seq, extra=5) == reference(seq, 4, 3, extra=5)
    assert lk.propose(seq) == [5, 6, 7]              # without extra the tail is 1,2,3,4 -> its follower
    # the last occurrence is the tail itself: nothing may be proposed from it
    assert SuffixLookup(m=4, n=3).propose([9, 9, 1, 2, 3, 4], extra=5) == []
    # a proposal may not invent the token under verification: only decided tokens are followers
    assert SuffixLookup(m=4, n=3).propose([1, 2, 3, 4, 1, 2, 3], extra=4) == reference([1, 2, 3, 4, 1, 2, 3], 4, 3,
                                                                                       extra=4)


def main():
    for fn in (test_silence, test_copy_the_continuation, test_caps, test_longest_context_wins, test_banned_truncates,
               test_extra_is_the_token_under_verification,
               test_incremental_matches_brute_force, test_truncated_sequence_is_dropped):
        fn()
        print(f"  ok  {fn.__name__}")
    print("server/lookup.py: the suffix proposer proposes what it should, and nothing when it should not")


if __name__ == "__main__":
    main()
