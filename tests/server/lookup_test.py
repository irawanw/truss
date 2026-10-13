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


def reference(seq, m, n, max_back=64, banned=()):
    """The same rule computed the slow obvious way: scan every start position of the prefix."""
    L = len(seq)
    if L < m + 1:
        return []
    tail = seq[L - m:]
    best_p, best_back, best_room = -1, 0, 0
    for p in range(0, L - m):
        if seq[p:p + m] != tail:
            continue
        back = 0
        while back < max_back and p - back - 1 >= 0 and seq[p - back - 1] == seq[L - m - back - 1]:
            back += 1
        room = min(n, L - (p + m))
        if (back, room) > (best_back, best_room):
            best_p, best_back, best_room = p, back, room
    if best_p < 0 or best_room <= 0:
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
    bad = 0
    for L in range(1, len(seq) + 1):
        got, want = lk.propose(seq[:L]), reference(seq[:L], m, n)
        if got != want:
            bad += 1
            if bad <= 3:
                print(f"  prefix {L}: got {got} want {want} tail {seq[max(0, L - m):L]}")
    assert bad == 0, f"{bad} of {len(seq)} prefixes disagree"


def test_truncated_sequence_is_dropped():
    lk = SuffixLookup(m=4, n=3)
    lk.propose([1, 2, 3, 4, 7, 8, 9, 1, 2, 3, 4])          # indexes this sequence
    assert lk.propose([5, 6, 7, 8, 1, 2]) == []            # shorter: the engine reset or restored, index must follow
    assert lk.propose([5, 6, 7, 8, 1, 2, 5, 6, 7, 8]) == [1, 2, 5]   # rebuilt over the new sequence only


def main():
    for fn in (test_silence, test_copy_the_continuation, test_caps, test_longest_context_wins, test_banned_truncates,
               test_incremental_matches_brute_force, test_truncated_sequence_is_dropped):
        fn()
        print(f"  ok  {fn.__name__}")
    print("server/lookup.py: the suffix proposer proposes what it should, and nothing when it should not")


if __name__ == "__main__":
    main()
