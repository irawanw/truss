"""Host-only test of server/words.py (no GPU): the DS41 tokenizer's word classes for the word-repeat penalty.
"Maybe" / " maybe" share a class and are penalized after each other; code symbols, digits, single letters and CJK
pieces have none; a glued piece ("els" in " vox"+"els") counts as its word but is never penalized itself.

  python3 tests/server/word_repeat_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from server.words import word_classes  # noqa: E402

TOK = os.environ.get("DS41_TOKENIZER",
                     os.path.expanduser("~/ML_projects/dsv41flash/20261019_truss_port/data/tokenizer/tokenizer.json"))


def main():
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=TOK)
    cls = word_classes(tok, len(tok))
    enc = lambda s: tok.encode(s, add_special_tokens=False)   # noqa: E731
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{name}: {detail}  {'PASS' if cond else 'FAIL'}")

    a, b = enc("Maybe maybe")
    word = lambda c: -c - 2 if c <= -2 else c   # noqa: E731  (the engine's reading of the previous token)
    check("Maybe then maybe: penalized", cls[b] >= 0 and word(cls[a]) == cls[b], f"{cls[a]} {cls[b]}")
    check("x = x + 1 1 unclassed", all(cls[i] == -1 for i in enc(" x = x + 1 1")))
    check("CJK unclassed", all(cls[i] == -1 for i in enc("谢谢 看看")))
    glued = [i for i in enc(" voxels") if tok.decode([i]) == "els"]
    check("glued piece counts but is not penalized", glued and cls[glued[0]] <= -2, f"{[cls[i] for i in glued]}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
