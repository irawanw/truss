"""Word classes for the word-repeat penalty (Sampling.set_word_repeat, truss_set_word_classes in truss.h).

A quantized model's reasoning can collapse into one word said over and over ("Maybe maybe maybe.", the DS41 agent
10-22, 14,512 tokens of it). The penalty takes TRUSS_WORD_REPEAT logits off every token that spells the same word as
the token just before it, so a word is not sampled twice in a row. Tokens are grouped by their letters: the leading
space or word-start marker is dropped and the rest lowercased, so "Maybe", " maybe" and " MAYBE" are one class.
Only tokens that are a whole run of 2+ ASCII letters get a class (CJK doublings like 谢谢 are normal text); digits,
punctuation, code symbols and single letters (-1) are never penalized, so "x = x + 1" or "1 1" are untouched. Only a
token that starts a new word (leading space) is penalized: a piece glued to the one before ("bon" + "bon") is part of
one word. Such pieces still count as the previous word, stored as -(class + 2) (<= -2; truss.h). Rows after
</think> are not penalized (word_lift).
Tests: tests/server/word_repeat_test.py (host only).
"""
import os

WORD_REPEAT = float(os.environ.get("TRUSS_WORD_REPEAT", "8"))   # logits off a repeated word in the reasoning; 0: off
# Filler recency (10-22: "Hmm 51200*1000 = 51,200,000? Hmm ..." x 43 at 67K; Flash-Next does it too): a filler word
# loses WORD_FILLER logits per use of it in the last WORD_FILLER_WINDOW tokens of the reasoning (0: off). The first use
# is free, so a "Hmm" now and then stays; a run of them gets expensive fast.
WORD_FILLER = float(os.environ.get("TRUSS_WORD_FILLER", "6"))
WORD_FILLER_WINDOW = int(os.environ.get("TRUSS_WORD_FILLER_WINDOW", "64"))
FILLERS = ("hmm", "hmmm", "maybe", "wait", "actually", "perhaps", "ok", "okay", "oh", "hm", "alright")
_STRIP = " \tĠ▁"   # space, tab, GPT-2 byte-level space (Ġ), SentencePiece word start (▁)


def word_key(piece: str):
    """The class key of one vocabulary piece, or None when it is not a word."""
    w = piece.lstrip(_STRIP)
    if len(w) < 2 or len(piece) - len(w) > 1 or not (w.isascii() and w.isalpha()):
        return None
    return w.lower()


def filler_seqs(tok):
    """Every spelling of FILLERS as its token sequence (" Hmm" is " H" + "mm" in DeepSeek-V4.1's vocab, so a filler is
    a sequence, not a token): word, " word", Word, " Word", deduplicated, for truss_set_filler_seqs."""
    out = []
    for w in FILLERS:
        for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            ids = tok.encode(v, add_special_tokens=False)
            if 1 <= len(ids) <= 8 and ids not in out:
                out.append(ids)
    return out


def word_classes(tok, n_vocab: int):
    """int list [n_vocab]: every token's word class: c >= 0 for a word-start piece, -(c + 2) for a glued piece of
    word c, -1 for no word. tok is a transformers tokenizer."""
    classes, keys = [-1] * n_vocab, {}
    for piece, i in tok.get_vocab().items():
        if 0 <= i < n_vocab:
            k = word_key(piece)
            if k is not None:
                c = keys.setdefault(k, len(keys))
                classes[i] = c if piece[0] in _STRIP else -(c + 2)
    return classes
