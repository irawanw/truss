"""Reasoning blocks: the end-of-turn ban inside an open <think> block, splitting a reply into reasoning and content,
and the streaming splitter.
"""
import os


# TRUSS_BAN_EOS_IN_THINK=1: no end of turn inside an open reasoning block (off: it looped a reply that never fed
# </think>, 10-13)
BAN_EOS_IN_THINK = os.environ.get("TRUSS_BAN_EOS_IN_THINK", "1") == "1"
BAN_MAX_TOKENS = 16384   # a reasoning block this long without </think> may end the turn after all


def split_reasoning(text: str, thinking: bool):
    """The template opens <think> in the prompt when thinking is on: text up to </think> is the reasoning."""
    if thinking and "</think>" in text:
        r, c = text.split("</think>", 1)
        return r.strip(), c.lstrip("\n")
    return (text.strip(), "") if thinking else ("", text)


class ThinkSplitter:
    """Streams text deltas as reasoning_content until </think>, then as content (leading newlines dropped). Holds
    back only a tail that could be the start of the tag."""

    TAG = "</think>"

    def __init__(self, thinking: bool):
        self.thinking, self.buf, self.started = thinking, "", False

    def _content(self, text):
        if not self.started:
            text = text.lstrip("\n")
            self.started = bool(text)
        return [("content", text)] if text else []

    def push(self, delta: str):
        if not self.thinking:
            return self._content(delta)
        self.buf += delta
        if self.TAG in self.buf:
            r, c = self.buf.split(self.TAG, 1)
            self.thinking, self.buf = False, ""
            return ([("reasoning_content", r)] if r else []) + self._content(c)
        keep = max((k for k in range(1, len(self.TAG)) if self.buf.endswith(self.TAG[:k])), default=0)
        out, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
        return [("reasoning_content", out)] if out else []

    def flush(self):
        out, self.buf = self.buf, ""
        if not out:
            return []
        return [("reasoning_content", out)] if self.thinking else self._content(out)
