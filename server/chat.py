"""Text side of serving: tokenizer, chat template, sampling, incremental detokenization.

The tokenizer is the model family's HF tokenizer.json (NFC-normalizing, as the model was trained; llama.cpp skips the
normalization). The chat template is the one stored in the GGUF, rendered by transformers exactly as HF would.
"""
import numpy as np


class Chat:
    def __init__(self, tokenizer_json: str, template: str):
        from transformers import PreTrainedTokenizerFast   # ~1.7 s (torch): Engine loads it beside the model load
        self.tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
        self.template = template

    def prompt(self, messages, tools=None, template_kwargs=None) -> str:
        return self.tok.apply_chat_template(messages, tools=tools, chat_template=self.template, tokenize=False,
                                            add_generation_prompt=True, **(template_kwargs or {}))

    def encode(self, text: str):
        return self.tok.encode(text, add_special_tokens=False)

    def token_id(self, text: str):
        i = self.tok.convert_tokens_to_ids(text)
        return i if isinstance(i, int) and i != self.tok.unk_token_id else None


class Detokenizer:
    """Turns generated ids into text deltas; holds back an incomplete UTF-8 character at the end."""

    def __init__(self, chat: Chat):
        self.chat, self.ids, self.sent = chat, [], 0

    def push(self, token: int) -> str:
        self.ids.append(token)
        text = self.chat.tok.decode(self.ids, skip_special_tokens=False)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta


def sample(logits: np.ndarray, temperature: float, top_p: float, top_k: int, rng: np.random.Generator) -> int:
    """temperature <= 0: greedy (the caller should use the device argmax instead)."""
    if temperature <= 0:
        return int(np.argmax(logits))
    x = logits.astype(np.float64) / temperature
    if top_k and top_k > 0:
        k = min(top_k, x.size)
        cut = np.partition(x, -k)[-k]
        x = np.where(x >= cut, x, -np.inf)
    x -= x.max()
    p = np.exp(x)
    p /= p.sum()
    if 0 < top_p < 1:
        order = np.argsort(-p)
        keep = np.cumsum(p[order]) - p[order] < top_p
        mask = np.zeros_like(p, dtype=bool)
        mask[order[keep]] = True
        p = np.where(mask, p, 0)
        p /= p.sum()
    return int(rng.choice(p.size, p=p))
