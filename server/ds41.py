"""DeepSeek-V4.1 chat for the OpenAI server.

The prompt and the reply format are DeepSeek's own: server/encoding.py is the official DeepSeek-V4.1 encoding
(MIT, from huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/encoding, vendored verbatim; see
docs/reference/runtime-api-server.md). It renders the chat template (system/user/assistant turns, the thinking
markers, the numeric reasoning effort) and the DSML tool-call markup, so a served prompt is byte-identical to
DeepInfer's.

Streaming follows DeepInfer's parsing recipe (doc/inference_deployment_qa.md): `</think>` and the DSML token are
recognized by TOKEN ID, so the same text spelled with ordinary tokens stays reasoning/content; the DSML block is then
parsed incrementally, emitting the tool name as soon as the invoke tag closes and each call's arguments when its
parameters are complete. Non-streaming requests parse the finished text with the official parser, with a lenient
fallback for truncated (finish_reason "length") replies, which the official parser rejects by design.
"""
import re
import uuid

from server import encoding as enc


def token_id(tok, text: str):
    """The id of a special token, or None (PreTrainedTokenizerFast returns the unk id for unknown tokens)."""
    i = tok.convert_tokens_to_ids(text)
    return i if isinstance(i, int) and i >= 0 and i != tok.unk_token_id else None


class Ds41Chat:
    """The DS41 prompt/parse adapter over the server's tokenizer (one per model)."""

    def __init__(self, tok):
        self.tok = tok
        self.eos = token_id(tok, enc.eos_token)
        self.think_start = token_id(tok, enc.thinking_start_token)
        self.think_end = token_id(tok, enc.thinking_end_token)
        self.dsml = token_id(tok, enc.dsml_token)
        self.calls_open = enc.tool_calls_block_name.lstrip()      # "calls"
        self.invoke_tag = enc.tool_call_tag_name.lstrip()         # "invoke"
        self.param_tag = enc.tool_parameter_tag_name.lstrip()     # "parameter"

    # ------------------------------------------------------------------ prompt

    def prompt(self, messages, tools=None, template_kwargs=None):
        """(prompt, thinking): the OpenAI messages through the official encoder, tools on the system message."""
        kw = dict(template_kwargs or {})
        thinking = kw.get("enable_thinking", True) is not False
        msgs = [dict(m) for m in messages if isinstance(m, dict)]
        if tools:
            # the encoder renders the tool schemas from a message's "tools" field (a system message, as DeepInfer
            # passes them); a request without a system message gets an empty one with the tools on it
            for m in msgs:
                if m.get("role") == "system":
                    m["tools"] = tools
                    break
            else:
                msgs.insert(0, {"role": "system", "content": "", "tools": tools})
        prompt = enc.encode_messages(msgs, thinking_mode="thinking" if thinking else "chat",
                                     reasoning_effort=kw.get("reasoning_effort"))
        return prompt, thinking

    # ------------------------------------------------------------------ reply

    def stream(self, thinking: bool) -> "Ds41Stream":
        return Ds41Stream(self, thinking)

    def parse(self, text: str, thinking: bool, finish: str = "stop"):
        """The finished reply as an OpenAI message dict (non-streaming path)."""
        mode = "thinking" if thinking else "chat"
        if finish != "length":
            try:
                msg = enc.parse_message_from_completion_text(text + enc.eos_token, mode)
                return self._clean(msg)
            except (AssertionError, ValueError):
                pass
        return self._lenient(text, thinking)

    @staticmethod
    def _tool_text(text: str) -> bool:
        return ("<" + enc.dsml_token + enc.tool_calls_block_name) in text or \
               ("<" + enc.dsml_token + "tool_calls") in text

    @staticmethod
    def _clean(msg):
        msg = dict(msg)
        msg["content"] = (msg.get("content") or "").lstrip("\n")
        msg["reasoning_content"] = msg.get("reasoning_content") or ""
        msg["tool_calls"] = with_ids(msg.get("tool_calls") or [])
        return msg

    def _lenient(self, text: str, thinking: bool):
        """Best effort on output the official parser refuses (a length stop, a malformed block)."""
        reasoning, content = "", text
        if thinking and enc.thinking_end_token in text:
            reasoning, content = text.split(enc.thinking_end_token, 1)
        elif thinking:   # stopped (length) before </think>: all of it is still reasoning, as the stream reports it
            reasoning, content = text, ""
        content = content.lstrip("\n")
        calls = []
        for start in ("\n\n<" + enc.dsml_token + enc.tool_calls_block_name,
                      "\n\n<" + enc.dsml_token + "tool_calls"):
            i = content.find(start)
            if i >= 0:
                block, content = content[i:], content[:i].rstrip("\n")
                open_end = block.find(">") + 1
                try:
                    _, _, calls = enc.parse_tool_calls(open_end, block)
                except (AssertionError, ValueError):
                    calls = []
                break
        return self._clean({"role": "assistant", "content": content, "reasoning_content": reasoning,
                            "tool_calls": enc.tool_calls_to_openai_format(calls)})


def with_ids(calls):
    """OpenAI tool calls carry an id the tool result answers (the official encoder's converter leaves it out)."""
    for c in calls:
        c.setdefault("id", "call_" + uuid.uuid4().hex[:24])
    return calls


class Ds41Stream:
    """The streaming reply: ("reasoning_content"|"content", text) deltas and ("call", fragment) events.

    Phases: reasoning (thinking on) until the `</think>` TOKEN, content, then the DSML calls block once the DSML
    TOKEN id arrives. In the block, a call's name is emitted when its invoke tag closes, its arguments when its
    parameters are complete.
    """

    HOLD = 3   # content held back: the model writes "\n\n<" before a calls block

    def __init__(self, chat: Ds41Chat, thinking: bool):
        self.chat = chat
        self.thinking = thinking
        self.phase = "reasoning" if thinking else "content"
        self.tail = ""       # content not yet emitted
        self.markup = ""     # the DSML block text, from "\n\n<"
        self.scan = 0        # parse position inside markup
        self.in_calls = False
        self.cur = None      # the open call: {"name": ..., "args": [(key, is_str, value)]}
        self.index = -1
        self.calls = []      # finished calls (internal format: name/arguments)
        self.done = False
        ds = re.escape(enc.dsml_token)
        self.invoke_re = re.compile(r"<" + ds + r" invoke name=\"([^\"]*)\">\n")
        self.param_re = re.compile(r"<" + ds + r" parameter name=\"([^\"]*)\" string=\"(true|false)\">(.*?)</" +
                                   ds + r" parameter>", re.S)
        self.invoke_close = "</" + enc.dsml_token + enc.tool_call_tag_name + ">"
        self.calls_close = "</" + enc.dsml_token + enc.tool_calls_block_name + ">"

    def push(self, tok, delta: str):
        """One generated token: its id (special-marker detection) and its text delta."""
        out = []
        if self.phase == "reasoning":
            if tok is not None and tok == self.chat.think_end:
                self.phase = "content"   # the marker itself is not content
            elif delta:
                out.append(("reasoning_content", delta))
            return out
        if self.phase == "content":
            if tok is not None and tok == self.chat.dsml:
                t = self.tail[:-1] if self.tail.endswith("<") else self.tail
                t = t.rstrip("\n")
                if t:
                    out.append(("content", t))
                self.tail = ""
                self.markup = "\n\n<" + delta
                self.phase = "tools"
                return out
            if delta:
                self.tail += delta
                if len(self.tail) > self.HOLD:
                    n = len(self.tail) - self.HOLD
                    out.append(("content", self.tail[:n]))
                    self.tail = self.tail[n:]
            return out
        self.markup += delta
        return self._advance()

    def _advance(self):
        out = []
        while not self.done:
            if self.markup.startswith("\n", self.scan):
                self.scan += 1   # the newline that ends a tag line
                continue
            if self.cur is None:
                m = self.invoke_re.search(self.markup, self.scan)
                if m is None:
                    i = self.markup.find(self.calls_close, self.scan)
                    if i >= 0:
                        self.done = True
                        self.scan = i + len(self.calls_close)
                    break
                self.scan = m.end()
                self.index += 1
                self.cur = {"name": m.group(1), "args": []}
                out.append(("call", {"index": self.index, "id": f"call_{self.index}", "type": "function",
                                     "function": {"name": m.group(1), "arguments": ""}}))
                continue
            m = self.param_re.match(self.markup, self.scan)
            if m is not None:
                self.cur["args"].append((m.group(1), m.group(2), m.group(3)))
                self.scan = m.end()
                continue
            j = self.markup.find(self.invoke_close, self.scan)
            if j < 0:
                break
            k = self.markup.find("<" + enc.dsml_token + enc.tool_parameter_tag_name, self.scan)
            if 0 <= k < j:
                break   # a parameter has started but is not complete
            args = enc.decode_dsml_to_arguments(self.cur["name"],
                                                {a[0]: (a[2], a[1]) for a in self.cur["args"]})
            out.append(("call", {"index": self.index, "function": {"arguments": args["arguments"]}}))
            self.calls.append({"name": self.cur["name"], "arguments": args["arguments"]})
            self.cur = None
            self.scan = j + len(self.invoke_close)
        return out

    def flush(self):
        """The end of the reply: the held content tail (a call still open was truncated and is dropped)."""
        out = []
        if self.tail:
            out.append(("content", self.tail))
            self.tail = ""
        return out
