"""OpenAI tool calling for the model's XML call format.

The GGUF chat template tells the model to call a function as

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

and renders past calls from `message.tool_calls[*].function.arguments|items`, i.e. arguments as a mapping. OpenAI
clients send arguments as a JSON string and expect calls back in `tool_calls`, not in `content`. This module does both
conversions (as llama-server's Qwen3-Coder parser does):

    messages = normalize_messages(body["messages"])        # string arguments -> dict, before templating
    content, calls = parse_calls(text, tools)               # after generation: calls out of the content
    ts = ToolStream(tools); ts.push(delta) ... ts.finish()       # the streaming form, incremental

A parameter value is kept as text when the tool's schema types it as a string (or the tool is not in the request);
otherwise it is parsed as JSON (numbers, booleans, objects, arrays), falling back to text.
"""
import json
import re
import uuid

OPEN = "<tool_call>"
_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)(?:</function>\s*)?(?:</tool_call>|$)", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def normalize_messages(messages):
    """Copies of the messages with each assistant tool call's JSON-string arguments parsed into a dict."""
    out = []
    for m in messages:
        calls = m.get("tool_calls") if isinstance(m, dict) else None
        if calls:
            m = dict(m)
            fixed = []
            for c in calls:
                c = dict(c)
                fn = dict(c.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:
                        args = {"arguments": args}
                    fn["arguments"] = args if isinstance(args, dict) else {"arguments": args}
                c["function"] = fn
                fixed.append(c)
            m["tool_calls"] = fixed
        out.append(m)
    return out


def strip_reasoning(messages, keep):
    """Copies of the messages with the reasoning of all but the last `keep` assistant turns dropped
    (`reasoning_content` / `reasoning`, and a `<think>...</think>` prefix in the content, which the template would
    otherwise turn back into reasoning). The chat template keeps every assistant turn's thinking since the last user
    message; in a long agent loop that is ~90 thinking blocks, and a degenerate habit in them ("✓ ✓ ✓", "hmm hmm")
    is copied and amplified by the next turn (TRACKER #105: loop sites 7/18 -> 0/18 with keep 0)."""
    idx = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"]
    drop = set(idx[:max(0, len(idx) - keep)])
    out = []
    for i, m in enumerate(messages):
        if i in drop:
            m = {k: v for k, v in m.items() if k not in ("reasoning_content", "reasoning")}
            c = m.get("content")
            if isinstance(c, str) and "</think>" in c:
                m["content"] = c.split("</think>", 1)[1].lstrip("\n")
        out.append(m)
    return out


def _schema_types(tools):
    """{function name: {parameter: JSON-schema type}} from the request's tools."""
    types = {}
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        props = ((fn.get("parameters") or {}).get("properties") or {}) if isinstance(fn, dict) else {}
        types[fn.get("name")] = {k: (v or {}).get("type") for k, v in props.items() if isinstance(v, dict)}
    return types


def _value(raw: str, typ):
    if typ == "string" or typ is None:
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def parse_calls(text: str, tools=None):
    """(content before the first call or None, [OpenAI tool_call]) for a finished message's content."""
    i = text.find(OPEN)
    if i < 0:
        return text, []
    types = _schema_types(tools)
    calls = []
    for m in _CALL.finditer(text, i):
        name = m.group(1).strip()
        known = types.get(name)
        args = {}
        for p in _PARAM.finditer(m.group(2)):
            key = p.group(1).strip()
            args[key] = _value(p.group(2), known.get(key) if known is not None else "string")
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    if not calls:
        return text, []
    content = text[:i].rstrip()
    return (content or None), calls


class ToolStream:
    """Streaming form, incremental like llama-server: content until a <tool_call>; then, per call, a first delta with
    the id and name as soon as `<function=NAME>` is complete, and the arguments JSON in fragments as the model writes
    them (a string value is streamed escaped as it arrives; other types are emitted at their </parameter>). A call that
    writes a whole file streams for minutes; holding it until the end (a229de0) stalled clients ("stream stalled while
    waiting for the next event"). Text after the first call is dropped (the template asks for no suffix).

        ts = ToolStream(tools)
        for kind, x in ts.push(delta): ...   # ("content", text) or ("call", OpenAI tool_calls delta with index)
        for kind, x in ts.finish(): ...      # closes an unterminated call
        ts.calls                             # number of calls emitted
    """

    def __init__(self, tools=None):
        self.types = _schema_types(tools)
        self.buf, self.state, self.calls = "", "text", 0
        self.first, self.strip_nl, self.typ = True, False, None

    def _arg(self, s):
        return ("call", {"index": self.calls - 1, "function": {"arguments": s}})

    @staticmethod
    def _partial(buf, tags):
        """buf could still become one of tags (a proper prefix)."""
        return any(t.startswith(buf) and t != buf for t in tags)

    def push(self, delta: str):
        self.buf += delta
        out = []
        while True:
            b = self.buf
            if self.state == "text":
                i = b.find(OPEN)
                if i >= 0:
                    if self.calls == 0 and b[:i]:
                        out.append(("content", b[:i]))
                    self.buf, self.state = b[i + len(OPEN):], "name"
                    continue
                keep = max((k for k in range(1, len(OPEN)) if b.endswith(OPEN[:k])), default=0)
                if self.calls == 0 and b[:len(b) - keep]:
                    out.append(("content", b[:len(b) - keep]))
                self.buf = b[len(b) - keep:]
                return out
            if self.state == "name":
                m = re.match(r"\s*<function=([^>\n]+)>", b)
                if not m:
                    return out
                name = m.group(1).strip()
                self.calls += 1
                self.known = self.types.get(name)
                out.append(("call", {"index": self.calls - 1, "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                                     "function": {"name": name, "arguments": ""}}))
                self.buf, self.state, self.first = b[m.end():], "params", True
                continue
            if self.state == "params":
                s = b.lstrip()
                m = re.match(r"<parameter=([^>\n]+)>", s)
                if m:
                    key = m.group(1).strip()
                    self.typ = self.known.get(key) if self.known is not None else "string"
                    lead = "{" if self.first else ", "
                    self.first = False
                    if self.typ in ("string", None):
                        out.append(self._arg(lead + json.dumps(key) + ": \""))
                        self.state = "str"
                    else:
                        out.append(self._arg(lead + json.dumps(key) + ": "))
                        self.state = "raw"
                    self.buf, self.strip_nl = s[m.end():], True
                    continue
                if s.startswith("</function>"):
                    out.append(self._arg("{}" if self.first else "}"))
                    self.buf, self.state = s[len("</function>"):], "end"
                    continue
                if not s or self._partial(s, ("<parameter=", "</function>")) or s.startswith("<parameter="):
                    return out
                self.buf = s[1:]   # stray text between parameters
                continue
            if self.state in ("str", "raw"):
                if self.strip_nl:
                    if not b:
                        return out
                    if b[0] == "\n":
                        b = b[1:]
                    self.buf, self.strip_nl = b, False
                i = b.find("</parameter>")
                if i < 0:
                    if self.state == "raw":
                        return out
                    # hold back a possible partial closing tag and one newline that may precede it
                    keep = max((k for k in range(1, len("\n</parameter>")) if b.endswith("\n</parameter>"[:k])), default=0)
                    keep = max(keep, max((k for k in range(1, len("</parameter>")) if b.endswith("</parameter>"[:k])), default=0))
                    if b[:len(b) - keep]:
                        out.append(self._arg(json.dumps(b[:len(b) - keep])[1:-1]))
                    self.buf = b[len(b) - keep:]
                    return out
                v = b[:i]
                if v.endswith("\n"):
                    v = v[:-1]
                if self.state == "str":
                    out.append(self._arg(json.dumps(v)[1:-1] + "\""))
                else:
                    out.append(self._arg(json.dumps(_value(v, self.typ), ensure_ascii=False)))
                self.buf, self.state = b[i + len("</parameter>"):], "params"
                continue
            if self.state == "end":
                i = b.find("</tool_call>")
                if i < 0:
                    if not self._partial(b.lstrip(), ("</tool_call>",)) and b.strip():
                        self.state = "text"   # no closing tag: treat what follows as text
                        continue
                    return out
                self.buf, self.state = b[i + len("</tool_call>"):], "text"
                continue

    def finish(self):
        """Closes whatever is open at the end of the reply."""
        out, b = [], self.buf
        self.buf = ""
        if self.state == "text":
            if self.calls == 0 and b:
                out.append(("content", b))
        elif self.state in ("str", "raw"):
            v = b[:-1] if b.endswith("\n") else b
            if self.state == "str":
                out.append(self._arg(json.dumps(v)[1:-1] + "\"}"))
            else:
                out.append(self._arg(json.dumps(_value(v, self.typ), ensure_ascii=False) + "}"))
        elif self.state == "params":
            out.append(self._arg("{}" if self.first else "}"))
        self.state = "text"
        return out
