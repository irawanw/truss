"""OpenAI tool calling for the models' call formats.

Two formats are parsed, both rendered back through the GGUF template's tool-call macro:

1. The XML-ish format some templates use:

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

2. LFM2.5's native Pythonic format (the template emits exactly this):

    <|tool_call_start|>[NAME(KEY='value', N=3), NAME2(ARGS_JSON)]<|tool_call_end|>

OpenAI clients send arguments as a JSON string and expect calls back in `tool_calls`, not in `content`. This module does both
conversions (as llama-server's Qwen3-Coder parser does):

    messages = normalize_messages(body["messages"])        # string arguments -> dict, before templating
    content, calls = parse_calls(text, tools)               # after generation: calls out of the content
    ts = ToolStream(tools); ts.push(delta) ... ts.finish()       # the streaming form, incremental

A parameter value is kept as text when the tool's schema types it as a string (or the tool is not in the request);
otherwise it is parsed as JSON (numbers, booleans, objects, arrays), falling back to text.
"""
import ast
import json
import re
import uuid

OPEN = "<tool_call>"
_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)(?:</function>\s*)?(?:</tool_call>|$)", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
LFM_OPEN = "<|tool_call_start|>"
LFM_CLOSE = "<|tool_call_end|>"


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


def _split_top(s):
    """Split on top-level commas (quotes, escapes and nested brackets respected)."""
    parts, depth, cur, q = [], 0, [], None
    i = 0
    while i < len(s):
        ch = s[i]
        if q:
            if ch == "\\" and i + 1 < len(s):
                cur.append(s[i:i + 2])
                i += 2
                continue
            cur.append(ch)
            if ch == q:
                q = None
        elif ch in ("'", '"'):
            q, cur = ch, cur + [ch]
        elif ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return parts


def _py_value(raw):
    """A template-emitted argument value: 'single-quoted' Python, or JSON ("...", 3, true, [...])."""
    t = raw.strip()
    if len(t) >= 2 and t[0] == "'" and t[-1] == "'":
        try:
            return ast.literal_eval(t)
        except (SyntaxError, ValueError):
            return t[1:-1]
    try:
        return json.loads(t)
    except ValueError:
        return t


def _lfm_list_body(s):
    """The '[...]' list after LFM_OPEN, quote-aware; (body, chars consumed) or (None, 0) if unterminated."""
    i = s.find("[")
    if i < 0:
        return None, 0
    depth, q, j = 0, None, i
    while j < len(s):
        ch = s[j]
        if q:
            if ch == "\\":
                j += 2
                continue
            if ch == q:
                q = None
        elif ch in ("'", '"'):
            q = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return None, 0


def _parse_lfm_calls(body, types):
    """[OpenAI tool_call] for an LFM '[NAME(k=v, ...), ...]' body."""
    calls = []
    for part in _split_top(body):
        m = re.match(r"\s*([A-Za-z_][\w.]*)\s*\((.*)\)\s*$", part, re.S)
        if not m:
            continue
        name, argstr = m.group(1), m.group(2)
        known = types.get(name)
        args = {}
        for kv in _split_top(argstr):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k = k.strip()
            v = _py_value(v)
            typ = known.get(k) if known is not None else None
            if typ == "string" and not isinstance(v, str):
                v = str(v)
            elif typ in ("number", "integer", "boolean", "object", "array") and isinstance(v, str):
                try:
                    v = json.loads(v)
                except ValueError:
                    pass
            args[k] = v
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


def parse_calls(text: str, tools=None):
    """(content before the first call or None, [OpenAI tool_call]) for a finished message's content."""
    li = text.find(LFM_OPEN)
    if li >= 0:
        types = _schema_types(tools)
        body = text[li + len(LFM_OPEN):]
        ci = body.find(LFM_CLOSE)
        lst, _ = _lfm_list_body(body[:ci] if ci >= 0 else body)
        calls = _parse_lfm_calls(lst, types) if lst is not None else []
        if calls:
            content = text[:li].rstrip()
            return (content or None), calls
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
    """Streaming form, incremental like llama-server: content until a call tag; then, per call, a first delta with
    the id and name as soon as `<function=NAME>` is complete (XML format), and the arguments JSON in fragments as the
    model writes them (a string value is streamed escaped as it arrives; other types are emitted at their
    </parameter>). LFM calls (`<|tool_call_start|>[...]`) are buffered to the closing `]` and emitted whole (one delta
    with id + name, one with the arguments JSON). A call that writes a whole file streams for minutes in the XML form;
    holding it until the end (a229de0) stalled clients ("stream stalled while waiting for the next event"). Text after
    the first call is dropped (the template asks for no suffix).

        ts = ToolStream(tools)
        for kind, x in ts.push(delta): ...   # ("content", text) or ("call", OpenAI tool_calls delta with index)
        for kind, x in ts.finish(): ...      # closes an unterminated call
        ts.calls                             # number of calls emitted
    """

    def __init__(self, tools=None):
        self.types = _schema_types(tools)
        self.buf, self.state, self.calls = "", "text", 0
        self.first, self.strip_nl, self.typ = True, False, None

    def _emit_lfm(self, body):
        """("call", ...) deltas for a complete LFM '[...]' body."""
        out = []
        for c in _parse_lfm_calls(body, self.types):
            self.calls += 1
            fn = c["function"]
            out.append(("call", {"index": self.calls - 1, "id": c["id"], "type": "function",
                                 "function": {"name": fn["name"], "arguments": ""}}))
            out.append(("call", {"index": self.calls - 1, "function": {"arguments": fn["arguments"]}}))
        return out

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
                i, j = b.find(OPEN), b.find(LFM_OPEN)
                if i >= 0 and (j < 0 or i < j):
                    if self.calls == 0 and b[:i]:
                        out.append(("content", b[:i]))
                    self.buf, self.state = b[i + len(OPEN):], "name"
                    continue
                if j >= 0:
                    if self.calls == 0 and b[:j]:
                        out.append(("content", b[:j]))
                    self.buf, self.state = b[j + len(LFM_OPEN):], "lfm"
                    continue
                tags = (OPEN, LFM_OPEN)
                keep = max((k for t in tags for k in range(1, len(t)) if b.endswith(t[:k])), default=0)
                if self.calls == 0 and b[:len(b) - keep]:
                    out.append(("content", b[:len(b) - keep]))
                self.buf = b[len(b) - keep:]
                return out
            if self.state == "lfm":
                body, used = _lfm_list_body(b)
                if body is None:
                    return out   # wait for the closing `]` (or finish())
                out.extend(self._emit_lfm(body))
                self.buf, self.state = b[used:], "lfm-end"
                continue
            if self.state == "lfm-end":
                s = b.lstrip()
                if s.startswith(LFM_CLOSE):
                    self.buf, self.state = s[len(LFM_CLOSE):], "text"
                    continue
                if s and not self._partial(s, (LFM_CLOSE,)):
                    self.state = "text"   # no closing tag: treat what follows as text
                    continue
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
        elif self.state == "lfm":
            body, _ = _lfm_list_body(b)
            if body is None and b.lstrip().startswith("["):
                body = b.lstrip()[1:]
            if body is not None:
                out.extend(self._emit_lfm(body))
        self.state = "text"
        return out
