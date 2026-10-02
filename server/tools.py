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
    split = ToolStream(); split.push(delta) ... split.finish(tools)   # the streaming form

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
    """Streams content deltas until a <tool_call> starts, then holds the rest for parse_calls at the end. Holds back
    only a tail that could be the start of the tag."""

    def __init__(self):
        self.held, self.calling = "", False

    def push(self, delta: str) -> str:
        if self.calling:
            self.held += delta
            return ""
        buf = self.held + delta
        i = buf.find(OPEN)
        if i >= 0:
            self.calling, self.held = True, buf[i:]
            return buf[:i]
        keep = max((k for k in range(1, len(OPEN)) if buf.endswith(OPEN[:k])), default=0)
        self.held = buf[len(buf) - keep:]
        return buf[:len(buf) - keep]

    def finish(self, tools=None):
        """(text still to send as content, [tool_call])."""
        held, self.held = self.held, ""
        if not self.calling:
            return held, []
        content, calls = parse_calls(held, tools)
        return ("" if calls else held), calls
