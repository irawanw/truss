"""Host-only test of the DS41 chat adapter (server/ds41.py): the official encoder drives the prompt, the DSML reply
parses and streams back to OpenAI calls. No GPU, no tokenizer file: the ids are the checkpoint's (128821 <think>,
128822 </think>, 128825 ｜DSML｜, 1 <｜end▁of▁sentence｜>) and a stub tokenizer maps them.

  python3 tests/server/ds41_encoding_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from server import encoding as enc  # noqa: E402
from server.ds41 import Ds41Chat  # noqa: E402

EOS, THINK_START, THINK_END, DSML = 1, 128821, 128822, 128825
DS = enc.dsml_token


class StubTok:
    unk_token_id = -1

    def convert_tokens_to_ids(self, text):
        return {enc.eos_token: EOS, enc.thinking_start_token: THINK_START,
                enc.thinking_end_token: THINK_END, enc.dsml_token: DSML}.get(text, self.unk_token_id)


TOOLS = [{"type": "function", "function": {
    "name": "lookup", "description": "Look up a value",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}}}]

REPLY = (" reason </think>summary\n\n<" + DS + " calls>\n<" + DS + " invoke name=\"lookup\">\n"
         "<" + DS + " parameter name=\"query\" string=\"true\">value</" + DS + " parameter>\n"
         "<" + DS + " parameter name=\"limit\" string=\"false\">2</" + DS + " parameter>\n"
         "</" + DS + " invoke>\n</" + DS + " calls>")


def test_prompt():
    chat = Ds41Chat(StubTok())
    p, thinking = chat.prompt([{"role": "system", "content": "S"}, {"role": "user", "content": "q"}], tools=TOOLS)
    assert thinking
    assert p.startswith(enc.bos_token + enc.SYSTEM_SP_TOKEN + "Reasoning Effort: 75 ")   # the default effort
    assert "## Tools" in p and "<" + DS + " calls>" in p and "<" + DS + " invoke name=" in p
    assert p.endswith(enc.USER_SP_TOKEN + "q" + enc.ASSISTANT_SP_TOKEN + enc.thinking_start_token)
    p, thinking = chat.prompt([{"role": "user", "content": "q"}], template_kwargs={"enable_thinking": False})
    assert not thinking and p == enc.bos_token + enc.USER_SP_TOKEN + "q" + enc.ASSISTANT_SP_TOKEN + enc.thinking_end_token
    p, _ = chat.prompt([{"role": "user", "content": "q"}], template_kwargs={"reasoning_effort": 42})
    assert "Reasoning Effort: 42 " in p
    p, _ = chat.prompt([{"role": "user", "content": "q"}], tools=TOOLS)   # tools ride on an inserted system message
    assert p.startswith(enc.bos_token + enc.SYSTEM_SP_TOKEN) and "## Tools" in p


def test_parse():
    chat = Ds41Chat(StubTok())
    msg = chat.parse(REPLY, True)
    assert msg["reasoning_content"] == " reason "
    assert msg["content"] == "summary"
    assert len(msg["tool_calls"]) == 1
    fn = msg["tool_calls"][0]["function"]
    assert fn["name"] == "lookup" and fn["arguments"] == '{"query": "value", "limit": 2}'
    msg = chat.parse("plain</think>text", True)   # truncated: the official parser refuses it, the lenient one splits
    assert msg["reasoning_content"] == "plain" and msg["content"] == "text" and not msg["tool_calls"]
    assert chat.parse("answer", False)["content"] == "answer"


def test_stream():
    chat = Ds41Chat(StubTok())
    st = chat.stream(True)
    events = []
    for tok, text in [(3986, " reason"), (223, " "), (THINK_END, "</think>"), (11235, "summary"),
                      (271, "\n\n"), (30, "<"), (DSML, DS), (10699, " calls"), (1018, ">\n"), (30, "<"),
                      (DSML, DS), (34756, " invoke"), (2329, " name"), (1281, "=\""), (12533, "look"),
                      (1425, "up"), (3816, "\">\n"), (30, "<"), (DSML, DS), (10767, " parameter"), (2329, " name"),
                      (1281, "=\""), (12066, "query"), (4, "\""), (3418, " string"), (1281, "=\""), (11476, "true"),
                      (3320, "\">"), (5197, "value"), (1718, "</"), (DSML, DS), (10767, " parameter"), (1018, ">\n"),
                      (1718, "</"), (DSML, DS), (34756, " invoke"), (1018, ">\n"), (1718, "</"), (DSML, DS),
                      (10699, " calls"), (32, ">"), (EOS, enc.eos_token)]:
        events += st.push(tok, text)
    events += st.flush()
    reasoning = "".join(x for k, x in events if k == "reasoning_content")
    content = "".join(x for k, x in events if k == "content")
    calls = [x for k, x in events if k == "call"]
    assert reasoning == " reason ", reasoning
    assert content == "summary", content
    assert calls[0]["function"] == {"name": "lookup", "arguments": ""}   # the name as soon as the tag closes
    assert calls[1]["function"] == {"arguments": '{"query": "value"}'}
    assert st.calls == [{"name": "lookup", "arguments": '{"query": "value"}'}]
    # a literal </think> spelled with ordinary tokens stays content: only the token id ends the reasoning
    st = chat.stream(True)
    out = st.push(1234, "</think>") + st.push(THINK_END, "</think>")
    assert out[0] == ("reasoning_content", "</think>") and len(out) == 1
    # a "普通" text reply in chat mode passes through as content
    st = chat.stream(False)
    ev = st.push(5, "hello") + st.flush()
    assert "".join(x for _, x in ev) == "hello" and all(k == "content" for k, _ in ev)


if __name__ == "__main__":
    test_prompt(), test_parse(), test_stream()
    print("ds41 encoding: OK")
