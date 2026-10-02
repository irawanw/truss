"""TRUSS OpenAI-compatible server (chat and text completions, streaming or not) for one model on one GPU.

One sequence at a time. A new prompt that extends the sequence already in the engine (same tokens, then more) only
evaluates the new tokens; anything else starts a new sequence (the GDN state cannot be rewound). Greedy decoding
takes the argmax on the device; sampling copies the logits.

Tool calls: the model's <tool_call> XML is returned as OpenAI `tool_calls` (finish_reason "tool_calls"), and
assistant tool calls in the history have their JSON-string arguments parsed before templating (server/tools.py).

Sampling defaults are llama-server's (temperature 0.8, top_k 40, top_p 0.95, min_p 0.05), so a client that sends no
sampling parameters gets what it got from llama-paw on this port; temperature 0 is greedy.

Each request logs one line (to stdout and --log) in the format flashnext_strata_bench.py parses, so that bench runs
unchanged against TRUSS; --log-tag sets its prefix (the bench counts "strata serve: prompt").

usage: CUDA_VISIBLE_DEVICES=2 python3 -m server.app --model <gguf> --tokenizer <tokenizer.json> [--port 8080]
"""
import argparse
import json
import os
import threading
import time
import uuid

import anyio
import asyncio
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from server.chat import Chat, Detokenizer
from server.tools import ToolStream, normalize_messages, parse_calls
from server.truss_ctypes import Model, Sampling, TrussError


class Engine:
    """The model plus the sequence in it; generate() is the only way in and holds the lock."""

    def __init__(self, a):
        self.model = Model(a.model, a.n_ctx, a.chunk, a.expert_usage, mtp=a.mtp, drafts=a.drafts, cpu_dir=a.cpu_dir,
                           cpu_share=a.cpu_share, cpu_threads=a.cpu_threads, draft_vocab=a.draft_vocab,
                           draft_min_p=a.draft_min_p)
        self.chat = Chat(a.tokenizer, self.model.meta_string("tokenizer.chat_template"))
        self.stop_ids = {i for i in (self.model.meta_int("tokenizer.ggml.eos_token_id"),
                                     self.chat.token_id("<|im_end|>"), self.chat.token_id("<|endoftext|>"))
                         if i is not None and i >= 0}
        self.name, self.log_path, self.log_tag = a.name, a.log, a.log_tag
        self.dump_dir = a.dump_dir
        self.cached = []               # tokens in the engine's sequence
        self.lock = threading.Lock()

    def log(self, line: str):
        print(line, flush=True)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def dump(self, prompt_ids, text, finish, sampling, keep=20):
        """--dump-dir: one JSON per request (prompt tail, generated text, finish, sampling), the newest `keep` kept;
        for diagnosing replies after the fact (e.g. a 32K-token reply that hit the client's cap)."""
        if not self.dump_dir:
            return
        try:
            os.makedirs(self.dump_dir, exist_ok=True)
            path = os.path.join(self.dump_dir, time.strftime("%Y%m%d-%H%M%S") + f"-{len(prompt_ids)}.json")
            with open(path, "w") as f:
                json.dump({"n_prompt": len(prompt_ids), "prompt_tail": self.chat.tok.decode(prompt_ids[-4000:]),
                           "finish": finish, "sampling": sampling, "text": text}, f, ensure_ascii=False)
            for old in sorted(os.listdir(self.dump_dir))[:-keep]:
                os.remove(os.path.join(self.dump_dir, old))
        except OSError as e:
            print(f"dump failed: {e}", flush=True)

    def generate(self, prompt_ids, max_tokens, temperature, top_p, top_k, seed, stop_strings, min_p=0.0):
        """Yields (token id, text delta, finish reason or None); the last item has the finish reason."""
        with self.lock:
            n_prompt = len(prompt_ids)
            if n_prompt == 0:
                raise ValueError("empty prompt")
            max_tokens = max(0, min(max_tokens, self.model.n_ctx - n_prompt))
            reuse = len(self.cached) if 0 < len(self.cached) < n_prompt and prompt_ids[:len(self.cached)] == self.cached else 0
            if not reuse:
                self.model.reset()
                self.cached = []
            greedy = temperature <= 0
            # sampled requests sample on the device and speculate too (Strata's sampled verify)
            sp = None if greedy else Sampling(temperature, top_p if top_p and top_p > 0 else 1.0, top_k or 0, min_p or 0.0,
                                              (seed if seed is not None else np.random.SeedSequence().entropy) % (1 << 64))

            def step(tokens):
                self.cached += list(tokens)
                if greedy:
                    return self.model.eval_argmax(tokens)
                return self.model.eval_sample(tokens, sp)

            t0 = time.time()
            try:
                nxt = step(prompt_ids[reuse:])
            except TrussError:
                self.cached = []   # the engine's sequence is unknown now: the next request starts over
                raise
            t1 = time.time()
            detok, text, n_gen, finish = Detokenizer(self.chat), "", 0, "length"
            spec = self.model.drafts > 0
            drafted = accepted = 0
            pending = []   # tokens the engine already holds that are still to be emitted (speculative rounds)
            try:
                while n_gen < max_tokens:
                    if spec and not pending:   # one round: nxt plus the accepted drafts enter the sequence
                        if self.model.position + 1 >= self.model.n_ctx:
                            break
                        emitted, after = self.model.spec_step(nxt) if greedy else self.model.spec_step_sampled(nxt, sp)
                        self.cached += emitted
                        drafted += self.model.drafts
                        accepted += len(emitted) - 1
                        pending, nxt = emitted, after
                    tok = pending.pop(0) if spec else nxt
                    n_gen += 1
                    if tok in self.stop_ids:
                        finish = "stop"
                        break
                    delta = detok.push(tok)
                    text += delta
                    hit = next((s for s in stop_strings if s and s in text), None)
                    if hit:
                        cut = text.index(hit)
                        yield tok, delta[:max(0, len(delta) - (len(text) - cut))], None
                        finish = "stop"
                        break
                    yield tok, delta, None
                    if not spec and n_gen < max_tokens and self.model.position < self.model.n_ctx:
                        nxt = step([nxt])
            except TrussError:
                self.cached = []
                raise
            finally:
                t2 = time.time()
                read, gen_s = t1 - t0, t2 - t1
                self.log(f"{self.log_tag}: prompt {n_prompt} tokens = {reuse} reused + {n_prompt - reuse} read in "
                         f"{read * 1e3:.0f} ms ({(n_prompt - reuse) / max(read, 1e-9):.1f} tok/s), {n_gen} generated "
                         f"in {gen_s * 1e3:.0f} ms ({n_gen / max(gen_s, 1e-9):.1f} tok/s), drafts accepted {accepted} of {drafted}")
                self.dump(prompt_ids, text, finish, dict(temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p))
            yield None, "", finish


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


def make_app(engine: Engine) -> FastAPI:
    app = FastAPI()

    def params(body):
        stop = body.get("stop") or []
        def get(key, default, cast):
            v = body.get(key)
            return cast(default if v is None else v)
        return dict(max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or engine.model.n_ctx),
                    temperature=get("temperature", 0.8, float), top_p=get("top_p", 0.95, float),
                    top_k=get("top_k", 40, int), min_p=get("min_p", 0.05, float), seed=body.get("seed"),
                    stop_strings=[stop] if isinstance(stop, str) else list(stop))

    _END, KEEPALIVE = object(), object()

    async def tokens(req: Request, ids, p, keepalive=None):
        """engine.generate driven from a worker thread, so the event loop never blocks on the engine (or its lock).
        On a client disconnect, or when the caller stops early, the generator is closed: it logs its line and releases
        the engine lock (a suspended, never-closed generator held the lock and deadlocked the next request).
        With `keepalive` (seconds), KEEPALIVE is yielded whenever the engine has produced nothing for that long (a
        long prompt read, a wait for the lock), so a streaming client's stall timer sees traffic."""
        gen = engine.generate(ids, **p)
        task = None
        try:
            while True:
                task = asyncio.ensure_future(anyio.to_thread.run_sync(next, gen, _END))
                while not (await asyncio.wait({task}, timeout=keepalive))[0]:
                    yield KEEPALIVE
                    if await req.is_disconnected():
                        return
                item, task = task.result(), None
                if item is _END:
                    return
                yield item
                if await req.is_disconnected():
                    return
        finally:
            if task is not None:   # the worker thread is inside next(gen): let it return before closing gen
                await asyncio.shield(task)
            gen.close()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": engine.name, "object": "model", "owned_by": "truss"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        body = await req.json()
        kw = dict(body.get("chat_template_kwargs") or {})
        thinking = kw.get("enable_thinking", True) is not False
        tools = body.get("tools") or None
        if body.get("tool_choice") == "none":
            tools = None
        prompt = await anyio.to_thread.run_sync(
            lambda: engine.chat.prompt(normalize_messages(body["messages"]), tools=tools, template_kwargs=kw))
        ids = await anyio.to_thread.run_sync(engine.chat.encode, prompt)
        p = params(body)
        engine.log(f"request: temperature {p['temperature']} top_p {p['top_p']} top_k {p['top_k']} min_p {p['min_p']}, "
                   f"{len(tools or [])} tools, thinking {'on' if thinking else 'off'}")
        rid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

        if body.get("stream"):
            async def events():
                def chunk(delta, finish=None):
                    return "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                                                  "model": engine.name,
                                                  "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"
                split, calls = ThinkSplitter(thinking), ToolStream(tools) if tools else None

                def emit(pairs):
                    for kind, x in pairs:
                        yield chunk({"tool_calls": [x]} if kind == "call" else {"content": x})

                def out(pairs):
                    for field, d in pairs:
                        if field == "content" and calls:
                            yield from emit(calls.push(d))
                        elif d:
                            yield chunk({field: d})
                yield chunk({"role": "assistant", "content": ""})
                async for item in tokens(req, ids, p, keepalive=5.0):
                    if item is KEEPALIVE:
                        yield chunk({"content": ""})
                        continue
                    tok, delta, finish = item
                    for c in out(split.push(delta)):
                        yield c
                    if finish:
                        for c in out(split.flush()):
                            yield c
                        if calls:
                            for c in emit(calls.finish()):
                                yield c
                            if calls.calls and finish == "stop":
                                finish = "tool_calls"
                        yield chunk({}, finish)
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")

        text, n, finish = "", 0, "length"
        async for tok, delta, fin in tokens(req, ids, p):
            if tok is not None:
                text += delta
                n += 1
            if fin:
                finish = fin
        reasoning, content = split_reasoning(text, thinking)
        found = []
        if tools:
            content, found = parse_calls(content, tools)
        msg = {"role": "assistant", "content": content}
        if found:
            msg["tool_calls"] = found
            if finish == "stop":
                finish = "tool_calls"
        if reasoning:
            msg["reasoning_content"] = reasoning
        return JSONResponse({"id": rid, "object": "chat.completion", "created": created, "model": engine.name,
                             "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                             "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}})

    @app.post("/v1/completions")
    async def completions(req: Request):
        body = await req.json()
        ids = await anyio.to_thread.run_sync(engine.chat.encode, body["prompt"])
        p = params(body)
        text, n, finish = "", 0, "length"
        async for tok, delta, fin in tokens(req, ids, p):
            if tok is not None:
                text += delta
                n += 1
            if fin:
                finish = fin
        return JSONResponse({"id": "cmpl-" + uuid.uuid4().hex[:24], "object": "text_completion", "created": int(time.time()),
                             "model": engine.name, "choices": [{"index": 0, "text": text, "finish_reason": finish}],
                             "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}})

    @app.exception_handler(TrussError)
    async def truss_error(_, e):
        return JSONResponse({"error": {"message": str(e), "type": "engine_error"}}, status_code=500)

    @app.exception_handler(ValueError)
    async def value_error(_, e):
        return JSONResponse({"error": {"message": str(e), "type": "invalid_request_error"}}, status_code=400)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="GGUF (first shard)")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer.json of the model family")
    ap.add_argument("--n-ctx", type=int, default=65536)
    ap.add_argument("--chunk", type=int, default=8192, help="prompt tokens per GPU pass")
    ap.add_argument("--expert-usage", default=None, help="routed counts per expert (float32 [layer][expert]); "
                    "keeps the most-used experts in VRAM (decode speed)")
    ap.add_argument("--mtp", default=None, help="MTP draft block GGUF: greedy requests decode speculatively")
    ap.add_argument("--drafts", type=int, default=3, help="MTP drafts per round")
    ap.add_argument("--draft-vocab", default=None, help="int32 token ids the draft head scores (data/draft_vocab_en.bin)")
    ap.add_argument("--draft-min-p", type=float, default=0.5, help="drafting stops below this MTP probability")
    ap.add_argument("--cpu-dir", default=None, help="CPU expert tier (q4s files); needs --expert-usage")
    ap.add_argument("--cpu-share", type=float, default=0.5)
    ap.add_argument("--cpu-threads", type=int, default=12)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--name", default="flash-next-truss")
    ap.add_argument("--log", default=None, help="append one line per request here")
    ap.add_argument("--log-tag", default="truss serve")
    ap.add_argument("--dump-dir", default=None, help="keep the last 20 requests (prompt tail + reply) here as JSON")
    a = ap.parse_args()
    engine = Engine(a)
    print(f"truss: model loaded, n_ctx {engine.model.n_ctx}, listening on {a.host}:{a.port}", flush=True)
    uvicorn.run(make_app(engine), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
