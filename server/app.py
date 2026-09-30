"""TRUSS OpenAI-compatible server (chat and text completions, streaming or not) for one model on one GPU.

One sequence at a time. A new prompt that extends the sequence already in the engine (same tokens, then more) only
evaluates the new tokens; anything else starts a new sequence (the GDN state cannot be rewound). Greedy decoding
takes the argmax on the device; sampling copies the logits.

Each request logs one line (to stdout and --log) in the format flashnext_strata_bench.py parses, so that bench runs
unchanged against TRUSS; --log-tag sets its prefix (the bench counts "strata serve: prompt").

usage: CUDA_VISIBLE_DEVICES=2 python3 -m server.app --model <gguf> --tokenizer <tokenizer.json> [--port 8080]
"""
import argparse
import json
import threading
import time
import uuid

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from server.chat import Chat, Detokenizer, sample
from server.truss_ctypes import Model, TrussError


class Engine:
    """The model plus the sequence in it; generate() is the only way in and holds the lock."""

    def __init__(self, a):
        self.model = Model(a.model, a.n_ctx, a.chunk)
        self.chat = Chat(a.tokenizer, self.model.meta_string("tokenizer.chat_template"))
        self.stop_ids = {i for i in (self.model.meta_int("tokenizer.ggml.eos_token_id"),
                                     self.chat.token_id("<|im_end|>"), self.chat.token_id("<|endoftext|>"))
                         if i is not None and i >= 0}
        self.name, self.log_path, self.log_tag = a.name, a.log, a.log_tag
        self.cached = []               # tokens in the engine's sequence
        self.lock = threading.Lock()

    def log(self, line: str):
        print(line, flush=True)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def generate(self, prompt_ids, max_tokens, temperature, top_p, top_k, seed, stop_strings):
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
            rng = np.random.default_rng(seed)
            greedy = temperature <= 0

            def step(tokens):
                self.cached += list(tokens)
                if greedy:
                    return self.model.eval_argmax(tokens)
                return sample(self.model.eval(tokens), temperature, top_p, top_k, rng)

            t0 = time.time()
            try:
                nxt = step(prompt_ids[reuse:])
            except TrussError:
                self.cached = []   # the engine's sequence is unknown now: the next request starts over
                raise
            t1 = time.time()
            detok, text, n_gen, finish = Detokenizer(self.chat), "", 0, "length"
            try:
                while n_gen < max_tokens:
                    n_gen += 1
                    if nxt in self.stop_ids:
                        finish = "stop"
                        break
                    delta = detok.push(nxt)
                    text += delta
                    hit = next((s for s in stop_strings if s and s in text), None)
                    if hit:
                        cut = text.index(hit)
                        yield nxt, delta[:max(0, len(delta) - (len(text) - cut))], None
                        finish = "stop"
                        break
                    yield nxt, delta, None
                    if n_gen < max_tokens and self.model.position < self.model.n_ctx:
                        nxt = step([nxt])
            except TrussError:
                self.cached = []
                raise
            finally:
                t2 = time.time()
                read, gen_s = t1 - t0, t2 - t1
                self.log(f"{self.log_tag}: prompt {n_prompt} tokens = {reuse} reused + {n_prompt - reuse} read in "
                         f"{read * 1e3:.0f} ms ({(n_prompt - reuse) / max(read, 1e-9):.1f} tok/s), {n_gen} generated "
                         f"in {gen_s * 1e3:.0f} ms ({n_gen / max(gen_s, 1e-9):.1f} tok/s), drafts accepted 0 of 0")
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
        return dict(max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024),
                    temperature=float(body.get("temperature", 1.0) if body.get("temperature") is not None else 1.0),
                    top_p=float(body.get("top_p") or 1.0), top_k=int(body.get("top_k") or 0), seed=body.get("seed"),
                    stop_strings=[stop] if isinstance(stop, str) else list(stop))

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
        prompt = engine.chat.prompt(body["messages"], tools=body.get("tools"), template_kwargs=kw)
        ids = engine.chat.encode(prompt)
        p = params(body)
        rid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

        if body.get("stream"):
            def events():
                def chunk(delta, finish=None):
                    return "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                                                  "model": engine.name,
                                                  "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"
                split = ThinkSplitter(thinking)
                for tok, delta, finish in engine.generate(ids, **p):
                    for field, d in split.push(delta):
                        yield chunk({field: d})
                    if finish:
                        for field, d in split.flush():
                            yield chunk({field: d})
                        yield chunk({}, finish)
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")

        text, n, finish = "", 0, "length"
        for tok, delta, fin in engine.generate(ids, **p):
            if tok is not None:
                text += delta
                n += 1
            if fin:
                finish = fin
        reasoning, content = split_reasoning(text, thinking)
        msg = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        return JSONResponse({"id": rid, "object": "chat.completion", "created": created, "model": engine.name,
                             "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                             "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}})

    @app.post("/v1/completions")
    async def completions(req: Request):
        body = await req.json()
        ids = engine.chat.encode(body["prompt"])
        p = params(body)
        text, n, finish = "", 0, "length"
        for tok, delta, fin in engine.generate(ids, **p):
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
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--name", default="flash-next-truss")
    ap.add_argument("--log", default=None, help="append one line per request here")
    ap.add_argument("--log-tag", default="truss serve")
    a = ap.parse_args()
    engine = Engine(a)
    print(f"truss: model loaded, n_ctx {engine.model.n_ctx}, listening on {a.host}:{a.port}", flush=True)
    uvicorn.run(make_app(engine), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
