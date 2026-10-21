"""TRUSS OpenAI-compatible server (chat and text completions, streaming or not) for one model on one GPU.

TRUSS_SLOTS sequences at once (1 without the flag, which is today's behaviour): each request owns a sequence for its
whole life and holds the engine lock for one call at a time, so two streams take turns per round (~30 ms) and
neither waits for the other's prompt. A new prompt that extends the sequence already in its slot (same tokens, then
more) only evaluates the new tokens; anything else starts a new sequence in that slot (the GDN state cannot be
rewound). Greedy decoding takes the argmax on the device; sampling copies the logits.

Tool calls: the model's <tool_call> XML is returned as OpenAI `tool_calls` (finish_reason "tool_calls"), and
assistant tool calls in the history have their JSON-string arguments parsed before templating (server/tools.py).

Sampling defaults are llama-server's (temperature 0.8, top_k 40, top_p 0.95, min_p 0.05; DeepSeek-V4.1: its own 1.0 / 1.0, no top_k, min_p 0.05), so a client that sends no
sampling parameters gets what it got from llama-paw on this port; temperature 0 is greedy.

Each request logs one line (to stdout and --log) in the format flashnext_strata_bench.py parses, so that bench runs
unchanged against TRUSS; --log-tag sets its prefix (the bench counts "strata serve: prompt", so the slot and the
round-gap clauses are appended at the END of the line, never before "prompt"). Progress lines carry the slot too.

usage: CUDA_VISIBLE_DEVICES=2 python3 -m server.app --model <gguf> --tokenizer <tokenizer.json> [--port 8080]
"""
import argparse
import json
import os
import time
import uuid
import anyio
import asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from server.tools import ToolStream, normalize_messages, parse_calls
from server.truss_ctypes import TrussError
from server.engine import Engine
from server.reasoning import ThinkSplitter, split_reasoning
from fastapi import HTTPException  # noqa: F401  (re-exported: tests use app.HTTPException)
from server.seq import FairGate, raw_ids  # noqa: F401  (re-exported: tests use app.FairGate, app.raw_ids)


def make_app(engine: Engine) -> FastAPI:
    app = FastAPI()

    # unset sampling fields: DeepSeek-V4.1's own defaults (DeepInfer sampling_params.py: temperature 1.0, top_p 1.0,
    # no top_k / min_p; its evals run at 1.0) for that family - the narrower llama-server defaults below were the
    # suspect for the agent's reasoning loops (10-21) - and llama-server's for every other family
    # min_p 0.05 on top for DS41: at 1.0 / 1.0 the quantized model sampled junk from its tail ("转基因" inside English
    # reasoning, 10-21); min_p drops what is under 5% of the top token and keeps the diversity
    dflt = (dict(temperature=1.0, top_p=1.0, top_k=0, min_p=0.05) if getattr(engine, "ds41", None)
            else dict(temperature=0.8, top_p=0.95, top_k=40, min_p=0.05))

    def params(body):
        stop = body.get("stop") or []
        def get(key, default, cast):
            v = body.get(key)
            return cast(default if v is None else v)
        p = dict(max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or engine.model.n_ctx),
                 temperature=get("temperature", dflt["temperature"], float), top_p=get("top_p", dflt["top_p"], float),
                 top_k=get("top_k", dflt["top_k"], int), min_p=get("min_p", dflt["min_p"], float), seed=body.get("seed"),
                 presence_penalty=get("presence_penalty", 0.0, float),
                 frequency_penalty=get("frequency_penalty", 0.0, float),
                 # llama-server's names too
                 repetition_penalty=get("repetition_penalty", body.get("repeat_penalty") or 1.0, float),
                 penalty_last_n=get("penalty_last_n", body.get("repeat_last_n") or 64, int),
                 # llama-server's ignore_eos: keep generating to max_tokens. Used by the decode-speed benches so that
                 # two streams decode at the same time instead of one ending its turn early.
                 ignore_eos=bool(body.get("ignore_eos", False)),
                 stop_strings=[stop] if isinstance(stop, str) else list(stop))
        # A repetition_penalty of 0 (or less) makes the engine's penalty divide by it, and it throws from inside the
        # first decode round - after the engine opened a verify window on this slot, which is the state review C2
        # found unrecoverable. Rejected here, as a 400 through the ValueError handler below, before a slot is taken or
        # a response starts. 0 is never valid: any other penalty set is off only when repetition_penalty is 1 too.
        if p["repetition_penalty"] <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if p["penalty_last_n"] < -1:   # -1 means "the whole request" (llama-server); the engine clamps the rest
            raise ValueError("penalty_last_n must be >= -1")
        return p

    _END, KEEPALIVE = object(), object()

    def too_long(ids):
        """llama-server's 400 for a prompt that does not fit (clients such as omp match its message and compact), sent
        before any response starts; the engine's own error came mid-stream and reached the client as a dead stream."""
        if len(ids) < engine.model.n_ctx:
            return None
        return JSONResponse({"error": {"code": 400, "message": "the request exceeds the available context size, try "
                                       "increasing it", "type": "exceed_context_size_error",
                                       "n_prompt_tokens": len(ids), "n_ctx": engine.model.n_ctx}}, status_code=400)

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

    @app.get("/debug/requests")
    def debug_requests():
        d = os.environ.get("TRUSS_REQ_LOG_DIR")
        return {"files": sorted(os.listdir(d)) if d and os.path.isdir(d) else []}

    @app.post("/debug/probe")
    async def debug_probe(req: Request):
        body = await req.json()
        return await anyio.to_thread.run_sync(lambda: engine.probe(body["file"], int(body.get("window", 400))))

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": engine.name, "object": "model", "owned_by": "truss"}]}

    @app.post("/debug/round_trace")
    def round_trace():
        """TRUSS_ROUND_TRACE (scripts/u_trace_report.py): flush the engine's buffered intervals to the file now,
        while the server is alive, so a bench can attribute a served run the way it attributes an in-process one.
        A shutdown hook would not do: uvicorn leaves on SIGTERM without running atexit handlers."""
        if not os.environ.get("TRUSS_ROUND_TRACE"):
            raise ValueError("TRUSS_ROUND_TRACE is not set")
        engine.model.trace_dump()
        return {"path": os.environ["TRUSS_ROUND_TRACE"]}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        body = await req.json()
        kw = dict(body.get("chat_template_kwargs") or {})
        # the OpenAI-style top-level reasoning_effort (omp's --thinking) reaches the template unless the kwargs set it;
        # DeepSeek-V4.1 knows low / high / max (50 / 75 / 100), the other names map to the nearest
        eff = body.get("reasoning_effort")
        if eff is not None and "reasoning_effort" not in kw:
            kw["reasoning_effort"] = {"minimal": "low", "low": "low", "medium": "high", "high": "high",
                                      "xhigh": "max", "max": "max"}.get(str(eff), eff)
        thinking = kw.get("enable_thinking", True) is not False
        tools = body.get("tools") or None
        if body.get("tool_choice") == "none":
            tools = None
        # vLLM-compatible `prompt`: an already-rendered prompt, as text or as token ids, bypasses the chat template.
        # A client that owns its prompt (an agent that renders its own template, a bench replaying recorded ids)
        # gets those exact tokens, which is what makes a served measurement comparable to an in-process one.
        raw = body.get("prompt")
        if isinstance(raw, list):
            ids = raw_ids(raw, engine.model.n_vocab)   # F1: these ids go to the embedding table unchecked
        elif raw is not None:
            ids = await anyio.to_thread.run_sync(engine.chat.encode, raw)
        else:
            msgs = normalize_messages(body["messages"])
            if engine.ds41 is not None:   # the official DeepSeek-V4.1 encoder, not a GGUF template
                prompt, thinking = await anyio.to_thread.run_sync(
                    lambda: engine.ds41.prompt(msgs, tools=tools, template_kwargs=kw))
            else:
                prompt = await anyio.to_thread.run_sync(
                    lambda: engine.chat.prompt(msgs, tools=tools, template_kwargs=kw))
            ids = await anyio.to_thread.run_sync(engine.chat.encode, prompt)
        if (r := too_long(ids)) is not None:
            return r
        p = params(body)
        p["think_open"] = thinking   # the generation prompt opens <think>: no end of turn before </think>
        pen = "".join(f" {k} {p[k]}" for k, off in (("presence_penalty", 0.0), ("frequency_penalty", 0.0),
                                                     ("repetition_penalty", 1.0)) if p[k] != off)
        # logged by generate() once the request has a slot, so the line says which sequence it belongs to (a queued
        # request that is abandoned before it gets a slot never logs one)
        p["note"] = (f"request: temperature {p['temperature']} top_p {p['top_p']} top_k {p['top_k']} "
                     f"min_p {p['min_p']}{pen}{' penalty_last_n ' + str(p['penalty_last_n']) if pen else ''}, "
                     f"{len(tools or [])} tools, thinking {'on' if thinking else 'off'}")
        rid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

        if body.get("stream"):
            async def events():
                def chunk(delta, finish=None):
                    return "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                                                  "model": engine.name,
                                                  "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"
                # DS41 parses its own format (server/ds41.py): the DSML token and </think> by token id, so the
                # splitter sees the token ids as well as the text deltas
                ds41 = engine.ds41
                split = ds41.stream(thinking) if ds41 else ThinkSplitter(thinking)
                calls = ToolStream(tools) if (tools and not ds41) else None

                def emit(pairs):
                    for kind, x in pairs:
                        yield chunk({"tool_calls": [x]} if kind == "call" else {"content": x})

                def out(pairs):
                    if ds41:
                        for kind, x in pairs:
                            yield chunk({"tool_calls": [x]}) if kind == "call" else chunk({kind: x})
                        return
                    for field, d in pairs:
                        if field == "content" and calls:
                            yield from emit(calls.push(d))
                        elif d:
                            yield chunk({field: d})
                yield chunk({"role": "assistant", "content": ""})
                n_gen, items = 0, tokens(req, ids, p, keepalive=5.0)
                while True:
                    try:
                        item = await items.__anext__()
                    except StopAsyncIteration:
                        break
                    except (TrussError, ValueError) as e:   # the response has started: report in-stream
                        yield "data: " + json.dumps({"error": {"code": 500, "message": str(e), "type": "server_error"}}) + "\n\n"
                        break
                    if item is KEEPALIVE:
                        yield chunk({"content": ""})
                        continue
                    tok, delta, finish = item
                    n_gen += tok is not None
                    for c in out(split.push(tok, delta) if ds41 else split.push(delta)):
                        yield c
                    if finish:
                        for c in out(split.flush()):
                            yield c
                        if ds41:
                            if split.calls and finish == "stop":
                                finish = "tool_calls"
                        elif calls:
                            for c in emit(calls.finish()):
                                yield c
                            if calls.calls and finish == "stop":
                                finish = "tool_calls"
                        yield chunk({}, finish)
                if (body.get("stream_options") or {}).get("include_usage"):
                    # omp sizes the context from this (without it, its bar read 19% at a full 262K)
                    yield "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                                                 "model": engine.name, "choices": [],
                                                 "usage": {"prompt_tokens": len(ids), "completion_tokens": n_gen,
                                                           "total_tokens": len(ids) + n_gen}}) + "\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")

        text, n, finish = "", 0, "length"
        async for tok, delta, fin in tokens(req, ids, p):
            if tok is not None:
                text += delta
                n += 1
            if fin:
                finish = fin
        if engine.ds41 is not None:
            msg = engine.ds41.parse(text, thinking, finish)
            if msg.get("tool_calls"):
                finish = "tool_calls"
            msg = {k: v for k, v in msg.items() if v or k in ("role", "content")}
        else:
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
        if (r := too_long(ids)) is not None:
            return r
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
