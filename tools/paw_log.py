#!/usr/bin/env python3
"""Follows the self-iteration agent's omp session (the newest *.jsonl under ~/.omp/agent/sessions/<dir>) and prints
one readable line per event, for `pm2 logs paw-ds41`: its thinking and replies (shortened), each tool call with its
intent and command, and each tool result's first lines. Switches to a newer session file when one appears.
  python3 tools/paw_log.py [--dir -trellis-kernel] [--width 220]
"""
import argparse
import glob
import json
import os
import time


def short(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 3] + "..."


def render(d, w):
    try:   # the session stores UTC; print local time like the other logs
        ts = time.strftime("%H:%M:%S", time.localtime(time.mktime(time.strptime(d["timestamp"][:19], "%Y-%m-%dT%H:%M:%S"))
                                                          - time.timezone))
    except (KeyError, ValueError):
        ts = ""
    if d.get("type") == "custom" and d.get("customType") == "tool_execution_start":
        x = d.get("data") or {}
        a = x.get("args") or {}
        what = a.get("command") or a.get("path") or a.get("file_path") or json.dumps(a)[:200]
        return [f"{ts} TOOL {x.get('toolName')} [{x.get('intent', '')}] {short(what, w)}"]
    if d.get("type") != "message":
        return []
    m = d.get("message") or {}
    role, out = m.get("role"), []
    for c in m.get("content") or []:
        t = c.get("type")
        if role == "assistant" and t == "thinking":
            out.append(f"{ts} think {short(c.get('thinking', ''), w)}")
        elif role == "assistant" and t == "text" and c.get("text", "").strip():
            out.append(f"{ts} PAW   {short(c['text'], w)}")
        elif role == "toolResult" and t == "text":
            out.append(f"{ts}   -> {short(c.get('text', ''), w)}")
        elif role == "user" and t == "text":
            out.append(f"{ts} USER  {short(c.get('text', ''), w)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="-trellis-kernel")
    ap.add_argument("--width", type=int, default=220)
    a = ap.parse_args()
    root = os.path.expanduser(f"~/.omp/agent/sessions/{a.dir}")
    cur, f = None, None
    while True:
        files = sorted(glob.glob(os.path.join(root, "*.jsonl")), key=os.path.getmtime)
        if files and files[-1] != cur:
            cur = files[-1]
            f = open(cur)
            print(f"=== session {os.path.basename(cur)}", flush=True)
        line = f.readline() if f else ""
        if not line:
            time.sleep(1)
            continue
        try:
            for s in render(json.loads(line), a.width):
                print(s, flush=True)
        except (ValueError, AttributeError):
            pass


if __name__ == "__main__":
    main()
