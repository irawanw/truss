#!/usr/bin/env python3
"""Holds TRUSS expert-store cold copies (memfds) between engine runs: see src/runtime/cold_keep.h.

usage:
  cold_keeper.py --socket PATH [--max-gb 90]   serve (foreground; start it detached with setsid -f)
  cold_keeper.py --socket PATH --status        what it holds
  cold_keeper.py --socket PATH --drop          free every copy (the RAM comes back once no engine maps it)

The RAM it holds stays taken while no engine runs (~82 GB for DS41): drop it, or stop the keeper, before running
something else that needs that memory. Entries are evicted least recently used first, so a request never makes the
total exceed --max-gb; a request for a slot with a different key replaces that slot's copy.
"""
import argparse
import os
import socket
import sys
import time


def serve(path, max_bytes):
    if os.path.exists(path):
        try:   # a keeper already answers there: do not start a second one
            socket.socket(socket.AF_UNIX).connect(path)
            print(f"cold_keeper: already running on {path}", file=sys.stderr)
            return 0
        except OSError:
            os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    os.chmod(path, 0o600)
    srv.listen(8)
    held = {}   # slot -> {"key", "bytes", "fd", "used"}

    def log(msg):
        print(time.strftime("%H:%M:%S"), "cold_keeper:", msg, flush=True)

    def drop(slot, why):
        e = held.pop(slot)
        os.close(e["fd"])
        log(f"dropped {slot} ({e['bytes'] / 1e9:.2f} GB, {why})")

    log(f"listening on {path}, cap {max_bytes / 1e9:.0f} GB")
    while True:
        c, _ = srv.accept()
        with c:
            c.settimeout(10)
            try:
                msg, fds, _, _ = socket.recv_fds(c, 256, 1)
                words = msg.decode().split()
                verb = words[0] if words else ""
                if verb == "GET" and len(words) == 4:
                    slot, key, size = words[1], words[2], int(words[3])
                    e = held.get(slot)
                    if e and e["key"] == key and e["bytes"] == size:
                        e["used"] = time.time()
                        socket.send_fds(c, [b"HIT\n"], [e["fd"]])
                        log(f"hit {slot} ({size / 1e9:.2f} GB)")
                        continue
                    if e:
                        drop(slot, "new key")
                    for s in sorted(held, key=lambda s: held[s]["used"]):
                        if sum(x["bytes"] for x in held.values()) + size <= max_bytes:
                            break
                        drop(s, "cap")
                    c.sendall(b"MISS\n")
                elif verb == "PUT" and len(words) == 4 and fds:
                    slot, key, size = words[1], words[2], int(words[3])
                    if slot in held:
                        drop(slot, "replaced")
                    held[slot] = {"key": key, "bytes": size, "fd": fds.pop(0), "used": time.time()}
                    c.sendall(b"OK\n")
                    log(f"holding {slot} ({size / 1e9:.2f} GB, {len(held)} copies, "
                        f"{sum(x['bytes'] for x in held.values()) / 1e9:.2f} GB)")
                elif verb == "STATUS":
                    lines = [f"{s} {e['key']} {e['bytes'] / 1e9:.2f} GB, used {time.ctime(e['used'])}"
                             for s, e in held.items()]
                    c.sendall(("\n".join(lines) or "empty").encode() + b"\n")
                elif verb == "DROP":
                    for s in list(held):
                        drop(s, "asked")
                    c.sendall(b"OK\n")
                else:
                    c.sendall(b"ERR\n")
                for fd in fds:
                    os.close(fd)
            except (OSError, ValueError, UnicodeDecodeError) as ex:
                log(f"bad request: {ex}")


def ask(path, verb):
    c = socket.socket(socket.AF_UNIX)
    c.connect(path)
    c.sendall(verb.encode() + b"\n")
    print(c.recv(65536).decode(), end="")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--max-gb", type=float, default=90)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--drop", action="store_true")
    a = ap.parse_args()
    if a.status or a.drop:
        ask(a.socket, "STATUS" if a.status else "DROP")
        return 0
    return serve(a.socket, int(a.max_gb * 1e9))


if __name__ == "__main__":
    sys.exit(main())
