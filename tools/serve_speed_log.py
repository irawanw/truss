#!/usr/bin/env python3
"""Per-request speed of the served model, one line each, for `pm2 logs ds41-speed`: follows the server's --log file
and prints prefill (tokens read, tok/s, reused), decode (tokens, tok/s) and draft acceptance.
  python3 tools/serve_speed_log.py [log file]    (default: the DS41 server's serve_8115-requests.log)
"""
import os
import re
import sys
import time

LOG = "/mnt/hdd/ml/dsv41flash/20261019_truss_port/logs/serve_8115-requests.log"
PAT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in "
                 r"(\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else LOG
    f = open(path)
    f.seek(0, os.SEEK_END)
    ino = os.fstat(f.fileno()).st_ino
    print(f"following {path}", flush=True)
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.5)
            try:   # rotated or recreated
                if os.stat(path).st_ino != ino:
                    f = open(path)
                    ino = os.fstat(f.fileno()).st_ino
            except FileNotFoundError:
                pass
            continue
        m = PAT.search(line)
        if not m:
            continue
        n, reused, read, rms, rtps, gen, gms, gtps, acc, dr = m.groups()
        acc_s = f"  drafts {100 * int(acc) / max(1, int(dr)):.0f}% ({1 + int(acc) / max(1, int(gen) - int(acc)):.2f} tok/pass)" \
            if int(dr) else ""
        print(f"{time.strftime('%H:%M:%S')}  prefill {int(read):>6} tok {float(rtps):7.1f} tok/s ({int(rms) / 1e3:5.1f} s, "
              f"{int(reused)} reused of {n})  |  decode {int(gen):>5} tok {float(gtps):5.1f} tok/s ({int(gms) / 1e3:5.1f} s)"
              f"{acc_s}", flush=True)


if __name__ == "__main__":
    main()
