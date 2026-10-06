#!/usr/bin/env bash
# Plan v3 0.3 runner: served-pattern bench on GPU 1 (dedicated X3.1 server :8193, brain's config).
# Lock + trap restart; brain down ~5-6 min. Steps log -> $LOGD/steps.jsonl, server log -> data/logs/e7_x31_server_paw03.log.
set -u
SELF=/home/green-gpu/ML_projects/flashnext/20261004_selfopt
WT=$SELF/work/agent
NOTIFY=$SELF/data/notify.txt
TS=$(date +%H%M%S)
LOGD=$SELF/data/logs/e7_03_$TS; mkdir -p "$LOGD"
SRV=""

exec 9>/tmp/selfopt_gpu2.lock
flock 9 || { echo "$(date '+%a %b %d %H:%M %Y') | GPU SKIP paw | 0.3 e7 lock busy" >> "$NOTIFY"; exit 1; }
cleanup() {
  [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null
  pm2 start truss-x31-brain >/dev/null 2>&1
  echo "$(date '+%a %b %d %H:%M %Y') | GPU END paw | 0.3 e7 served-pattern bench (trap restart)" >> "$NOTIFY"
}
trap cleanup EXIT
echo "$(date '+%a %b %d %H:%M %Y') | GPU START paw | 0.3 e7 served-pattern bench GPU1 server :8193 brain-down ~5min" >> "$NOTIFY"
pm2 stop truss-x31-brain >/dev/null
sleep 3

"$WT/scripts/e7_serve_x31.sh" paw03 > "$LOGD/server_stdout.log" 2>&1 &
SRV=$!
READY=0
for i in $(seq 1 240); do
  curl -s -m 1 127.0.0.1:8193/health >/dev/null 2>&1 && { READY=1; break; }
  kill -0 "$SRV" 2>/dev/null || break
  sleep 1
done
if [ "$READY" != 1 ]; then
  echo "$(date '+%a %b %d %H:%M %Y') | STUCK | 0.3 e7 server did not come up (see $LOGD/server_stdout.log)" >> "$NOTIFY"
  exit 1   # trap restarts the brain
fi
echo "$(date '+%H:%M:%S') server ready after ${i}s" >> "$LOGD/status.txt"
python3 "$WT/scripts/e7_steps_x31.py" --port 8193 --steps 8 > "$LOGD/steps.jsonl" 2>&1
RC=$?
echo "$(date '+%H:%M:%S') steps rc=$RC" >> "$LOGD/status.txt"
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""
pm2 start truss-x31-brain >/dev/null
trap - EXIT
echo "$(date '+%a %b %d %H:%M %Y') | GPU END paw | 0.3 e7 served-pattern bench rc=$RC logs=$LOGD" >> "$NOTIFY"
