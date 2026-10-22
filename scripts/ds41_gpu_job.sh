#!/usr/bin/env bash
# One GPU job on our GPUs (TRUSS_GPUS, default "1 3"; any number, in CUDA device order, e.g. "1 3 0 2" for the
# 4-stage split; which GPUs are ours changes: see ~/gpu-park/parked.json) while the served DS41 (pm2 truss-ds41) is the self-iteration agent's own model:
#   scripts/ds41_gpu_job.sh <log dir> <cmd...>
# Stops truss-ds41 when it is online, runs <cmd> under run_watched2.sh (renter watchdog), then starts truss-ds41
# again and RETURNS ONLY WHEN IT ANSWERS (health + a 4-token reply, <= 5 min), so the agent's next request finds its
# model up. Exit code = the job's. The served copy is ~/trellis-serve (promoted by the planner only), so a broken
# dev build here never breaks the agent's model. scripts/bench_ds41.sh and the golden runs go through this.
set -uo pipefail
OUT=$1; shift
WATCH=${TRUSS_WATCH:-$HOME/ML_projects/dsv41flash/20261018_poc/scripts/run_watchedN.sh}
GPUS=${TRUSS_GPUS:-1 3}
# a foreign GPU pid that is gone within this many seconds is a probe (10-22: two golden runs stopped by pids with no
# exe, gone in < 3 s, likely the vast agent in our own parking containers), not a renter
export FOREIGN_PERSIST_S=${FOREIGN_PERSIST_S:-4}

online=0
if pm2 jlist 2>/dev/null | python3 -c 'import json,sys; sys.exit(0 if any(p["name"]=="truss-ds41" and p["pm2_env"]["status"]=="online" for p in json.load(sys.stdin)) else 1)'; then
  online=1
  echo "stopping pm2 truss-ds41 for the job (started again after it)"
  pm2 stop truss-ds41 >/dev/null
fi
back() {
  [ $online = 1 ] || return 0
  echo "starting pm2 truss-ds41, waiting until it answers"
  pm2 start truss-ds41 >/dev/null
  for _ in $(seq 300); do
    if curl -sf -m 120 http://127.0.0.1:8115/v1/chat/completions -H 'Content-Type: application/json' \
         -d '{"model":"ds41-flash","max_tokens":4,"messages":[{"role":"user","content":"Hi"}]}' >/dev/null 2>&1; then
      echo "truss-ds41 answers"; return 0
    fi
    sleep 1
  done
  echo "WARNING: truss-ds41 did not answer within 5 min (pm2 logs truss-ds41)"
}
trap back EXIT

# the stopped server's CUDA contexts leave the GPUs a few seconds after it exits
for _ in $(seq 120); do
  n=0
  for G in $GPUS; do
    U="$(nvidia-smi -i "$G" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
    n=$((n + $(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d ' ' | grep -cxF "$U" || true)))
  done
  [ $n = 0 ] && break
  sleep 0.5
done
bash "$WATCH" "${GPUS// /,}" "$OUT" "$@"
