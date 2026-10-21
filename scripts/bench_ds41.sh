#!/usr/bin/env bash
# The DS41 arms bench (tools/tk-bench/arms.py) on the served configuration (scripts/ds41_env.sh): one model load
# (~31 s with the cold keeper full), every arm on that model. GPUs 0 + 2 under run_watched2.sh.
#   scripts/bench_ds41.sh [--quick] --arm base --arm skip_weight=0.12 --arm d3:drafts=3 ...
# The pm2 server truss-ds41 holds those GPUs: when it is online this stops it first and starts it again after the
# bench (whatever the bench's result). Results: $H/logs/bench_arms/<time>/ (run.log, arms.json).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/ds41_env.sh"
cd "$REPO"
OUT="$H/logs/bench_arms/$(date +%m%d-%H%M%S)"
WATCH=${TRUSS_WATCH:-$HOME/ML_projects/dsv41flash/20261018_poc/scripts/run_watched2.sh}

online=0
if pm2 jlist 2>/dev/null | python3 -c 'import json,sys; sys.exit(0 if any(p["name"]=="truss-ds41" and p["pm2_env"]["status"]=="online" for p in json.load(sys.stdin)) else 1)'; then
  online=1
  echo "stopping pm2 truss-ds41 for the bench (restarted after it)"
  pm2 stop truss-ds41 >/dev/null
fi
restart() { [ $online = 1 ] && { echo "restarting pm2 truss-ds41"; pm2 start truss-ds41 >/dev/null; }; return 0; }
trap restart EXIT

# the stopped server's CUDA contexts leave the GPUs a few seconds after it exits
for _ in $(seq 120); do
  n=0
  for G in 0 2; do
    U="$(nvidia-smi -i "$G" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
    n=$((n + $(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d ' ' | grep -cxF "$U" || true)))
  done
  [ $n = 0 ] && break
  sleep 0.5
done

rc=0
bash "$WATCH" 0 2 "$OUT" python3 tools/tk-bench/arms.py "${DS41_ARGS[@]}" --json "$OUT/arms.json" "$@" || rc=$?
sed -n '/^== arms/,$p' "$OUT/run.log"
echo "log: $OUT/run.log"
exit $rc
