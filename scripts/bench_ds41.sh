#!/usr/bin/env bash
# The DS41 arms bench (tools/tk-bench/arms.py) on the served configuration (scripts/ds41_env.sh) with THIS checkout's
# build: one model load (~21 s + ~21 s page-locking), every arm on that model. GPUs 0 + 2 via scripts/ds41_gpu_job.sh
# (stops the served truss-ds41, starts it again and waits until it answers).
#   scripts/bench_ds41.sh [--quick] --arm base --arm skip_weight=0.12 --arm d4:drafts=4 ...
# Results: $H/logs/bench_arms/<time>/ (run.log, arms.json); the table is printed at the end. Exit 1 = health FAIL.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/ds41_env.sh"
cd "$REPO"
OUT="$H/logs/bench_arms/$(date +%m%d-%H%M%S)"
rc=0
bash scripts/ds41_gpu_job.sh "$OUT" python3 tools/tk-bench/arms.py "${DS41_ARGS[@]}" --drafts-open 5 \
  --json "$OUT/arms.json" "$@" || rc=$?
sed -n '/^== arms/,$p' "$OUT/run.log"
echo "log: $OUT/run.log"
exit $rc
