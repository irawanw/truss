#!/usr/bin/env bash
# Engine golden (tests/golden/run.sh, Flash-Next slice, 7 configurations) for THIS checkout's build-dsv41, compared
# with the reference golden_1019b; GPUs via scripts/ds41_gpu_job.sh (the served truss-ds41 is stopped and waited
# back). ~5 min. Exit 0 only when every configuration is IDENTICAL. Required before any shared engine code merges.
#   scripts/golden_ds41.sh [name]     -> $T/data/logs/golden_<name or time>
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/ds41_env.sh"
cd "$REPO"
NAME=${1:-$(date +%m%d-%H%M%S)}
O="$T/data/logs/golden_$NAME"
REF=${GOLDEN_REF:-$T/data/logs/golden_1019b}
(cd build-dsv41 && ninja engine_golden >/dev/null) || { echo "build failed"; exit 1; }
bash scripts/ds41_gpu_job.sh "$H/logs/golden_$NAME" bash tests/golden/run.sh build-dsv41 "$O"
bash tests/golden/run.sh compare "$REF" "$O"
