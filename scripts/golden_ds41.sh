#!/usr/bin/env bash
# Engine golden (tests/golden/run.sh, Flash-Next slice, 7 configurations) for THIS checkout's build-dsv41, compared
# with the reference golden_1019b; GPUs via scripts/ds41_gpu_job.sh (the served truss-ds41 is stopped and waited
# back). ~5 min. Exit 0 only when every configuration is IDENTICAL. Required before any shared engine code merges.
#   scripts/golden_ds41.sh [name]     -> $T/data/logs/golden_<name or time>
set -uo pipefail
# Paths only: sourcing ds41_env.sh leaked the served config into the slice (TRUSS_EXPERT_OVERLAY replaced DS41 expert
# tensors inside the 8-layer Flash-Next model -> "open: qwen4exp: blk.0.ffn_gate_exps is 256 x 2304x5120"), and the
# golden's own configuration is tests/golden/run.sh's. Keep the device order the golden was recorded with.
T=${TRUSS_PORT_DIR:-$HOME/ML_projects/dsv41flash/20261019_truss_port}
H=${TRUSS_HOME:-/mnt/hdd/ml/dsv41flash/20261019_truss_port}
REPO=${TRUSS_REPO:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$REPO"
NAME=${1:-$(date +%m%d-%H%M%S)}
O="$T/data/logs/golden_$NAME"
REF=${GOLDEN_REF:-$T/data/logs/golden_1019b}
(cd build-dsv41 && ninja engine_golden >/dev/null) || { echo "build failed"; exit 1; }
bash scripts/ds41_gpu_job.sh "$H/logs/golden_$NAME" bash tests/golden/run.sh build-dsv41 "$O"
bash tests/golden/run.sh compare "$REF" "$O"
