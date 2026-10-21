#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash-1017 on port 8115 (OpenAI-compatible, server/app.py), the model's own encoder driving
# the prompt and the reply (server/encoding.py, MIT: DeepSeek-V4.1's official encoding).
#
# The configuration (env + model arguments) lives in scripts/ds41_env.sh, shared with the bench
# (scripts/bench_ds41.sh), so what is benched is what is served.
#
# GPUs: 0 and 2 (1 and 3 belong to a renter). Start it under run_watched2.sh:
#   cd ~/ML_projects/dsv41flash/20261019_truss_port
#   H=/mnt/hdd/ml/dsv41flash/20261019_truss_port
#   nohup bash ~/ML_projects/dsv41flash/20261018_poc/scripts/run_watched2.sh 0 2 $H/logs/serve_8115 \
#     ~/trellis-kernel/scripts/serve_ds41.sh > $H/logs/serve_8115.nohup 2>&1 &
# Never run two instances: one model load takes all of both GPUs' free memory.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/ds41_env.sh"
cd "$REPO"

# the first request after a load is slower (the background page-locking of the cold copy, ~20 s after listening, and
# the first decode refilling the expert ring; was ~35 s before #220, ~6 s since): a warm-up request takes it
( for _ in $(seq 1 180); do
      curl -sf -m 2 "http://127.0.0.1:8115/health" >/dev/null 2>&1 && break; sleep 5
  done
  curl -s -m 600 "http://127.0.0.1:8115/v1/chat/completions" -H 'Content-Type: application/json' \
       -d '{"model":"ds41-flash","max_tokens":4,"messages":[{"role":"user","content":"Hi"}]}' >/dev/null 2>&1 ) &

exec python3 -m server.app "${DS41_ARGS[@]}" \
  --host 127.0.0.1 --port 8115 --name ds41-flash \
  --log "$H/logs/serve_8115-requests.log"
