#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash-1017 on port 8115 (OpenAI-compatible, server/app.py), the model's own encoder driving
# the prompt and the reply (server/encoding.py, MIT: DeepSeek-V4.1's official encoding).
#
# The env is the best arm measured in E15 item 9 (20261019_truss_port/exec/E15_report.md, section 9): the e14g CPU
# tier (experts split VRAM + pinned RAM), layer split 14 across GPUs 0 and 2, skip 0.20, TRUSS_HINT_K=0 (the
# adaptive tier already prefetches), the DSpark drafter with 5 drafts and the item-7 tail prefill.
#   spec, 5 drafts, hints off, skip 0.20:  42.1 tok/s on the bench (plain 32.1)
# Greedy spec output differs from plain greedy by design (the call-scoped CPU weight-share skip); sampled serving is
# the normal path and is unaffected in quality (ds41_kl 0.0687 vs 0.0728 budget).
#
# GPUs: 0 and 2 (1 and 3 belong to a renter). Start it under run_watched2.sh:
#   cd ~/ML_projects/dsv41flash/20261019_truss_port
#   H=/mnt/hdd/ml/dsv41flash/20261019_truss_port
#   nohup bash ~/ML_projects/dsv41flash/20261018_poc/scripts/run_watched2.sh 0 2 $H/logs/serve_8115 \
#     ~/trellis-kernel/scripts/serve_ds41.sh > $H/logs/serve_8115.nohup 2>&1 &
# Never run two instances: one model load takes all of both GPUs' free memory.
set -euo pipefail
T=${TRUSS_PORT_DIR:-$HOME/ML_projects/dsv41flash/20261019_truss_port}
H=${TRUSS_HOME:-/mnt/hdd/ml/dsv41flash/20261019_truss_port}
REPO=${TRUSS_REPO:-$HOME/trellis-kernel}
cd "$REPO"

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# the model: the served dense GGUF + the X3.1 expert/Engram overlays (#118), as data/e14g_cpu.sh and item 9
export TRUSS_EXPERT_OVERLAY="$T/data/pack/ds41-experts.gguf,$T/data/pack/ds41-engram-L1.gguf,$T/data/pack/ds41-engram-L14.gguf,$T/data/pack/ds41-dense-x3.gguf"
export TRUSS_DSPARK_EXPERT_OVERLAY="$T/data/drafter/ds41-dspark-experts.gguf"
export TRUSS_LAYER_SPLIT=14 TRUSS_LAYER_SPLIT_DEVICE=1
# the CPU tier (E14): pinned RAM threads, the 42 GB/s trellis, the per-layer adaptation of the resident set
export TRUSS_CPU_TRELLIS=1 TRUSS_CPU_DYNAMIC=1 TRUSS_CPU_PIN=1 TRUSS_CPU_THREADS=20 TRUSS_AUX_CPUS=auto
export TRUSS_SKIP_WEIGHT=0.20 TRUSS_PCIE_FRAC=0.1 TRUSS_ADAPT_EVERY=1 TRUSS_ADAPT_SWAPS=8
# item 9: the adaptive tier already prefetches, so the hint router only costs; item 7: the consumer layers' tail prefill
export TRUSS_HINT_K=0 TRUSS_TAIL_PREFILL=134
# a short prompt runs as decode-path chunks (only its routed experts move) instead of a prompt chunk that streams
# every cold expert layer. Measured 10-21 (warm server): the stream costs ~5.3 s + ~0.4 ms/token (2,769 tokens 6.4 s,
# 5,195 tokens 7.4 s), the decode path ~43 ms/token (109 tokens 5.2 s, 387 tokens 16.6 s): it only wins below ~110
export TRUSS_FETCH_PROMPT=${TRUSS_FETCH_PROMPT:-96}
# served default: 16-bit dense activations (the C API sets it too; kept explicit for the record)
export TRUSS_ACT=fp16
export TRUSS_LIB="$REPO/build-dsv41/libtruss.so"

# the kept cold copy (src/runtime/cold_keep.h): the ~82 GB of cold experts stay in RAM in a keeper process between
# runs, so a restart maps them instead of re-reading them (the NVMe reads ~1 GB/s here: ~85 s of a ~115 s load).
# The keeper runs detached (setsid: a pm2 restart does not stop it); free its RAM with
#   python3 ~/trellis-kernel/tools/cold_keeper.py --socket $TRUSS_COLD_KEEP --drop    (or kill it). TRUSS_COLD_KEEP=0: off
export TRUSS_COLD_KEEP=${TRUSS_COLD_KEEP:-/run/user/$(id -u)/truss-cold.sock}
if [ "$TRUSS_COLD_KEEP" != 0 ] && ! python3 tools/cold_keeper.py --socket "$TRUSS_COLD_KEEP" --status >/dev/null 2>&1; then
  setsid -f python3 tools/cold_keeper.py --socket "$TRUSS_COLD_KEEP" >> "$H/logs/cold_keeper.log" 2>&1 < /dev/null
  for _ in $(seq 50); do [ -S "$TRUSS_COLD_KEEP" ] && break; sleep 0.1; done
fi

# the first requests after a load pay a one-time ~35-45 s (measured 10-21; the cause is not diagnosed yet: likely the
# background page-locking of the cold copy): a background warm-up request takes it, so a user's first request does not
( for _ in $(seq 1 180); do
      curl -sf -m 2 "http://127.0.0.1:8115/health" >/dev/null 2>&1 && break; sleep 5
  done
  curl -s -m 600 "http://127.0.0.1:8115/v1/chat/completions" -H 'Content-Type: application/json' \
       -d '{"model":"ds41-flash","max_tokens":4,"messages":[{"role":"user","content":"Hi"}]}' >/dev/null 2>&1 ) &

exec python3 -m server.app \
  --model "$T/data/pack/ds41-dense.gguf" \
  --tokenizer "$T/data/tokenizer/tokenizer.json" \
  --n-ctx 131072 --chunk 8192 \
  --expert-usage "$T/data/bench_in/usage_calib256.f32" \
  --mtp "$T/data/drafter/ds41-dspark-dense.gguf" --drafts 5 --draft-min-p 0 \
  --host 127.0.0.1 --port 8115 --name ds41-flash \
  --log "$H/logs/serve_8115-requests.log"
