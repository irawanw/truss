#!/usr/bin/env bash
# Plan v3 0.3 (port of 20261003_x31/scripts/e7_serve.sh): served-pattern server for X3.1 on GPU 1, port 8193.
# Served config = the brain's exactly (selfopt_brain.sh env + cmdline), only the port differs. GPU 1 must be free.
# usage: e7_serve_x31.sh <tag> [VAR=value ...]   log: data/logs/e7_x31_server_<tag>.log
set -euo pipefail
tag=${1:?tag}; shift
cd /home/green-gpu/trellis-kernel
export CUDA_VISIBLE_DEVICES=1
export TRUSS_CPU_TRELLIS=1 TRUSS_CPU_DYNAMIC=1 TRUSS_PCIE_FRAC=0.2 TRUSS_HINT_K=3 TRUSS_CPU_THREADS=22 TRUSS_CPU_PIN=1
export TRUSS_KV_INT8=1 TRUSS_FETCH_PROMPT=96 TRUSS_SPLIT_ROWS=2048 TRUSS_KV_LEND=1 TRUSS_ADMIT_IDLE=64 TRUSS_LOAD_TIMES=1
for kv in "$@"; do export "$kv"; done
D=$HOME/ML_projects/flashnext
LOG=$D/20261004_selfopt/data/logs/e7_x31_server_$tag.log
nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1 | awk '{ if ($1 > 1000) { print "GPU 1 is in use"; exit 1 } }'
exec python3 -m server.app \
  --model $D/20261003_x31/data/flashnext-x31.gguf \
  --tokenizer /data/www/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/models/Qwen3.8-27B-EXL3-3.5bpw/tokenizer.json \
  --n-ctx 262144 --chunk 8192 --expert-usage data/usage_strata_rank.f32 \
  --mtp $D/20260930_truss_tg/data/mtp_x3/flashnext-mtp-x3k3.gguf --drafts 3 --draft-vocab data/draft_vocab_en.bin --draft-min-p 0.5 \
  --host 127.0.0.1 --port 8193 --name e7x31-$tag --log "$LOG"
