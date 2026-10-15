#!/usr/bin/env bash
# TRUSS OpenAI-compatible server for the LFM K3.5 pack on physical GPU 2, port 8115, 256K context.
# Bench-identical config (Q8_1 activations, GPU-resident experts, chunk 2048); no MTP/draft (none exists for LFM).
# pm2: pm2 start scripts/lfm_serve_8114.config.js
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=2
D=$HOME/ML_projects/lfm24b/20261014_x31/data
exec python3 -m server.app \
  --model $D/gguf/lfm25-24b-x31-k3.5.gguf \
  --tokenizer $D/lfm-tok/tokenizer.json \
  --n-ctx 262144 --chunk 2048 \
  --host 127.0.0.1 --port 8115 --name lfm-k35 \
  --log $HOME/.pm2/logs/lfm-k35-requests.log
