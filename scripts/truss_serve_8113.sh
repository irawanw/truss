#!/usr/bin/env bash
# TRUSS OpenAI-compatible server on llama-paw's port (8113), physical GPU 2 only, 256K context, the bench config of
# TRACKER #85 (trellis CPU tier, Strata split, PCIe share 0.2, hints 2, 22 pinned threads, Strata's expert profile).
# Thinking: the GGUF chat template defaults to enable_thinking on, reasoning_effort xhigh (clients may override with
# chat_template_kwargs). pm2: pm2 start scripts/truss_serve_8113.config.js (only one of this and paw-x3-8113 can run).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=2
export TRUSS_CPU_TRELLIS=1 TRUSS_CPU_DYNAMIC=1 TRUSS_PCIE_FRAC=0.2 TRUSS_HINT_K=2 TRUSS_CPU_THREADS=22 TRUSS_CPU_PIN=1
export TRUSS_KV_INT8=1   # Strata's int8 KV (TRACKER #87): +2,340 resident experts at 256K
D=$HOME/ML_projects/flashnext
exec python3 -m server.app \
  --model $D/20260918_ngram_q8/data/qwen38-flash-next-paw-x3-q8_0-00001-of-00002.gguf \
  --tokenizer /data/www/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/models/Qwen3.8-27B-EXL3-3.5bpw/tokenizer.json \
  --n-ctx 262144 --chunk 4096 \
  --expert-usage data/usage_strata_rank.f32 \
  --mtp $D/20260930_truss_tg/data/mtp_x3/flashnext-mtp-x3k3.gguf --drafts 3 \
  --draft-vocab data/draft_vocab_en.bin --draft-min-p 0.5 \
  --host 127.0.0.1 --port 8113 --name flash-next-truss \
  --log $HOME/.pm2/logs/truss-8113-requests.log
