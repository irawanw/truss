#!/usr/bin/env bash
# Serve PAW-125B-FLASH-NEXT-X3.1 with the best settings measured so far (262,144 context), on one 24 GB GPU + CPU
# or on two 24 GB GPUs + CPU (layer split: the first GPU runs layers 24-47, the output head and the MTP drafter, the
# second runs layers 0-23; prompt chunks are pipelined across both).
# Usage:  MODEL_DIR=/path/with/the/two/ggufs TOKENIZER=/path/tokenizer.json tools/serve/serve_flashnext_x31.sh
# Optional: GPU=<index> (one GPU) or GPUS=<a>,<b> (two GPUs), PORT=8080 NCTX=262144 THREADS=<n>
#           VIT=<mmproj gguf> to accept images, VIT_GPU=<index into CUDA_VISIBLE_DEVICES, default the last card>,
#           IMAGES=1 to also allow local image paths (default: data URI and http(s) only)
#           DRY=1 (print the command, do not start)
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${MODEL_DIR:?directory holding flashnext-x31-0000{1,2}-of-00002.gguf and flashnext-mtp-x3k3.gguf (put it on NVMe)}"
: "${TOKENIZER:?path to tokenizer.json (the one in Qwen/Qwen3.8-Flash-Next)}"
GPU=${GPU:-0}
PORT=${PORT:-8080}
NCTX=${NCTX:-262144}
# CPU threads for the expert tier: physical cores minus 2 (the rest keep the driver and the OS responsive)
PHYS=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
THREADS=${THREADS:-$(( PHYS > 6 ? PHYS - 2 : 4 ))}

[ -f "$MODEL_DIR/flashnext-x31-00002-of-00002.gguf" ] || { echo "missing the n-gram shard $MODEL_DIR/flashnext-x31-00002-of-00002.gguf" >&2; exit 1; }
[ -f "$MODEL_DIR/flashnext-x31-00001-of-00002.gguf" ] || { echo "missing $MODEL_DIR/flashnext-x31-00001-of-00002.gguf" >&2; exit 1; }
[ -f build/libtruss.so ] || { echo "build first: cmake -S . -B build -G Ninja && cmake --build build" >&2; exit 1; }
case "$MODEL_DIR" in /mnt/*|/media/*) echo "warning: $MODEL_DIR looks like removable/slow storage; the n-gram table is read from disk while decoding" >&2;; esac

DRAFTS=3
if [ -n "${GPUS:-}" ]; then   # two GPUs: layer split, 7 drafts (more rows per pass are cheap with two GPUs)
  export CUDA_VISIBLE_DEVICES=$GPUS
  export TRUSS_LAYER_SPLIT=24 TRUSS_LAYER_SPLIT_DEVICE=1 TRUSS_HINT_MIN_ROWS=8
  export TRUSS_HC_SILU=1 TRUSS_HC_COLLAPSE=1 TRUSS_ROUTE_PUBLISH=1   # fused decode kernels (bit-exact)
  DRAFTS=7
else
  export CUDA_VISIBLE_DEVICES=$GPU
fi
export TRUSS_CPU_TRELLIS=1 TRUSS_CPU_DYNAMIC=1 TRUSS_PCIE_FRAC=0.2 TRUSS_HINT_K=3
export TRUSS_CPU_THREADS=$THREADS TRUSS_CPU_PIN=1
export TRUSS_KV_INT8=1 TRUSS_KV_LEND=1 TRUSS_ADMIT_IDLE=64 TRUSS_FETCH_PROMPT=96 TRUSS_SPLIT_ROWS=2048

CMD=(python3 -m server.app
  --model "$MODEL_DIR/flashnext-x31-00001-of-00002.gguf" --tokenizer "$TOKENIZER"
  --n-ctx "$NCTX" --chunk 8192
  --expert-usage data/usage_strata_rank.f32
  --mtp "$MODEL_DIR/flashnext-mtp-x3k3.gguf" --drafts "$DRAFTS"
  --draft-vocab data/draft_vocab_en.bin --draft-min-p 0.5
  --host 127.0.0.1 --port "$PORT")

if [ -n "${VIT:-}" ]; then   # image input: the ViT (mmproj) on its own card when there is one
  [ -f "$VIT" ] || { echo "missing $VIT (hf download lackonendes/PAW-125B-FLASH-NEXT-X3.1 flashnext-mmproj-f16.gguf)" >&2; exit 1; }
  if [ -z "${VIT_GPU:-}" ]; then [ -n "${GPUS:-}" ] && VIT_GPU=${GPUS##*,} || VIT_GPU=$GPU; fi
  CMD+=(--mmproj "$VIT" --mmproj-device "$VIT_GPU")
  [ -n "${IMAGES:-}" ] && CMD+=(--allow-local-images)
fi

echo "GPU(s) ${GPUS:-$GPU}, $THREADS CPU threads, context $NCTX, $DRAFTS drafts, port $PORT${VIT:+, ViT $VIT on device $VIT_GPU}"
[ -n "${DRY:-}" ] && { echo "${CMD[@]}"; exit 0; }
exec "${CMD[@]}"
