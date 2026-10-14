#!/usr/bin/env bash
# Engine golden: run engine_golden under every engine configuration and write one file per configuration.
#   tests/golden/run.sh <build dir> <out dir>            run all configurations
#   tests/golden/run.sh compare <dir a> <dir b>          byte-compare two runs (exit 0 = identical)
# The model is the 8-layer Flash-Next slice ($S, X3 experts) plus the MTP block; configurations cover the expert
# store paths (all resident, cold experts streamed and fetched, the CPU tier), both activation formats, int8 K/V,
# two sequence slots, a layer split on one device and speculative decoding. Each must be identical run to run on
# one build (checked by running the old build twice) before it can gate a refactor.
set -u
if [ "${1:-}" = compare ]; then
  rc=0
  for f in "$2"/*.bin; do
    b="$3/$(basename "$f")"
    if cmp -s "$f" "$b"; then echo "IDENTICAL $(basename "$f")"; else echo "DIFFER    $(basename "$f")"; rc=1; fi
  done
  exit $rc
fi
B=$1; O=$2
S=${S:-/mnt/hdd/ml/flashnext/20260930_truss_cp2/data/slice8/flashnext-x3-slice8-00001-of-00002.gguf}
MTP=${MTP:-$HOME/ML_projects/flashnext/20261014_engine_refactor/data/mtp_slice8.gguf}   # blk.48 renumbered blk.8
TOK=${TOK:-$HOME/ML_projects/flashnext/20261014_engine_refactor/data/chunk8_256.i32}
mkdir -p "$O"
run() {   # name, then env assignments; mtp args from $ARGS
  local name=$1; shift
  [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]] && return 0   # ONLY="a b": just those configurations
  if env "$@" "$B/engine_golden" "$S" "$TOK" "$O/$name.bin" ${ARGS:-} > "$O/$name.log" 2>&1; then
    echo "ok   $name"
  else
    echo "FAIL $name (see $O/$name.log)"; tail -3 "$O/$name.log"
  fi
}
run base       TRUSS_X=0
run q8         TRUSS_ACT=q8
run cold       TRUSS_EXPERT_GB=3 TRUSS_RING_GB=0.5
run cold_q8    TRUSS_EXPERT_GB=3 TRUSS_RING_GB=0.5 TRUSS_ACT=q8
run cpu        TRUSS_EXPERT_GB=3 TRUSS_RING_GB=0.5 TRUSS_CPU_TRELLIS=1 TRUSS_CPU_DYNAMIC=1 TRUSS_PCIE_FRAC=0.2 TRUSS_CPU_THREADS=8
run kv8_slots  TRUSS_KV_INT8=1 TRUSS_SLOTS=2
run split      TRUSS_LAYER_SPLIT=4 TRUSS_LAYER_SPLIT_DEVICE=-1
ARGS="$MTP 3 0.5"
run mtp        TRUSS_X=0
run mtp_cold   TRUSS_EXPERT_GB=3 TRUSS_RING_GB=0.5
run mtp_split  TRUSS_LAYER_SPLIT=4 TRUSS_LAYER_SPLIT_DEVICE=-1 TRUSS_SLOTS=2
