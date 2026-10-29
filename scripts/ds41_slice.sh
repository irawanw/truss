#!/usr/bin/env bash
# usage: slice.sh <tag> [env...] : 8-layer DS41 slice, 4-row decode windows, GPU 0
cd ~/ML_projects/dsv41flash/20261019_truss_port; P=data/pack; S=${SLICE_LOG:-/tmp/e18_slice}
tag=$1; shift
env TRUSS_COLD_KEEP=0 TRUSS_SLICE_ROWS=4 TRUSS_ACT=fp16 TRUSS_EXPERT_OVERLAY=$P/ds41-experts.gguf,$P/ds41-engram-L1.gguf,$P/ds41-engram-L14.gguf,$P/ds41-dense-x3.gguf "$@" \
  bash ~/ML_projects/dsv41flash/20261018_poc/scripts/run_watchedN.sh ${SLICE_GPU:-0} $S/slice_$tag ${SLICE_BIN:-$HOME/trellis-kernel/build-dsv41/tk-bench-ds41-slice} $P/ds41-dense.gguf data/dspark/prompt_0.i32 ${SLICE_N:-8} 256 64 256 data/bench_in/usage_calib256.f32 >/dev/null
echo "== $tag"; grep -E "^decode|decode sections" $S/slice_$tag/run.log | cut -c1-200
