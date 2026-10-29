#!/usr/bin/env bash
# e18 microbench batch: dense groups + ceiling + slice, one GPU job (rules: one job per iteration).
# usage: e18_iter.sh <tag>   (runs in one GPU job on GPUs 1,3)
set -u
TAG=${1:-base}
P=$HOME/ML_projects/dsv41flash/20261019_truss_port/data/pack
B=$HOME/trellis-kernel/build-dsv41
L=/tmp/e18_$TAG
mkdir -p "$L"
GRP="blk.21.attn_q_b.weight blk.21.attn_output_b.weight blk.21.attn_q_a.weight+blk.21.attn_kv.weight blk.21.ffn_gate_shexp.weight+blk.21.ffn_up_shexp.weight blk.21.ffn_down_shexp.weight"
echo "== dense bench $TAG"
$B/tk-bench-trellis-dense $P/ds41-dense-x3.gguf $GRP 2>&1 | tee "$L/dense.log" | grep -E "group|blk"
echo "== ceiling $TAG"
$B/tk-bench-ceiling 2>&1 | tee "$L/ceiling.log" | grep -E "mul1|stream" | head -30
echo "== slice $TAG"
SLICE_GPU=${SLICE_GPU:-0} SLICE_LOG=$L bash ~/trellis-kernel/scripts/ds41_slice.sh $TAG 2>&1 | tee "$L/slice.log" | grep -E "^decode|decode sections|== "
echo "== done $TAG"
