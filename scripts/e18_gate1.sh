#!/usr/bin/env bash
# E18 gate 1: bisect td_gemv on q_b at n=4 (brief "Past failures to avoid"). One GPU job.
# variants: lo = loads only (decode replaced by a cheap use of the words), reg = decode from registers no weight loads,
# nomma = no mma/fold. base = production (TD_BISECT=0).
set -u
P=$HOME/ML_projects/dsv41flash/20261019_truss_port/data/pack
L=${1:-/tmp/e18_gate1}
mkdir -p "$L"
GRP="blk.21.attn_q_b.weight blk.21.attn_output_b.weight"
for v in base lo reg nomma; do
  case $v in base) B=$HOME/trellis-kernel/build-dsv41;; *) B=$HOME/trellis-kernel/build-dsv41-$v;; esac
  echo "== $v"
  $B/tk-bench-trellis-dense $P/ds41-dense-x3.gguf $GRP 2>&1 | tee "$L/$v.log" | grep -E "^blk|^group|sum"
done
echo "== gate1 done"
