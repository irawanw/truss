#!/usr/bin/env bash
# e18 profile batch: ncu td_gemv (q_b n=4), ncu ceiling (K3 WNT8), SASS of td_gemv. One GPU job.
set -u
P=$HOME/ML_projects/dsv41flash/20261019_truss_port/data/pack
B=$HOME/trellis-kernel/build-dsv41
L=${1:-/tmp/e18_ncu}
mkdir -p "$L"
NCU="sudo -n /usr/local/cuda/bin/ncu"
echo "== ncu td_gemv q_b n=4"
$NCU --set full -k regex:td_gemv --launch-skip 5 --launch-count 1 -o "$L/td_qb" \
  $B/tk-bench-trellis-dense $P/ds41-dense-x3.gguf blk.21.attn_q_b.weight > "$L/td_qb_run.log" 2>&1
sudo -n chown -R "$(id -u):$(id -g)" "$L"
/usr/local/cuda/bin/ncu --import "$L/td_qb.ncu-rep" --page details > "$L/td_qb.txt" 2>&1
echo "== ncu ceiling K3 WNT8"
$NCU --set full -k regex:decode_bench --launch-skip 72 --launch-count 1 -o "$L/ceil" \
  $B/tk-bench-ceiling > "$L/ceil_run.log" 2>&1
sudo -n chown -R "$(id -u):$(id -g)" "$L"
/usr/local/cuda/bin/ncu --import "$L/ceil.ncu-rep" --page details > "$L/ceil.txt" 2>&1
echo "== sass td_gemv"
/usr/local/cuda/bin/cuobjdump -sass $B/tk-bench-trellis-dense > "$L/sass_all.txt" 2>&1
awk '/Function : .*td_gemv/{p=1} p{print} /^$/&&p{c++} c>2{p=0;c=0}' "$L/sass_all.txt" | head -400 > "$L/sass_td.txt"
echo "== done ncu"
