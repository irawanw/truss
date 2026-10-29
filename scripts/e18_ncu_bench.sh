#!/usr/bin/env bash
# e18 F.1: ncu on the served bench, metrics bytes/time for window_kernel / td_gemv / gemv_mma_a16.
set -u
cd "$HOME/trellis-kernel"
source scripts/ds41_env.sh >/dev/null 2>&1
L=${1:-/tmp/e18_ncu_bench}
mkdir -p "$L"
NCU="sudo -n -E /usr/local/cuda/bin/ncu"   # -E: sudo drops TRUSS_EXPERT_OVERLAY/HOME; the bench needs them
$NCU --metrics dram__bytes_read.sum,gpu__time_duration.sum \
  -k regex:window_kernel\|td_gemv\|gemv_mma_a16 --launch-skip 2000 --launch-count 300 \
  --csv -o "$L/bench" \
  python3 tools/tk-bench/arms.py "${DS41_ARGS[@]}" --tasks code --no-check --repeat 1 --tokens 192 \
  > "$L/run.log" 2>&1
RC=$?
sudo -n -E chown -R "$(id -u):$(id -g)" "$L"
/usr/local/cuda/bin/ncu --import "$L/bench.ncu-rep" --csv --page raw > "$L/metrics.csv" 2>&1
echo "== rc=$RC metrics=$L/metrics.csv"
