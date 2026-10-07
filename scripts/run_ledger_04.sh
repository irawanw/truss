#!/usr/bin/env bash
# Plan v3 0.4 runner: decode bandwidth ledger capture. nsys over ~20 clean decode passes at 150K
# (TRUSS_BENCH_PROFRANGE=1 brackets exactly the decode loops; prefill excluded via capture-range).
# Lock + trap restart. Served env, NO section timers. Output: /tmp/ledger_decode_<ts>.nsys-rep (+ copy to logs dir).
set -u
SELF=/home/green-gpu/ML_projects/flashnext/20261004_selfopt
WT=$SELF/work/agent
NOTIFY=$SELF/data/notify.txt
TS=$(date +%H%M%S)
LOGD=$SELF/data/logs/ledger04_$TS; mkdir -p "$LOGD"
REP=$LOGD/ledger_decode
M=/home/green-gpu/ML_projects/flashnext/20261003_x31/data/flashnext-x31.gguf
MTP=/home/green-gpu/ML_projects/flashnext/20260930_truss_tg/data/mtp_x3/flashnext-mtp-x3k3.gguf
EVAL=/home/green-gpu/ML_projects/flashnext/20261003_x31/data/e1/eval_clean_150k.i32

exec 9>/tmp/selfopt_gpu2.lock
flock 9 || { echo "$(date '+%a %b %d %H:%M %Y') | GPU SKIP paw | 0.4 ledger lock busy" >> "$NOTIFY"; exit 1; }
trap 'pm2 start truss-x31-brain >/dev/null 2>&1; echo "$(date "+%a %b %d %H:%M %Y") | GPU END paw | 0.4 decode ledger capture (trap restart)" >> "$NOTIFY"' EXIT
echo "$(date '+%a %b %d %H:%M %Y') | GPU START paw | 0.4 decode bandwidth ledger nsys@150K ~20 passes brain-down ~4min" >> "$NOTIFY"
pm2 stop truss-x31-brain >/dev/null
sleep 3

cd /home/green-gpu/trellis-kernel
CUDA_VISIBLE_DEVICES=1 TRUSS_BENCH_PLAIN=0 TRUSS_BENCH_PROFRANGE=1 \
TRUSS_KV_INT8=1 TRUSS_KV_LEND=1 TRUSS_ADMIT_IDLE=64 TRUSS_CPU_DYNAMIC=1 TRUSS_CPU_PIN=1 TRUSS_CPU_THREADS=22 \
TRUSS_CPU_TRELLIS=1 TRUSS_FETCH_PROMPT=96 TRUSS_HINT_K=3 TRUSS_LOAD_TIMES=1 TRUSS_PCIE_FRAC=0.2 TRUSS_SPLIT_ROWS=2048 \
  /usr/local/cuda-12.6/bin/nsys profile -c cudaProfilerApi --capture-range-end=stop --stats=false -o "$REP" -f true \
  ./build/tk-bench-spec \
    "$M" "$MTP" "@$EVAL" 60 3 data/usage_strata_rank.f32 262144 8192 data/draft_vocab_en.bin - 0.5 \
    > "$LOGD/bench.log" 2>&1
RC=$?
pm2 start truss-x31-brain >/dev/null
trap - EXIT
echo "$(date '+%a %b %d %H:%M %Y') | GPU END paw | 0.4 decode ledger rc=$RC rep=$REP.nsys-rep log=$LOGD/bench.log" >> "$NOTIFY"
