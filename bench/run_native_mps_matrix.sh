#!/usr/bin/env bash
# Paired ordinary-context versus private uncapped MPS native-worker benchmark.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
N="${N:-8}"
STEPS="${STEPS:-50}"
CPUS="${CPUS:-2}"
REPS="${REPS:-1}"
BATCH="${BATCH:-1}"
MAXSEQ="${MAXSEQ:-256}"
TIMEOUT="${TIMEOUT:-1800}"
WORKLOAD="${WORKLOAD:-workload_grpo.py}"
export NATIVE_REFERENCE_WARMUP="${NATIVE_REFERENCE_WARMUP:-1}"
RUN_CONTROL="${RUN_CONTROL:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
MPS_PIPE="${MPS_PIPE:-/tmp/native-mps-${UID}-${RUN_TAG}}"
MPS_LOG="${MPS_LOG:-$HOME/bench_mps/native-${RUN_TAG}}"

install -d -m 700 "$MPS_PIPE" "$MPS_LOG"

mps_started=0
stop_mps() {
    if [[ "$mps_started" == "1" ]]; then
        CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
        CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" \
            sh -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 || true
        mps_started=0
    fi
}
trap stop_mps EXIT INT TERM

common=(
    --arm native
    --n "$N"
    --steps "$STEPS"
    --reps "$REPS"
    --cpus "$CPUS"
    --batch "$BATCH"
    --maxseq "$MAXSEQ"
    --timeout "$TIMEOUT"
    --workload "$WORKLOAD"
)

if [[ "$RUN_CONTROL" == "1" ]]; then
    env -u CUDA_MPS_PIPE_DIRECTORY \
        -u CUDA_MPS_LOG_DIRECTORY \
        -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
        "$BENCH_DIR/bench.sh" "${common[@]}" --tag "grpo-native-nomps-${RUN_TAG}"
fi

export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE"
export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG"
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
nvidia-cuda-mps-control -d
mps_started=1

"$BENCH_DIR/bench.sh" "${common[@]}" --tag "grpo-native-mps-${RUN_TAG}"

stop_mps
echo "MPS logs: $MPS_LOG"
