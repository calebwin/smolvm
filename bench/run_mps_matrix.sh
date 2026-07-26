#!/usr/bin/env bash
# Paired ordinary-context versus NVIDIA MPS fork-pool benchmark.
#
# This runner owns a private MPS controller directory and never modifies a
# controller in the default/system directory. It intentionally leaves MPS
# active-thread percentage uncapped: the DPO investigation found NaNs at 33%
# even in the native arm, so resource caps are not generically transparent.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
N="${N:-8}"
STEPS="${STEPS:-50}"
VCPU="${VCPU:-2}"
REPS="${REPS:-1}"
BATCH="${BATCH:-8}"
MAXSEQ="${MAXSEQ:-1024}"
TIMEOUT="${TIMEOUT:-1800}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
MPS_PIPE="${MPS_PIPE:-/tmp/smolvm-mps-${UID}-${RUN_TAG}}"
MPS_LOG="${MPS_LOG:-$HOME/bench_mps/$RUN_TAG}"

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
    --arm fork
    --n "$N"
    --steps "$STEPS"
    --reps "$REPS"
    --cpus 4
    --vcpu "$VCPU"
    --batch "$BATCH"
    --maxseq "$MAXSEQ"
    --timeout "$TIMEOUT"
    --share on
)

echo "CONTROL: MPS off"
env -u CUDA_MPS_PIPE_DIRECTORY \
    -u CUDA_MPS_LOG_DIRECTORY \
    -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
    "$BENCH_DIR/bench.sh" "${common[@]}"

export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE"
export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG"
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
nvidia-cuda-mps-control -d
mps_started=1

echo "MPS: private controller, uncapped clients"
echo get_default_active_thread_percentage | nvidia-cuda-mps-control
"$BENCH_DIR/bench.sh" "${common[@]}"

stop_mps
echo "MPS logs: $MPS_LOG"
