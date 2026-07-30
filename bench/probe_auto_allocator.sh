#!/usr/bin/env bash
set -euo pipefail

SMOLVM="${SMOLVM:?set SMOLVM to the candidate binary}"
SMOLVM_AGENT_ROOTFS="${SMOLVM_AGENT_ROOTFS:?set SMOLVM_AGENT_ROOTFS to the candidate rootfs}"
SMOLVM_LIB_DIR="${SMOLVM_LIB_DIR:-$HOME/smolvm/lib/linux-x86_64}"
PACK="${PACK:-$HOME/grpo-vllm-toolchain.smolmachine}"
COORD_HOST="${COORD_HOST:?set COORD_HOST to a new result directory}"
MPS_PIPE="${MPS_PIPE:-/tmp/smolvm-mps-1000-2190482}"
SOCKET="/tmp/smolvm/cuda-daemon.sock"
NAMES=(auto-alloc-pool auto-alloc-ordinary auto-alloc-optout)

export SMOLVM_AGENT_ROOTFS SMOLVM_LIB_DIR
daemon_pid=""

cleanup() {
    for name in "${NAMES[@]}"; do
        "$SMOLVM" machine rm --name "$name" --cascade >/dev/null 2>&1 || true
    done
    if [[ -n "$daemon_pid" ]]; then kill "$daemon_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT HUP INT TERM

[[ -d "$MPS_PIPE" ]] || { echo "missing MPS pipe: $MPS_PIPE" >&2; exit 1; }
[[ -f "$PACK" ]] || { echo "missing pack: $PACK" >&2; exit 1; }
mkdir "$COORD_HOST"
rm -f "$SOCKET"

env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
    SMOLVM_CUDA_DAEMON_IDLE_SECS=0 \
    "$SMOLVM" _cuda-daemon "$SOCKET" >"$COORD_HOST/daemon.log" 2>&1 &
daemon_pid=$!
for _index in $(seq 1 100); do [[ -S "$SOCKET" ]] && break; sleep 0.1; done
[[ -S "$SOCKET" ]] || { echo "CUDA daemon failed to start" >&2; exit 1; }

run_case() {
    local name="$1"
    local mode="$2"
    local override="${3:-}"
    local result_dir="$COORD_HOST/$name"
    mkdir "$result_dir"

    create_args=(machine create --name "$name" --cuda --cpus 2 --mem 4096
        -v "$result_dir:/opt/result:rw" --from "$PACK" --storage 30 --overlay 20)
    if [[ -n "$override" ]]; then
        create_args+=(-e "$override")
    fi
    create_args+=(-- sh -c
        "printf '%s|%s\\n' \"\${PYTORCH_CUDA_ALLOC_CONF-}\" \"\${PYTORCH_ALLOC_CONF-}\" > /opt/result/allocator; sleep 120")
    "$SMOLVM" "${create_args[@]}" >/dev/null

    start_args=(machine start --name "$name")
    if [[ "$mode" == "pool" ]]; then
        start_args+=(--forkable --fork-pool-size 2)
    fi
    env SMOLVM_CUDA_SHARED=1 "$SMOLVM" "${start_args[@]}" >/dev/null
    for _index in $(seq 1 60); do
        [[ -f "$result_dir/allocator" ]] && break
        sleep 0.2
    done
    [[ -f "$result_dir/allocator" ]] || { echo "$name did not record its environment" >&2; exit 1; }
    "$SMOLVM" machine exec --name "$name" -- sh -c \
        "printf '%s|%s\\n' \"\${PYTORCH_CUDA_ALLOC_CONF-}\" \"\${PYTORCH_ALLOC_CONF-}\" > /opt/result/allocator-exec" \
        >/dev/null
    [[ -f "$result_dir/allocator-exec" ]] || { echo "$name exec did not record its environment" >&2; exit 1; }
    "$SMOLVM" machine rm --name "$name" --cascade >/dev/null
}

run_case auto-alloc-pool pool
run_case auto-alloc-ordinary ordinary
run_case auto-alloc-optout pool SMOLVM_CUDA_EXPANDABLE_SEGMENTS=off

pool_value=$(<"$COORD_HOST/auto-alloc-pool/allocator")
ordinary_value=$(<"$COORD_HOST/auto-alloc-ordinary/allocator")
optout_value=$(<"$COORD_HOST/auto-alloc-optout/allocator")
pool_exec_value=$(<"$COORD_HOST/auto-alloc-pool/allocator-exec")
ordinary_exec_value=$(<"$COORD_HOST/auto-alloc-ordinary/allocator-exec")
optout_exec_value=$(<"$COORD_HOST/auto-alloc-optout/allocator-exec")

[[ "$pool_value" == "expandable_segments:True|" ]]
[[ "$ordinary_value" == "|" ]]
[[ "$optout_value" == "|" ]]
[[ "$pool_exec_value" == "expandable_segments:True|" ]]
[[ "$ordinary_exec_value" == "|" ]]
[[ "$optout_exec_value" == "|" ]]
printf 'pool=%s\npool_exec=%s\nordinary=%s\nordinary_exec=%s\noptout=%s\noptout_exec=%s\n' \
    "$pool_value" "$pool_exec_value" "$ordinary_value" "$ordinary_exec_value" \
    "$optout_value" "$optout_exec_value"
