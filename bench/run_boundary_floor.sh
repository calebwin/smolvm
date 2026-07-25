#!/usr/bin/env bash
# Measure the cost left when either the daemon or guest shim acknowledges but
# does not execute LaunchKernel requests. Wrong-results diagnostic.
set -euo pipefail

S="${SMOLVM_BIN:-$HOME/smolvm/smolvm}"
PACK="${SMOLVM_TEST_PACK:-$HOME/qlora-baked.smolmachine}"
SOCK="${SMOLVM_CUDA_SOCK:-/tmp/smolvm/cuda-daemon.sock}"
ROOT="${BOUNDARY_FLOOR_ROOT:-$HOME/boundary_floor}"
ITERS="${BOUNDARY_FLOOR_ITERS:-200000}"

export SMOLVM_LIB_DIR="${SMOLVM_LIB_DIR:-$HOME/smolvm/lib/linux-x86_64}"
rm -rf "$ROOT"
mkdir -p "$ROOT"

run_arm() {
    local arm="$1"
    local co="$ROOT/$arm"
    local vm="bf-$arm"
    mkdir -p "$co"
    cp "${BOUNDARY_FLOOR_SCRIPT:-$HOME/bench/launch_rate.py}" "$co/probe.py"
    "$S" machine rm --name "$vm" --force >/dev/null 2>&1 || true
    rm -f "$SOCK"

    local daemon_env=(
        SMOLVM_CUDA_DAEMON_IDLE_SECS=0
        SMOLVM_CUDA_RPC_STATS=1
        RUST_LOG=warn
    )
    if [[ "$arm" == "host-noop" ]]; then
        daemon_env+=(SMOLVM_CUDA_NOOP_LAUNCHES=1)
    fi
    env "${daemon_env[@]}" "$S" _cuda-daemon "$SOCK" >"$co/daemon.log" 2>&1 &
    local daemon_pid=$!

    for _ in $(seq 1 100); do
        [[ -S "$SOCK" ]] && break
        sleep 0.1
    done
    local guest_noop=""
    if [[ "$arm" == "client-noop" ]]; then
        guest_noop="SMOLVM_CUDA_CLIENT_NOOP_LAUNCHES=1"
    fi
    "$S" machine create --name "$vm" --cuda --net -v "$co:/opt/coord:rw" \
        --from "$PACK" --storage 30 --overlay 20 -- \
        sh -c "export COORD=/opt/coord FORK=0 LEARNER_ID=$arm LR_ITERS=$ITERS \
        $guest_noop; \
        /home/ubuntu/ptwork/bin/python /opt/coord/probe.py 2>>/opt/coord/g.err" >/dev/null
    env SMOLVM_CUDA_SHARED=1 "$S" machine start --name "$vm" >/dev/null

    for _ in $(seq 1 240); do
        [[ -f "$co/lr_$arm.json" ]] && break
        sleep 0.5
    done
    cat "$co/lr_$arm.json"
    grep '\[rpc-stats\]' "$co/daemon.log" | tail -1 || true

    "$S" machine rm --name "$vm" --force >/dev/null 2>&1 || true
    kill "$daemon_pid" >/dev/null 2>&1 || true
    wait "$daemon_pid" >/dev/null 2>&1 || true
}

run_arm normal
run_arm host-noop
run_arm client-noop
