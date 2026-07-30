#!/usr/bin/env bash
set -euo pipefail

N="${N:-4}"
STEPS="${STEPS:-5}"
VCPU="${VCPU:-4}"
VMEM="${VMEM:-16384}"
MODEL="${MODEL:-unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit}"
POOL_GPU_UTIL="${POOL_GPU_UTIL:-0.14}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:False}"
AUTO_ALLOCATOR="${AUTO_ALLOCATOR:-0}"
DECLARE_POOL="${DECLARE_POOL:-1}"
ADAPTER_EXPORT_MODE="${ADAPTER_EXPORT_MODE:-peft}"
DAEMON_LINGER_SECS="${DAEMON_LINGER_SECS:-0}"
MPS_PIPE="${MPS_PIPE:-/tmp/smolvm-mps-1000-2190482}"
SMOLVM="${SMOLVM:-$HOME/smolvm-vmm-cow-prod-bin}"
SMOLVM_LIB_DIR="${SMOLVM_LIB_DIR:-$HOME/smolvm/lib/linux-x86_64}"
PACK="${PACK:-$HOME/grpo-vllm-toolchain.smolmachine}"
BENCH_DIR="${BENCH_DIR:-$HOME/bench}"
COORD_HOST="${COORD_HOST:?set COORD_HOST to a new result directory}"
SOCKET="/tmp/smolvm/cuda-daemon.sock"

export SMOLVM_LIB_DIR
DAEMON_PID=""
SERVER_PID=""
SAMPLER_PID=""

cleanup() {
    touch "$COORD_HOST/sample-stop" 2>/dev/null || true
    if [[ -n "$SAMPLER_PID" ]]; then wait "$SAMPLER_PID" 2>/dev/null || true; fi
    "$SMOLVM" machine rm --name bench-g --cascade >/dev/null 2>&1 || true
    if [[ "$DAEMON_LINGER_SECS" != "0" ]]; then sleep "$DAEMON_LINGER_SECS"; fi
    if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; fi
    if [[ -n "$DAEMON_PID" ]]; then kill "$DAEMON_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT HUP INT TERM

[[ -d "$MPS_PIPE" ]] || { echo "missing MPS pipe: $MPS_PIPE" >&2; exit 1; }
[[ -f "$PACK" ]] || { echo "missing pack: $PACK" >&2; exit 1; }
[[ -f "$BENCH_DIR/probe_grpo_pool_server.py" ]] || { echo "missing server probe" >&2; exit 1; }
[[ -f "$BENCH_DIR/probe_grpo_pool_trainer.py" ]] || { echo "missing trainer probe" >&2; exit 1; }
mkdir "$COORD_HOST"
mkdir "$COORD_HOST/pool"
cp "$BENCH_DIR/probe_grpo_pool_trainer.py" "$COORD_HOST/workload.py"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "$used" -lt 500 ]] || { echo "GPU is not clean: ${used} MiB" >&2; exit 1; }
if "$SMOLVM" machine list 2>/dev/null | awk '$1 ~ /^bench-/ {found=1} END {exit !found}'; then
    echo "bench machine already exists" >&2
    exit 1
fi
rm -f "$SOCKET"

env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
    SMOLVM_CUDA_DAEMON_IDLE_SECS=0 \
    SMOLVM_CUDA_FORK_SHARE_WEIGHTS=1 \
    SMOLVM_CUDA_FORK_WORKERS=1 \
    SMOLVM_CUDA_FORK_ISOLATE=1 \
    RUST_LOG=warn,smolvm::cuda_daemon=info \
    "$SMOLVM" _cuda-daemon "$SOCKET" >"$COORD_HOST/daemon.log" 2>&1 &
DAEMON_PID=$!
for _index in $(seq 1 100); do [[ -S "$SOCKET" ]] && break; sleep 0.1; done
[[ -S "$SOCKET" ]] || { echo "CUDA daemon failed to start" >&2; exit 1; }

allocator_env="PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_ALLOC_CONF"
if [[ "$AUTO_ALLOCATOR" == "1" ]]; then
    allocator_env=""
fi
guest="export HF_HOME=/home/ubuntu/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false \
POOL_ROOT=/opt/coord/pool COORD=/opt/coord FORK=1 NSLOTS=$N STEPS=$STEPS \
BATCH=4 NGEN=4 MAXSEQ=256 MODEL=$MODEL $allocator_env; \
printf '%s\\n' \"\${PYTORCH_CUDA_ALLOC_CONF-}\" > /opt/coord/pytorch-alloc-conf; \
export ADAPTER_EXPORT_MODE=$ADAPTER_EXPORT_MODE; \
/home/ubuntu/ptwork/bin/python /opt/coord/workload.py >>/opt/coord/golden.log 2>&1"

"$SMOLVM" machine create --name bench-g --cuda --net --cpus "$VCPU" --mem "$VMEM" \
    -v "$COORD_HOST:/opt/coord:rw" --from "$PACK" --storage 30 --overlay 20 \
    -- sh -c "$guest" >/dev/null
start_args=(machine start --forkable --name bench-g)
if [[ "$DECLARE_POOL" == "1" ]]; then
    start_args+=(--fork-pool-size "$N")
fi
env SMOLVM_CUDA_SHARED=1 "$SMOLVM" "${start_args[@]}" >/dev/null

for _index in $(seq 1 360); do
    [[ -f "$COORD_HOST/golden_ready" ]] && break
    kill -0 "$DAEMON_PID" 2>/dev/null || { tail -100 "$COORD_HOST/daemon.log"; exit 1; }
    if grep -q 'Traceback (most recent call last)' "$COORD_HOST/golden.log" 2>/dev/null; then
        tail -100 "$COORD_HOST/golden.log"
        exit 1
    fi
    sleep 1
done
[[ -f "$COORD_HOST/golden_ready" ]] || { echo "golden did not become ready" >&2; exit 1; }

for clone in $(seq 0 $((N - 1))); do
    created=0
    : >"$COORD_HOST/fork-c$clone.log"
    for attempt in 1 2 3; do
        echo "attempt=$attempt" >>"$COORD_HOST/fork-c$clone.log"
        if "$SMOLVM" machine fork --golden bench-g --name "bench-c$clone" --share-weights \
            >>"$COORD_HOST/fork-c$clone.log" 2>&1; then
            created=1
            break
        fi
        if "$SMOLVM" machine list 2>/dev/null | awk -v name="bench-c$clone" '$1 == name {found=1} END {exit !found}'; then
            echo "failed fork left a partial machine; refusing retry" >>"$COORD_HOST/fork-c$clone.log"
            break
        fi
        sleep 1
    done
    if [[ "$created" != "1" ]]; then
        tail -100 "$COORD_HOST/fork-c$clone.log"
        exit 1
    fi
done

nohup env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
    HF_HOME="$HOME/hf" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    POOL_ROOT="$COORD_HOST/pool" MODEL="$MODEL" MAXSEQ=256 MAX_POLICIES="$N" \
    VLLM_GPU_MEMORY_UTILIZATION="$POOL_GPU_UTIL" BATCH_WINDOW_MS=50 \
    "$HOME/rlwork/bin/python" "$BENCH_DIR/probe_grpo_pool_server.py" \
    >"$COORD_HOST/pool/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$COORD_HOST/pool/server.pid"
for _index in $(seq 1 180); do
    [[ -f "$COORD_HOST/pool/ready.json" ]] && break
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$COORD_HOST/pool/server.log"; exit 1; }
    sleep 1
done
[[ -f "$COORD_HOST/pool/ready.json" ]] || { echo "rollout server did not become ready" >&2; exit 1; }

(
    peak=0
    while [[ ! -f "$COORD_HOST/sample-stop" ]]; do
        value=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
        (( value > peak )) && peak=$value
        sleep 0.2
    done
    echo "$peak" >"$COORD_HOST/peak-gpu-mib"
) &
SAMPLER_PID=$!

date +%s.%N >"$COORD_HOST/release-time"
touch "$COORD_HOST/go"
deadline=$(( $(date +%s) + 900 ))
while [[ $(find "$COORD_HOST" -maxdepth 1 -name 'pool_trainer_*.json' | wc -l) -lt "$N" ]]; do
    [[ $(date +%s) -lt $deadline ]] || { echo "trainer timeout" >&2; exit 1; }
    if grep -q 'Traceback (most recent call last)' "$COORD_HOST/golden.log" 2>/dev/null; then
        tail -100 "$COORD_HOST/golden.log"
        exit 1
    fi
    kill -0 "$DAEMON_PID" 2>/dev/null || { tail -100 "$COORD_HOST/daemon.log"; exit 1; }
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$COORD_HOST/pool/server.log"; exit 1; }
    sleep 1
done
date +%s.%N >"$COORD_HOST/done-time"
touch "$COORD_HOST/sample-stop"
wait "$SAMPLER_PID"
SAMPLER_PID=""

python3 - "$COORD_HOST" "$N" <<'PY'
import glob
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
n = int(sys.argv[2])
records = [json.loads(pathlib.Path(path).read_text()) for path in sorted(glob.glob(str(root / "pool_trainer_*.json")))]
batches = [json.loads(line) for line in (root / "pool" / "batches.jsonl").read_text().splitlines()]
release = float((root / "release-time").read_text())
done = float((root / "done-time").read_text())
result = {
    "n": n,
    "completed": len(records),
    "scheduled_s": done - release,
    "peak_gpu_mib": int((root / "peak-gpu-mib").read_text()),
    "steps": sum(record["steps"] for record in records),
    "rollout_tokens": sum(record["rollout_tokens"] for record in records),
    "adapter_export_s": sum(record["adapter_export_s"] for record in records),
    "pool_roundtrip_s": sum(record["pool_roundtrip_s"] for record in records),
    "distinct_initial_adapters": len({record["initial_adapter_sha256"] for record in records}),
    "distinct_final_adapters": len({record["final_adapter_sha256"] for record in records}),
    "changed_adapters": sum(record["initial_adapter_sha256"] != record["final_adapter_sha256"] for record in records),
    "reward_std_max": max(
        (record["reward_std_max"] for record in records if record["reward_std_max"] is not None),
        default=None,
    ),
    "pool_batches": len(batches),
    "pool_requests": sum(batch["requests"] for batch in batches),
    "max_batch_requests": max(batch["requests"] for batch in batches),
    "pool_tokens": sum(batch["tokens"] for batch in batches),
    "pool_busy_s": sum(batch["batch_s"] for batch in batches),
}
result["scheduled_steps_s"] = result["steps"] / result["scheduled_s"]
result["scheduled_rollout_tok_s"] = result["rollout_tokens"] / result["scheduled_s"]
(root / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY

grep 'M2: shared weight ranges' "$COORD_HOST/daemon.log" | tail -1 || true
