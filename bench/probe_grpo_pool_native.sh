#!/usr/bin/env bash
set -euo pipefail

N="${N:-4}"
STEPS="${STEPS:-20}"
MODEL="${MODEL:-unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit}"
POOL_GPU_UTIL="${POOL_GPU_UTIL:-0.14}"
ADAPTER_EXPORT_MODE="${ADAPTER_EXPORT_MODE:-peft}"
MPS_PIPE="${MPS_PIPE:-/tmp/smolvm-mps-1000-2190482}"
BENCH_DIR="${BENCH_DIR:-$HOME/bench}"
COORD="${COORD:?set COORD to a new result directory}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$HOME/ptwork/bin/python}"
POOL_PYTHON="${POOL_PYTHON:-$HOME/rlwork/bin/python}"

SERVER_PID=""
SAMPLER_PID=""
LEARNER_PIDS=()

cleanup() {
    touch "$COORD/sample-stop" 2>/dev/null || true
    if [[ -n "$SAMPLER_PID" ]]; then wait "$SAMPLER_PID" 2>/dev/null || true; fi
    for pid in "${LEARNER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT HUP INT TERM

[[ -d "$MPS_PIPE" ]] || { echo "missing MPS pipe: $MPS_PIPE" >&2; exit 1; }
mkdir "$COORD"
mkdir "$COORD/pool"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "$used" -lt 500 ]] || { echo "GPU is not clean: ${used} MiB" >&2; exit 1; }

env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
    POOL_ROOT="$COORD/pool" MODEL="$MODEL" MAXSEQ=256 MAX_POLICIES="$N" \
    VLLM_GPU_MEMORY_UTILIZATION="$POOL_GPU_UTIL" BATCH_WINDOW_MS=50 \
    "$POOL_PYTHON" "$BENCH_DIR/probe_grpo_pool_server.py" \
    >"$COORD/pool/server.log" 2>&1 &
SERVER_PID=$!
for _index in $(seq 1 180); do
    [[ -f "$COORD/pool/ready.json" ]] && break
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$COORD/pool/server.log"; exit 1; }
    sleep 1
done
[[ -f "$COORD/pool/ready.json" ]] || { echo "rollout server did not become ready" >&2; exit 1; }

(
    peak=0
    while [[ ! -f "$COORD/sample-stop" ]]; do
        value=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
        (( value > peak )) && peak=$value
        sleep 0.2
    done
    echo "$peak" >"$COORD/peak-gpu-mib"
) &
SAMPLER_PID=$!

for learner in $(seq 0 $((N - 1))); do
    env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" HF_HOME="$HOME/hf" HF_HUB_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false POOL_ROOT="$COORD/pool" COORD="$COORD" \
        FORK=0 WAIT_FOR_GO=1 LEARNER_ID="$learner" STEPS="$STEPS" BATCH=4 NGEN=4 \
        MAXSEQ=256 MODEL="$MODEL" ADAPTER_EXPORT_MODE="$ADAPTER_EXPORT_MODE" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
        "$TRAIN_PYTHON" "$BENCH_DIR/probe_grpo_pool_trainer.py" \
        >"$COORD/learner-$learner.log" 2>&1 &
    LEARNER_PIDS+=("$!")
done

deadline=$(( $(date +%s) + 600 ))
while [[ $(find "$COORD" -maxdepth 1 -name 'native_ready_*.json' | wc -l) -lt "$N" ]]; do
    [[ $(date +%s) -lt $deadline ]] || { echo "native preload timeout" >&2; exit 1; }
    for learner in $(seq 0 $((N - 1))); do
        if grep -q 'Traceback (most recent call last)' "$COORD/learner-$learner.log" 2>/dev/null; then
            tail -100 "$COORD/learner-$learner.log"
            exit 1
        fi
    done
    sleep 1
done

date +%s.%N >"$COORD/release-time"
touch "$COORD/go"
deadline=$(( $(date +%s) + 900 ))
while [[ $(find "$COORD" -maxdepth 1 -name 'pool_trainer_*.json' | wc -l) -lt "$N" ]]; do
    [[ $(date +%s) -lt $deadline ]] || { echo "native trainer timeout" >&2; exit 1; }
    for learner in $(seq 0 $((N - 1))); do
        if grep -q 'Traceback (most recent call last)' "$COORD/learner-$learner.log" 2>/dev/null; then
            tail -100 "$COORD/learner-$learner.log"
            exit 1
        fi
    done
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$COORD/pool/server.log"; exit 1; }
    sleep 1
done
date +%s.%N >"$COORD/done-time"
touch "$COORD/sample-stop"
wait "$SAMPLER_PID"
SAMPLER_PID=""
for pid in "${LEARNER_PIDS[@]}"; do wait "$pid"; done
LEARNER_PIDS=()

python3 - "$COORD" "$N" <<'PY'
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
