#!/usr/bin/env bash
# Repeated fresh-clone CUDA-graph reliability matrix for the H100 testbed.
set -euo pipefail

S="${SMOLVM_BIN:-$HOME/smolvm/smolvm}"
PACK="${SMOLVM_TEST_PACK:-$HOME/qlora-baked.smolmachine}"
TRIALS="${GRAPH_TRIALS:-12}"
REPLAYS="${GRAPH_TRIAL_REPLAYS:-1}"
CO="${GRAPH_TRIAL_COORD:-$HOME/coord_graph_fresh}"
SOCK="${SMOLVM_CUDA_SOCK:-/tmp/smolvm/cuda-daemon.sock}"
GOLDEN="gft-g"

export SMOLVM_LIB_DIR="${SMOLVM_LIB_DIR:-$HOME/smolvm/lib/linux-x86_64}"

rm -rf "$CO"
mkdir -p "$CO"
cp "${GRAPH_TRIAL_SCRIPT:-$HOME/bench/graph_fresh_trial.py}" "$CO/probe.py"

python3 - "$CO" "$TRIALS" "$REPLAYS" <<'PY'
import json
import os
import random
import sys

coord, count, replays = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
base = [
    {"k": 1, "state": "standard"},
    {"k": 1, "state": "prealloc"},
    {"k": 1, "state": "doublewarm"},
    {"k": 500, "state": "standard"},
    {"k": 500, "state": "prealloc"},
    {"k": 500, "state": "doublewarm"},
]
cases = [base[i % len(base)].copy() for i in range(count)]
random.Random(741742).shuffle(cases)
for i, case in enumerate(cases):
    case["trial"] = i
    case["replays"] = replays
    with open(os.path.join(coord, f"case_{i}.json"), "w") as f:
        json.dump(case, f)
PY

"$S" machine rm --name "$GOLDEN" --cascade >/dev/null 2>&1 || true
for i in $(seq 0 $((TRIALS - 1))); do
    "$S" machine rm --name "gft-c$i" --force >/dev/null 2>&1 || true
done
rm -f "$SOCK"

env SMOLVM_CUDA_FORK_WORKERS=1 \
    SMOLVM_CUDA_FORK_ISOLATE=1 \
    SMOLVM_CUDA_DAEMON_IDLE_SECS=0 \
    SMOLVM_CUDA_FORK_SHARE_WEIGHTS=1 \
    SMOLVM_CUDA_LOG_ERRORS=1 \
    RUST_LOG=warn,smolvm::cuda_daemon=info \
    "$S" _cuda-daemon "$SOCK" >"$CO/daemon.log" 2>&1 &
DAEMON_PID=$!

cleanup() {
    "$S" machine rm --name "$GOLDEN" --cascade >/dev/null 2>&1 || true
    for i in $(seq 0 $((TRIALS - 1))); do
        "$S" machine rm --name "gft-c$i" --force >/dev/null 2>&1 || true
    done
    kill "$DAEMON_PID" >/dev/null 2>&1 || true
    wait "$DAEMON_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 100); do
    [[ -S "$SOCK" ]] && break
    sleep 0.1
done

GUEST="export COORD=/opt/coord NSLOTS=$TRIALS; \
/home/ubuntu/ptwork/bin/python /opt/coord/probe.py 2>>/opt/coord/g.err"
"$S" machine create --name "$GOLDEN" --cuda --net \
    -v "$CO:/opt/coord:rw" --from "$PACK" --storage 30 --overlay 20 \
    -- sh -c "$GUEST" >/dev/null
env SMOLVM_CUDA_SHARED=1 "$S" machine start --forkable --name "$GOLDEN" >/dev/null

for _ in $(seq 1 240); do
    [[ -f "$CO/golden_ready" ]] && break
    sleep 0.5
done
if [[ ! -f "$CO/golden_ready" ]]; then
    echo "golden did not reach the fork gate" >&2
    tail -40 "$CO/g.err" >&2 || true
    exit 1
fi

for i in $(seq 0 $((TRIALS - 1))); do
    "$S" machine fork --golden "$GOLDEN" --name "gft-c$i" >/dev/null
done
touch "$CO/go"

for _ in $(seq 1 240); do
    found=$(find "$CO" -maxdepth 1 -type f -name 'trial_*.json' | wc -l)
    [[ "$found" -eq "$TRIALS" ]] && break
    sleep 0.5
done
found=$(find "$CO" -maxdepth 1 -type f -name 'trial_*.json' | wc -l)
if [[ "$found" -ne "$TRIALS" ]]; then
    echo "only $found/$TRIALS trials produced results" >&2
fi

python3 - "$CO" <<'PY'
import collections
import glob
import json
import os
import sys

coord = sys.argv[1]
rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(coord, "trial_*.json")))]
groups = collections.defaultdict(lambda: [0, 0])
for row in rows:
    key = (row["k"], row["state"])
    groups[key][1] += 1
    groups[key][0] += int(row["ok"])
    failed = next((s["stage"] for s in row["steps"] if not s["ok"]), "-")
    print(
        f"trial={row['lid']:02d} k={row['k']:3d} state={row['state']:<10} "
        f"ok={row['ok']} failed_stage={failed} "
        f"graph_us_per_op={row.get('graph_us_per_op', float('nan')):.4f}"
    )
print("SUMMARY")
for key in sorted(groups):
    passed, total = groups[key]
    print(f"k={key[0]:3d} state={key[1]:<10} {passed}/{total}")
PY

echo "FIRST HOST FAILURES"
sed -e 's/\x1b\[[0-9;]*m//g' "$CO/daemon.log" | grep '\[op-err\]' | head -40 || true
