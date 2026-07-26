#!/bin/bash
# Reproducible comparison: the SAME post-training workload run three ways on one GPU.
#
#   native — N learner processes on bare metal, each loading its own base
#   fork   — one smolvm golden VM loads the base once, then N --share-weights
#            clones, CUDA remoted to the host daemon
#   queue  — one native process keeps the base and compiled kernels resident,
#            then resets and trains N independent LoRA jobs sequentially
#
# All arms run the identical workload file (workload_dpo.py by default) with identical
# STEPS/BATCH/MAXSEQ/MODEL/seed, and are measured identically (wall clock from
# t0 to last learner done, 1 Hz nvidia-smi peak sampling, per-learner tok/s from
# the workload's own JSONL). Results append to results/<run-id>.json.
#
# FAIRNESS KNOBS (the confounds that made the first ad-hoc comparison invalid):
#   --cpus K   pin EACH native learner to its own K-core set, matching the
#              per-VM vCPU count in the fork arm. Default 4 = fork parity.
#              Use --cpus 0 to leave native unpinned (whole host).
#   --cold     drop the page cache before the run (default: warm, i.e. the
#              model file is pre-read so neither arm pays first-touch disk I/O)
#   --reps R   repeat R times; summarize.py reports the median and the spread
#              (the golden load is known to be bimodal ~15s vs ~156s)
#
# Usage: ./bench.sh --arm native|fork|queue --n 4 [--steps 20] [--reps 3] [--cpus 4]
#        [--cold] [--share on|off|default] [--batch 2] [--maxseq 256]
#        [--workload workload_sft.py] [--tag sft]
set -u

ARM=""; N=4; STEPS=20; REPS=1; CPUS=4; COLD=0; BATCH=2; MAXSEQ=256; TIMEOUT=600; SHARE=default; VCPU=""; VMEM=""; WORKLOAD_ARG=""; TAG_ARG=""
GRPO_WARM_STEPS="${GRPO_WARM_STEPS:-1}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm) ARM="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --reps) REPS="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --cold) COLD=1; shift ;;
        --batch) BATCH="$2"; shift 2 ;;
        --maxseq) MAXSEQ="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --workload) WORKLOAD_ARG="$2"; shift 2 ;;
        --tag) TAG_ARG="$2"; shift 2 ;;
        # on|off|default — sets SMOLVM_CUDA_FORK_SHARE_WEIGHTS for the run and
        # VERIFIES the daemon actually honoured it (see check_share_mode).
        --share) SHARE="$2"; shift 2 ;;
        # vCPU/RAM given to the golden (clones inherit). Separate from
        # --cpus, which pins the NATIVE arm. Thin clones matter at high N:
        # N x vcpu must stay near the host core count or the clones starve
        # each other feeding the GPU (N=16 x 4 vCPU on 26 cores regressed
        # below N=8, while a single clone gains nothing from 8 over 4).
        --vcpu) VCPU="$2"; shift 2 ;;
        --vmem) VMEM="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done
[[ "$ARM" == "native" || "$ARM" == "fork" || "$ARM" == "queue" ]] || {
    echo "need --arm native|fork|queue"; exit 2;
}

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$BENCH_DIR/results"; mkdir -p "$RESULTS"
WORKLOAD="${WORKLOAD_ARG:-${BENCH_WORKLOAD:-$BENCH_DIR/workload_dpo.py}}"
[[ "$WORKLOAD" = /* ]] || WORKLOAD="$BENCH_DIR/$WORKLOAD"
[[ -f "$WORKLOAD" ]] || { echo "workload not found: $WORKLOAD"; exit 2; }
WORKLOAD_TAG="${TAG_ARG:-$(basename "$WORKLOAD" .py)}"
WORKLOAD_TAG="${WORKLOAD_TAG#workload_}"
[[ "$WORKLOAD_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || {
    echo "bad workload tag: $WORKLOAD_TAG"; exit 2;
}
WORKLOAD_MD5="$(md5sum "$WORKLOAD" | awk '{print $1}')"
MODEL="${MODEL:-unsloth/Qwen2.5-7B-bnb-4bit}"
PY_VENV="${PY_VENV:-$HOME/ptwork/bin/python}"
S="${SMOLVM:-$HOME/smolvm/smolvm}"
export SMOLVM_LIB_DIR="${SMOLVM_LIB_DIR:-$HOME/smolvm/lib/linux-x86_64}"
SOCK=/tmp/smolvm/cuda-daemon.sock
PACK="${PACK:-$HOME/qlora-baked.smolmachine}"
# expandable_segments:True makes torch allocate via CUDA VMM, which bypasses
# the daemon alloc-table that marks weight ranges "loaded" -> fork weight
# sharing degrades to private copies. Override to test the sharing path.
ALLOC_CONF="${ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------- preconditions
# A leaked context or a live clone from a previous run silently changes both
# load time and peak memory, so refuse to measure until the box is clean.
preflight() {
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ "$used" -gt 500 ]]; then
        echo "PREFLIGHT FAIL: GPU already using ${used} MiB (leaked context?). Aborting."
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv | head
        exit 1
    fi
    pkill -f "smolvm _cuda-daemo[n]" 2>/dev/null
    pkill -f "_cuda-clone-worke[r]" 2>/dev/null
    # --cascade removes a golden together with its clones (no name guessing).
    $S machine rm --name bench-g --cascade >/dev/null 2>&1
    for m in $($S machine list 2>/dev/null | awk '/^bench-/{print $1}'); do
        $S machine rm --name "$m" --force >/dev/null 2>&1
    done
    rm -f "$SOCK"
    if [[ "$COLD" == "1" ]]; then
        sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || echo "  (cache drop needs sudo; continuing warm)"
    else
        # Warm both arms identically: page in the model weights once.
        find "${HF_HOME:-$HOME/hf}" -name "*.safetensors" -exec cat {} + > /dev/null 2>&1
    fi
}

# Environment manifest: everything that could move a number between runs.
manifest() {
    python3 - "$@" <<'PY'
import json, subprocess, sys, os, shlex
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
rootfs = os.environ.get(
    "SMOLVM_AGENT_ROOTFS",
    os.path.expanduser("~/.local/share/smolvm/agent-rootfs"),
)
rootfs_q = shlex.quote(rootfs)
print(json.dumps({
  "smolvm_version": sh(f"{sys.argv[1]} --version"),
  "smolvm_binary_md5": sh(f"md5sum {sys.argv[1]} | cut -d' ' -f1"),
  "agent_rootfs": rootfs,
  "proto_hash_rootfs": sh(f"cat {rootfs_q}/usr/local/lib/smolvm-cuda/proto-hash"),
  "shim_md5": sh(f"md5sum {rootfs_q}/usr/local/lib/smolvm-cuda/libcudart-shim.so | cut -d' ' -f1"),
  "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
  "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
  "torch": sh(f"{sys.argv[2]} -c 'import torch;print(torch.__version__)'"),
  "python": sh(f"{sys.argv[2]} --version"),
  "unsloth": sh(f"{sys.argv[2]} -c 'import importlib.metadata as m;print(m.version(\"unsloth\"))'"),
  "trl": sh(f"{sys.argv[2]} -c 'import importlib.metadata as m;print(m.version(\"trl\"))'"),
  "transformers": sh(f"{sys.argv[2]} -c 'import importlib.metadata as m;print(m.version(\"transformers\"))'"),
  "bitsandbytes": sh(f"{sys.argv[2]} -c 'import importlib.metadata as m;print(m.version(\"bitsandbytes\"))'"),
  "pip_freeze_sha256": sh(f"{sys.argv[2]} -m pip freeze | sha256sum | cut -d' ' -f1"),
  "host_cores": os.cpu_count(),
  "host_mem_gb": round(os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/1e9),
  "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
  "cuda_mps_log_directory": os.environ.get("CUDA_MPS_LOG_DIRECTORY"),
  "cuda_mps_active_thread_percentage": os.environ.get(
      "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
  ),
}))
PY
}

# 1 Hz peak sampler; writes the max MiB seen to $1 when told to stop via $2.
sample_gpu() {
    local out="$1" stopflag="$2" maxv=0 v
    while [[ ! -f "$stopflag" ]]; do
        v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        [[ "$v" -gt "$maxv" ]] 2>/dev/null && maxv=$v
        sleep 1
    done
    echo "$maxv" > "$out"
}

# ---------------------------------------------------------------------- arms
run_native() {
    local CO="$1"
    export HF_HOME="${HF_HOME:-$HOME/hf}" HF_HUB_OFFLINE=1 COORD=$CO ARM=native FORK=0 OUTBASE=$CO
    export STEPS=$STEPS MODEL=$MODEL BATCH=$BATCH MAXSEQ=$MAXSEQ
    export GRPO_WARM_STEPS=$GRPO_WARM_STEPS
    export PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" TORCHINDUCTOR_COMPILE_THREADS=1
    # Collect the learner PIDs and wait on THOSE only: a bare `wait` would also
    # block on the GPU sampler running in this same shell, which by design does
    # not exit until after this function returns (deadlock).
    local pids=()
    for i in $(seq 0 $((N-1))); do
        if [[ "$CPUS" -gt 0 ]]; then
            # Give each learner its own K-core set, mirroring one VM's vCPUs.
            local nproc_n cores
            nproc_n=$(nproc)
            cores=$(python3 -c "print(','.join(str(($i*$CPUS+k) % $nproc_n) for k in range($CPUS)))")
            LEARNER_ID=$i taskset -c "$cores" "$PY_VENV" "$CO/workload.py" > "$CO/o$i.log" 2>"$CO/e$i.log" &
        else
            LEARNER_ID=$i "$PY_VENV" "$CO/workload.py" > "$CO/o$i.log" 2>"$CO/e$i.log" &
        fi
        pids+=($!)
    done
    wait "${pids[@]}"
}

run_queue() {
    local CO="$1"
    export HF_HOME="${HF_HOME:-$HOME/hf}" HF_HUB_OFFLINE=1 COORD=$CO ARM=queue FORK=0 OUTBASE=$CO
    export STEPS=$STEPS MODEL=$MODEL BATCH=$BATCH MAXSEQ=$MAXSEQ QUEUE_JOBS=$N
    export GRPO_WARM_STEPS=$GRPO_WARM_STEPS
    export PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" TORCHINDUCTOR_COMPILE_THREADS=1
    if [[ "$CPUS" -gt 0 ]]; then
        local nproc_n cores
        nproc_n=$(nproc)
        cores=$(python3 -c "print(','.join(str(k) for k in range(min($CPUS, $nproc_n))))")
        LEARNER_ID=0 taskset -c "$cores" "$PY_VENV" "$CO/workload.py" \
            > "$CO/o0.log" 2> "$CO/e0.log"
    else
        LEARNER_ID=0 "$PY_VENV" "$CO/workload.py" > "$CO/o0.log" 2> "$CO/e0.log"
    fi
}

run_fork() {
    local CO="$1"
    # Set the mode explicitly rather than relying on ambient env: an arm that
    # silently runs the wrong configuration is worse than no arm at all (this
    # bit me — a "copy mode" control that actually shared, because the variable
    # never reached the daemon). check_share_mode below proves what ran.
    local share_setting=""
    case "$SHARE" in
        on)  share_setting=1 ;;
        off) share_setting=0 ;;
        default) ;;
        *) echo "bad --share: $SHARE (want on|off|default)"; exit 2 ;;
    esac
    # CONTEXT MODE. Default (workers=1) gives each clone its own CUDA context,
    # which is isolated but makes the GPU TIME-SLICE between clones. Setting
    # BENCH_FORK_WORKERS=0 selects the legacy shared-context path, where clones
    # run on separate streams of ONE context and their kernels can actually
    # overlap -- the only way to fill a GPU that a single learner leaves ~64%
    # idle. Native cannot do this at all: separate processes = separate
    # contexts (absent MPS).
    local daemon_env=(SMOLVM_CUDA_DAEMON_IDLE_SECS=0 RUST_LOG=warn,smolvm::cuda_daemon=info)
    # This is daemon/worker policy, not guest or VMM policy. Leaking the
    # all-private kill switch through machine create/fork can perturb clone
    # reconnect before a worker exists, making the control time out without
    # measuring either sharing mode.
    [[ -n "$share_setting" ]] \
        && daemon_env+=(SMOLVM_CUDA_FORK_SHARE_WEIGHTS="$share_setting")
    if [[ "${BENCH_FORK_WORKERS:-1}" == "1" ]]; then
        daemon_env+=(SMOLVM_CUDA_FORK_WORKERS=1 SMOLVM_CUDA_FORK_ISOLATE=1)
    fi
    env "${daemon_env[@]}" "$S" _cuda-daemon "$SOCK" > "$CO/daemon.log" 2>&1 &
    for i in $(seq 1 100); do [[ -S "$SOCK" ]] && break; sleep 0.1; done
    # No manual CUDA env: smolvm injects SMOLVM_CUDA_ZEROCOPY and stages the
    # guest shims (libcuda.so.1 + the unversioned dev names) itself.
    # Optional guest-side prelude (e.g. `unset SMOLVM_CUDA_ZEROCOPY;`) so a run
    # can disable the zero-copy upload path, whose crc=0 coverage makes every
    # weight chunk unshareable.
    local GUEST="${BENCH_GUEST_EXTRA:-} export HF_HOME=/opt/hfcache HF_HUB_OFFLINE=0 COORD=/opt/coord ARM=fork FORK=1 \
STEPS=$STEPS NSLOTS=$N MODEL=$MODEL OUTBASE=/root BATCH=$BATCH MAXSEQ=$MAXSEQ \
GRPO_WARM_STEPS=$GRPO_WARM_STEPS \
PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF TORCHINDUCTOR_COMPILE_THREADS=1; \
/home/ubuntu/ptwork/bin/python /opt/coord/workload.py 2>>/opt/coord/g.err"
    local vmargs=()
    [[ -n "$VCPU" ]] && vmargs+=(--cpus "$VCPU")
    [[ -n "$VMEM" ]] && vmargs+=(--mem "$VMEM")
    $S machine create --name bench-g --cuda --net "${vmargs[@]}" -v "$CO:/opt/coord:rw" \
        --from "$PACK" --storage 30 --overlay 20 -- sh -c "$GUEST" >/dev/null 2>&1
    env SMOLVM_CUDA_SHARED=1 $S machine start --forkable --name bench-g >/dev/null 2>&1
    local tgold
    for i in $(seq 1 300); do [[ -f "$CO/golden_ready" ]] && break; sleep 1; done
    [[ -f "$CO/golden_ready" ]] || { echo "GOLDEN-LOAD-FAILED"; return 1; }
    tgold=$(date +%s.%N); echo "$tgold" > "$CO/.t_golden"
    for c in $(seq 0 $((N-1))); do
        # Fork startup has occasionally failed transiently under high fan-out.
        # Never discard that failure and then wait the full learner timeout for
        # a clone that does not exist. Retry an absent clone twice, retain every
        # attempt's output, and fail the run before releasing `go` if it still
        # cannot be created.
        local fork_log="$CO/fork-c$c.log" fork_ok=0 attempt
        : > "$fork_log"
        for attempt in 1 2 3; do
            echo "attempt $attempt" >> "$fork_log"
            if $S machine fork --golden bench-g --name bench-c$c --share-weights \
                >> "$fork_log" 2>&1; then
                fork_ok=1
                break
            fi
            # Do not retry over a partially created machine. Preserve it and
            # fail so the product state can be inspected rather than hidden.
            if $S machine list 2>/dev/null \
                | awk -v name="bench-c$c" '$1 == name { found=1 } END { exit !found }'; then
                echo "fork command failed but machine exists; refusing blind retry" >> "$fork_log"
                break
            fi
            sleep 1
        done
        if [[ "$fork_ok" != "1" ]]; then
            echo "FORK-FAILED: bench-c$c (see $fork_log)"
            tail -n 20 "$fork_log"
            $S machine rm --name bench-g --cascade >/dev/null 2>&1
            pkill -f "smolvm _cuda-daemo[n]" 2>/dev/null
            pkill -f "_cuda-clone-worke[r]" 2>/dev/null
            return 1
        fi
    done
    echo "$(date +%s.%N)" > "$CO/.t_forked"
    touch "$CO/go"
    local deadline=$(( $(date +%s) + TIMEOUT ))
    while [[ "$(grep -l '"event": "done"' "$CO"/learner_*.jsonl 2>/dev/null | wc -l)" -lt "$N" ]]; do
        [[ $(date +%s) -ge $deadline ]] && { echo "  TIMEOUT after ${TIMEOUT}s (learners still unfinished)"; break; }
        sleep 2
    done
    local share_mode_ok=1
    check_share_mode "$CO" || share_mode_ok=0
    $S machine rm --name bench-g --cascade >/dev/null 2>&1
    pkill -f "smolvm _cuda-daemo[n]" 2>/dev/null; pkill -f "_cuda-clone-worke[r]" 2>/dev/null
    [[ "$share_mode_ok" == "1" ]]
}

# The daemon logs "M2: shared weight ranges shared=<n> private=<m>" per clone.
# Confirm it agrees with --share, so a result can never be attributed to a mode
# that did not actually run.
check_share_mode() {
    local CO="$1" line shared
    # The daemon logs through tracing, which interleaves ANSI escapes between
    # the field name and its value ("shared\e[0m\e[2m=\e[0m260"), so strip
    # escapes before matching -- a plain grep silently finds nothing and the
    # check reports "<none>" for every run, which is how this guard first
    # shipped broken.
    # A workload may have several CUDA processes and therefore several clone
    # workers (Unsloth SFT has a preprocessing process with zero VMM maps).
    # Report the weight-bearing worker, not whichever worker logged last.
    line=$(sed -e 's/\x1b\[[0-9;]*m//g' "$CO/daemon.log" 2>/dev/null \
           | grep -o "shared=[0-9]* private=[0-9]*" \
           | awk -F'[= ]+' 'NR == 1 || $2 > max { max=$2; line=$0 } END { print line }')
    shared=$(echo "$line" | sed -E "s/shared=([0-9]*).*/\1/")
    echo "  share-mode requested=$SHARE  daemon reported: ${line:-<none>}"
    if [[ -z "$line" ]]; then
        echo "  !! CONTROL FAILED: daemon emitted no sharing verdict"
        return 1
    elif [[ "$SHARE" == "off" && "${shared:-0}" -gt 0 ]]; then
        echo "  !! CONTROL FAILED: asked for copy mode but the daemon shared $shared ranges"
        return 1
    elif [[ "$SHARE" == "on" && "${shared:-0}" -eq 0 ]]; then
        echo "  !! SHARING DID NOT ENGAGE: asked to share but the daemon shared 0 ranges"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------- driver
MF=$(manifest "$S" "$PY_VENV")
for rep in $(seq 1 "$REPS"); do
    RUNID="${ARM}_${WORKLOAD_TAG}_n${N}_s${STEPS}_c${CPUS}_$(date +%Y%m%d-%H%M%S)_r${rep}"
    CO="$HOME/bench_run/$RUNID"; rm -rf "$CO"; mkdir -p "$CO"; cp "$WORKLOAD" "$CO/workload.py"
    echo "=== $RUNID (arm=$ARM workload=$WORKLOAD_TAG n=$N steps=$STEPS cpus=$CPUS cold=$COLD) ==="
    preflight
    STOP="$CO/.stop"; PEAK="$CO/.peak"; rm -f "$STOP"
    sample_gpu "$PEAK" "$STOP" &
    SAMPLER=$!
    T0=$(date +%s.%N)
    case "$ARM" in
        native) run_native "$CO" ;;
        queue)  run_queue "$CO" ;;
        fork)   run_fork "$CO" ;;
    esac || { touch "$STOP"; wait "$SAMPLER" 2>/dev/null; exit 1; }
    T1=$(date +%s.%N)
    touch "$STOP"; wait $SAMPLER 2>/dev/null
    WALL=$(echo "$T1 - $T0" | bc)
    GOLD=""; [[ -f "$CO/.t_golden" ]] && GOLD=$(echo "$(cat "$CO/.t_golden") - $T0" | bc)
    FORKED=""; [[ -f "$CO/.t_forked" ]] && FORKED=$(echo "$(cat "$CO/.t_forked") - $(cat "$CO/.t_golden")" | bc)
    MPS_MODE_RECORD=off
    if sed -e 's/\x1b\[[0-9;]*m//g' "$CO/daemon.log" 2>/dev/null \
        | grep -q 'private uncapped NVIDIA MPS active'; then
        MPS_MODE_RECORD=managed-uncapped
    elif [[ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]]; then
        MPS_MODE_RECORD=external
    fi
    case "$SHARE" in on) SW_RECORD_VALUE=1 ;; off) SW_RECORD_VALUE=0 ;; *) SW_RECORD_VALUE=unset ;; esac
    VCPU_RECORD="$VCPU" VMEM_RECORD="$VMEM" FW_RECORD="${BENCH_FORK_WORKERS:-1}" SHARE_RECORD="$SHARE" SHARED_RANGES="$(sed -e 's/\x1b\[[0-9;]*m//g' "$CO/daemon.log" 2>/dev/null | grep -o 'shared=[0-9]*' | cut -d= -f2 | sort -nr | head -1)" BATCH_RECORD="$BATCH" MAXSEQ_RECORD="$MAXSEQ" ALLOC_CONF_RECORD="$ALLOC_CONF" SW_RECORD="$SW_RECORD_VALUE" MPS_MODE_RECORD="$MPS_MODE_RECORD" WORKLOAD_RECORD="$WORKLOAD_TAG" WORKLOAD_MD5_RECORD="$WORKLOAD_MD5" python3 - "$CO" "$RESULTS/$RUNID.json" "$ARM" "$N" "$STEPS" "$CPUS" "$COLD" "$WALL" "${GOLD:-null}" "${FORKED:-null}" "$(cat "$PEAK" 2>/dev/null || echo 0)" "$MF" <<'PY'
import sys, json, glob
co, out, arm, n, steps, cpus, cold, wall, gold, forked, peak, mf = sys.argv[1:13]
learners = []
for f in sorted(glob.glob(f"{co}/learner_*.jsonl")):
    d = {}
    for line in open(f):
        e = json.loads(line); d[e["event"]] = e
    if "done" in d:
        z = d["done"]
        learners.append({k: z.get(k) for k in (
            "lid", "method", "tok_s", "examples_s", "train_s", "step_ms",
            "loss0", "lossN", "reward0", "rewardN", "peak_gb",
            "reward_max", "reward_min", "rollout_tokens", "rollout_sha256",
            "rollout_step_sha256", "rollout_step_rewards",
            "rollout_step_tokens",
            "initial_parameter_sha256", "parameter_sha256",
            "parameter_count", "parameter_sum", "parameter_abs_sum",
            "parameter_l2", "parameter_max_abs",
            "model_output_sha256", "model_output_sum", "model_output_l2",
            "dataset_sha256", "cpu_rng_sha256", "cuda_rng_sha256",
            "final_cpu_rng_sha256", "final_cuda_rng_sha256",
            "trainer_init_s", "trainer_train_s", "reward_step_elapsed_s",
            "trainable_grad_tensors",
            "bf16_supported", "device_capability", "model_dtype", "model_snapshot",
            "compile_cache_scoped", "trainable_dtypes",
        ) if z.get(k) is not None})
rec = {
  "arm": arm, "n": int(n), "steps": int(steps), "cpus_per_learner": int(cpus),
  "workload": __import__("os").environ.get("WORKLOAD_RECORD", "dpo"),
  "workload_md5": __import__("os").environ.get("WORKLOAD_MD5_RECORD"),
  "cold_cache": bool(int(cold)),
  "mps_mode": __import__("os").environ.get("MPS_MODE_RECORD", "off"),
  "native_reference_warmup": __import__("os").environ.get(
      "NATIVE_REFERENCE_WARMUP", "0"
  ) == "1",
  "grpo_warm_steps": int(__import__("os").environ.get("GRPO_WARM_STEPS", "1")),
  "cuda_mps_pipe_directory": __import__("os").environ.get(
      "CUDA_MPS_PIPE_DIRECTORY"
  ),
  "cuda_mps_active_thread_percentage": __import__("os").environ.get(
      "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
  ),
  "alloc_conf": __import__("os").environ.get("ALLOC_CONF_RECORD", ""),
  "batch": int(__import__("os").environ.get("BATCH_RECORD", "0")),
  "maxseq": int(__import__("os").environ.get("MAXSEQ_RECORD", "0")),
  "share_weights_env": __import__("os").environ.get("SW_RECORD", ""),
  "share_mode": __import__("os").environ.get("SHARE_RECORD", ""),
  "fork_workers": __import__("os").environ.get("FW_RECORD", "1"),
  "golden_vcpu": __import__("os").environ.get("VCPU_RECORD", ""),
  "golden_vmem": __import__("os").environ.get("VMEM_RECORD", ""),
  "shared_ranges": __import__("os").environ.get("SHARED_RANGES", ""),
  "wall_s": round(float(wall), 2),
  "golden_load_s": None if gold == "null" else round(float(gold), 2),
  "fork_s": None if forked == "null" else round(float(forked), 2),
  "peak_gpu_mib": int(peak),
  "learners_done": len(learners), "learners_expected": int(n),
  "agg_tok_s": sum(l["tok_s"] for l in learners if l["tok_s"]),
  "learners": learners,
  "env": json.loads(mf),
}
timed = [l for l in learners if l.get("train_s") and l.get("train_s") > 0]
rollout_timed = [l for l in timed if l.get("rollout_tokens") is not None]
if rollout_timed:
    rec["exact_agg_tok_s"] = round(sum(
        l["rollout_tokens"] / l["train_s"] for l in rollout_timed
    ), 3)
if timed:
    rec["train_tail_s"] = (
        sum(l["train_s"] for l in timed)
        if arm == "queue"
        else max(l["train_s"] for l in timed)
    )
    rec["aggregate_step_s"] = round(
        sum(int(l.get("steps") or steps) for l in timed) / rec["train_tail_s"], 3
    )
if rollout_timed and len(rollout_timed) == len(timed):
    rec["tail_agg_tok_s"] = round(
        sum(l["rollout_tokens"] for l in rollout_timed) / rec["train_tail_s"], 3
    )
if arm == "queue" and "tail_agg_tok_s" in rec:
    rec["exact_agg_tok_s"] = rec["tail_agg_tok_s"]
    rec["agg_tok_s"] = round(rec["tail_agg_tok_s"])
json.dump(rec, open(out, "w"), indent=2)
tail_metrics = ""
for key in ("exact_agg_tok_s", "tail_agg_tok_s", "aggregate_step_s"):
    if key in rec:
        tail_metrics += f" {key}={rec[key]}"
print(f"  wall={rec['wall_s']}s done={rec['learners_done']}/{n} agg_tok_s={rec['agg_tok_s']}{tail_metrics} peak_gpu={rec['peak_gpu_mib']}MiB" + (f" golden_load={rec['golden_load_s']}s" if rec["golden_load_s"] else "") + f" mps={rec['mps_mode']}")
print(f"  -> {out}")
PY
done
