#!/usr/bin/env bash
# Build the pinned host and guest environments used by the GRPO reproduction.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$BENCH_DIR/.." && pwd)"
PY_VENV_DIR="${PY_VENV_DIR:-$HOME/ptwork}"
HF_HOME="${HF_HOME:-$HOME/hf}"
PACK="${PACK:-$HOME/qlora-baked.smolmachine}"
MODEL="${MODEL:-unsloth/Qwen2.5-7B-bnb-4bit}"
MODEL_REVISION="${MODEL_REVISION:-8664e5c8c25614048aec0b89415a5986053fde5c}"
BUILD_RUNTIME="${BUILD_RUNTIME:-1}"
BUILD_PACK="${BUILD_PACK:-1}"
BUILD_VM_NAME="${BUILD_VM_NAME:-grpo-repro-build}"
PACK_EXPORT_MAX_BYTES="${PACK_EXPORT_MAX_BYTES:-21474836480}"
REPRO_ENV="${REPRO_ENV:-$BENCH_DIR/.grpo-repro.env}"

need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "missing required command: $1" >&2
        exit 1
    }
}

for command_name in cargo curl git python3 rustup; do
    need "$command_name"
done
git lfs env >/dev/null 2>&1 || {
    echo "git-lfs is required" >&2
    exit 1
}
[[ -r /dev/kvm && -w /dev/kvm ]] || {
    echo "/dev/kvm is not readable and writable by $(id -un)" >&2
    exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi is required on the GPU host" >&2
    exit 1
}

if [[ "$BUILD_RUNTIME" == "1" ]]; then
    git -C "$REPO_DIR" lfs pull
    rustup target add x86_64-unknown-linux-musl
    (
        cd "$REPO_DIR"
        LIBKRUN_BUNDLE="$REPO_DIR/lib/linux-x86_64" cargo build --release -p smolvm
        ./scripts/build-agent-rootfs.sh --arch x86_64
    )
fi

SMOLVM="${SMOLVM:-$REPO_DIR/target/release/smolvm}"
export SMOLVM_LIB_DIR="${SMOLVM_LIB_DIR:-$REPO_DIR/lib/linux-x86_64}"
export SMOLVM_AGENT_ROOTFS="${SMOLVM_AGENT_ROOTFS:-$REPO_DIR/target/agent-rootfs}"
[[ -x "$SMOLVM" ]] || { echo "smolvm binary not found: $SMOLVM" >&2; exit 1; }

python3 -m venv "$PY_VENV_DIR"
"$PY_VENV_DIR/bin/pip" install pip==26.1.2
"$PY_VENV_DIR/bin/pip" install -r "$BENCH_DIR/grpo-requirements.txt"
HF_HOME="$HF_HOME" "$PY_VENV_DIR/bin/python" - "$MODEL" "$MODEL_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(sys.argv[1], revision=sys.argv[2]))
PY

if [[ "$BUILD_PACK" == "1" ]]; then
    pack_stub="${PACK%.smolmachine}"
    if [[ -e "$pack_stub" || -e "$pack_stub.smolmachine" ]]; then
        echo "refusing to replace existing pack: $pack_stub" >&2
        exit 1
    fi
    if "$SMOLVM" machine list 2>/dev/null \
        | awk -v name="$BUILD_VM_NAME" '$1 == name { found=1 } END { exit !found }'; then
        echo "refusing to replace existing machine: $BUILD_VM_NAME" >&2
        exit 1
    fi
    "$SMOLVM" machine create --name "$BUILD_VM_NAME" --net \
        --image ubuntu:22.04 --storage 30 --overlay 20 \
        -v "$BENCH_DIR:/opt/grpo-repro:ro"
    "$SMOLVM" machine start --name "$BUILD_VM_NAME"
    "$SMOLVM" machine exec --stream --name "$BUILD_VM_NAME" -- sh -lc \
        'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates gcc git python3 python3-dev python3-venv'
    "$SMOLVM" machine exec --stream --name "$BUILD_VM_NAME" -- sh -lc \
        'python3 -m venv /home/ubuntu/ptwork && /home/ubuntu/ptwork/bin/pip install pip==26.1.2 && /home/ubuntu/ptwork/bin/pip install -r /opt/grpo-repro/grpo-requirements.txt && /home/ubuntu/ptwork/bin/pip check'
    "$SMOLVM" machine exec --stream --name "$BUILD_VM_NAME" -- sh -lc \
        "HF_HOME=/opt/hfcache /home/ubuntu/ptwork/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download(\"$MODEL\", revision=\"$MODEL_REVISION\"))'"
    "$SMOLVM" machine stop --name "$BUILD_VM_NAME"
    SMOLVM_FILE_TRANSFER_MAX_BYTES="$PACK_EXPORT_MAX_BYTES" \
        "$SMOLVM" pack create --from-vm "$BUILD_VM_NAME" -o "$pack_stub"
fi

{
    printf 'export SMOLVM=%q\n' "$SMOLVM"
    printf 'export SMOLVM_LIB_DIR=%q\n' "$SMOLVM_LIB_DIR"
    printf 'export SMOLVM_AGENT_ROOTFS=%q\n' "$SMOLVM_AGENT_ROOTFS"
    printf 'export PY_VENV=%q\n' "$PY_VENV_DIR/bin/python"
    printf 'export HF_HOME=%q\n' "$HF_HOME"
    printf 'export PACK=%q\n' "$PACK"
} > "$REPRO_ENV"
chmod 600 "$REPRO_ENV"

echo "GRPO environment ready; repro_grpo.sh will load $REPRO_ENV"
