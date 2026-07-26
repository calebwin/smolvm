# Reproduce the GRPO result from a fresh H100

This procedure compares the same pinned Unsloth GRPO workload in three modes:
ordinary native processes, native processes under uncapped NVIDIA MPS, and
weight-sharing smolvm forks under managed MPS. The full run checks deterministic
setup, reward and adapter quality, completion throughput, and whole-GPU peak
memory from machine-readable result files.

## Fresh host

Use an exclusive Linux x86-64 NVIDIA GPU host with KVM. The reference system is
Ubuntu 22.04, an H100 80 GB, and driver 570.148.08.

```sh
sudo apt-get update
sudo apt-get install -y bc build-essential curl e2fsprogs git git-lfs \
  libssl-dev pkg-config python3-dev python3-venv zstd
sudo usermod -aG kvm "$USER"
# Log out and back in if the current shell cannot read and write /dev/kvm.

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"

git clone https://github.com/smol-machines/smolvm.git
cd smolvm
git checkout cuda-graph-capture-investigation
./bench/setup_grpo_repro.sh
```

The setup builds smolvm and its matching CUDA guest shims on the target host,
creates the pinned Python environment, downloads model revision
`8664e5c8c25614048aec0b89415a5986053fde5c`, and creates the same auditable guest
environment as a local `.smolmachine`. It does not use the original benchmark
host or its caches. The setup raises the pack export limit to 20 GiB because
the flattened pinned Torch and model environment legitimately exceeds the
runtime's conservative 4 GiB default.

The setup writes the local paths to an ignored environment file that the runner
loads automatically. Run a plumbing check:

```sh
./bench/repro_grpo.sh --smoke
```

The smoke run is not a quality or performance claim. Run the qualification on
an otherwise idle GPU:

```sh
./bench/repro_grpo.sh --full
```

The full run uses N=8, 200 optimizer steps, batch 1, sequence length 256, two
CPU cores per worker, and a 20-step representative warmup. It emits ordinary
native, native-MPS, and smolvm-fork JSON under `bench/results/`; the final
comparison exits nonzero if setup, correctness, or quality checks fail.

The expected signal is not that smolvm beats equally scheduled native workers.
Native MPS should explain the gain over ordinary native, while smolvm should
retain about 83% of native-MPS step throughput with about 64% lower peak GPU
memory. Treat meaningful deviations as results to investigate, not numbers to
force.

To audit the result, inspect `env`, `workload_md5`, `model_snapshot`,
`learners_done`, `shared_ranges`, `mps_mode`, `peak_gpu_mib`,
`tail_agg_tok_s`, and `aggregate_step_s` in each JSON. The comparison also
requires identical frozen-model output, adapter initialization, dataset, and
CPU/CUDA RNG fingerprints before applying bounded sampled-RL quality checks.
