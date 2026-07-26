# Fork-pool benchmark harness

Measures the same post-training workload two ways on one GPU, so claims about
smolvm's fork pool can be checked rather than believed:

- **native** — N learner processes on bare metal, each loading its own base model
- **fork** — one smolvm golden VM loads the base once, then N `--share-weights`
  clones, CUDA remoted to the host daemon

Both arms run the *same* workload file with the same
`STEPS/BATCH/MAXSEQ/MODEL`, and are measured the same way: wall clock from t0 to
the last learner finishing, 1 Hz `nvidia-smi` peak sampling, and per-learner
tok/s taken from the workload's own JSONL. Every run writes
`results/<run-id>.json` including an environment manifest (smolvm version +
binary md5, rootfs proto-hash, shim md5, driver, torch, GPU, host cores).

## Prerequisites

The harness does not build or install anything. You need, on the GPU host:

| Requirement | Default | Override |
|---|---|---|
| smolvm binary | `~/smolvm/smolvm` | `SMOLVM=` |
| libkrun dir | `~/smolvm/lib/linux-x86_64` | `SMOLVM_LIB_DIR=` |
| Python venv with torch + unsloth + trl | `~/ptwork/bin/python` | `PY_VENV=` |
| Machine image (`.smolmachine`) with that venv baked in | `~/qlora-baked.smolmachine` | `PACK=` |
| Model in the HF cache (`~/hf`) | `unsloth/Qwen2.5-7B-bnb-4bit` | `MODEL=` |
| NVIDIA GPU + driver | — | — |

**The agent rootfs must carry CUDA guest shims built from the same tree as the
smolvm binary.** If they disagree you get
`PROTOCOL MISMATCH: client wire hash … != server …` and the golden never loads.
Stage them together:

```sh
T=<repo>/target/release
R=~/.local/share/smolvm/agent-rootfs/usr/local/lib/smolvm-cuda
cp $T/libcudart.so $R/libcudart-shim.so
cp $T/libcuda.so   $R/libcuda.so.1
cp $T/libnvidia_ml.so $R/libnvidia-ml.so.1
ln -sf libcuda.so.1 $R/libcuda.so          # Triton's JIT links -lcuda
ln -sf libcudart-shim.so $R/libcudart.so
$T/smolvm-cuda-run --proto-hash > $R/proto-hash
```

## Running

```sh
./bench.sh --arm native --n 4 --steps 20 --cpus 4
./bench.sh --arm fork   --n 4 --steps 20 --cpus 4 --share on
./bench.sh --arm native --n 4 --steps 200 --cpus 2 --batch 1 \
  --workload workload_grpo.py --tag grpo-reference
./bench.sh --arm fork --n 4 --steps 200 --cpus 2 --batch 1 \
  --workload workload_grpo.py --tag grpo
./compare_grpo.py results/native_grpo-reference_....json \
  results/fork_grpo_....json      # correctness, quality, throughput, and density gate
./summarize.py                    # source-identical groups, medians, spreads, ratios
```

`workload_grpo.py` is the real sampled-GRPO qualification workload. It records
the frozen model and adapter state, per-step rollout/reward digests, final RNG
state, and generated-token throughput. Its installed Unsloth 2026.7.3 stack
requires `ACCELERATE_MIXED_PRECISION=bf16` to agree with
`GRPOConfig(bf16=True)`; the workload establishes that contract before importing
Unsloth in both arms. This is a framework requirement for this workload, not a
smolvm-wide environment injection.

`compare_grpo.py` requires deterministic setup and final CPU RNG state to match
exactly, then applies explicit tolerances to reward means and final adapter L2
norms. Sampled RL trajectories can cross a token boundary after small numerical
differences, so final adapter bytes are retained as evidence but are not used as
an inappropriate bitwise pass/fail rule. The command exits nonzero on any failed
gate.

Flags that exist because leaving them out produces misleading numbers:

- `--cpus K` — pin **each** native learner to its own K-core set, matching the
  per-VM vCPU count in the fork arm. Without this, native silently gets the
  whole host (26 cores here) while each clone VM gets 4. `--cpus 0` = unpinned.
- `--share on|off|default` — sets `SMOLVM_CUDA_FORK_SHARE_WEIGHTS` **and checks
  the daemon honoured it**, printing `!! CONTROL FAILED` if a copy-mode arm
  actually shared. An earlier control silently ran the wrong configuration; this
  is the guard against repeating that.
- `--reps R` — repeat; `summarize.py` reports median and min–max. Single runs are
  not evidence: the golden load is bimodal (~15s vs ~156s) and one N=16 run in
  three produced `nan` learners.
- `--cold` — drop the page cache first (default warms both arms identically).
- `--batch` / `--maxseq` — kernel size. The default (2 × 256) is launch-latency
  bound, which flatters neither arm honestly.

`matrix.sh` (scaling + kernel-size sweep) and `highn.sh` (native vs fork at high
N) drive `bench.sh` sequentially — the GPU must be exclusive or every number is
noise. `bench.sh` refuses to start if the GPU already holds >500 MiB.

## Reading the output

```
wall=253.79s done=4/4 agg_tok_s=546 exact_agg_tok_s=546.2 \
tail_agg_tok_s=531.8 aggregate_step_s=1.6 peak_gpu=14550MiB golden_load=165.59s
```

- `done` — learners that reported `event: done`. **Less than N usually means OOM**;
  `summarize.py` flags those rows, since a partial aggregate looks like a real
  datapoint otherwise.
- `exact_agg_tok_s` — sum of each learner's unrounded token rate.
- `tail_agg_tok_s` — all learner tokens divided by the slowest learner's training
  time. This is the pool's useful completion throughput and is the default metric
  selected by `summarize.py` when available.
- `aggregate_step_s` — completed learner-steps divided by the same tail time. For
  sampled workloads, this prevents a low-quality run that emits many extra tokens
  from looking faster merely because it generated more text.
- `peak_gpu` — whole-device peak, so it includes the golden's own context in the
  fork arm. For a per-process split use `probe_mem.sh` during a run.
- `golden_load` — fork arm only; paid once per pool, not per learner.

Analysis helpers: `nan_census.py` (nan count per run across `results/`),
`parse_losses.py <run-dir>` (per-learner losses for one run).

## Caveats a reader should know

1. **Throughput and density are separate questions.** Weight sharing changes
   VRAM, not FLOPs — measured +6% aggregate at N=4, i.e. noise.
2. **CPU becomes the limit before VRAM does.** At 4 vCPU per clone on a 26-core
   host, aggregate throughput peaks near N=8 and falls at N=16. High-N results
   measure a memory ceiling, not useful throughput.
3. **`nan` learners appear intermittently at N=16** (1 run in 3; never at N≤8).
   Not yet attributed — a copy-mode control at N=16 is impossible because it
   needs ~118 GiB on an 80 GiB card.
4. **A ~750 MiB CUDA context can leak after a fork run.** The preflight check
   catches it and aborts rather than measuring on a dirty GPU; kill stray
   `_cuda-daemon` / `_cuda-clone-worker` processes between runs.
5. The workload (`workload_dpo.py`) writes its trainer output under `$OUTBASE`;
   it is a copy of the DPO fork workload with that path made configurable so the
   same file runs both in-VM (as root) and on the host.
