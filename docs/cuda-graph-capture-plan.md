# CUDA fork-pool throughput investigation & graph-capture plan

Status: investigation complete, root cause fully attributed, one blocking assumption
(kernel-argument stability) not yet resolved. Nothing in §6/§7 is implemented.

## 1. TL;DR

- The fork pool is now **production-safe for density**: weight sharing was silently
  inert (a real bug, fixed), is on by default, and a post-fork base write now fails
  loudly in one clone instead of corrupting every sibling silently. Shipped in
  [#741](https://github.com/smol-machines/smolvm/pull/741) and
  [#742](https://github.com/smol-machines/smolvm/pull/742) (both merged).
- **Native still wins raw throughput at every configuration measured.** The gap is
  fully root-caused: it is CUDA op *count* × a small per-op remoting cost, not any
  slow path. Five other hypotheses were tested and eliminated (§4).
- The only large lever left is **CUDA graph capture at the remoting boundary**. Measured
  through the boundary it nearly erases the per-op tax (1.241 µs/op remoted vs 1.224
  native — a 3.7x reduction). It is blocked by two unresolved questions, not a wall:
  clone-side capture reliability, and whether kernel arguments are stable enough to
  replay. See §6 for the validated numbers and §7 for the phased plan with go/no-go
  gates.
- Separately, fork's *saturated* ceiling (~13.7k tok/s regardless of N) is a different,
  architectural issue — per-clone CUDA contexts time-slice the GPU. Fixing that needs a
  single-context/multi-stream redesign, which is a real project, not started, and needs
  an explicit decision before any work begins (§5).
- **There is real uncommitted work on disk right now** (§8) — the diagnostics that
  produced §6's numbers are not on any branch. Read that section before it's lost.

## 2. Shipped (merged)

| PR | commit in main | what |
|---|---|---|
| [#741](https://github.com/smol-machines/smolvm/pull/741) | `5d0896b` | crc-hash fix so zero-copy H2D uploads are shareable (was `shared=0 private=420` unconditionally); weight sharing on by default; `machine rm --cascade`; rootfs Triton `-lcuda` symlinks; CI guard for the no-`host`-feature guest-shim build |
| [#742](https://github.com/smol-machines/smolvm/pull/742) | `181248f` | shared weight chunks mapped **read-only** by default (a stray post-fork base write now faults in the offending clone instead of racing across every sibling); reproducible `bench/` harness + results; console warning + `SMOLVM_CUDA_FILE_RING=force` for the silent ring-mount skip |

Validated density numbers (H100 80 GiB, Qwen2.5-7B-bnb-4bit DPO):

- Sharing verdict: `shared=260 private=148` (previously `0/420`, unconditionally).
- Per-clone marginal VRAM: **6928 → 1648 MiB** (4.2x).
- N=4: total VRAM ~35.3 → **14.5 GiB**, final losses **bit-identical** to all-private mode.
- N=16: **16/16 learners complete in 35.4 GiB**; native completes **2/16** before hitting
  the 80 GiB wall at the same N.
- Read-only mapping: real workload, N=8, `SMOLVM_CUDA_FORK_SHARE_WEIGHTS=1` +
  read-only shared chunks → 8/8 clean, 0 faults, identical `260/148` split and identical
  VRAM to read-write sharing.

## 3. Throughput investigation — root cause, fully attributed

Cost decomposition (steps 20 vs 100, N=1, batch 8×1024, isolates fixed cost from
per-step cost):

| | fixed cost | per-step cost |
|---|---|---|
| native | 14.9 s | 0.615 s |
| fork clone | **48.1 s** | **0.797 s** (~30% tax) |

The daemon's own profiler (`SMOLVM_CUDA_HOST_PROF=1`) attributes the per-step cost:

```
ops=1,114,112 (= 55,706 ops/STEP)
idle=9410ms (73%)   <- daemon WAITING on the guest
exec=2245ms (18%)   <- actual GPU/driver work
respond=1019ms (8%)
decode=141ms (1%)
```

The daemon is idle 73% of the time — **guest-bound, not host- or GPU-bound.** An
op-size-matched launch-rate benchmark (tiny kernel, tight loop, one sync) shows the
clone's per-op *work* is at native parity (**4.38 µs/launch vs native 4.47**), so the
30% tax is not a slow operation anywhere — it is **op count × a small per-crossing
cost**, multiplied across ~55.7k ops/step.

The ~33s/clone fixed-cost gap is **not** module reload (`prewarm_clone_worker` already
eagerly reloads all 482 modules in 1.4–1.7 s) — root cause not fully pinned, deprioritized
once the graph-capture lever was found (§6), since capture removes the steady tax
entirely and the fixed cost only matters for very short runs.

## 4. Eliminated hypotheses (with evidence — do not re-try these)

| hypothesis | test | result |
|---|---|---|
| Allocation churn in the launch path (`params.to_vec()`, double-copy) | Implemented an allocation-free launch path (`encode_launch_into` + reusable scratch buffer), byte-equality tested, deployed, measured | **No gain** (2665→2549 tok/s, noise). Reverted — it also gratuitously rotated `PROTO_HASH`. |
| Doorbell/park thrash | Read the host serve loop | Already spins 20,000 iterations before parking; measured idle is 8.2 µs/op, consistent with spin not park/wake. |
| VM CPU/memory limits (4 vCPU / 8 GiB) | Swept golden `--cpus 4/8/16` and `--mem 8192/32768` | **Flat**: 2502 / 2464 / 2506 tok/s across 4→16 vCPU. 32 GiB made it *worse* (1906) — clone has idle CPU. |
| Guest virtualization overhead (single-thread CPU) | Int-loop / method-call / alloc microbenchmark, host vs guest | Parity: 1.02–1.12x. |
| Per-op *latency* (round-trip cost) | `sync_empty`, `sync_after_kernel`, `event_query`, 4-byte D2H | Parity: +3 µs, not the 50–100 µs a slow path would need. |
| A pathological cuBLASLt path ("157x slower") | Re-measured with an op-size-matched benchmark instead of a 200-iteration 4096³-matmul loop | **My own benchmark artifact** — the first run timed the clone's *first* post-fork ops (real one-time cost) and divided by 200, mistaking a fixed cost for a per-op rate. |
| `torch.compile` in the unsloth+TRL workload | Instrumented `torch._dynamo` state before/after `import unsloth` | Not a smolvm issue: unsloth's patching sets `dynamo.suppress_errors = True`, which makes `torch.compile` **silently fall back to eager** — `first_call_s` went from 6.29s (real compile) to 0.0s (no-op) purely from the import. Flags are accepted; nothing traces. This is why boundary-level capture (§6) is the only approach that works for *any* workload — it never asks the framework's permission. |

## 5. The structural ceiling (separate from §3/§4 — not addressed by graphs)

Saturation curve (batch 8×1024, steps=50):

| arm | N | vCPU each | agg tok/s | SM util | GPU mem |
|---|---|---|---|---|---|
| native | 1 | 4 | 8,943 | 35% | 7.6 GB |
| native | 4 | 4 | **20,796** | 92% | 30.4 GB |
| fork | 1 | 4 | 4,782 | 28% | 9.8 GB |
| fork | 4 | 4 | **13,702** | 89% | 16.9 GB |
| fork | 8 | 2 | 11,008 | 86% | 26.3 GB |
| fork | 16 | 2 | 12,399 | 89% | 45.1 GB |

**Fork plateaus at ~13.7k tok/s (66% of native) regardless of N** — N=4 beats N=8 and
N=16, so oversubscribing further *hurts*. Not CPU-bound (§4). Cause: each clone worker
owns its own CUDA context; the GPU **time-slices** between them (N clones + the golden).

Shared-context mode exists (`BENCH_FORK_WORKERS=0` — all clones on streams of *one*
context) but is not a drop-in fix: tested at N=4, result was `0/4 learners, timeout`,
`shared=0`. By design this mode is for *"resume the golden's exact work"*
(checkpoint/continue, a single successor) — independent serving needs per-clone address
translation the shared-context path doesn't do. **Closing this ceiling needs a real
single-context/multi-stream redesign with per-clone translation — not started, and
should be an explicit decision, not something attempted incidentally inside this work.**

## 6. CUDA graph capture — the validated lever

Per-op work is already at native parity (§3/§4); the only way left to cut the ~30% tax
is to **issue fewer ops** — CUDA graphs collapse thousands of launches into one replay.

### A1 — does replay help *through* the remoting boundary? **Yes, decisively.**

| arm | eager µs/op | graph µs/op | speedup |
|---|---|---|---|
| native | 4.551 | 1.224 | 3.7x |
| VM (remoted, GPA ring, no fork) | 3.304 | **1.241** | 2.7x |

Through the boundary, replayed per-op cost is within **1.4%** of native's own graph
number. One `GraphLaunch` crossing carries K kernels, so the ~3 µs/op crossing tax
stops mattering once K is large enough. Capture itself: 40 ms for K=500.

### A2/A3 — is the real workload's op stream capturable? **Mixed — this is the gate.**

Instrumented the daemon (`SMOLVM_CUDA_OPSTREAM_PROBE=1`) to hash every op's bytes into
two rolling hashes per sync boundary, on the real DPO workload:

- `ops_hash` (op bytes only) → **does the sequence repeat?** — **YES.** 291 iterations
  recorded, only 138 distinct `ops_hash` values (one repeated 77 times consecutively).
  The op *sequence* is periodic (A3 confirmed).
- `full_hash` (whole encoded request, i.e. including arguments) → **do the arguments
  repeat too?** — **NO.** All 291 iterations had a distinct `full_hash`. CUDA graphs
  bake in device addresses, so naive capture-and-replay would use stale/wrong values.

**Open question, not yet answered:** is the `full_hash` divergence caused by device
*pointers* moving between iterations (fatal — every step would need a fresh capture,
probably no net win) or by legitimate *scalars* changing (step counters, stream/event
handles, loss scale — fixable by marshalling those into a small patchable region before
replay)? The probe as written conflates the two. **This is Phase 0 below and should be
run before any capture-engine code is written.**

### Clone-side capture reliability — currently unreliable, not fundamentally broken

- A hand-rolled bisection (stream create → side-stream warm → `CUDAGraph()` alloc →
  pre-resolve kernel → capture → replay) **succeeded once, fully**, in a fresh clone.
- A capture-length sweep (k=1,4,16,...,500) **failed at k=1** in a separate run with
  `CUDA error: unknown error` — same operation, different preceding state.
- ⇒ **order/state-dependent bug, not a capability gap.** Fixable.
- Diagnosed with a first-failure reporter added to the daemon
  (`SMOLVM_CUDA_LOG_ERRORS=1`, covers both a non-zero dispatch status *and* a
  `LibResult` with a non-zero `lib_status` — the latter is missed by naively checking
  dispatch status alone, and is exactly the kind of failure a fresh cuBLAS/cuDNN handle
  in a clone would produce). Reproduction with this logging armed was in progress when
  this doc was written and needs to be re-run (§8).

## 7. Implementation plan (phased, each with a go/no-go gate)

### Phase 0 — Pointer-vs-scalar disambiguation (~half day, do this first)

Extend the op-stream probe with a **third hash over pointer-like argument bytes only**
(8-byte windows resolving inside a known device allocation — the daemon already runs
this exact scan for `LaunchKernel` arg translation, see `xlat` in
`crates/smolvm-cuda/src/host.rs`), plus a per-arg-slot variance map.

| outcome | verdict |
|---|---|
| pointers stable, only scalars vary | proceed — capture is viable if scalars are marshalled into a small patchable region before each replay |
| pointers vary every iteration | **stop** — re-capture-per-step is probably not a net win; do not build the engine |
| both vary with no structure | **stop**, feature is dead as designed |

Do not write Phase 2 before this gate.

### Phase 1 — Fix clone graph capture (prerequisite, 1–2 days)

A fork pool that can't capture in its clones gets none of this benefit. Use the
`SMOLVM_CUDA_LOG_ERRORS=1` reporter to attribute the `k=1` failure to a specific op +
driver/lib status, then fix that path. Deliverable: the `graph_len.py` sweep (1→500)
passes reliably in a clone, matching native's behavior.

### Phase 2 — Capture engine (3–5 days, only after Phase 0 says "proceed")

- **Segment detection**: hash the op stream between syncs; identify the repeating cycle
  (A3 already proves one exists for this workload).
- **Capture**: on the *n*-th identical `ops_hash`, wrap the segment in
  `cuStreamBeginCapture` in the daemon's context, instantiate, cache by hash.
- **Replay gate**: replay only when `ops_hash` **and** the pointer-arg hash (from Phase
  0) match the captured instance; any mismatch → fall back to eager for that iteration.
  Fail-safe by construction — mirrors how weight-sharing verification already gates on
  content (§2).
- Reuses existing primitives from the P3b clone-graph work (`#695`): `capture_rec`,
  `clone_graph_oplogs`, `replay_capture_graph`, `rebuild_clone_graphs`.

### Phase 3 — Correctness hardening (2–3 days)

- Segment boundaries must exclude non-capture-safe ops (H2D from host memory,
  allocations, event queries) — split segments around them, don't try to capture through
  them.
- Scalar-argument handling per Phase 0's finding (static-buffer marshalling, or refuse to
  capture segments whose scalars can't be pinned to a stable location).
- **Numerical equivalence test**: same workload, graphs on vs off, losses must match —
  same bar used to validate weight sharing (§2's bit-identical losses).

### Phase 4 — Measurement & rollout (1 day)

Re-run the saturation curve (§5's table). Target: move fork's plateau from 13.7k toward
native's 20.8k (A1 showed the per-op tax nearly vanishes under replay — this is where
that shows up in a real training loop). Ship **opt-in** first
(`SMOLVM_CUDA_AUTOGRAPH=1`); default-on only once Phase 3's equivalence test is green —
same discipline used for weight-sharing's default (§2).

### Explicit non-goals for this work

- No more per-op micro-optimization — measured at parity twice (§4), there is nothing
  left to shave there.
- Not bundling the single-context/multi-stream redesign (§5) into this — that's a
  separate, larger architectural decision.
- Not depending on framework cooperation (`torch.compile`, `reduce-overhead`) — §4 shows
  why that's a dead end for at least the unsloth+TRL stack; capture must happen
  transparently at the daemon, below any framework.

## 8. Current repo state — uncommitted work, read before it's lost

Branch `fork-share-safety` is now **stale**: PR #742 squash-merged as `181248f`, so the
branch's own commit history is no longer an ancestor of `origin/main` (expected for a
squash merge — the content is in main, the commits aren't). `origin/main` has also moved
past that point (`711a12c` GPU/DRI libkrunfw bump, `a31810e` README docs — neither is
mine).

**Uncommitted on disk right now** (this is what produced every number in §3/§6 — if it's
lost, the diagnostic tooling has to be rewritten from scratch):

| file | state | contents |
|---|---|---|
| `crates/smolvm-cuda/src/host.rs` | modified | `[op-err]` first-failure diagnostic (`SMOLVM_CUDA_LOG_ERRORS=1`, covers dispatch status *and* `LibResult` lib-status); op-stream periodicity/pointer-stability probe (`SMOLVM_CUDA_OPSTREAM_PROBE=1`) |
| `bench/bench.sh` | modified | self-verifying `--share on\|off\|default` (checks the daemon's own `M2` log line, ANSI-escape-stripped), `--vcpu`/`--vmem` for the golden, `--batch`/`--maxseq`, `BENCH_FORK_WORKERS` shared-context switch, `--timeout` |
| `bench/workload_dpo.py` | modified | `GRAPHS=1` attempts `torch_compile(reduce-overhead)` — does not currently engage, see §4's unsloth finding |
| `bench/mm_bench.py`, `op_bench.py`, `lat_bench.py`, `launch_rate.py`, `cpu_bench.py`, `graph_rate.py`, `graph_probe.py`, `graph_len.py` | untracked | the microbenchmarks behind every number in §3, §4, §6 |

**Recommendation:** before continuing this work, cut a fresh branch off current
`origin/main` and commit the above (host.rs diagnostics as one focused commit, the bench
scripts as another) — same pattern used for `fork-pool-ops` → `fork-share-safety`
earlier in this investigation. Not done automatically here since it wasn't asked for.

## 9. Where the supporting data lives

- **On the H100 box** (`ubuntu@192.222.53.56`): `~/bench/` (harness + scripts),
  `~/bench_run/*/daemon.log` (per-run logs — `M2` sharing verdict, `[op-err]`,
  `[opstream]`, `[serve-prof]` lines), `~/bench/results/*.json` (raw run data with full
  environment manifests).
- **In this repo**: `bench/results/*.json` (11 committed in #742), `bench/README.md`
  (harness usage), `bench/RESULTS.md` (generated summary table).
- **Session memory** (cross-conversation continuity, not in this repo):
  `fork-vs-native-benchmark.md`, `cuda-fork-weight-sharing-fix.md`,
  `cuda-clone-procmem-transport.md`.
