# CUDA fork-pool throughput investigation & graph-capture plan

Status: the original graph-placement gates are resolved. Transparent daemon-side
auto-capture is a **no-go**; explicit CUDA graphs initiated above the per-op API
boundary are a **go**. Before changing a workload, the remaining transparent layers
are being exhausted and bounded (§7). No production performance feature is
implemented.

## 1. TL;DR

- The fork pool is now **production-safe for density**: weight sharing was silently
  inert (a real bug, fixed), is on by default, and a post-fork base write now fails
  loudly in one clone instead of corrupting every sibling silently. Shipped in
  [#741](https://github.com/smol-machines/smolvm/pull/741) and
  [#742](https://github.com/smol-machines/smolvm/pull/742) (both merged).
- **Native still wins raw throughput at every configuration measured.** The steady
  gap tracks CUDA op count and per-call guest/framework work. Host-side launch
  suppression does not improve it; even returning at the start of the guest shim only
  moves 3.76 to 2.89 µs/launch. The original claim that the entire gap was a
  ~3 µs transport tax was too strong (§3).
- **Explicit CUDA graphs are still the one validated large lever.** Through the
  boundary, a K=500 graph measures 1.241 µs/op versus native's 1.224. Fresh fork
  clones now pass capture and exact-readback tests reliably (§6).
- **Transparent daemon auto-capture is ruled out.** It sees all K guest crossings
  before it can recognize a segment, so it cannot remove the dominant guest-side
  work. In addition, the upgraded real-DPO probe found moving device pointers in
  every repeated structural group. Hash-and-replay would therefore be unsafe (§6).
- **Generic transparent transport optimizations have now been tested and rejected:**
  socket batching is ~2x slower than the ring, compound ring records do not improve
  launch rate, 4 KiB records remove 1,126 blocking launches but regress DPO throughput,
  and an exact TMA-descriptor cache hits ~89% without an end-to-end win (§7).
- The remaining pre-workload investigation is deliberately narrow: make the previous
  silent `torch.compile` fallback fail loudly, determine whether runtime-only graph
  activation can work without source changes, bound safe synchronization elision, and
  calculate the maximum possible benefit of the residual blocking classes (§7).
- If those transparent gates fail, the next path is an explicit graph at the
  workload/framework boundary: fixed-shape batches copied into static device buffers,
  warm up once per clone, capture a large training region, and replay it. The target
  application must stop issuing the K eager calls; daemon recognition cannot do that
  (§8).
- Separately, fork's *saturated* ceiling (~13.7k tok/s regardless of N) remains an
  architectural question. Per-clone contexts and scheduling are the leading
  explanation, but the current data do not prove that they are the only cause. A
  single-context/multi-stream redesign remains a separate project (§5).
- The original diagnostics are committed on the investigation branch. New boundary,
  pointer-classification, and fresh-clone probes are currently uncommitted (§8).

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

## 3. Throughput investigation — mechanism located, attribution bounded

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
clone's complete per-op path at native parity (**4.38 µs/launch vs native 4.47**).
That establishes that there is no catastrophic host slow path, but it does not assign
all guest time to transport.

Two deliberately wrong-result lower-bound probes locate that time more precisely:

| arm | behavior | µs/launch | daemon RPC count |
|---|---|---:|---:|
| normal | full guest → daemon → driver path | 3.76 | ~244k |
| host no-op | daemon acknowledges `LaunchKernel`, no driver submission | 4.37 | ~244k |
| guest-shim no-op | return before encoding/ring publication | **2.89** | ~42k |

Four normal/host-no-op pairs gave normal **3.16–3.95** and host-no-op
**3.55–4.37 µs/launch**: host suppression is never materially faster. Returning at the
start of the guest shim removes the 200k launch RPCs but saves only ~0.87 µs/launch.
The remaining ~2.89 µs is above that boundary (PyTorch/dispatcher/kernel launch
preparation and call overhead). Therefore:

- daemon-side recognition cannot deliver the graph result because all crossings have
  already happened;
- a transparent guest-shim trace cache has a limited ceiling because the framework
  still prepares and invokes every CUDA call;
- explicit graph replay wins because the framework itself issues one `GraphLaunch`
  instead of K eager launches.

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
| Default `torch.compile` attempt in the unsloth+TRL workload | Instrumented `torch._dynamo` state before/after `import unsloth` | The measured arm was **not a valid compile result**: unsloth's patching sets `dynamo.suppress_errors = True`, so it silently fell back to eager (`first_call_s` 6.29s before import, 0.0s after). Do not cite this as proof that compile cannot work. The next diagnostic disables suppression and captures the first real failure (§7). |

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
N=16, so oversubscribing further *hurts*. It is not explained by the VM CPU limits
tested in §4. Each clone worker owns its own CUDA context, and multi-context scheduling
is the leading explanation, but this sweep does not prove causality or exclude another
shared bottleneck. The six-clone graph stress test also produced K=500 replay rates
from 1.26 to 2.68 µs/op under concurrency, consistent with scheduling variance.

Shared-context mode exists (`BENCH_FORK_WORKERS=0` — all clones on streams of *one*
context) but is not a drop-in fix: tested at N=4, result was `0/4 learners, timeout`,
`shared=0`. By design this mode is for *"resume the golden's exact work"*
(checkpoint/continue, a single successor) — independent serving needs per-clone address
translation the shared-context path doesn't do. **Testing or closing this ceiling needs
a real single-context/multi-stream experiment with per-clone translation — not started,
and should be an explicit decision, not something attempted incidentally inside graph
work.**

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

### A2/A3 — can the daemon safely recognize and replay DPO segments? **No.**

The first probe proved that opcode sequences repeat but conflated pointers and scalars.
The upgraded probe is per connection and records, per synchronization segment:

- opcode, full-request, structural-shape, device-pointer, non-pointer, and handle
  hashes;
- pointer-like 8-byte words only when they resolve inside a device allocation known to
  that session;
- structural metadata for kernel launches and library calls without hashing their
  changing argument values.

On the real eight-step DPO workload it recorded **291 segments / 169 structural
groups**. Repeated groups existed (the largest appeared nine times), but **every
repeated group had a distinct pointer hash on every occurrence**. Handles were often
stable; device pointers were not. The old open question is resolved:

- periodic opcode/shape sequences: **yes**;
- stable device-pointer arguments: **no**;
- safe daemon hash-and-replay: **no-go**.

Capturing an observed segment and replaying it on the next matching opcode hash would
use stale addresses. Correcting that below the framework would require understanding
tensor semantics and patching graph nodes, not merely hashing the wire stream.

### Placement probe — where must graph recognition happen?

The no-op measurements in §3 give an independent no-go:

- daemon recognition is too late because K requests have already crossed;
- guest-shim recognition is also below most of the remaining ~2.89 µs/call;
- only explicit graph replay above the eager API sequence prevents the framework from
  preparing and issuing K calls.

This invalidates the original daemon capture-engine design even independently of the
moving-pointer result.

### Clone-side capture reliability — **go**

`graph_fresh_trial.py` performs exactly one capture in each fresh clone, independently
varying K=1/K=500 and standard/preallocated/double-warm state, then replays and checks a
device readback.

- Corrected gated harness: **12/12** fresh clones passed across two six-clone runs.
- Stress arm: six clones × 400 replays, **6/6 exact readbacks**.
- K=500 under concurrency: **1.26–2.68 µs/op**. K=1: 4.69–6.67 µs/op, confirming that
  graphs must cover a substantial region to amortize replay overhead.
- One attempted 12-clone run stopped at clone 5 because the VM agent exited during
  startup, before capture. The immediate six-clone retry passed; track this as a
  separate high-fan-out clone-start reliability issue.

The harness originally released the golden before all clones were forked, creating a
snapshot-state race. It now forks every clone while the guest waits, then releases the
gate. `[op-err] GpaDtoH ... st=500` still appears during successful readback because
the fast GPA path can fail before the normal D2H fallback succeeds; it is not a graph
failure.

## 7. Transparent smolvm investigation before workload changes

This section separates three materially different product experiences:

1. **Pure smolvm transparency:** changes stay below the application's CUDA/framework
   calls. The user runs the same program with the same arguments.
2. **Runtime/framework activation:** smolvm supplies environment or runtime
   interception, but the workload source is unchanged. This may still affect framework
   behavior and must be disclosed and safely fall back.
3. **Workload integration:** the user or an adapter changes configuration/code to use
   static buffers and an explicit captured region.

Only (1) qualifies as "without injection." The experiments below test that layer
before proceeding to (2) or (3).

### Completed transparent transport/cache prototypes — all no-go

All prototypes preserved CUDA results and were measured on the same H100. Prototype
code that failed its gate is reverted rather than left in the production path.

| prototype | measured result | decision |
|---|---|---|
| Disable the shared-memory ring and use the deferred socket batching path | Ring: 4.10 / 3.92 / 3.56 µs per launch. Socket: 8.49 / 7.74 / 8.42. | **No-go.** Blocking socket round trips are roughly 2x slower; the ring is the correct generic transport. |
| Put 2/4/8/16 quiet requests in one compound ring record | Baseline 3.87 µs; B2 4.25; B4 3.87; B8 5.21; B16 4.04. Paired B4 comparisons also straddled baseline noise. | **No-go.** Fewer publications do not remove framework call preparation and add buffering cost. |
| Increase ring records from 1 KiB to 4 KiB so all kernel-launch payloads can be quiet | Blocking ops fell 8,049→6,921 and all 1,126 formerly blocking launches became quiet. Twenty-step DPO runs were 2,417 and 2,355 tok/s versus historical 1 KiB median ~2,546. | **No-go.** It removes the intended barriers but regresses end-to-end throughput ~5–7%, consistent with worse queue locality. |
| Exact guest-side cache for 152-byte `cuTensorMapEncodeTiled` inputs and 128-byte descriptors, invalidated on clone reconnect | Clone hit ratio reached 2,280/2,560 (~89%). Twenty-step DPO runs were 2,574 / 2,475 / 2,459 tok/s, median 2,475 versus baseline median ~2,546; losses remained exact. | **No-go for performance.** High repetition does not imply material wall-clock cost. The reconnect-generation mechanism was required because a Firecracker clone preserves the guest PID. |

### Unchanged DPO operation census

A two-step, N=1 fork run produced 116,409 boundary operations:

- 108,360 quiet and 8,049 blocking with the production 1 KiB ring;
- 26,228 quiet plus 1,126 blocking `LaunchKernel` calls;
- 2,688 blocking `LibCall(6:0)` calls (`cuTensorMapEncodeTiled`);
- 2,332 blocking D2H copies;
- 784 context synchronizations, 295 stream synchronizations, and smaller event/VMM
  classes.

All 1,126 blocking launches were only 1.1–3.5 KiB; the 4 KiB experiment therefore
isolated record capacity cleanly. Its throughput regression and the TMA cache result
show why call counts alone are not enough: every remaining proposal needs a measured
end-to-end gate.

### Remaining pure-smolvm questions

These are not yet validated improvements:

1. **Safe barrier elision/coalescing.** Classify which D2H, synchronization, and event
   calls actually expose results to the CPU or order later work. Elide only barriers
   whose data dependencies prove them redundant; call-type-only suppression is
   incorrect.
2. **Upper bound by residual class.** Measure time, not just count, for each blocking
   class. Stop if eliminating an entire class cannot materially move step time.
3. **Earlier transparent sequence handling.** Determine whether any stable sequence
   can be recognized before PyTorch performs its per-op preparation. The current
   evidence already rules out the daemon and puts a tight ceiling on a CUDA-shim-only
   trace cache; the investigation should close this with a quantified bound rather
   than assume it.

### Unchanged-source runtime activation (category 2, diagnostic only)

The previous Unsloth compile arm silently fell back to eager because
`torch._dynamo.config.suppress_errors` was enabled. Run the unchanged benchmark source
with suppression disabled and capture the first graph break/unsupported operation.
Then determine whether configuration supplied outside the workload can activate a real
compile/graph path. This does **not** establish pure smolvm transparency; it determines
whether source edits can be avoided.

Go only if logs and boundary counts prove that tracing/capture occurred, numerical
results match eager, and end-to-end throughput improves. An accepted flag followed by
eager execution is a failed result.

## 8. Revised implementation path (phased, each with a go/no-go gate)

### Phase 0 — daemon autographs: **complete, no-go**

Do not implement segment detection/capture/replay in the daemon. Both required
assumptions failed: recognition is below the cost that needs to be removed, and
repeated real-workload segments do not retain pointer arguments.

### Phase 1 — explicit graph substrate in fork clones: **complete, go**

Current cudart capture/instantiate/launch forwarding works in fresh isolated clone
contexts and retains exact results in the synthetic graph probe. No clone graph fix is
currently justified. Preserve the corrected reliability test as a regression.

### Phase 2 — actual DPO graphability prototype (next; no engine code first)

Prototype one real training region using PyTorch's explicit `torch.cuda.CUDAGraph`
contract, which smolvm already forwards:

1. Force fixed batch/sequence shapes and preallocate static device buffers.
2. Copy each new batch into those buffers; do not expose changing tensor addresses to
   captured kernels.
3. Warm the exact forward/backward/optimizer region on a side stream in each clone.
4. Capture the largest safe region. If the whole step contains a capture-unsafe host
   operation, split at that operation rather than shrinking immediately to tiny
   graphs.
5. Replay for at least 20 steps with eager versus graph loss/parameter checks and
   daemon op counts.

Go only if all of these hold:

- exact or tolerance-defined numerical equivalence is green;
- K is large enough to amortize replay (the K=1 probe is explicitly insufficient);
- boundary-visible op count collapses by at least an order of magnitude;
- end-to-end N=1 fork throughput improves materially, not just the microbenchmark.

If capture fails, first attribute the exact unsupported CUDA/runtime operation. Add a
missing graph API to smolvm only when the real workload proves it is required. Do not
build a general graph-update engine speculatively.

### Phase 3 — fork saturation measurement

After Phase 2 passes, run eager versus graph at N=1 and N=4 first, then N=8/N=16:

- throughput, step latency, SM utilization, and aggregate VRAM;
- final losses/parameters;
- graph capture/replay counts and daemon op counts;
- per-clone fairness under concurrent contexts.

This separates removal of the eager-call tax from the independent saturation ceiling.
The target is to move the N=4 fork result toward native's 20.8k tok/s; A1 proves the
mechanism but does not guarantee that end-to-end result.

### Phase 4 — productization choice

If the DPO prototype wins, choose the narrowest useful integration:

- an explicit graph-enabled workload adapter/static-buffer runner for known training
  stacks; or
- a small application-visible capture contract for serving/training loops that can
  guarantee fixed shapes and stable storage.

Keep it opt-in until eager/graph equivalence and fallback behavior are tested. A
transparent `SMOLVM_CUDA_AUTOGRAPH=1` daemon feature is no longer proposed.

### Separate decision — single-context/multi-stream architecture

Only after graph-enabled saturation data exist should the larger per-clone address
translation/shared-context project be approved. Graphs reduce API issue overhead; they
do not by themselves prove or remove a multi-context scheduling ceiling.

### Explicit non-goals for this work

- No daemon-side hash-and-replay engine.
- No more per-op host micro-optimization; host launch suppression already falsified
  that path as a material lever.
- No speculative graph-node update API work before the actual DPO capture identifies a
  concrete need.
- No bundled single-context redesign.

## 9. Current repo state

Branch: `cuda-graph-capture-investigation`, based on `a31810e`.

The investigation and validation work is preserved in the following commits and at
the current branch tip:

| commit | contents |
|---|---|
| `39b8a35` | host first-failure and original opstream diagnostics |
| `c6ed98b` | reproducible benchmark/probe tooling |
| `9bc21e1` | the original investigation plan |
| `cdf7337` | placement no-ops, upgraded pointer probe, and corrected fresh-clone graph matrix |
| current branch tip | transparent transport/cache no-go results and operation-census tooling |

The latest validation files are:

| file | contents |
|---|---|
| `crates/smolvm-cuda/src/host.rs` | per-session structural/pointer/non-pointer/handle opstream hashes; diagnostic host launch no-op |
| `crates/smolvm-cuda/src/client.rs` | diagnostic guest-shim launch no-op |
| `bench/analyze_opstream.py` | groups repeated structures and classifies pointer stability |
| `bench/run_boundary_floor.sh` | normal/host-no-op/guest-no-op placement benchmark |
| `bench/graph_fresh_trial.py` | one capture per fresh clone with exact readback and replay timing |
| `bench/run_graph_fresh_trials.sh` | corrected fork-before-release clone matrix |
| `bench/analyze_oplog.py` | counts quiet/blocking operations by process and operation class |
| `bench/run_transport_matrix.sh` | paired shared-memory-ring versus socket transport launch-rate harness |

The untracked `demo/*` files belong to separate user work and were not modified.
Safety snapshots:

- `/home/binsquare/smolvm-graph-snapshots/demo-untracked-20260725.tgz`
- `/home/binsquare/smolvm-graph-snapshots/validation-pre-h100-20260725.tgz`

## 10. Where the supporting data lives

- **On the H100 box** (`ubuntu@192.222.53.56`): `~/bench/` (harness + scripts),
  `~/bench_run/*/daemon.log` (per-run logs — `M2` sharing verdict, `[op-err]`,
  `[opstream]`, `[serve-prof]` lines), `~/bench/results/*.json` (raw run data with full
  environment manifests), `~/boundary_floor_guest_1/` (placement result),
  `~/coord_os/daemon.log` (upgraded real-DPO opstream), and
  `~/coord_graph_perf_2/` / `~/coord_graph_reliability_corrected_2/` (corrected clone
  graph results). The unchanged-DPO operation censuses are in
  `~/bench_run/fork_n1_s2_c4_20260725-200836_r1/` (production 1 KiB ring) and
  `~/bench_run/fork_n1_s2_c4_20260725-202926_r1/` (temporary 4 KiB ring).
- **In this repo**: `bench/results/*.json` (11 committed in #742), `bench/README.md`
  (harness usage), `bench/RESULTS.md` (generated summary table).
- **Session memory** (cross-conversation continuity, not in this repo):
  `fork-vs-native-benchmark.md`, `cuda-fork-weight-sharing-fix.md`,
  `cuda-clone-procmem-transport.md`.
