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
  an exact TMA-descriptor cache hits ~89% without an end-to-end win, deferred VMM
  unmaps only move the wait, and direct clone-RAM copies regress throughput (§7).
- **NVIDIA MPS is the first validated workload-transparent speed lever.** At N=4,
  two 50-step controls measured 12,905 / 13,058 aggregate tok/s and two MPS runs
  measured 14,993 / 15,495: a **17.4% median gain**, 4/4 completion, the same
  `shared=260 private=148` split, and only 39 MiB more peak VRAM. All four clone
  workers were observed as clients of one MPS server. At N=8, a paired arm improved
  **17,152→22,597 tok/s (+31.7%)**, completed 8/8, and retained bit-identical loss
  endpoints per learner. At N=16, a 20-step density pair improved
  **12,552→14,915 tok/s (+18.8%)** with 16/16 completion. MPS improves multi-context
  scheduling without changing the workload, but it does not remove the eager
  per-operation tax (§5/§7).
- Runtime-only `torch.compile` activation and the remaining synchronization/transport
  alternatives have now been closed as no-go. The remaining transparent validation is
  MPS numerical-repeatability bounds and an immutable-build confirmation; basic
  lifecycle/restart/fallback probes are green (§7).
- After the uncapped-MPS implementation, the next *additional* lever is an explicit
  graph at the workload/framework boundary: fixed-shape batches copied into static
  device buffers, warm up once per clone, capture a large training region, and replay
  it. That later step changes/integrates with the workload because the target
  application must stop issuing K eager calls; daemon recognition cannot do that (§8).
- The old ~13.7k "hard ceiling" is superseded: the current paired N=8 control reached
  17,152 and uncapped MPS reached **22,597**. That directly validates multi-context
  scheduling as a material limiter. A full single-context/multi-stream redesign is now
  lower priority than productizing the much narrower MPS path (§5).
- The investigation diagnostics and transparent transport results are committed on
  the investigation branch; the latest MPS/lifecycle findings and clone-RAM tracing
  are in the current working tree (§9).

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

**The original no-MPS sweep appeared to plateau at ~13.7k tok/s (66% of native)** —
N=4 beat N=8 and N=16, so oversubscribing further hurt in that sweep. It was not
explained by the VM CPU limits tested in §4. Each clone worker owns its own CUDA
context, and multi-context scheduling was the leading explanation. The six-clone graph
stress test also produced K=500 replay rates from 1.26 to 2.68 µs/op under concurrency,
consistent with scheduling variance. The current paired N=8 control later disproved
13.7k as a stable hard ceiling, but not the scheduling attribution.

A paired MPS prototype now provides direct evidence for that attribution. With the
unchanged DPO workload at N=4, steps=50, batch=8, maxseq=1024:

| mode | run 1 | run 2 | median | peak VRAM |
|---|---:|---:|---:|---:|
| ordinary per-clone contexts | 12,905 | 13,058 | 12,981.5 tok/s | 16,852 MiB |
| same workers under NVIDIA MPS | 14,993 | 15,495 | **15,244 tok/s** | 16,891 MiB |

That is a **17.4% median throughput improvement**. Every run completed 4/4 learners
and reported `shared=260 private=148`; live MPS queries listed the daemon plus all four
clone workers as clients of the same server. The controller/server were stopped after
each arm and the GPU returned to 0 MiB. This is a pure-smolvm-transparent candidate:
smolvm can manage the host MPS lifecycle and environment while the guest runs the same
program and arguments.

The N=4 result raises the current fork rate from 62% to 73% of the paired native N=4
reference (20,796 tok/s), but does not close that same-N gap. At N=8, the next paired
observation was stronger:

| mode | aggregate tok/s | per-clone range | completion | peak VRAM |
|---|---:|---:|---:|---:|
| ordinary per-clone contexts | 17,152 | 2,133–2,189 | 8/8 | 26,260 MiB |
| same workers under NVIDIA MPS | **22,597** | 2,718–2,915 | 8/8 | 26,321 MiB |

That is **+31.7%** with only 61 MiB additional peak memory. Every learner's reported
loss endpoints were bit-identical between the paired arms. Live control queries listed
the daemon plus all eight clone workers under one MPS server, and cleanup again
returned the GPU to 0 MiB.

The N=8 ordinary control is itself much faster than the older 11,008 tok/s result in
the original sweep, despite matching its documented configuration. Therefore the old
13.7k "hard plateau" is not stable enough to retain as a universal ceiling; only
same-build paired arms should be used for the MPS decision. MPS addresses inter-context
scheduling; explicit graphs address the independent eager operation-issue tax.

The N=16 density pair (20 steps, so compared only within its pair) completed 16/16 and
improved 12,552→14,915 tok/s (**+18.8%**) at essentially unchanged peak memory
(45,070→45,054 MiB). MPS attached all 16 workers plus the daemon. The result remains
below N=8/MPS because 16 two-vCPU guests oversubscribe the 26-core host and per-clone
rates fall to 868–1,055 tok/s. The best measured performance point is therefore
**N=8 with MPS**, while N=16 remains the density point.

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
| Defer status-only `cuMemUnmap` and preserve unmap→release order | Guest sync timing attributed 1,390 ms to 220 inherited golden+clone unmaps. Correctness passed, but clean 20-step baseline was 2,530 / 2,497 tok/s (median 2,513.5) and deferred-unmap was 2,440 / 2,583 (median 2,511.5). All four losses were exactly 0.6862→0.4095. | **No-go.** The apparent synchronous time merely moves into a later dependency because the host must still execute the unmaps in order; paired throughput changes by effectively 0%. |
| Activate the existing direct clone-RAM `/proc/<pid>/mem` transport instead of bounce copies | The default warm dial accidentally starts a clone worker before its proc-mem advert arrives, so 2,332 D2H operations fall back to bounce copies. Disabling warm dial made all 2,332 use `MemcpyGpaDtoH` and made 139 H2D GPA copies succeed. A 20-step gate nevertheless fell from 2,616 to 2,196 tok/s with identical 0.6862→0.4095 loss. | **No-go for performance.** Thousands of tiny direct proc-mem reads are slower than the shared bounce path, and successful H2D GPA copies save only millisecond-scale time. The warm-dial advert bug is a correctness/transport cleanup, not a speed feature. |

### Validated transparent scheduling prototype — MPS go

NVIDIA MPS is below the guest workload and requires no CUDA/framework interception.
The prototype used dedicated pipe/log directories, let the smolvm daemon and clone
workers inherit them, and always sent `quit` through an exit trap. On the H100, MPS
accepted smolvm's existing CUDA VMM/imported-memory behavior, and all clone workers
were verified as MPS clients during the run.

At N=4, two paired 50-step controls and two MPS arms produced the 17.4% median gain
reported in §5. Per-clone throughput improved together rather than through an outlier:
the first pair was 3,208–3,245 tok/s per control clone versus 3,660–3,806 under MPS.
Weight density and completion were unchanged.

At N=8, the paired arm improved 17,152→22,597 tok/s (+31.7%). All eight clone workers
were visible as MPS clients, per-clone throughput rose from 2,133–2,189 to
2,718–2,915 tok/s, and each learner's reported loss endpoints matched its control
bit-for-bit. This both strengthens the scheduling attribution and shows that the MPS
gain does not trade aggregate throughput for a starving tail.

At N=16/20 steps, MPS improved 12,552→14,915 tok/s (+18.8%) with 16/16 completion and
no memory increase. This confirms compatibility at the maximum density point already
validated by the fork pool, but N=8 remains faster in aggregate because N=16
oversubscribes host CPUs. Several per-learner loss endpoints varied across the N=16
pair, as they also do across ordinary repeated N=4 runs; this is why a deterministic
numerical probe remains a productization gate rather than claiming bitwise DPO
repeatability from the current harness.

#### Active-thread partitioning — generic no-go

MPS can cap the fraction of active SM threads available to each client. A targeted
N=8 sweep tested whether explicit partitioning could improve on default MPS:

| mode | aggregate tok/s | per-clone range | peak VRAM | numerical result |
|---|---:|---:|---:|---|
| no MPS | 17,152 | 2,133–2,189 | 26,260 MiB | finite |
| MPS, uncapped | 22,597 | 2,718–2,915 | 26,321 MiB | bit-identical to paired control |
| MPS, 12% | 21,772 | 2,658–2,804 | 21,603 MiB | finite |
| MPS, 25% | 23,415 | 2,864–2,977 | 22,321 MiB | finite, endpoints drifted |
| MPS, 33% | 23,520 | 2,845–3,031 | 22,701 MiB | **all 8 learners NaN from first logged loss** |

The apparent 33% throughput lead is invalid. There were zero daemon operation errors,
and an isolation run of the same Unsloth workload **natively** under MPS/33% also
produced NaN from its first loss. The failure is therefore a workload/kernel
interaction with the MPS resource cap, not smolvm transport or clone state.

Decision: **do not set `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` transparently.** Uncapped
MPS is the generic candidate. A cap can save several GiB and 25% happened to be finite
and ~3.6% faster in this workload, but resource partitions are workload-visible in
practice and require explicit qualification/opt-in. They are not part of the pure
smolvm performance proposal.

This is a **go for further validation**, not yet a default-on implementation. Before
productizing:

1. repeat the positive N=8 point on the immutable production candidate; it is the
   current performance optimum, while N=16 is the validated density point;
2. compare repeated numerical endpoints against repeated ordinary and native runs;
   the harness already shows endpoint variation between ordinary runs, so exact
   bitwise cross-run equality is not a valid gate without first making the workload
   deterministic;
3. test controller discovery, stale-controller handling, daemon crash cleanup, and
   safe fallback when MPS is unavailable;
4. keep MPS scoped to compatible NVIDIA/Linux pool configurations and never silently
   change a system-wide MPS service owned by another user.
5. leave active-thread percentage uncapped by default; the 33% native isolation
   failure proves this tuning is not generically transparent.

The first lifecycle/fallback probe is green:

- a second controller start against the same private pipe directory fails explicitly
  with `An instance of this daemon is already running`;
- after terminating the owned controller abruptly, a new controller successfully
  reclaims the same directory despite stale entries;
- a real CUDA tensor operation succeeds with both a nonexistent controller directory
  and the stale post-crash directory, demonstrating ordinary CUDA fallback instead of
  a hang or initialization failure;
- graceful `quit` removes both controller and server processes; all performance arms
  also returned the GPU to 0 MiB.

Product code still needs an ownership lock/supervisor and must only stop a controller
it started. The safe design is a private smolvm pipe/log directory and fallback to
ordinary contexts when no owned controller is available.

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

The guest-side synchronous-call profiler then ranked full round-trip wall time. In
the useful post-fork interval:

- ordinary D2H readbacks dominated: 1,579 observed calls added ~2.42 s by the next
  snapshot (~1.53 ms/call, including the GPU dependency they wait on);
- stream synchronizations were frequent but cheap: 240 calls added ~0.4 ms;
- the separate driver-shim tally attributed 1,390 ms/220 calls to VMM unmap and
  601 ms/3,821 calls to TMA encoding across inherited golden+clone state.

The deferred-unmap paired result above is the important interpretation: synchronous
wall time is not automatically removable wall time. A barrier that waits on required
ordered work can only move that wait to the next consumer.

### Remaining pure-smolvm questions

The per-operation transport questions are now bounded. The remaining material
pure-smolvm question is how far MPS can move the multi-context ceiling:

1. **MPS policy confirmation.** Repeat N=8 on the immutable production candidate and
   define a private, ownership-aware controller lifecycle. N=8 is the current
   throughput optimum; N=16 remains the density configuration. Missing/stale
   controllers already fall back successfully in the driver probe.
2. **Numerical-repeatability bound.** Separate pre-existing stochastic/nondeterministic
   DPO variation from an MPS-specific effect. The current harness seeds its data and
   trainer but does not request deterministic CUDA algorithms.
3. **No speculative barrier suppression.** Ordinary D2H calls dominate observed
   synchronous wall time because the CPU consumes their results. Stream sync is cheap,
   deferred VMM unmap only moves the wait, and direct proc-mem D2H is slower. No
   dependency proof has exposed a material safely removable class.

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

This gate is now resolved **no-go for the installed Unsloth path**:

- `GRAPHS=1`, `torch_compile=True`, reduce-overhead mode, and
  `UNSLOTH_COMPILE_IGNORE_ERRORS=0` completed without an exception;
- the clone emitted exactly **116,409 boundary operations**, identical to the eager
  census, with **zero** `StreamBeginCapture`, `GraphInstantiate`, or `GraphLaunch`
  operations;
- installed Unsloth replaces Accelerate's compile kwargs and explicitly supplies
  `"triton.cudagraphs": False`.

The accepted configuration therefore did not activate graphs or reduce the eager
boundary sequence. Changing that override would be runtime/framework injection
(category 2), not a pure-smolvm improvement.

## 8. Revised implementation path (phased, each with a go/no-go gate)

### Phase 0 — daemon autographs: **complete, no-go**

Do not implement segment detection/capture/replay in the daemon. Both required
assumptions failed: recognition is below the cost that needs to be removed, and
repeated real-workload segments do not retain pointer arguments.

### Phase 1 — explicit graph substrate in fork clones: **complete, go**

Current cudart capture/instantiate/launch forwarding works in fresh isolated clone
contexts and retains exact results in the synthetic graph probe. No clone graph fix is
currently justified. Preserve the corrected reliability test as a regression.

### Phase 2 — transparent MPS scheduling validation: **go, uncapped only**

The performance, scaling, compatibility, and basic lifecycle/fallback gates in §7 are
green for uncapped MPS. The narrow next implementation is host-side MPS service
coordination for fork pools. It must require no guest/workload change, use a private
ownership-aware controller directory, leave active-thread percentage unset, and safely
fall back to the current per-context path.

This phase is complementary to graphs, not a replacement: the validated N=4 result
recovers 17.4%; the N=8 result recovers 31.7% and is the best current aggregate point.
Re-run N=8 on the actual implementation artifact before enabling it by default.

### Phase 3 — actual DPO graphability prototype (next workload-visible step; no engine code first)

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

### Phase 4 — graph-enabled fork saturation measurement

After Phase 3 passes, run eager versus graph at N=1 and N=4 first, then N=8/N=16:

- throughput, step latency, SM utilization, and aggregate VRAM;
- final losses/parameters;
- graph capture/replay counts and daemon op counts;
- per-clone fairness under concurrent contexts.

This separates removal of the eager-call tax from the independent saturation ceiling.
The target is to move the N=4 fork result toward native's 20.8k tok/s; A1 proves the
mechanism but does not guarantee that end-to-end result.

### Phase 5 — graph productization choice

If the DPO prototype wins, choose the narrowest useful integration:

- an explicit graph-enabled workload adapter/static-buffer runner for known training
  stacks; or
- a small application-visible capture contract for serving/training loops that can
  guarantee fixed shapes and stable storage.

Keep it opt-in until eager/graph equivalence and fallback behavior are tested. A
transparent `SMOLVM_CUDA_AUTOGRAPH=1` daemon feature is no longer proposed.

### Separate decision — single-context/multi-stream architecture

MPS should now be evaluated before approving the larger per-clone address
translation/shared-context project. It recovers part of the multi-context scheduling
loss without changing smolvm's per-clone isolation model. Graphs reduce API issue
overhead; they do not replace either scheduling approach.

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
| `75e12a1` | plan revised from graph placement, pointer, and clone-reliability results |
| `4fb3f0b` | transparent transport/cache no-go results and operation-census tooling |
| this commit | MPS scaling/lifecycle/cap findings, durable result summary, and clone-RAM/sync diagnostics |

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
| `bench/run_mps_matrix.sh` | private-controller, uncapped MPS versus ordinary-context paired harness with guaranteed cleanup |
| `bench/results/mps-h100-20260725.json` | durable machine-readable MPS performance, correctness, cap, and lifecycle summary |
| `bench/bench.sh` | benchmark manifests now record MPS pipe directory and active-thread percentage |
| `crates/smolvm-cuda/src/client.rs` | synchronous-call profiler now retains enough ranked classes to expose low-count barriers hidden by one-time module loads |
| `src/cuda_host.rs` | opt-in clone-RAM advert trace that exposed the warm-dial/proc-mem ordering bug |

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
  `~/bench_run/fork_n1_s2_c4_20260725-202926_r1/` (temporary 4 KiB ring). Runtime
  compile boundary evidence is in
  `~/bench_run/fork_n1_s2_c4_20260725-212612_r1/`; expanded sync timing is in
  `~/bench_run/fork_n1_s2_c4_20260725-214633_r1/`; the paired deferred-unmap gate is
  in the four `~/bench_run/fork_n1_s20_c4_20260725-{220029,220441,220910,221321}_r1/`
  directories and matching `~/bench/results/*.json` files. Clone-RAM advert and
  direct-proc-mem evidence is in `~/coord_clone_ram_trace/`,
  `~/coord_clone_ram_nowarm/`, and
  `~/bench_run/fork_n1_s{2,20}_c4_20260725-{222720,223210,223633}_r1/`.
  The paired MPS runs are:
  `fork_n4_s50_c4_20260725-{224347,224916,225448,230012}_r1`,
  `fork_n8_s50_c4_20260725-{230633,231305}_r1`, and
  `fork_n16_s20_c4_20260725-{231950,232655}_r1`. The active-thread sweep is in
  `fork_n8_s50_c4_20260725-{233715,234319,234922}_r1` (25%, 12%, 33%,
  respectively), and the native 33%-cap NaN isolation is
  `native_n1_s2_c4_20260725-235606_r1`. Matching JSON files are under
  `~/bench/results/`; MPS controller/server logs are under the corresponding
  `/tmp/smolvm-mps-log-20260725-*` directories.
- **In this repo**: `bench/results/*.json` (11 committed in #742), `bench/README.md`
  (harness usage), `bench/RESULTS.md` (generated summary table), and
  `bench/results/mps-h100-20260725.json` (the durable summary for this investigation).
- **Session memory** (cross-conversation continuity, not in this repo):
  `fork-vs-native-benchmark.md`, `cuda-fork-weight-sharing-fix.md`,
  `cuda-clone-procmem-transport.md`.
