# CUDA fork-pool throughput investigation & graph-capture plan

Status: the transparent MPS optimization is implemented and release-qualified on the
H100. At the N=8 throughput point, the final managed-smolvm artifact is **6.1% faster
than the fair same-N native control while using 56.7% less GPU memory**. All three
qualified managed N=8 runs exceed native; the slowest is still +1.0%. Transparent
daemon-side auto-capture remains a no-go, and the installed Unsloth DPO backward path
is also proven non-capturable by an explicit workload-level prototype (§6/§8).

Broader Unsloth SFT qualification is now complete at N=1 and N=4. It exposed two
fork-safety bugs not exercised by DPO: a VM can contain several independent CUDA
process lineages, and a late-attached channel needs the clone's live-RAM advert. Both
fixes pass real SFT, while exact model-output, RNG, loss, and 40.4-million-parameter
digests match fair native controls. SFT remains below native throughput, so this is a
correctness/reliability candidate rather than a new performance claim (§7).

## 1. TL;DR

- The fork pool is now **production-safe for density**: weight sharing was silently
  inert (a real bug, fixed), is on by default, and a post-fork base write now fails
  loudly in one clone instead of corrupting every sibling silently. Shipped in
  [#741](https://github.com/smol-machines/smolvm/pull/741) and
  [#742](https://github.com/smol-machines/smolvm/pull/742) (both merged).
- **The same-N throughput target is reached at N=8 and survives repeat
  qualification.** Managed artifacts measured **22,088, 23,015, and 23,195 tok/s**
  versus **21,865 tok/s native** with the same two CPU cores per learner (+1.0% to
  +6.1%), while using **26,281 versus 60,740 MiB** of GPU memory (56.7% less). Every
  qualified run completed 8/8 and every learner's loss endpoints matched native
  exactly (§5/§7).
- **Real Unsloth SFT is now a qualified second workload.** A normal N=4 fork run
  completed 4/4 while correctly creating eight workers (trainer + preprocessing
  lineage per clone), with no CUDA errors and 19,025 MiB peak versus native's 29,762
  MiB. Every learner's final adapter digest and losses matched a fair native control
  exactly. This found and fixed blind cross-process worker attachment, missing
  proc-mem metadata on late attachment, ambiguous tokenless routing, and a broken
  all-private kill switch (§7).
- **Synthetic explicit CUDA graphs remain fast through the boundary**, but the real
  installed Unsloth training path is not graphable today. A K=500 graph measures
  1.241 µs/op versus native's 1.224, yet forced Inductor graphs contained one op each,
  fullgraph tracing rejects `PeftModel_fast_forward`, and explicit backward capture
  fails on an intrinsic legacy-stream dependency (§6).
- **Transparent daemon auto-capture is ruled out.** It sees all K guest crossings
  before it can recognize a segment, so it cannot remove the dominant guest-side
  work. In addition, the upgraded real-DPO probe found moving device pointers in
  every repeated structural group. Hash-and-replay would therefore be unsafe (§6).
- **Generic transparent transport optimizations have now been tested and rejected:**
  socket batching is ~2x slower than the ring, compound ring records do not improve
  launch rate, 4 KiB records remove 1,126 blocking launches but regress DPO throughput,
  an exact TMA-descriptor cache hits ~89% without an end-to-end win, deferred VMM
  unmaps only move the wait, and direct clone-RAM copies regress throughput (§7).
- **Weight sharing is not the SFT speed limiter.** A valid all-private in-Trainer
  control measured 241 tok/s versus 246 shared while increasing peak VRAM from about
  11.1 to 16.3 GiB. A later snapshot inside a live Trainer improves 10-step SFT
  187→246 tok/s and 50-step SFT reaches 537 tok/s, but native remains 1,052/1,103.
  Exact final adapter bytes match in every fair pair. The residual is a steady eager
  execution/context cost, not reconstruction corruption or imported-weight access
  (§7).
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
- Runtime-only `torch.compile`, explicit DPO graph capture, and the remaining
  synchronization/transport alternatives are now closed as no-go for the installed
  stack. The graph probe moved TRL/Transformers host decisions outside capture and
  disabled both checkpointing and Unsloth compile; backward still attempted an
  illegal dependency from the legacy stream to the capture stream (§6/§8).
- Managed uncapped MPS is now implemented below the workload: a private PID-scoped
  controller, ownership supervisor, crash/TERM cleanup, external-controller
  non-ownership, explicit opt-out, and ordinary-context fallback. Release gates also
  prove failed-start cleanup and preservation of a colliding foreign PID path. The
  deployed candidate reproduced the earlier external-MPS N=4 result and repeatedly
  passed N=8 against a fair native control (§7).
- The old ~13.7k "hard ceiling" is superseded: the current paired N=8 control reached
  17,152 and uncapped MPS reached **22,597**. That directly validates multi-context
  scheduling as a material limiter. A full single-context/multi-stream redesign is now
  lower priority than productizing the much narrower MPS path (§5).
- The investigation diagnostics, transparent transport results, implementation, and
  release evidence are preserved on the investigation branch (§9).

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

The implemented, ownership-managed candidate was first built and deployed as immutable
binary `e1d7b6d123b81fd1d3b41eea59d515fc`. Its N=4 confirmation measured **15,489
tok/s**, within 0.04% of the prior best external-MPS arm (15,495), with 4/4 completion,
`shared=260 private=148`, and the same four loss endpoints. N=8 was then repeated
across rebuilds and daemon/controller restarts:

| arm / immutable binary | aggregate tok/s | completion | peak GPU memory |
|---|---:|---:|---:|
| managed MPS, initial (`e1d7b6d1…`) | **22,088** | 8/8 | **26,281 MiB** |
| managed MPS, post-main merge (`b5fddf23…`) | **23,015** | 8/8 | **26,281 MiB** |
| managed MPS, final (`889f76f3…`) | **23,195** | 8/8 | **26,281 MiB** |
| native, two CPU cores/learner | 21,865 | 8/8 | 60,740 MiB |

The final artifact is **+6.1% throughput versus native at the same N and CPU
allocation**; even the slowest qualified managed run is +1.0%. GPU memory is
**56.7% lower** (2.31x native/smolvm memory ratio). Per-learner native and smolvm loss
endpoints match exactly in every qualified run. The final run used byte-identical
checked-in harness/workload files, all eight forks succeeded on their first recorded
attempt, and live MPS control listed the daemon plus all eight workers under one
server.

Two qualification incidents are explicitly excluded from that comparison. In
`fork_n8_s50_c2_20260726-015002_r1`, seven learners were released before a missing
clone was manually recovered; its reported 25,848 tok/s is invalid because concurrency
was staggered. The harness now retains each fork command's output, retries only when no
partial machine exists, and fails before the synchronized release if creation still
fails. In `fork_n8_s50_c2_20260726-020224_r1`, all eight first-attempt forks completed
synchronously at 22,611 tok/s with exact losses, but a successful teardown's final
`pkill` status prevented JSON emission; `run_fork` now returns success explicitly.
Neither incident contributes to the qualified result set.

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

## 6. CUDA graph capture — substrate validated, current Unsloth DPO no-go

Per-op work is already at native parity (§3/§4); graphs can cut eager issue overhead by
collapsing thousands of launches into one replay. The substrate works, but the
application must expose a capture-safe region. The installed Unsloth DPO stack does
not currently do so.

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

### Real Unsloth DPO graphability — **no-go for the installed stack**

Three progressively more explicit arms close the gap between the synthetic substrate
result and the actual workload:

1. **Forced regional Inductor CUDA graphs.** Overriding Unsloth's
   `"triton.cudagraphs": False` produced 32 real captures, 32 instantiations, and 31
   launches. Every captured graph contained exactly **one CUDA operation**. The clone
   still issued 116,382 boundary operations versus 116,409 eager, so the accepted
   configuration did not coalesce the training region and incurred large compile
   overhead. No-go.
2. **Forced fullgraph.** Static shapes, `fullgraph=True`, and suppressed fallback
   failed at the first model call because Dynamo intentionally marks Unsloth's
   `PeftModel_fast_forward` as skipped/untraceable. Zero graphs launched. No-go.
3. **Explicit `CUDAGraph`/`make_graphed_callables` prototype.** The probe cached frozen
   reference log-probabilities, pre-expanded a fixed attention mask, and moved TRL's
   `flush_left` and Transformers' `torch.all(mask == 1)` host decisions outside
   capture. Direct forward/backward capture then ran ~3x faster for the tiny measured
   region but produced NaN on replay, so it was rejected. PyTorch's training-specific
   graph wrapper gave the exact cause while capturing backward:

   `CUDA operation would make the legacy stream depend on a capturing blocking stream`

   The same failure remained after using Unsloth's own API to disable gradient
   checkpointing and after a separate process-start arm disabled Unsloth regional
   compilation entirely. It is therefore intrinsic to the installed Unsloth backward
   execution path, not a missing smolvm graph API or a Dynamo-only limitation.

Decision: do not build a workload adapter around this stack until Unsloth/PyTorch can
provide a backward path that is capture-safe on one stream (or explicitly joins its
auxiliary streams to capture). The synthetic graph substrate remains useful for a
different framework/workload that already satisfies CUDA graph constraints.

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

This is now implemented for Linux/NVIDIA fork-worker pools:

1. `SMOLVM_CUDA_FORK_WORKERS` enables a private uncapped controller automatically;
   `SMOLVM_CUDA_MPS=0` disables it and `SMOLVM_CUDA_MPS=1` explicitly enables it.
2. The daemon creates a mode-0700, UID/daemon-PID-scoped pipe directory and a private
   log directory before loading CUDA. Both must be newly created: a PID-path collision
   falls back without adopting, deleting, or changing the existing path. Clone workers
   inherit the endpoint.
3. A hidden supervisor lives in a separate process group, owns the controller only
   when its start succeeds, watches a kernel lifecycle channel, and sends `quit` after
   daemon EOF even for daemon crash/SIGKILL. It removes only NVIDIA's known nodes and
   the private PID log directory. Graceful TERM, pipe cleanup, and log cleanup passed
   on the H100.
4. An ambient `CUDA_MPS_PIPE_DIRECTORY` is treated as externally owned: smolvm uses it
   but never starts, stops, or removes it. Controller-start failure clears the private
   environment, cleans the directories it created, and keeps the CUDA daemon available
   on ordinary contexts.
5. Active-thread percentage remains unset. The 33% native isolation failure proves
   that tuning is not generically transparent.

The immutable-candidate confirmation and fair native comparison are recorded in §5
and `bench/results/mps-managed-h100-20260726.json`.

Lifecycle/fallback qualification is green:

- a second controller start against the same private pipe directory fails explicitly
  with `An instance of this daemon is already running`;
- after terminating the owned controller abruptly, a new controller successfully
  reclaims the same directory despite stale entries;
- a real CUDA tensor operation succeeds with both a nonexistent controller directory
  and the stale post-crash directory, demonstrating ordinary CUDA fallback instead of
  a hang or initialization failure;
- forced controller-start failure leaves the daemon socket live, removes its private
  PID pipe/log directories, and uses ordinary contexts;
- a pre-existing daemon-PID pipe path causes fallback while its foreign sentinel
  remains byte-identical;
- `SMOLVM_CUDA_MPS=0` keeps the fork-worker daemon live without creating a controller
  path;
- an external controller remains queryable after the smolvm daemon exits and stops
  only when the external owner sends `quit`;
- graceful managed shutdown removes controller/server processes and its private
  pipe/PID-log paths; all performance arms also returned the GPU to 0 MiB.

These gates were rerun on final immutable binary
`889f76f3d72d84f7e565154084902392`. No global/system-wide MPS controller is mutated.

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

### Real Unsloth SFT qualification and the remaining throughput gap

The original release work used real Unsloth DPO. A second workload now exercises
Unsloth + TRL `SFTTrainer`, Qwen2.5-7B 4-bit LoRA, the 8-bit AdamW optimizer,
gradient checkpointing, dataset preprocessing, and multiple CUDA-using processes in
one VM. `bench/workload_sft.py` runs unchanged in native and fork arms and records
exact SHA-256 digests of all 40,370,176 trainable values plus deterministic CPU-side
norms, dataset and RNG digests, loss endpoints, and throughput.

The first real fork attempt found that the daemon assumed one CUDA process per VM.
Unsloth preprocessing created a second process-scoped golden layout; the daemon
attached the trainer to that worker, reconstructed zero maps, and crashed it. The
candidate now attaches across lineage tokens only when both tokens refer to the same
process-scoped `GoldenLayout`. A normal N=4 run subsequently created exactly eight
workers (tokens 1 and 3 for each of four clone IDs) and completed cleanly:

| arm | N | steps | aggregate tok/s | peak GPU memory | completion |
|---|---:|---:|---:|---:|---:|
| native, same one-step reference setup | 4 | 2 | **931** | 29,762 MiB | 4/4 |
| smolvm fork + managed MPS | 4 | 2 | 143 | **19,025 MiB** | 4/4 |

Every learner matched its same-LID native reference exactly: initial adapter digest,
processed-dataset digest, CPU and CUDA RNG digests, both loss endpoints, final adapter
digest, sum, absolute sum, and L2 norm. The weight-bearing worker reported
`shared=260 private=169`; the preprocessing worker correctly had zero maps. The
benchmark now records the maximum shared verdict instead of incorrectly reporting
the zero-map worker that happened to log last.

The same investigation found a second independent reliability bug. An eagerly
spawned worker predates the real connection's `/proc/<pid>/mem` advert, so a
late-attached channel lacked access to the clone's live COW RAM. The control channel
now sends the accepted fd and proc-mem advert in one `SOCK_SEQPACKET` message. The
first prototype retained a one-byte receive iovec and truncated every advert; an
end-to-end fd+metadata test reproduced that exact live failure before the corrected
version was deployed. On the H100, the corrected attach path reduced GPA-copy errors
from 13 to zero and advanced to the expected read-only safety fault in an intentionally
too-early snapshot. The ordinary warmed SFT and unchanged DPO regression both pass.

Numerical equivalence required a fair control. Native without the golden's one-step
safe-point setup produced a different, but deterministic, final adapter. The clone
matched the golden exactly, and dataset/RNG/initial-adapter state matched native.
When native executed the same setup, every checked field—including fixed model-output
and final adapter SHA-256—matched the fork exactly. The apparent discrepancy was a
control mismatch, not clone corruption.

SFT performance does not yet match native:

| placement | steps | native tok/s | fork tok/s | fork/native | exact final state |
|---|---:|---:|---:|---:|---|
| rebuild `SFTTrainer` after safe point | 10 | 741 | 187 | 25.2% | yes |
| fork from callback inside live Trainer | 10 | 1,052 | 246 | 23.4% | yes |
| fork from callback inside live Trainer | 50 | 1,103 | 537 | 48.7% | yes |

The in-Trainer placement is a diagnostic upper-bound prototype, not a proposed
source requirement. It removes a second Trainer construction and improves the
10-step fork rate by 31.6%, but it does not close the steady gap. From 10→50 steps,
native adds about 0.459 s/step and the fork about 0.671 s/step (~46% steady tax), with
roughly 14 s additional post-fork fixed cost. This is consistent with a small-op eager
path remaining materially more latency-sensitive than DPO's N=8 saturation point.

The strongest remaining transparent alternate explanation is rejected. The daemon's
documented `SMOLVM_CUDA_FORK_SHARE_WEIGHTS=0` kill switch was being overwritten by the
fork preamble; the candidate now keeps explicit `0` authoritative and the harness
scopes that policy to the daemon/worker rather than the VM. A valid all-private
10-step continuation completed at 241 tok/s versus 246 shared, with identical loss and
final adapter SHA-256, while peak memory increased from about 11.1 to 16.3 GiB.
Imported read-only weight mappings therefore buy density without causing the SFT
throughput gap.

Additional fail-safe behavior is now explicit:

- a tokenless channel attaches only when exactly one live worker exists for the
  clone; several workers make it fail closed instead of guessing a process;
- malformed late-attach metadata closes the received fd rather than leaking it;
- snapshotting before trainer setup remains unsupported without GPU-write COW: a
  diagnostic no-warm arm correctly faults on a shared read-only base chunk rather
  than silently corrupting siblings;
- managed MPS is currently required for this SFT fork path: an N=1 no-MPS isolation
  hit CUDA status 710 and completed 0/1, whereas managed MPS arms are clean. This is
  recorded as a compatibility boundary, not generalized into a new performance
  claim.

### Remaining pure-smolvm questions

The per-operation transport questions and MPS policy/lifecycle gates are now bounded.
The managed candidate repeatedly exceeds native throughput at N=8. Remaining work is
a separate architecture project or independent reliability follow-up, not another
untested transport tweak:

1. **Repeated release qualification is complete.** Three manifest-backed managed N=8
   runs across rebuilds/controller restarts measured 22,088–23,195 tok/s, all 8/8 with
   native-matching per-learner endpoints. The final binary also produced a separate
   synchronized raw 22,611 tok/s run before the harness bookkeeping fix.
2. **Track clone-start reliability independently.** One N=4 and one N=8 attempt
   initially lacked a clone before CUDA/MPS work. The N=8 clone succeeded on explicit
   retry, but the old harness had discarded the original error. It now logs every
   attempt, retries only a wholly absent clone, and refuses to release the workload
   barrier otherwise. Both final synchronized N=8 executions created all eight clones
   on their first attempt.
3. **Single-context/multi-stream redesign only by explicit decision.** It is the
   remaining architectural way to seek a higher ceiling below the workload, but it
   requires per-clone address translation and changes the isolation design. The legacy
   shared-context mode is not a viable proxy (0/4 learners).
4. **No speculative barrier suppression.** Required D2H consumers, cheap stream sync,
   deferred-unmap equivalence, and slower direct proc-mem copies leave no proven
   safely removable material class.
5. **Treat SFT parity as a new workload-specific performance project.** Correctness,
   later snapshot placement, and sharing/private controls are closed. The 50-step
   continuation still measures 537 versus 1,103 tok/s native, so DPO's N=8 parity
   must not be generalized to every training method. The remaining transparent
   architectural lever is the explicitly undecided single-context/per-clone-address-
   translation design; the current Unsloth backward remains non-capturable.

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
(category 2), not a pure-smolvm improvement. That injection was nevertheless tested:
non-fullgraph produced only one-op graphs and fullgraph rejected
`PeftModel_fast_forward`, as detailed in §6. Runtime flags alone cannot deliver useful
coalescing for this installed workload.

## 8. Revised implementation path (phased, each with a go/no-go gate)

### Phase 0 — daemon autographs: **complete, no-go**

Do not implement segment detection/capture/replay in the daemon. Both required
assumptions failed: recognition is below the cost that needs to be removed, and
repeated real-workload segments do not retain pointer arguments.

### Phase 1 — explicit graph substrate in fork clones: **complete, go**

Current cudart capture/instantiate/launch forwarding works in fresh isolated clone
contexts and retains exact results in the synthetic graph probe. No clone graph fix is
currently justified. Preserve the corrected reliability test as a regression.

### Phase 2 — transparent MPS scheduling: **release-qualified, go**

The private ownership-aware controller, supervisor, opt-out, external-controller
non-ownership, uncapped policy, and ordinary-context fallback are implemented. H100
lifecycle, failed-start cleanup, collision-preservation, and external-ownership tests
are green. Managed N=8 measured 22,088–23,195 tok/s across three qualified runs versus
21,865 native with 56.7% less GPU memory. Preserve the byte-pinned N=8 arm as a release
regression; no workload change or injection is required.

### Phase 2b — multi-process fork routing and attachment: **candidate, go**

Real Unsloth SFT exposed and now passes the process-lineage routing and live-RAM
attachment fixes described in §7. Unit gates cover process lineage, atomic fd+advert
delivery, tokenless ambiguity, and the sharing kill switch. H100 gates cover normal
SFT at N=1/N=4, exact fair-native state, in-Trainer resume at 10/50 steps, private
copy mode, and unchanged DPO regression. Final immutable binary
`18d33faae5fa996822e07cf1d407576f` passed both final smokes: DPO 1/1 at 40 tok/s,
`shared=260 private=160`, and SFT 1/1 at 45 tok/s with two correctly separated
workers, `shared=260 private=169`, exact qualified adapter digest, and zero operation
errors. Keep these changes separate from any claim that SFT throughput is solved.

### Phase 3 — actual DPO graphability prototype: **complete, no-go for current Unsloth**

The prototype forced real compiler graphs, forced fullgraph, and then built a
fixed-address explicit training region with preprocessing outside capture. It also
tested PyTorch's training-specific graph wrapper, checkpointing disabled, and Unsloth
compile disabled. The accepted regional compiler graphs had one op each; fullgraph
refused the skipped PEFT forward; explicit backward capture always hit the
legacy-stream dependency documented in §6.

Do not add a smolvm graph engine or ship the adapter. Reopen this phase only for a
different framework/workload with a capture-safe backward path, or after an upstream
Unsloth/PyTorch change removes the stream violation.

### Phase 4 — graph-enabled fork saturation measurement: **blocked by Phase 3**

After Phase 3 passes, run eager versus graph at N=1 and N=4 first, then N=8/N=16:

- throughput, step latency, SM utilization, and aggregate VRAM;
- final losses/parameters;
- graph capture/replay counts and daemon op counts;
- per-clone fairness under concurrent contexts.

This separates removal of the eager-call tax from the independent saturation ceiling.
The target is to move the N=4 fork result toward native's 20.8k tok/s; A1 proves the
mechanism but does not guarantee that end-to-end result.

### Phase 5 — graph productization choice: **not approved**

If a future capture-safe workload prototype wins, choose the narrowest useful
integration:

- an explicit graph-enabled workload adapter/static-buffer runner for known training
  stacks; or
- a small application-visible capture contract for serving/training loops that can
  guarantee fixed shapes and stable storage.

Keep it opt-in until eager/graph equivalence and fallback behavior are tested. A
transparent `SMOLVM_CUDA_AUTOGRAPH=1` daemon feature is no longer proposed.

### Separate decision — single-context/multi-stream architecture

MPS has now reached same-N native throughput without changing smolvm's per-clone
isolation model. A larger per-clone address-translation/shared-context project is
therefore not required for the current performance target. Consider it only as an
explicit next-generation architecture project seeking margin beyond parity; the
legacy shared-context mode cannot validate it.

### Explicit non-goals for this work

- No daemon-side hash-and-replay engine.
- No more per-op host micro-optimization; host launch suppression already falsified
  that path as a material lever.
- No speculative graph-node update API work before the actual DPO capture identifies a
  concrete need.
- No bundled single-context redesign.

## 9. Current repo state

Branch: `cuda-graph-capture-investigation`, forked from `a31810e` and merged with
`origin/main` through `4ba25f7`. The SFT qualification/fork-safety change described
below passed its final local and H100 release gates on immutable binary `18d33faa…`.

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
| `0c98808` | MPS scaling/lifecycle/cap findings, durable result summary, and clone-RAM/sync diagnostics |
| `5201032` | managed-MPS implementation, immutable-candidate parity result, and real-DPO graph no-go |
| `55cf22d` | release qualification, owned-path cleanup/collision hardening, and fail-loud fork benchmark retries |
| current change | multi-process lineage routing; atomic late-attach live-RAM metadata; fail-closed tokenless routing; sharing kill-switch fix; generalized real-workload harness; SFT qualification and in-Trainer placement probe |

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
| `bench/results/mps-managed-h100-20260726.json` | managed-candidate N=4/N=8/native comparison and compiler/explicit-graph verdicts |
| `bench/bench.sh` | benchmark manifests record external/managed/off MPS mode; fork errors are retained/retried before synchronized release |
| `bench/workload_dpo.py` | opt-in forced-compiler and explicit fixed-region graph diagnostics |
| `bench/workload_sft.py` | real Unsloth/TRL SFT qualification with exact adapter, model-output, dataset, and RNG fingerprints |
| `bench/workload_sft_resume.py` | diagnostic fork-from-live-Trainer placement upper bound |
| `src/cuda_daemon.rs` | managed private MPS policy, ownership supervisor, bounded path cleanup/collision refusal, fallback, and tests |
| `src/main.rs` | hidden MPS supervisor process entry point |
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
  `/tmp/smolvm-mps-log-20260725-*` directories. Managed-candidate results are
  `fork_n4_s50_c4_20260726-002110_r1`,
  `fork_n8_s50_c4_20260726-002620_r1`,
  `fork_n8_s50_c2_20260726-013610_r1`, and final release result
  `fork_n8_s50_c2_20260726-020902_r1`; the fair native control is
  `native_n8_s50_c2_20260726-003233_r1`. Excluded staggered qualification is under
  `fork_n8_s50_c2_20260726-015002_r1`, and the synchronized raw-only harness-status
  run is `fork_n8_s50_c2_20260726-020224_r1`. The one-op forced-compiler graph run is
  `fork_n1_s2_c4_20260726-003739_r1`; fullgraph rejection is in
  `fork_n1_s2_c4_20260726-004224_r1/g.err`; explicit graphability attempts are the
  native N=1 runs from `20260726-005207` through `20260726-010600`.
  Real-SFT evidence is under `~/bench_run/` and matching `~/bench/results/`:
  `native_sft_fingerprint_n1_s2_c2_20260726-042503_r1` plus its exact repeat,
  `fork_sft_divergence_probe_n1_s2_c2_20260726-044117_r1` and fair control
  `native_sft_reference_warmup_n1_s2_c2_20260726-044526_r1`, N=4 pair
  `fork_sft_multilineage_n4_n4_s2_c2_20260726-044630_r1` /
  `native_sft_multilineage_n4_reference_n4_s2_c2_20260726-045034_r1`, ordinary
  10-step pair `native_sft_profile_reference_n1_s10_c2_20260726-045132_r1` /
  `fork_sft_profile_n1_s10_c2_20260726-045207_r1`, live-Trainer 10-step pair
  `native_sft_resume_reference_n1_s10_c2_20260726-045945_r1` /
  `fork_sft_resume_n1_s10_c2_20260726-050018_r1`, live-Trainer 50-step pair
  `native_sft_resume50_reference_n1_s50_c2_20260726-050401_r1` /
  `fork_sft_resume50_n1_s50_c2_20260726-050452_r1`, and the valid all-private
  gate `fork_sft_private_reconnect_probe_n1_s10_c2_20260726-054605_r1`.
  Final immutable-candidate smokes are
  `fork_dpo_final_candidate_n1_s4_c2_20260726-055615_r1` and
  `fork_sft_final_candidate_n1_s2_c2_20260726-060007_r1`, both on binary
  `18d33faae5fa996822e07cf1d407576f`.
- **In this repo**: `bench/results/*.json` (11 committed in #742), `bench/README.md`
  (harness usage), `bench/RESULTS.md` (generated summary table), and
  `bench/results/mps-h100-20260725.json` plus
  `bench/results/mps-managed-h100-20260726.json` (durable prototype and managed
  candidate summaries).
- **Session memory** (cross-conversation continuity, not in this repo):
  `fork-vs-native-benchmark.md`, `cuda-fork-weight-sharing-fix.md`,
  `cuda-clone-procmem-transport.md`.
