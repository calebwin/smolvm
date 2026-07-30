# CUDA fork-pool throughput investigation & graph-capture plan

Status: the transparent MPS optimization is implemented and release-qualified on the
H100. At the DPO N=8 throughput point, the final managed-smolvm artifact is **6.1%
faster than the same-N ordinary native control while using 56.7% less GPU memory**.
The earlier 7B equal-scheduling GRPO control shows why that claim must not be
generalized: native workers under their own uncapped MPS server are faster than
smolvm, while smolvm retains a roughly 64% memory reduction. The newer vLLM-enabled
0.5B N=2 gate reaches native-MPS steady throughput after the compatibility fixes,
but does not restore ordinary-allocation CoW density. A follow-up transparent golden-
eviction candidate now removes the inactive source context after durably staging its
private state, reducing paired N=4 vLLM GRPO peak VRAM by 15.7% with approximately
preserved throughput. The same transparent policy reduces true-sharing DPO peak VRAM
by 11.4% while preserving throughput. A separate `--auto-graph` candidate now enables
framework-supported TorchInductor graphs and improves a fixed-shape eight-matmul region
by about 39% median through smolvm. Transparent daemon-side auto-capture remains a
no-go, and the installed Unsloth DPO backward path is also proven non-capturable by an
explicit workload-level prototype (§6/§8).

Broader Unsloth SFT qualification is now complete through N=8. In addition to the
multi-process-lineage and late-channel fixes, exact pre-training probes exposed an
exported-fd shuffle collision and mutable reuse inside shared VMM chunks. The corrected
runtime now passes 8/8 exact model hashes and 200-step adapter/loss gates at 3,036 tok/s
and 28,280 MiB, versus native's 3,393 tok/s and 60,236 MiB: 89.5% throughput with 53.0%
less GPU memory (§7).

Unsloth + TRL GRPO is now qualified as a third real workload on Qwen2.5-7B. It
completes at N=1, N=4, and N=8 with exact frozen-policy and initial-state checks.
A two-step pair is byte-exact through the final 40.4-million-value adapter;
longer stochastic runs show small, bounded remoted numerical variation in sampled
text. Against ordinary native, the long N=8 gate reaches 74.75 versus 52.63 tail
aggregate tok/s while using 64.0% less peak VRAM. Against native MPS, smolvm reaches
75.5% of rollout throughput with the default warmup and 82.6% with a representative
warm snapshot. The qualification gate rejects independently observed low-reward
sampled trajectories rather than misreporting their extra generated tokens as speed
(§7).

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
- **Real Unsloth SFT is now a qualified second workload through N=8.** The corrected
  200-step N=8 run completed 8/8 with exact initial model hashes, finite decreasing
  losses, and distinct final adapters at **3,036 tok/s and 28,280 MiB**, versus
  native's **3,393 tok/s and 60,236 MiB**: 89.5% throughput using 47.0% of the memory.
  The gate covers independent process lineage, late-channel metadata, exported-fd
  handoff, read-only shared mappings, and transparent address-preserving VMM COW (§7).
- **Real Unsloth GRPO exceeds ordinary native, but not equally scheduled native
  MPS.** The 7B N=8, 200-step quality-qualified fork run reaches **74.75 tail
  aggregate tok/s versus 52.63 ordinary native (+42.1%)**. The stronger uncapped-MPS
  native control reaches 99.13 tok/s and 4.057 learner-steps/s versus smolvm's 74.87
  and 3.015. A representative warm snapshot narrows this to 85.10 versus 102.99 tok/s
  and 3.501 versus 4.215 steps/s while preserving the roughly 64% VRAM reduction.
  Deterministic setup and final CPU RNG match exactly; the maximum per-learner reward
  mean delta is 0.0114 and adapter-L2 relative delta is 0.000218. One separate fork
  execution produced a low-reward learner from its first sampled step and fails the
  explicit quality gate; its larger token count is excluded from performance claims.
  The installed Unsloth stack still requires its explicit BF16 environment contract
  in both native and fork arms (§7).
- **Unsloth GRPO with in-process vLLM now passes transparent fork continuity at
  native-MPS steady throughput.** On Qwen2.5-0.5B, two isolated managed-MPS clones
  each completed 20 real updates with distinct data/rollouts and changed parameter
  digests at 111.613 exact aggregate tok/s, versus 106.292 for two native workers
  under uncapped MPS at the same 0.12 cache setting (+5.0%). This validates the
  runtime fixes and hot-state reuse. Ordinary allocations remain private, and the
  resident frozen golden originally made smolvm use 26,215 versus 21,676 MiB. The
  follow-up host-backed eviction candidate now reclaims 2,564 MiB directly from that
  source context and reduces a paired N=4 peak from 34,702 to 29,256 MiB (15.7%). A
  post-eviction replacement reconstructs 97 mappings and completes with exact hashes.
  vLLM still reports `shared=0 private=33`, so this removes inactive-source overhead;
  it does not turn vLLM's private worker state into shared weights (§7).
- **Forked GRPO also beats a production-style resident-base queue.** Eight
  source-identical 200-step jobs finish 41.3% sooner, with 1.70x jobs/hour, 2.38x
  effective rollout-token throughput, and 2.34x learner-step throughput. This is
  direct evidence for rollouts generated inside independent trainer forks. A
  follow-up rollout-only comparison is a no-go as a homogeneous performance
  optimization: one fused vLLM engine is 4.15x faster and uses 28.6% less memory
  than four isolated fork workers at the same small KV-cache setting. The queue
  uses 7,443 MiB versus the training fork pool's 21,479 MiB (§7).
- **Synthetic explicit CUDA graphs remain fast through the boundary**, but the real
  installed Unsloth training path is not graphable today. A K=500 graph measures
  1.241 µs/op versus native's 1.224, yet forced Inductor graphs contained one op each,
  fullgraph tracing rejects `PeftModel_fast_forward`, and explicit backward capture
  fails on an intrinsic legacy-stream dependency (§6).
- **A bounded framework auto-graph option is validated.** `--auto-graph` implies CUDA,
  sets TorchInductor's supported CUDA-graph policy, and preserves framework fallback
  rather than synthesizing capture around eager calls. On the H100, an identical
  fixed-shape eight-matmul compiled region measured 12,288--12,428 iterations/s versus
  8,683--9,094 without graphs (about +39% median), with the same checksum. Named-machine
  and direct packed execution both pass; the latter exposed and fixed a missing guest
  CUDA-shim sentinel in the dynamic packed launcher. This does not change the Unsloth
  no-go above because its current training region still cannot be coalesced (§6).
- **Transparent daemon auto-capture is ruled out.** It sees all K guest crossings
  before it can recognize a segment, so it cannot remove the dominant guest-side
  work. In addition, the upgraded real-DPO probe found moving device pointers in
  every repeated structural group. Hash-and-replay would therefore be unsafe (§6).
- **Generic transparent transport optimizations have now been tested and rejected:**
  socket batching is ~2x slower than the ring, compound ring records do not improve
  launch rate, 4 KiB records remove 1,126 blocking launches but regress DPO throughput,
  a 128-page request ring stays inside the 32-page completed-step band and fails the
  GRPO quality gate, an exact TMA-descriptor cache hits ~89% without an end-to-end
  win, repeated cuBLAS state elision remains inside run noise, deferred VMM unmaps
  only move the wait, direct clone-RAM copies regress throughput, and blind
  clone-context replay cannot restore safely from CUDA-visible state alone (§7).
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

### Framework `--auto-graph` policy — **go for compatible compiled regions**

The product candidate exposes the safe part of this mechanism without guessing at
eager operation boundaries. `--auto-graph` (and `auto_graph = true`) implies CUDA,
sets `TORCHINDUCTOR_CUDAGRAPHS=1`, and leaves graph-region selection and fallback with
TorchInductor. It does not capture arbitrary eager calls or alter workload source.

A source-identical H100 A/B used a fullgraph, fixed-shape module containing eight
matrix multiplications per invocation and 200 timed invocations. The no-graph runs
measured 9,094 and 8,683 iterations/s; auto-graph measured 12,428 and 12,288
iterations/s, a **39.0% median gain**. Both arms produced the exact same 1108.736816
checksum. The workload observed both `SMOLVM_CUDA_AUTO_GRAPH=1` and
`torch._inductor.config.triton.cudagraphs=True` only in the requested arm.

The same policy passes direct ephemeral packed execution with `device=cuda:0` and the
expected result. That gate found a real packed-launcher bug: it wired the host CUDA
bridge but did not send the guest feature sentinel, so the agent never staged its
bundled shim. The dynamic launcher now uses the shared protocol constant and carries
the same sentinel as the normal launcher. A stale July 12 packed artifact required its
cached shim to be temporarily replaced for this compatibility smoke; the cache was
restored afterward.

This is a useful opt-in for already-compilable fixed-shape workloads, not evidence that
current Unsloth SFT/DPO/GRPO becomes graphable. Those paths retain the no-go findings
below, and auto-graph must not be enabled by default until broader compiled-workload
compatibility and memory overhead are qualified.

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
| Increase the GRPO request ring from 32 to 128 pages after observing backpressure | Source-identical N=8 candidate: 2.981 aggregate learner-steps/s versus the 32-page 2.967–3.075 release band; -3.0% versus the passing reference, and two reward-mean deltas fail the quality gate. | **No-go.** A full queue proves burst pressure, not that a deeper queue removes guest operation preparation or increases useful throughput. Keep 32 pages. |
| Exact guest-side cache for 152-byte `cuTensorMapEncodeTiled` inputs and 128-byte descriptors, invalidated on clone reconnect | Clone hit ratio reached 2,280/2,560 (~89%). Twenty-step DPO runs were 2,574 / 2,475 / 2,459 tok/s, median 2,475 versus baseline median ~2,546; losses remained exact. | **No-go for performance.** High repetition does not imply material wall-clock cost. The reconnect-generation mechanism was required because a Firecracker clone preserves the guest PID. |
| Elide repeated cuBLAS handle state on each clone connection | N=4 GRPO control, math-mode-only, and combined math/stream/workspace arms measured **0.809 / 0.817 / 0.817 steps/s** and **25.721 / 25.986 / 25.962 tail tok/s**. The math-only pair matched all 80 reward steps, final parameters, and CUDA RNG exactly. | **No-go.** The apparent ~1% is inside the established 0.803–0.822 band; broader setter elision adds no gain and is not generically transparent because `cublasSetStream` unconditionally resets workspace state. Both prototypes were removed. |
| Replay recent golden CUDA intervals in each new clone context, backing up and restoring every private GPU range | A 105,789-op suffix without H2D writes asserted at op 175; adding H2D/device writes produced 102,917 ops but asserted at kernel op 3,334. Both returned CUDA 710, poisoned the context, and prevented restoration from synchronizing. | **No-go.** CUDA synchronization boundaries do not expose the CPU/framework state or beginning-of-step device snapshot needed to replay data-dependent work. The runtime prototype was removed; blind replay cannot be a default feature. |
| NVIDIA CUDA process checkpoint/restore for a warmed worker | The H100 is on driver 570.148.08 with CUDA 12.8. NVIDIA's driver-570 implementation explicitly does not support IPC memory created with `cuMemExportToShareableHandle()`, which is the mechanism used for smolvm's shared VMM allocations. Checkpointing also copies device memory into host allocations and restores one process's resources rather than creating CoW contexts. | **No-go.** It is incompatible with the current shared allocations and would duplicate restored GPU contents instead of preserving smolvm's density advantage. Driver 610 adds only `cuIpcGetMemHandle` IPC support; NVIDIA still lists exported VMM memory as unsupported. |
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

#### SFT parity opportunities and next validation gate

The existing SFT measurements do not yet answer whether smolvm can match **aggregate**
native throughput at a realistic concurrency and run length. The steady measurement is
N=1 for 50 steps, while the only N=4 correctness pair runs for two steps and is
dominated by startup. DPO reached parity only at N=8 under managed MPS, so generalizing
the short SFT result in either direction would be premature.

The 10→50-step fit also bounds what startup work alone can achieve. Removing the
entire ~14 s fork intercept would leave approximately 0.671 s/step versus native's
0.459 s/step: about 68% of native asymptotically. Startup/reconstruction work is worth
reducing, especially for short jobs, but it cannot by itself produce N=1 parity.

The next pure-smolvm SFT work is therefore ordered as follows:

1. Run source-identical, exact-state-checked SFT for at least 200 steps at N=1, N=4,
   and N=8 under the existing uncapped managed-MPS policy. Report both aggregate and
   per-learner throughput; compare only synchronized, same-build pairs. This is the
   missing test of whether density plus scheduling reaches aggregate parity as it did
   for DPO.
2. Collect an SFT-specific operation census and guest/daemon timing profile over the
   post-warm steady interval. The current ~0.212 s/step residual is consistent with an
   eager small-operation tax, but its exact SFT call classes have not yet been
   attributed and must not be inferred solely from DPO.
3. Phase-time the ~14 s post-fork intercept (worker reconstruction, framework/trainer
   setup, optimizer state, and first-use library work) and optimize only a component
   with a measured end-to-end contribution. The live-Trainer snapshot remains an
   upper-bound diagnostic, not a workload requirement.
4. If long N=4/N=8 SFT remains below native after those bounded fixes, require an
   explicit architecture decision for single-context/multi-stream execution with
   per-clone address translation. This is the principal remaining workload-transparent
   design lever, but it is unvalidated and changes the isolation model.

These gates distinguish two product claims: aggregate parity may still be attainable
transparently through concurrency and MPS; per-learner/N=1 parity is not currently
demonstrated and likely needs the architecture work above or a future capture-safe
framework path. The installed Unsloth backward remains non-capturable, so graph
integration is not counted as a transparent near-term SFT opportunity.

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

### Real Unsloth GRPO qualification

The third workload uses Unsloth 2026.7.3, TRL 0.24.0, Transformers 4.57.6, and
Qwen2.5-7B 4-bit LoRA. It performs genuine sampled GRPO updates on a deterministic
arithmetic dataset with four completions per prompt and a dense correctness reward.
`bench/workload_grpo.py` records the model revision and fixed frozen-policy output,
initial/final hashes and norms for all 40,370,176 trainable values, dataset and
CPU/CUDA RNG hashes, per-step rollout hashes/rewards, final RNG hashes, and actual
generated-token throughput. The native and fork arms execute the same file.

#### Installed-framework precision contract

The first golden failed before forking in Unsloth's fused LoRA forward:

`self and mat2 must have the same dtype, but got Half and Float`

The diagnostic established all of the following instead of attributing it to CUDA
remoting:

- the VM reports H100 capability 9.0 and BF16 support correctly;
- `GRPOConfig` and Accelerator both report BF16, and the first fused-LoRA call has
  BF16 activations/autocast plus FP32 adapters in both native and VM;
- a later generated GRPO-loss region silently selects FP16 because installed Unsloth
  reads `ACCELERATE_MIXED_PRECISION` and defaults it to `fp16`, independently of
  `GRPOConfig(bf16=True)`;
- `UNSLOTH_COMPILE_DISABLE=1` fails identically, so this is not an Inductor-only bug.

The qualification workload now sets `ACCELERATE_MIXED_PRECISION=bf16` before importing
Unsloth and explicitly loads the quantized policy with BF16 compute, matching its
declared trainer configuration in both arms. This is an installed-framework/runtime
contract, not a smolvm-only mathematical workaround. It does mean arbitrary GRPO
workloads that request BF16 only through `GRPOConfig` are **not yet transparently
compatible** with this exact installed stack; smolvm must not claim otherwise or
inject a framework-specific variable silently.

#### Correctness, stochastic equivalence, and density

The first Qwen2.5-7B N=1, two-step fair pair passed the strongest exact gate:

| arm | measured train | peak GPU memory | rollout | final adapter |
|---|---:|---:|---|---|
| native | 11.55 s | 7,443 MiB | reference | reference |
| fork + managed MPS | 27.52 s | 9,406 MiB | byte-identical | byte-identical |

The model snapshot, frozen-policy output, initial adapter, dataset, initial CPU/CUDA
RNG, rewards, sampled completion bytes, and final adapter SHA-256 all match. The clone
reports `shared=260 private=162` and no daemon operation errors.

Longer stochastic execution requires a more appropriate equivalence rule than blindly
requiring byte-identical sampled text forever. In the final-source, compiler-cache-
scoped four-step N=1 pair:

- rollout steps 1, 2, and 4 match byte-for-byte; step 3 samples different text with
  the same reward;
- every per-step reward and both final CPU/CUDA RNG hashes match;
- adapter L2 is 32.329819064 native versus 32.329778662 fork (about 1.25 ppm apart),
  though final adapter SHA-256 differs;
- native under its own private external MPS controller remains byte-identical to
  ordinary native, so MPS does not explain the difference.

The N=4 gate completes 4/4 in both arms. Every learner matches its native model output,
initial state, reward sequence, and final RNG state; two learners retain exact rollout
and final-adapter bytes, while two cross sampled-text boundaries during training. The
density result is material:

| arm | completion | aggregate rollout tok/s | peak GPU memory |
|---|---:|---:|---:|
| native | 4/4 | 28 | 29,762 MiB |
| fork + managed MPS | 4/4 | 9 | **14,515 MiB** |

That is 51.2% less peak memory. The short four-step throughput is directional only:
this N=4 pair predates explicit per-run compiler-cache scoping and is dominated by
trainer/compile setup, so it is not a steady-state ratio.

An all-private N=1 control positively reports `shared=0 private=422`, takes 56.82 s
versus the final shared control's 56.71 s, and increases peak memory from 9,408 to
14,608 MiB. Its rollout sequence and final adapter match the shared control exactly.
Imported read-only weights therefore cause neither the GRPO throughput gap nor the
observed numerical boundary crossing.

#### GRPO long-run performance and quality verdict

The 200-step, N=8 source-identical gate resolves the startup-dominated short-run
result. Both arms use two CPU cores per learner and the same per-run compiler-cache
policy:

| arm | completion | tail aggregate tok/s | aggregate learner-steps/s | peak GPU memory |
|---|---:|---:|---:|---:|
| native | 8/8 | 52.627 | 2.154 | 59,522 MiB |
| fork + managed MPS | 8/8 | **74.755** | **3.075** | **21,459 MiB** |

The valid fork execution is +42.1% by useful tail token throughput, +42.8% by the
token-count-independent completed-step rate, and uses 64.0% less peak VRAM. The
comparison requires exact model snapshot/output, initial adapter, dataset, initial
CPU/CUDA RNG, final CPU RNG, and parameter count. It then bounds the stochastic
quantities: maximum per-learner reward-mean delta is 0.011319 (gate 0.02), and maximum
final parameter-L2 relative delta is 0.0002173 (gate 0.001). Final CUDA RNG is retained
but is not required to match after a sampled token boundary changes.

An independent N=8 fork execution is explicitly excluded: learner 0 had mean reward
0.1064 versus 0.9728 native, beginning at its first sampled step, while the other
seven learners matched an earlier run byte-for-byte. Its 11,627 generated tokens
would inflate a token-rate-only comparison. `bench/compare_grpo.py` rejects that run
and reports completed-step throughput alongside tokens so a poor policy cannot look
faster merely by generating more text. Across three long 32-page N=8 fork executions,
two have healthy reward/adapter endpoints and one fails this quality gate; performance
and density were otherwise stable. This is a stochastic-quality reliability signal,
not evidence of memory corruption, and release claims use only gate-passing runs.

#### Production hot-base queue comparison

The native N-process baseline above is not the strongest alternative used by a
fine-tuning service. A second control models a production LoRA queue: one native
process loads and prewarms the base once, retains its compiled kernels and fixed-shape
LoRA allocation, then trains eight jobs sequentially. Before each job it restores the
same initial adapter and post-warmup CPU/CUDA RNG state, creates a fresh
`GRPOTrainer` and optimizer, and fingerprints the final adapter before resetting the
slot. This is an optimistic homogeneous queue—there is no repeated model load and no
adapter artifact upload.

The queue and fork executions use the same workload hash
`ae61dbd5d2672921475b1d9336ef7b4d`, model snapshot, batch 1, 200 steps, and 16 total
CPU cores (16 for the queue; eight two-vCPU forks):

| arm | completion | end-to-end wall | jobs/hour | effective rollout tok/s | learner-steps/s | peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| resident-base queue | 8/8 | 1,279.61 s | 22.507 | 31.433 | 1.286 | **7,443 MiB** |
| fork + managed MPS | 8/8 | **751.16 s** | **38.341** | **74.870** | **3.015** | 21,479 MiB |

Against the queue, smolvm finishes the eight-job batch 41.3% sooner, provides 1.70x
jobs/hour, 2.38x effective rollout-token throughput, and 2.34x completed-step
throughput. It spends 2.89x the peak VRAM (13.71 GiB more) because eight private
adapter/optimizer/CUDA states are live concurrently. An individual queued learner is
faster—the median train time is 154.70 s versus 489.33 s per fork—but the other seven
jobs wait. The benefit is concurrent batch completion and lower queue latency, not a
faster learner.

The source-identical comparison passes every quality gate: exact model output,
initial adapter, dataset, initial CPU/CUDA RNG, final CPU RNG, and parameter count;
maximum reward-mean delta 0.011319 (limit 0.02); maximum final parameter-L2 relative
delta 0.0002173 (limit 0.001). This establishes value over a sequential hot-base
queue when GPU memory is available and one GRPO learner leaves compute idle. It does
**not** establish superiority over a fused concurrent multi-adapter trainer or
several hot queues co-located under MPS; those remain separate application-aware
baselines.

The open-source baseline audit narrows that caveat. [mLoRA](https://github.com/TUDB-Labs/mLoRA)
is a real shared-base concurrent training system, but its public runner supports
LLaMA-family SFT and DPO/CPO/CIT rather than source-identical Unsloth/TRL GRPO; using
it changes the training engine and workload contract. [tLoRA](https://arxiv.org/abs/2602.07263)
describes a newer fused elastic super-model for heterogeneous LoRA training, but the
paper page does not expose a runnable artifact. Hugging Face PEFT can activate and
train several adapters together, but that composes them in one model/optimizer rather
than executing independent experiments. [Punica](https://github.com/punica-ai/punica)
and similar segmented-LoRA kernels are multi-tenant inference systems. Therefore the
manually tuned native MPS worker pool remains the strongest runnable, source-identical
GRPO control; a fused GRPO comparison requires an explicitly application-aware port,
not another smolvm runtime flag.

Because each GRPO fork generates its own sampled completions, the measured 2.38x
effective token-rate gain directly covers rollouts embedded in independent trainer
processes. A dedicated rollout-only fleet has now been tested separately. The exact
Unsloth vLLM+tensor-LoRA path was snapshotted after engine initialization and graph
capture, then restored into four isolated workers. Inheriting a pre-fork
`LoRARequest` failed on vLLM's first post-restore input-preparation event; recreating
that small mutable request in each worker, as `GRPOTrainer` already does, passed. A
one-worker probe produced the exact native output hash for the same prompt and seed,
and the four-worker run completed 4/4 with four distinct rollout hashes.

The strong control batches the same four logical jobs through one hot vLLM engine.
Both arms use a 0.06 GPU-memory-utilization setting, four prompts by four generations,
32 completion tokens, and 20 rounds:

| rollout topology | completion | useful scheduled tok/s | sampled peak | median hot batch |
|---|---:|---:|---:|---:|
| one fused vLLM engine | 4/4 logical jobs | **4,447.079** | **6,532 MiB** | approximately 0.24 s |
| four isolated fork workers | 4/4 | 1,071.578 | 9,154 MiB | approximately 0.80 s per worker |

The fused engine is **4.15x faster and uses 28.6% less memory**. Isolated workers also
pay 6.7--8.9 s on their first generated batch and approximately 9 s to reconstruct a
worker-local tensor-backed LoRA request. Therefore rollout-only pools are a no-go as
the default homogeneous performance path. They remain useful only when independent
CUDA state, failure domains, heterogeneous policies, or environment isolation are the
product requirement; smolvm should route homogeneous rollout batches to one engine.

Durable evidence is in
`bench/results/queue_grpo-queue-rngfixed-long200_n8_s200_c16_20260726-184329_r1.json`,
`bench/results/fork_grpo-fork-vsqueue-final-long200_n8_s200_c2_20260726-190515_r1.json`,
`bench/results/grpo-queue-vs-fork-h100-20260726.json`,
`~/bench/results/queue_grpo-rollout-fused-kv06_n4_a4_s20_c4_20260729-212644_r1.json`,
and `~/bench/results/fork_grpo-rollout-isolated-local-lora_n4_a4_s20_c4_20260729-212122_r1.json`.

#### Equal-scheduling control and cold-path root cause

The ordinary native control above gives each process a separate CUDA context but does
not place those contexts under MPS. That is not the strongest concurrent hot-worker
baseline. A paired 50-step screen prewarmed every native process through the same
zero-update GRPO step, held memory at about 59.5 GiB, and changed only whether eight
native workers used a private uncapped MPS server:

| native mode | tail rollout tok/s | learner-steps/s | peak GPU memory |
|---|---:|---:|---:|
| ordinary contexts | 51.838 | 1.951 | 59,522 MiB |
| uncapped MPS | **89.410** | **3.364** | 59,548 MiB |

The 1.72x gain at unchanged memory proves that MPS scheduling—not CoW—caused
smolvm's throughput lead over ordinary native processes. The 200-step,
source-identical native-MPS versus managed-smolvm comparison passes every deterministic
setup, reward, and adapter gate:

| one-step warmup | wall | tail rollout tok/s | learner-steps/s | peak GPU memory |
|---|---:|---:|---:|---:|
| native + MPS | **425.66 s** | **99.128** | **4.057** | 59,384 MiB |
| smolvm + managed MPS | 751.16 s | 74.870 | 3.015 | **21,479 MiB** |

Smolvm therefore provides 75.5% of native-MPS rollout throughput and 74.3% of its
step rate while using 63.8% less GPU memory. This supersedes the earlier broad GRPO
“beats native” interpretation; that statement applies only to ordinary native
contexts and the sequential resident queue.

Two run lengths locate most of the residual in a per-job cold path. Median 50/200-step
times fit a 20.19-second fixed component plus 1.8202 seconds per step for native MPS,
versus 125.05 seconds fixed plus 1.8214 seconds per step for smolvm. Phase timestamps
confirm the mechanism at N=1: with one warmup step, native reaches its first reward in
1.68 seconds and smolvm in 10.34; smolvm then encounters several lazy-shape stalls.
Giving the VM eight vCPUs instead of two does not move the result.

A diagnostic 20-step zero-update warmup before snapshot eliminates the later stalls.
At N=1, post-warm median reward-step intervals are 0.925 seconds native and 0.948
smolvm; the last-ten means are 0.860 and 0.850 seconds. The remaining clone first-use
delay is not inherited-module reload: every worker eagerly reloaded 1,207 modules in
1.3–1.7 seconds before release. A first-step host profile then observed about 115,000
proxy operations but only about 0.41 seconds in backend dispatch and 0.06 seconds
responding. The rest is the high-volume guest/proxy cold sequence. The profiler also
omitted blocked ring waits from its `idle` counter; the diagnostic counter is corrected
in the investigation branch, but the locally linked binary required newer glibc than
the H100 and is not used for any performance claim.

The final N=8, 200-step representative-warmup pair passes the full quality gate:

| 20-step warm snapshot | wall | tail rollout tok/s | learner-steps/s | peak GPU memory |
|---|---:|---:|---:|---:|
| native + MPS | **475.70 s** | **102.987** | **4.215** | 59,510 MiB |
| smolvm + managed MPS | 790.51 s | 85.105 | 3.501 | **21,507 MiB** |

Relative to smolvm's one-step warmup, hot throughput improves 13.7% by tokens and
16.1% by steps, reaching 82.6%/83.1% of equally warmed native while preserving a
63.9% memory reduction. The deeper golden takes 302.44 versus 188.53 seconds, so one
cold batch is 5.2% slower end to end. If the golden is retained, its estimated
per-wave time falls from 562.63 to 488.07 seconds and the extra preparation amortizes
after 1.53 eight-job waves. Decision: **conditional go for a persistent, workload-
aware hot snapshot serving at least two waves; no transparent runtime activation and
no cold one-batch claim.**

Durable root-cause evidence and exclusions are in
`bench/results/grpo-native-mps-rootcause-h100-20260726.json`.

#### Continued transparent optimization screens — no new release candidate

The next screens tested whether smolvm's memory headroom or host scheduling could
close the remaining equal-MPS gap without changing GRPO:

| screen | result | decision |
|---|---|---|
| Higher-density concurrency | Native MPS N=10 completed 10/10 at **3.797 steps/s**, 100.737 tail tok/s, and 74,254 MiB. Smolvm N=12 completed 12/12 at **2.543 steps/s**, 67.899 tail tok/s, and 28,359 MiB. | **No-go for throughput.** Density survives, but N>8 adds context/CPU contention faster than useful rollout parallelism. |
| `CUDA_MODULE_LOADING=EAGER` | At N=1, first reward was 16.958→16.815 s and last-ten cadence 0.850→0.853 s, while peak memory rose 9,456→11,052 MiB. | **No-go.** The clone cold path is not deferred module loading. |
| Ring wait spin 20,000→2,000 | Three 400k-launch pairs changed 3.84→4.08, 3.63→3.91, and 3.62→3.75 µs/launch. | **No-go.** Earlier parking saves host CPU but regresses every critical launch path by 3.6–7.7%; the prototype was removed and release artifacts restored byte-for-byte. |
| CPU-affinity isolation | N=8 default versus guest/runtime-separated cores changed 2.277→2.214 steps/s, 61.347→59.351 tail tok/s, and 505.31→514.31 s wall; both completed 8/8 at 21,491/21,493 MiB. | **No-go.** Unrestricted scheduling is better for the mixed VM/proxy workload. |
| One CUDA work queue per clone context | `CUDA_DEVICE_MAX_CONNECTIONS=1` produced 2.268 steps/s, 61.090 tail tok/s, and 516.71 s wall versus the paired-default 2.277, 61.347, and 505.31. | **No-go.** Hopper work-queue over-subscription is not the remaining gap; keep CUDA's default. |
| One vCPU per clone | Repeated clean one-vCPU forkable boots failed inside libkrun's first `KVM_RUN` with `ENOMEM`; paired two-vCPU controls booted. The failure was later isolated to a kernel timing bug and fixed with a one-time 5 ms first-entry delay; see the high-fanout resolution below. | **Original screen rejected; superseded by the release-qualified first-entry fix.** The upstream every-entry 5 ms workaround remains inappropriate for CUDA, but one-time startup delay has no steady-state tax. |
| Two-second clone-release staggering | The N=4, 20-step diagnostic changed 0.813→0.821 steps/s, 25.833→25.570 tail tok/s, and 98.44→97.42 s train tail. | **No-go.** The ±1% movement is noise and the same learner remained the straggler; synchronized first use is not the cause. |
| Host-shared Inductor and Unsloth caches | A source-identical N=4 pair changed 0.817→0.803 steps/s, 25.455→25.532 tail tok/s, and 97.86→99.60 s train tail. | **No-go.** PyTorch's FX/AOT caches were already enabled, the selected Inductor directory stayed empty, and sharing generated trainer files did not remove a stall. |
| Host-shared guest kernel-cache paths | The follow-up completed at 0.822 steps/s, 26.133 tail tok/s, and 97.31 s train tail versus the private control's 0.817, 25.455, and 97.86. Neither requested kernel-cache directory was created. | **No-go.** The 0.6% step-rate movement is noise and the stall sequence is unchanged; this workload is not consulting those disk caches. |
| Repeated cuBLAS handle-state cache | Source-identical N=4 control, math-mode-only, and math/stream/workspace arms measured 0.809/0.817/0.817 steps/s and 25.721/25.986/25.962 tail tok/s. | **No-go.** Exact state matched, but the ~1% result is within the existing band and caching additional setters adds nothing. The broader form also cannot preserve arbitrary callers' workspace-reset semantics. |

The cache/release screens also sharpen the attribution. In three source-identical
private/shared-cache runs, learner 1 repeatedly encountered six post-first-reward
gaps in the same bands: about 7.4–7.9, 18.2–19.1, 5.4–7.1, 11.7–13.1, 5.4–5.6,
and 8.9–9.6 seconds. Its first reward remained 22.2–22.6 seconds. The N=8 paired
phase data shows the distinction from native more clearly: warmed native contexts
reach first reward in 2.9–4.2 seconds and have only two later gaps above four
seconds, while new smolvm clone contexts take 30.2–31.9 seconds and exhibit repeated
7–21 second lazy-shape pairs. The initial model/adapter fingerprints match in every
screen.

This closes disk-cache placement and clone-release order as transparent levers. The
remaining state is context-local CUDA/library initialization or execution-plan state:
it is rebuilt for every isolated clone context and is not emitted to the tested disk
caches. Removing that cost without workload cooperation requires either complete
clone-scoped CUDA execution-state snapshot/replay or the safe single-context namespace
described below; it is not an environment or scheduler tweak.

A glibc-2.35-compatible cold-path profiler now resolves the earlier profiler gap on
the actual H100. In a one-clone, one-step run, the clone reached its first reward in
10.595 seconds. At its first 8,192 operations the worker had spent 10 ms executing
kernels, 124 ms in library calls, 42 ms in read/synchronization calls, 3 ms in writes,
3 ms in allocation, and 17.405 seconds waiting for the guest. The library census was
1,863 `cublasSetMathMode`, 685 `cublasSetStream`, 685 `cublasSetWorkspace`, and 589
`cublasGemmEx` calls. At 131,072 operations those setter counts reached 23,261,
8,550, and 8,550 respectively. This proves the repetition but also explains the
paired cache result: setter execution itself is nearly free, most elapsed time is
above or between boundary calls, and removing them cannot remove context-local lazy
shape initialization. The profiler and cache are diagnostic-only and are not in the
production runtime.

#### Clone-local priming — leverage confirmed, transparent replay rejected

A workload-level feasibility probe executed one zero-learning-rate GRPO step inside
each clone, restored the trainable-parameter digest plus CPU/CUDA RNG, emptied the
allocator cache, and then started the measured 20 steps. Against the 0.809 steps/s,
25.721 tail tok/s control, its measured phase reached **1.021 steps/s (+26.1%)** and
**31.781 tail tok/s (+23.6%)** at the same 14,607 MiB peak. The tail train time fell
98.87→78.38 seconds. This confirms that clone-context warm state—not transport,
module reload, disk caches, or base-weight sharing—is a material performance lever.

It is not a shippable optimization as tested. The extra prime made one short wave
425.57→434.35 seconds end to end (**2.1% slower**) and changed sampled trajectories;
learner 2 exceeded the reward-mean quality threshold by 0.0705. More importantly,
the corresponding below-workload replay+restore prototype failed twice. It retained
complete CUDA intervals at synchronization boundaries, backed up all 162 private GPU
ranges, replayed in the fresh clone context, and attempted to restore the exact fork
bytes. Without input writes it asserted at op 175/105,789; with H2D and device writes
included it asserted at `LaunchKernel` op 3,334/102,917. CUDA error 710 poisoned each
context, so even restoration could not synchronize.

The boundary is now explicit: a valid prime requires the framework's logical step
boundary, CPU-side inputs/decisions, and the device state from the beginning of that
step. A generic CUDA proxy sees none of those together. Clone-local priming remains a
useful future workload/runtime contract for long or multi-wave jobs, but automatic
rolling replay is neither transparent nor production-safe. No replay code remains in
the runtime.

NVIDIA's process-checkpoint API does not supply that missing clone primitive on this
system. The installed driver 570/CUDA 12.8 API suspends and restores one process by
copying its device contents through host memory. NVIDIA explicitly excludes IPC memory
created with `cuMemExportToShareableHandle()`, which is the foundation of smolvm's
shared VMM mappings. Restoring multiple full checkpoints would also allocate each
worker's device contents independently, trading away the CoW memory benefit. This API
is useful for preemption/migration and warm restart, not for concurrent lightweight
forks; it is not a hidden transparent path to context cloning.

#### Shared-context GRPO prototypes — correctness no-go

The remaining single-context idea was tested directly with the classic allocator so
the existing inherited-pointer translation could copy mutable allocations while
sharing loaded base weights. A one-clone, five-step control completed correctly with
65 private allocations (580,911,104 bytes) and 82 shared allocations
(5,807,013,888 bytes). Two concurrent clones failed during their first generation
with invalid sampling probabilities, proving that pointer translation alone is
insufficient.

Three increasingly strict namespace prototypes then isolated the missing state:

| prototype | observed result | decision |
|---|---|---|
| Recreate stream 0, explicit streams/events, and top-level library handles per clone | Both clones reconstructed the same 65 private/82 shared allocation split and private execution handles, but N=2 still produced NaN/invalid-probability failures. | **No-go.** Tensor isolation plus the obvious execution handles is incomplete. |
| Route forwarded library stream 0 through the private stream map | The failure changed to out-of-bounds indexing but remained immediate at N=2. | **No-go.** The stream leak was real but not the only shared state. |
| Reload every inherited CUDA module and function per clone | Each clone received 582 module instances and 22,817 function bindings. The immediate assertion disappeared, but both learners made no step progress for more than four minutes while the GPU remained at 100%. | **No-go.** Reloaded code does not reproduce initialized module/runtime state. |
| Allow one clone operation epoch at a time and switch only after a host-visible synchronization | N=2 again failed immediately with invalid probabilities. | **No-go.** Synchronization-boundary scheduling does not make context-global state clone-safe. |

Hopper green contexts do not provide a transparent escape hatch. A direct H100 driver
probe created two green resource groups, converted both with `cuCtxFromGreenCtx`, and
observed distinct context IDs (`2` and `3`). Reserving the same requested virtual address
in the second context succeeded only at a different address. The CUDA header also defines
the conversion as producing a primary context backed by the green resources, not another
stream namespace inside one address space. Green contexts can partition scheduling
resources, but they do not remove the same-process virtual-address collision or provide
the missing per-clone module/library namespace. Isolated contexts under MPS therefore
remain the only qualified transparent architecture.

The H100 runtime was restored byte-for-byte to the production artifact after every
screen (`18d33faae5fa996822e07cf1d407576f`). No shared-context prototype remains in
the source tree. A production design now requires a clone-scoped namespace shared
across all of one guest's connections plus complete snapshot/restore of initialized
module-global and library runtime state; it is not an incremental stream or scheduler
change.

The first clean shared-context run also found an independent daemon routing bug: a
warm-dial clone preamble could enter the worker-spawn branch before the worker-mode
gate. The fix consumes warm dials when workers are disabled while allowing the real
clone connection to continue through the legacy server. Treat that as a correctness
fix, not as evidence for shared-context performance.

#### vLLM-enabled GRPO compatibility and fork gate

The unchanged Unsloth vLLM path exposed a separate transparent hardware-detection
bug. PyTorch loaded the staged CUDA driver with `RTLD_LOCAL`, so the NVML shim's
`RTLD_DEFAULT` lookup could not see it and silently returned its generic fallback
(`NVIDIA GPU`, compute capability 8.0). vLLM therefore selected FlashAttention 2 on
the H100; its selected fatbin failed with driver status 222 and later surfaced as an
invalid device pointer. The shim now explicitly opens `libcuda.so.1` when the global
lookup misses. A clean guest then reported `NVIDIA H100 80GB HBM3` and capability
9.0 through PyTorch, NVML, and vLLM, selected FlashAttention 3, compiled the model,
and captured 70 piecewise plus 38 full decode graphs without a workload patch.

The full GRPO runtime also needs a C++ compiler because Unsloth's first loss compile
enters TorchInductor. The original packed environment lacked `g++` and failed after
graph capture with `InvalidCxxCompiler`; repacking the otherwise identical runtime
with `g++` 11.4 fixed that packaging prerequisite. This is not a CUDA-remoting or
fork-runtime defect, but a reproducible artifact requirement for this workload.

With clone workers disabled and the warm-dial routing fix installed, an unchanged
Qwen2.5-0.5B Unsloth GRPO run completed its zero-learning-rate golden step in 94.11
seconds, forked, preserved compute capability 9.0 and its parameter/model-output
fingerprints, and completed the clone's real Trainer step in 2.08 seconds (7.15
seconds including cloned trainer setup and fingerprints). There were no CUDA errors.
The harness itself returned a control failure only because this legacy shared-context
arm intentionally emits no `M2: shared weight ranges` verdict. This validates N=1
fork continuity across vLLM's captured graphs; it does **not** qualify N>1 independent
learners, which still require the isolated-worker path.

The remaining isolation gate is precise. Under the default managed-MPS worker path,
the golden exports the expected memory/modules/graphs, but a fresh worker fails many
fixed-VA VMM maps with error 1; eager graph pre-warm then reaches an illegal address
and tears down the MPS client. The identical isolated fork with MPS disabled disproved
MPS as the primary cause: it still failed 24 fixed-VA maps, reconstructed only nine
VMM ranges plus 64 translated non-VMM allocations, and poisoned the worker while
pre-replaying the inherited graph logs. The guest then failed its first post-fork
parameter read with `cudaErrorInitializationError`.

The golden contains two distinct CUDA process layouts. Connections for sibling
tokens correctly attach while the process worker is alive, but a fatal worker error
causes later connections to rebuild partial workers from their individual layouts.
The serialized layout itself is internally sound: the primary process reported 58
reservations and 33 non-overlapping mappings, with every mapping contained in a
reservation. Disabling eager graph pre-replay let the clone preserve its initial
parameter/model fingerprints and emit `ready`, but it still failed during GRPO
generation after only 15/33 VMM mappings were recreated.

The failure is pinned to address placement. `cuMemAddressReserve` treats the requested
address as a hint; a direct H100 driver probe returned success while moving every
unaligned request, and smolvm's `mem_address_reserve_fixed` helper neither checked nor
used the returned address. It then called `cuMemMap` at the unreserved golden VA,
which explains error 1 and the missing ranges. Reserving each range before context
creation improved reconstruction from 15/33 to 22/33 mappings but exposed a second
constraint: this H100 driver places nonzero hints exactly only when the requested
envelope is aligned to 32 MiB.

The aligned-envelope prototype merged all VMM reservations and non-VMM pointer guards
into 32 MiB-aligned VA-only spans and reserved them before creating the clone context.
It then reconstructed all 33 VMM mappings plus all 64 non-VMM allocations at their
golden addresses. An unchanged isolated Unsloth GRPO clone preserved capability 9.0
and its initial parameter/model-output fingerprints, emitted `ready`, and completed a
real Trainer step in 2.58 seconds with no CUDA failure; the golden warm step took 92.68
seconds. The harness returned only its expected sharing gate because this control used
`expandable_segments:False`: all 33 VMM ranges and 843,055,104 bytes of non-VMM
allocations were private, so this proves isolated hot-state compatibility and startup
reuse but not base-weight density. A direct expandable-allocator control is not a
transparent route for this workload: Unsloth 2026.7.4 detects vLLM standby mode and
removes `expandable_segments` before allocation because the two modes are incompatible.
The next density gate is therefore safe cross-process sharing of verified ordinary
CUDA allocations, followed by default MPS and N>1 isolation. Do not ship the hard-coded
H100 alignment or claim shared vLLM density until those gates pass.

A source-identical allocation census repeated the successful isolated clone and found
16/64 ordinary allocations (616,562,688 of 843,055,104 bytes, 73%) marked as load-time
upload candidates after the real golden GRPO step. That is a meaningful upper bound
for an ordinary-allocation sharing layer, but the existing boolean is not a sufficient
safety verdict: production sharing must record complete upload coverage, verify the
bytes again at fork time, map only verified pages read-only, and leave every other page
private. The same run replaced the H100 constant with a runtime alignment probe; it
tried the device granularity upward, selected 32 MiB, restored all 33 VMM mappings, and
completed the clone again. The probe needs broader GPU/driver qualification before it
is production-ready.

The stricter ordinary-allocation verifier invalidated that upper bound as a direct
implementation route. Only 6/64 allocations (12,582,912 bytes) had complete verifiable
upload coverage, and none remained byte-identical after the warm step. PyTorch's
ordinary allocator mixes or reuses static and mutable storage, so whole-allocation
sharing is a **no-go**. The next density probe operates at the driver's 2 MiB mapping
granularity: a page qualifies only when every live allocation byte it overlaps is
covered by matching upload hashes; mixed pages remain private.

That page-level probe is also a **no-go**: only 24 pages / 50,331,648 bytes of the
843,055,104-byte staged snapshot (6%) remained completely covered and byte-identical
after vLLM initialization, graph capture, and the real warm GRPO step. A cached mixed
shared/private page remapper would add substantial lifecycle and allocator complexity
for negligible density. Do not implement it. With vLLM standby forcing ordinary CUDA
allocations, transparent sharing below the workload/allocator boundary is exhausted;
the remaining default-runtime work is isolated multi-channel/graph correctness, not a
credible memory optimization for this particular configuration.

That tracking control also reproduced the separate graph-replay reliability gate. The
main worker lazily recaptured many inherited graphs, then a late attached CUDA channel
encountered error 700 while resolving modules, killed the worker, and the guest later
surfaced `CUBLAS_STATUS_NOT_INITIALIZED`. No sharing was active, so this is not a CoW
failure; it keeps default graph replay and multi-channel vLLM outside the production
gate.

An immediate source-identical all-private replay control then passed: no late channel
appeared, the same 26 lazily used graphs recaptured, and the unchanged clone completed
at 9.697 exact aggregate tok/s with preserved fingerprints. This makes the late-channel
fault intermittent/topology-dependent rather than a deterministic consequence of the
recorded graphs. It still blocks production qualification: a transparent runtime must
handle every CUDA connection the unchanged workload opens, not only runs that remain
on the initial channel.

The intended default eager pre-replay is independently **no-go** even after complete VA
reconstruction: it recaptured 0/1,840 inherited graph logs and every attempt eventually
returned error 700, poisoning the worker before guest work resumed. Eager replay must
therefore require explicit opt-in; lazy first-launch replay is the safe default. The
late-channel root cause is also concrete: the worker stored graph logs in a draining
thread-local, so the first channel consumed all 1,840 and a later channel could not
rebuild an unseen graph. The active fix keeps immutable logs process-wide, fetches only
the launched graph for each session, and serializes first capture across channels. It
is not production-ready until unchanged N=1 and N>1 runs pass.

The first fixed-default H100 qualification passed unchanged at N=1. It retained all
1,840 inherited logs process-wide, performed zero eager replay attempts, lazily rebuilt
only the 26 graphs the clone actually launched, preserved capability 9.0 plus all
initial model/parameter fingerprints, and completed at 9.786 exact aggregate tok/s
with no CUDA error. No late channel appeared in that run, so it validates the safe
default and primary-channel path.

The source-identical N=2 qualification then passed both isolated clone workers with
MPS disabled and no sharing active. Each clone retained all 1,840 graph logs and lazily
rebuilt the same 26 graphs with no eager replay, CUDA error, or process failure. The run
completed at 17.363 exact aggregate tok/s and 79,292 MiB peak device memory; distinct
dataset and rollout fingerprints confirm independent work. This validates concurrent
workers, but neither clone opened the intermittent late channel. The toy reward was
constant, so the zero-loss step also left parameter fingerprints unchanged; an
observed late-channel pass and a nonconstant-reward parameter-update run remain before
production qualification.

The automatic managed-MPS default passed unchanged at N=1: all 26 used graphs rebuilt
lazily, the learner completed at 8.815 exact tok/s, and the private MPS controller and
server exited with the daemon. The first longer N=2 managed-MPS run did **not** pass:
one learner completed three steps and changed its trainable-parameter fingerprint, but
the other worker received SIGSEGV after its 26th successful re-capture and first launch
of the large 2,305-op graph. Its later connections then failed initialization because
the owning context was gone. Peak memory was 79,243 MiB, effectively identical to the
passing no-MPS control, and the NVIDIA MPS server log recorded an ordinary client exit
rather than a server/GPU fault. This rules out OOM, missing graph logs, and an MPS
service crash; concurrent execution exposes a client-side first-launch lifetime or
ordering fault.

`CUDA_LAUNCH_BLOCKING=1` made the same N=2 managed-MPS configuration pass 2/2 at
15.471 exact aggregate tok/s. That is a diagnostic control, not a proposed default:
global synchronous launches are too broad. The current narrower probe synchronizes
only the first launch of each newly re-captured inherited graph, leaving every later
launch asynchronous. It is also a **no-go**: learner 0 completed three steps with
nonconstant rewards and a changed parameter fingerprint, while learner 1 still hit the
same worker SIGSEGV. The unsafe window is therefore later asynchronous graph overlap,
not graph construction or only the first launch. The next probe synchronizes every
inherited replayed-graph launch while leaving non-graph kernels, copies, and library
work asynchronous. It is a third **no-go**: one learner again completed three updates
while the sibling worker crashed. Global launch blocking changes more than graph
ordering, so the next discriminator is an operation-ring trace of the ordinary
kernel/library/memory traffic immediately before the fault.

The operation ring resolved the fault completely. After the 26th graph rebuilt, the
worker successfully freed a series of temporary allocations, then a deferred
`MemCreateVh` for `0xb1e00000` bytes (2.78 GiB) returned status 2 / CUDA OOM. Peak
sampling was 79,133 MiB, so that request genuinely did not fit. The guest had already
pipelined its dependent `MemMap`; because the host sticky-error path kept dispatching
the rest of the quiet batch, the map's never-created virtual handle fell through to
libcuda and produced SIGSEGV. Graph capture and MPS were timing witnesses, not the
cause. The production fix is two-layered: stop dispatching a deferred batch after its
first failure until the fence reports it, and reject any tagged VMM handle absent from
both the session and reconstructed-worker maps. The workload must then receive a clean
OOM rather than losing its CUDA host process.

The capacity finding is separate from that correctness fix. This N=2 all-private run
holds the frozen golden plus two clone copies; only one learner can obtain the extra
2.78 GiB update allocation at a vLLM utilization of 0.15. A lower fitting cache budget
and a native N=2 control are required to distinguish smolvm's retained-golden tax from
the workload's intrinsic two-worker footprint. Longer term, evicting a frozen golden's
private mutable snapshot to host memory is the transparent architectural lever: it
would retain future forkability without charging its full inactive footprint to HBM.

The first H100 validation of the deferred-batch/VMM-handle guard passed its failure
contract on the exact over-capacity case. One learner again completed all three updates;
the other surfaced `CUDA error: out of memory` in Python. The daemon recorded zero
fatal worker signals and zero worker respawns, and NVIDIA MPS stayed healthy. This turns
the former host-process crash into the correct workload-visible OOM without perturbing
the successful sibling. A fitting two-learner run is still required for the positive
functional gate.

The first cache-budget probe (`gpu_memory_utilization=0.10`) was invalid as a
capacity lever and is excluded: Unsloth standby clamps every supplied value to its
H100 target unless `UNSLOTH_VLLM_STANDBY_UTIL_OVERRIDE=1` is set. The workload reported
the requested 0.10, but peak HBM remained 79,169 MiB and the same one-of-two OOM
recurred. The next diagnostic explicitly enables Unsloth's supported override to test
whether the retained footprint is actually tunable; smolvm must not silently set this
workload-specific variable as a product fix.

With that supported override explicitly enabled, the fitting 0.10 run passed both
managed-MPS clones through three real GRPO updates. It completed at 40.141 exact
aggregate tok/s (39.344 tail aggregate), peaked at 16,177 MiB, rebuilt exactly 26
graphs per clone, and recorded no CUDA error or worker failure. Both parameter digests
changed; dataset and rollout digests differed between learners, and learner 0 observed
nonconstant rewards (0.7945–1.0). This is the first unchanged-workload concurrent vLLM
GRPO qualification that proves independent parameter updates. It also confirms the
previous failure was entirely capacity/error-propagation, not MPS concurrency or graph
replay correctness. A source-identical native N=2 control is required before making a
throughput claim.

An unchanged repeat passed the same quality gates at 41.228 exact aggregate tok/s
(40.808 tail) with both parameter digests updated, distinct dataset/rollout digests,
zero CUDA errors, and zero worker failures. Its sampled peak was 22,835 MiB rather
than 16,177 MiB, so the short-run throughput is reproducible but a tight peak-memory
number is not. Both runs reported `shared=0 private=33`; this qualification proves
transparent fork continuity and hot-state reuse, not ordinary-allocation CoW density.

The first native control was invalid and is excluded because the harness selected the
host `ptwork` environment, which did not contain vLLM. The packed guest actually uses
torch 2.10.0, Unsloth 2026.7.4, vLLM 0.19.1, TRL 0.24.0, and Transformers 4.57.6;
the host `rlwork` environment matches those versions. With that corrected stack,
ordinary native N=2 at utilization 0.10 passed at 12.013 exact aggregate tok/s and
17,507 MiB. Native under a private uncapped MPS server could not initialize both
vLLM caches at 0.10: one learner cleanly reported that no cache blocks were available.
The harness also exposed and fixed a diagnostic-only false-pass bug: `wait pid0 pid1`
returns only the final PID's status, so it now waits each learner and fails if any one
fails.

Raising the native-MPS cache budget to the smallest tested passing value, 0.12, made
both learners pass the three-step screen at 30.507 exact aggregate tok/s and 13,254
MiB. The meaningful steady pair used 20 updates per learner: native MPS reached
106.292 exact aggregate tok/s (103.602 tail), 1.619 learner-steps/s, and 21,676 MiB;
smolvm at utilization 0.10 reached **111.026 exact aggregate tok/s** (109.918 tail),
1.717 learner-steps/s, and 22,857 MiB. Both smolvm learners updated their parameter
digests, produced distinct 1,280-token rollouts and datasets, observed nonconstant
rewards, lazily rebuilt 26 inherited graphs apiece, and completed with no CUDA or
worker error.

The final equal-budget smolvm run at utilization 0.12 passed the same gates at
**111.613 exact aggregate tok/s** (110.202 tail), 1.722 learner-steps/s, and 26,215
MiB. It is 5.0% faster than native MPS at the same setting, with zero errors and 52
successful lazy graph re-captures. Smolvm has therefore reached native-MPS steady
throughput for this real N=2 GRPO workload. Its remaining cost is memory, not speed:
it uses 4,539 MiB more than native because the forkable frozen golden remains resident
alongside both private learners. Evicting the inactive golden's private mutable GPU
snapshot to host memory is now the most meaningful transparent optimization target.

The validated production changes are now isolated from diagnostics on the rebased
`fix-vllm-fork-compatibility` branch: safe process-global lazy graph logs, serialized
late-channel capture, pre-context exact-address reservation with a runtime alignment
probe, and deferred-batch/VMM-handle failure containment. The graph-launch
synchronization probes, allocation/page-sharing census, operation trace, workload,
harness, and result artifacts remain investigation-only.

The exact pushed revision `52c8d577ac40c79e06317d354d1991a577f078da` is now
qualified independently of the diagnostic binary. Its release binary
(`164a98a95020d7cecb12416b60d85e24`) and a rootfs restaged from the same source both
reported protocol `c5631758382d21af`. The corrected N=2 utilization-0.12 smoke passed
2/2 real updates at 39.837 exact aggregate tok/s and 18,063 MiB peak, with 52 lazy
graph re-captures, capability 9.0, distinct datasets/rollouts/final parameter digests,
and zero CUDA/worker errors. The first smoke attempt is excluded: it used the old
1.6.13 rootfs protocol `71cace562af4e81f` against the 1.7.0 host and failed the
explicit handshake. A second excluded harness invocation enabled Unsloth's override
gate but accidentally left the workload at its default utilization 0.3; it is not a
release regression or a performance result.

The first generic capacity-planning prototype also passes. The existing
`SMOLVM_CUDA_VRAM_LIMIT_MB` guard enforced allocations but still reported the physical
device size through `cuDeviceTotalMem` and `cuMemGetInfo`, so frameworks sized caches
for the whole H100 and hit the limit later. A host-only change now advertises the same
limit and subtracts the current session's tracked allocations from reported free
memory. Through the real shared-daemon path, PyTorch reported free, total, and device
property memory as exactly 10,737,418,240 bytes for a 10 GiB budget while retaining the
truthful H100 name and capability 9.0.

With only that generic host budget set—no Unsloth standby override and no workload
utilization setting—the unchanged N=2 GRPO gate passed at 44.353 exact aggregate
tok/s (43.636 tail), 0.682 learner-steps/s, and 16,417 MiB peak. Both learners changed
to the same deterministic per-learner final digests as the explicit-0.12 production
smoke, with distinct data/rollouts, 52 successful lazy graph rebuilds, and zero runtime
errors. Unsloth visibly reduced the golden capture from 30 piecewise + 18 full graphs
to 22 + 14. This validates virtual VRAM reporting as a framework-agnostic sizing
primitive; it does **not** solve automatic policy. Smolvm still needs the intended
fork-pool size before golden initialization to choose a safe per-replica budget. A
later incremental `machine fork` cannot transparently shrink an already allocated KV
cache, so guessing a universal default would trade away single-worker capacity or
still overcommit larger pools. The validated host-side reporting fix is pushed as
`bfa048f`; automatic pool-budget selection remains separate and unimplemented.

The 20-update repeat rules out a setup-only gain. It passed 2/2 at **112.207 exact
aggregate tok/s**, 112.182 tail, 1.753 learner-steps/s, and 23,273 MiB with zero
runtime errors and 52 lazy graph rebuilds. That is 0.5% faster and 2,942 MiB lower
than the explicit-0.12 smolvm run, and 5.6% faster than native MPS while using 1,597
MiB more than native. Both final parameter hashes, rollout hashes, and reward ranges
match the explicit-smolvm and native controls learner-for-learner. The generic virtual
view therefore changes capacity planning without changing the sampled training path.

#### Automatic fork-pool capacity prototype — correctness passes, tail gate still open

An uncommitted product prototype adds `machine start --fork-pool-size N` (implying
`--forkable`) plus an explicit `--cuda-vram-limit-mib` override, persists the policy
on the golden and its clones, and carries it per VM through the shared daemon. For an
80 GiB H100 and `N=2`, the density policy advertises and enforces exactly 10 GiB per
CUDA session. The first three-update screen passed 2/2 at 16,012 MiB with independent
datasets, rollouts, and updated adapter hashes. A later 20-update attempt is excluded:
both learners reached roughly 10--12 updates before one context surfaced CUDA unknown
/ cuBLAS-not-initialized errors. There was no clean OOM or fatal worker signal, so the
three-update screen was not a sufficient reliability gate.

The meaningful discriminator used the same new per-VM handshake with an explicit
10 GiB limit. It passed 2/2 through 20 updates at 114.542 sum-of-per-learner tok/s and
23,346 MiB, with deterministic final parameter/rollout hashes, zero CUDA errors, and
zero worker deaths. Automatic sizing differed only by querying physical device memory
again on allocation paths. Resolving the automatic share once at CUDA session init
made its clean candidate pass the same gate at 112.329 sum-of-per-learner tok/s and
22,700 MiB. Its learner-for-learner final hashes exactly match the explicit control;
it also recorded zero CUDA errors and zero worker deaths. This validates stable
one-time policy resolution as the correct automatic implementation shape.

Those two new runs are **correctness and capacity evidence, not steady-throughput
passes**. In both, one learner completed in about 27 seconds while its sibling took
130--132 seconds, reducing tail aggregate throughput to 38.6--39.1 tok/s and aggregate
step rate to about 0.30/s. The misleading 112--115 figure sums each learner's local
average and ignores that long tail. The delayed clone reconnects a second inherited
CUDA lineage (`token 5`) with zero allocations, hundreds of staged modules, and no
inherited graphs. That lineage is consistent with an auxiliary CUDA-using compiler
or engine process, but correlation is not causation: it appears after the fast learner
finishes and is not yet established as the sibling's stall. The earlier generic
10 GiB repeat had no such reconnect and sustained 112.182 tail tok/s.

The daemon-global 10 GiB control has now resolved the policy discriminator. With no
fork-pool preamble and the limit supplied only through
`SMOLVM_CUDA_VRAM_LIMIT_MB=10240`, it passed 2/2 with the same deterministic final
state and 23,346 MiB peak, but reproduced the long tail at 38.746 aggregate tok/s and
0.304 learner-steps/s. It also produced the same late empty-layout `token 5`
reconnect. The absence of a per-VM preamble clears the new policy **value** and its
one-time limit resolution, but it does not clear the candidate binary itself because
the new daemon still runs the policy-detection path for every connection.

An exact old-binary repeat on the same live box closes that distinction. Binary
`aec8c273...`, the verified `5af851c7...` vLLM workload, vLLM 0.19.1, the same guest
venv mount, and the same daemon-global 10 GiB limit passed 2/2 at **110.631 tail
tok/s**, 1.729 learner-steps/s, and 23,400 MiB. Both learners finished in about 23
seconds with the expected deterministic hashes. The old binary lazily re-captured 26
inherited inference/training graphs per worker (52 total); the first regressed
candidate completed only one per worker before the asymmetric stall. That contrast
initially implicated graph resolution, but the unique-handle trace disproved it: the
primary clone path sees all 26 inherited graph handles as tagged and backed by
immutable operation logs, and successfully re-captures the used graphs. Graph
continuity is not the regression.

The trace instead exposes a deterministic worker-lifecycle storm in the second
lineage. `token 5` has no allocations, streams, events, or graphs, but stages 456
modules and 21,747 functions. Its first real channel is attached to the reconstructed
worker just as the startup channel reaches clean EOF. `run_clone_worker` then returns,
the process exits with code 0, and the attached channel is killed with it. The same
lineage reconnects about every 3.5 seconds, causing smolvm to rebuild those 456 modules
again each time. The guest failure stack includes Torch Inductor's
`compile_worker/subproc_pool.py`, turning the earlier auxiliary-compiler hypothesis
into direct evidence. The transparent fix is to retain the reconstructed worker while
attached channels are active and for a bounded idle grace period after clean channel
turnover; it must restore the exact 20-step tail before the capacity policy can ship.

The first production-shaped fix screen passed. The worker now counts its primary and
attached serving channels, keeps the reconstructed context alive while any are active,
and exits after a configurable 30-second all-idle grace. The unchanged N=2,
three-update workload completed 2/2 in lockstep at 44.189 tail tok/s and 16,478 MiB,
with independent datasets/rollouts, expected final adapter hashes, and zero CUDA or
worker error. No `token 5` reconnect occurred in this short screen, so it validates
correctness and removes the immediate regression but does not by itself prove that the
storm is resolved. The exact 20-update repeat remains the go/no-go gate.

The first exact 20-update repeat also passed: **115.602 exact aggregate tok/s**
(115.576 tail), 1.806 learner-steps/s, and 16,550 MiB peak. Both learners finished in
22.15 seconds, rebuilt exactly 26 inherited graphs apiece (52 total), produced the
known learner-specific final parameter and rollout hashes, and logged zero CUDA error,
graph miss, worker exit, or fatal signal. On this run the candidate is 4.5% faster than
the exact old-binary tail and 11.6% faster than native MPS, while using 23.6% less peak
VRAM than native. `token 5` did not appear, so the run proves that the regression is
absent under the changed lifecycle but does not directly exercise a recurrent helper
channel. Two further source-identical repetitions are required before release
qualification; the unexpectedly lower peak also needs repeatability before it is
attributed to the fix.

The second sustained measurement agrees: 117.298 exact aggregate tok/s
(116.948 tail), 1.827 learner-steps/s, and 16,554 MiB peak, again with 2/2 exact
learner completion. The harness's automatic next repetition was excluded before any
workload started because preflight caught the prior private MPS server still holding
1,506 MiB during its asynchronous shutdown; it released the context moments later.
This is a measurement-isolation success, not a workload failure, and the third run is
being launched separately from a clean 58 MiB standing-MPS baseline.

The third sustained run closes the throughput gate and corrects the memory claim. It
passed at 117.064 exact aggregate tok/s (116.047 tail), 1.813 learner-steps/s, 2/2
known-correct final states, and 52 successful lazy graph re-captures with zero runtime
error. The three tail results are 115.576, 116.948, and 116.047 tok/s: a tight,
repeatable recovery. The third 1 Hz sampler caught 23,380 MiB, however, despite the
same two workers, 33 private/zero shared ranges each, and no auxiliary lineage. The
first two 16.55 GiB peaks therefore missed a short overlap window; they are not a
memory optimization. Use the conservative 23.38 GiB maximum until a higher-frequency
sampler proves otherwise. For this vLLM GRPO shape, the lifecycle fix makes smolvm
about 12% faster than native MPS but still roughly 8% higher in peak VRAM because
vLLM's allocations are not currently classified as shareable and the frozen golden
remains resident.

The final immutable source and actual automatic CLI path now pass the same sustained
gate. Release binary `b437f4d3...`, started with only `--fork-pool-size 2` (no daemon
VRAM override), completed 2/2 at 110.004 exact aggregate tok/s (108.751 tail), 1.699
learner-steps/s, and 22,704 MiB. It produced the same final parameter and rollout
hashes, rebuilt 52 inherited graphs, and logged no helper lineage, CUDA error, graph
miss, worker exit, or fatal signal. This is about 5% faster than native MPS and restores
the automatic path to the pre-regression performance band. It is lower than the three
explicit-control tails (median 116.047), so N=2 automatic policy timing still has a
roughly 6% control delta to characterize; it is no longer a correctness or catastrophic
tail issue.

Automatic N=4 scales productively on the same final binary. With 4 vCPUs per clone
(16 assigned on a 26-core host, so no CPU oversubscription), all four learners
completed exact updates at **198.558 aggregate tok/s** (196.998 tail), 3.079
learner-steps/s, and 28,166 MiB peak. Their four datasets and final parameter hashes
are distinct and deterministic; the workers rebuilt 26 inherited graphs each (104
total) with zero CUDA error or worker exit. Aggregate tail is 81% above the automatic
N=2 run. One empty-layout `token 5` compiler lineage appeared once after the learner
phase, reconstructed once, and did not reconnect or storm before cleanup—the first
real-workload exercise of the lifecycle fix's target beyond the unit regression.

The equal-N native control does not start. Four unchanged native processes each size
vLLM against the full 79.2 GiB device; together they exhaust VRAM during KV-cache
allocation (only 1.39 GiB remained, with individual processes already at roughly
11--33 GiB), and all four abort before training. Supplying the prior nominal 0.12
utilization is still insufficient with Unsloth standby: two workers reject a zero-size
cache, one reaches training, and one fails vLLM's sleep assertion because global GPU
usage increased while it slept. This is the exact multi-tenant capacity-coordination
failure the per-session logical device view removes transparently. A native control
with standby disabled is still needed to measure a manually reconfigured alternative;
it is not equivalent to the unchanged workload experience.

That manually reconfigured native alternative succeeds when standby is disabled and
the workload explicitly sets utilization 0.12. At N=4 it reaches 201.446 exact
aggregate tok/s, 178.265 tail tok/s, 2.786 learner-steps/s, 22,319 MiB, and 79.88
seconds full wall time. Against it, automatic smolvm is 1.4% lower by the misleading
sum of learner-local averages, but **10.5% higher in tail throughput and aggregate
step rate** (196.998 tok/s and 3.079 steps/s), because its four learners finish more
uniformly. Smolvm uses 26.2% more peak VRAM here (28,166 MiB) because vLLM has zero
shareable ranges and the golden remains resident. Its 374.85-second first-wave wall
time is also much worse because it includes a 322.97-second golden initialization;
after the golden is hot, the forked wave occupies about 51.9 seconds versus native's
79.9 seconds. The honest current value is therefore transparent capacity coordination,
hot-wave startup, and better tail throughput—not vLLM memory sharing or first-run
latency. DPO remains the workload with validated true sharing and lower VRAM.

Automatic N=8 with 3 vCPUs per clone is correct but not throughput-optimal. All eight
learners complete with distinct deterministic parameter hashes, 208 total inherited
graph re-captures, and zero CUDA/worker error at 61,644 MiB peak. Seven finish in
31.97--33.71 seconds; learner 7 takes 74.03 seconds, pulling tail throughput to 138.106
tok/s and aggregate step rate to 2.161, both below N=4. Seven clones each spawn one
empty-layout `token 5` helper. This run happened to survive the candidate's fixed
30-second idle grace, but the 2-vCPU stress control proved that elapsed channel silence
is not a valid local-worker lifetime signal.

On that first 2-vCPU control, all eight learners reached `ready` but none completed.
Six helper lineages repeatedly reattached (116 route/attach events), two primary
workers exited cleanly when the 30-second grace elapsed during legitimate live-VM
CUDA gaps, and the private MPS server entered `Teardown in progress, client creation
denied`; subsequent training forwards aborted. There was no allocation failure,
daemon CUDA error, or fatal worker signal. The root cause is therefore the fixed idle
timer under CPU pressure, not graph replay, memory capacity, or MPS scheduling.

The corrected worker follows the clone VM host PID advertised by the existing
proc-mem transport. A live VM keeps its reconstructed context indefinitely; after the
VM dies the worker gets a five-second cleanup grace. Only transports without a local
PID use a configurable bounded fallback (300 seconds by default). The source-identical
N=8, 2-vCPU repeat passes **8/8** at **290.043 exact aggregate tok/s**, **139.710 tail
tok/s**, 2.186 learner-steps/s, and 51,394 MiB sampled peak. Its parameter, rollout,
per-step rollout, and dataset hashes are byte-identical to the prior N=8 run. It
performs 208 successful inherited-graph re-captures and logs zero CUDA/worker/MPS
error, zero mid-run lifetime exit, and zero old idle-grace exit. All 15 primary/helper
workers exit only after machine deletion. The one straggler still takes 73.18 seconds
while the other seven take 31.75--33.55, confirming a host-scheduling tail rather than
a CUDA lifecycle failure. Do not attribute the lower one-run sampled peak as a memory
optimization without repetition.

A high-fanout capacity regression is now fixed and release-gated. The original
`min(90%/(N+1), 1/8)` policy was applied to the golden as well as its clones, so an
N=24 pool advertised only `72 GiB / 25 = 2.88 GiB` while the 7B golden needed roughly
6.7 GiB just for its shareable base load. Transformers consequently tried CPU/disk
offload and rejected the unchanged workload before any fork existed. Capacity is now
role-aware: on an 80 GiB H100 the golden retains the validated 10 GiB density budget
needed to load the one shared base, while each clone retains the tighter pool share for
private post-fork growth; an explicit user limit still takes precedence. Clone-marked
connections receive this policy in both isolated-worker and legacy shared-context modes.

The role-split candidate passed a real N=24 DPO gate: the golden loaded all 339 weight
shards, all 24 clones reached training, and 24/24 completed with finite loss `0.6931`,
`shared=260 private=148`, and 63,047 MiB sampled peak. This is a one-step correctness,
density, and high-fanout policy gate—not a throughput result. Its long per-step times
again reflect severe host-CPU oversubscription at N=24. The production commit is
`c11567c`; local gates passed strict workspace clippy plus 426 core, 113 guest-agent,
42 CUDA, and 11 Smolfile tests.

#### High-fanout CPU and clone-restore resolution

The N=24 rollover was directly attributable to host CPU oversubscription, but the
first apparent workaround was correctly rejected. The matched two-vCPU control
completed 24/24 at **13,422 tok/s**, 519.94 seconds wall, and 63,561 MiB; host CPU
averaged 63.3% busy and spent 312/522 samples (59.8%) above 90%. Pinning only the
guest workload to CPU 0 reached 14,001 tok/s (+4.3%) but reduced wall time only 0.7%
and produced NaN losses in four learners. Do not revive affinity injection: it is not
a correctness-preserving product optimization.

The one-vCPU boot failure was then reproduced without CUDA, including a bare N=1 VM:
libkrun's first `KVM_RUN` returned `ENOMEM` on a host with ample memory. The existing
`KRUN_ENOMEM_WORKAROUND` avoids the kernel bug by sleeping 5 ms before *every* KVM
entry, which would impose an unacceptable steady tax on a CUDA RPC workload. The first
mitigation, `KRUN_FIRST_RUN_DELAY`, slept once before only the first entry and passed
the original qualification. A later repeat on the same H100 showed that the delay alone
is not reliable: untraced one-vCPU boots could still lose the race and fail immediately.

The production fix now combines that one-time delay with at most 100 ms of retries only
after the otherwise-impossible `KVM_RUN ENOMEM`; normal guest exits never sleep. Delay
alone and retry alone each failed the repeat, while the combination booted five explicit
and five zero-configuration candidate VMs consecutively in 1.66--1.76 seconds. libkrun
commits `9cb89fe` and `86daef5` on `enomem-retry-probe` implement the bounded fallback;
smolvm commit `c6929cf` enables both defenses automatically only for Linux x86 one-vCPU
VMs. The 31 runnable VMM tests, 438 smolvm core tests, 44 CUDA tests, formatting, and
strict clippy gates pass.

The first N=24 one-vCPU DPO measurement exposed a separate pre-existing correctness
bug rather than a CPU-policy failure. It completed at 16,556 tok/s, but learners 1
and 8 had NaN losses. Running those exact learner/data seeds natively and serially
produced finite losses (0.6857→0.4026 and 0.6864→0.4091), ruling out the synthetic
dataset. The invalid run logged 13 cases where a shared imported CUDA mapping failed,
fell back to a private copy, then failed its temporary source mapping with CUDA 801.
The worker warned and continued with the private destination uninitialized. Historical
N=16/N=24 NaNs occurred under the same fail-soft reconstruction code, so no prior
NaN-bearing throughput result is release evidence.

The production candidate now retries imported allocation mappings for a bounded 25 ms
and fails clone startup on any incomplete import, reservation, access, copy,
synchronization, unmap, or release. With automatic host-backed golden snapshots enabled,
all 24 workers restored `shared=260 private=148` with zero map failures; the daemon then
evicted the frozen golden CUDA context. Two corrected N=24 runs passed 24/24 with every
loss finite:

| arm | aggregate tok/s | wall | sampled peak | CPU mean / >90% | versus 13,422 control |
|---|---:|---:|---:|---:|---:|
| explicit one-vCPU qualification | **18,792** | **439.60 s** | 62,067 MiB | 54.5% / 48.4% | **+40.0%** |
| transparent policy, machine configured for four vCPUs | **17,054** | 458.74 s | **61,571 MiB** | 55.9% / 47.0% | **+27.1%** |

The transparent policy caps a declared CUDA fork pool at the configured vCPU count or
an even host share, whichever is smaller: N=24 on this 26-CPU host becomes one effective
vCPU per golden/clone with no workload change. `SMOLVM_CUDA_FORK_CPU_POLICY=off` is the
rollback switch. End-to-end, a machine created with `--cpus 4` and started with
`--fork-pool-size 24` reported `nproc=1`; the final workload used no vCPU, eviction, or
KVM-workaround flags. The honest corrected range is **17.1--18.8k tok/s**, or 82.0--90.4%
of the 20,796 tok/s native ceiling—not a native win yet, but a material transparent
recovery while preserving the fork pool's memory density.

Supporting artifacts:

- `~/bench/results/fork_dpo-current-n24-control_n24_s20_c4_20260729-050703_r1.json`
- `~/bench/results/fork_dpo-current-n24-affinity1_n24_s20_c4_20260729-051614_r1.json`
- `~/bench/results/fork_dpo-vcpu1-first-entry-n24_n24_s20_c4_20260729-055427_r1.json` (invalid NaN/root-cause run)
- `~/bench/results/fork_dpo-vcpu1-restore-retry-n24_n24_s20_c4_20260729-061211_r1.json`
- `~/bench/results/fork_dpo-auto-resources-n24_n24_s20_c4_20260729-062755_r1.json`
- CPU traces with the matching `cpu-dpo-*.log` names under `~/bench/`.

The retry also unlocks a source-identical vLLM GRPO high-fanout A/B that the first-entry
failure previously made impossible. With ordinary-allocation golden eviction and the
exact Qwen2.5-0.5B workload, N=16 completed 16/16 in both arms:

| effective vCPU per clone | exact aggregate tok/s | tail tok/s | learner-steps/s | wall | sampled peak |
|---|---:|---:|---:|---:|---:|
| **1 (automatic policy)** | **426.746** | **211.566** | **3.313** | **477.82 s** | 81,075 MiB |
| 2 (policy disabled) | 417.600 | 198.939 | 3.116 | 486.74 s | 81,071 MiB |

One vCPU improves steady tail token throughput by **6.35%** and completed-step throughput
by **6.32%** at identical VRAM, while reducing wall time 1.83%. All learners are finite
and have distinct final adapters and rollouts. Fifteen pairs are byte-exact through final
parameters and rollouts. Learner 15 differs only at rollout steps 17--18, retains identical
per-step rewards, and its final parameter-L2 relative delta is `6.35e-10`.

N=24 is excluded for this exact vLLM shape rather than reported as a CPU result. Eighteen
learners reached ready state, then the automatic 2.85 GiB clone budget was smaller than
the workload's roughly 3.1 GiB PyTorch state and physical usage reached 80,802 MiB. The
run correctly failed capacity before producing throughput evidence. Supporting results:

- `~/bench/results/fork_grpo-vllm-ordinary-evict-n16-vcpu1_n16_s20_c1_20260729-114616_r1.json`
- `~/bench/results/fork_grpo-vllm-ordinary-evict-n16-vcpu2-control_n16_s20_c2_20260729-115457_r1.json`
- `~/bench_run/fork_grpo-vllm-ordinary-evict-n24-vcpu1_n24_s20_c1_20260729-113801_r1/` (capacity exclusion)

N=4 remains this host's measured throughput optimum (196.998 versus 139.710 tail
tok/s). The automatic path and worker lifecycle are now correct at N=2/N=4/N=8, but
the current `min(90%/(N+1), 1/8)` density formula remains H100/workload-specific and
must not be presented as a universal safe default for smaller GPUs or larger base
models.

The architecture-derived fair-share fallback is also validated rather than assumed.
At N=4, an explicit 14,680 MiB limit (approximately 90% of device memory divided by
the frozen golden plus four clones) completes 4/4 with byte-identical parameter,
rollout, per-step rollout, and dataset hashes. It reaches 195.521 exact / 190.615 tail
tok/s and 2.980 learner-steps/s at 46,742 MiB peak. The density default reaches 198.558
exact / 196.998 tail and 3.079 steps/s at 28,166 MiB: 3.2% better tail throughput and
39.7% less memory. Therefore fair-share is a working explicit larger-model escape
hatch, not the right automatic default for a model that fits the density budget.

Golden-VRAM reclamation is a measured opportunity, but the first prototype is a
**no-go** for production. After all eight N=4 primary/helper workers existed, stopping
the otherwise-frozen golden reduced the daemon's allocation from 4,068 to 1,504 MiB
(2,564 MiB reclaimed) while every imported worker allocation remained resident. A
stronger 100-step test stopped the golden during active training after only the four
primary workers existed and reproduced the same 2,566 MiB reduction. All four learners
eventually completed, so active workers do not depend on the golden context for their
already-imported state.

That stronger test also exposed the unsafe boundary: vLLM's second process-scoped
`token 5` lineage first connects late in the job. With the golden session gone, the
daemon no longer had its weak `GoldenLayout`; worker creation failed with `no golden
layout for token` and entered a 1,365-attempt reconnect storm. Three learners completed
normally, but the fourth suffered repeated roughly 9.5-second stalls before completing.
Therefore neither VM teardown nor golden-CUDA-session teardown may be automatic merely
because every primary worker exists.

Eagerly materializing every process layout is also a **no-go**. The first implementation
started both the memory-bearing `token 1` context and the empty-memory `token 5` context
from the same warm dial. All four learners reached `ready`, but none completed useful
training. The prematurely initialized helper contexts later failed module work with
CUDA 716/719 errors. A process layout being known at fork time therefore does not mean
its CUDA context can safely be constructed before that guest process runs.

The safe prototype retains only a frozen layout proven to have no GPU-memory dependency:
no VMM maps, reservations, `cudaMalloc` allocations, or VMM allocation ranges. It keeps
the helper's module/function reconstruction metadata in host memory without creating a
CUDA context. The warm dial still materializes the one unambiguous memory-bearing
primary worker. A unit test proves the metadata survives golden-session exit under all
tokens for that process; a separate negative test proves a layout with a GPU reservation
is refused.

The source-identical N=4, 100-step H100 validation stopped the frozen golden after the
four primary workers were resident. Daemon VRAM fell from 4,068 to 1,504 MiB (2,564 MiB
reclaimed). All four late `token 5` helpers then reconstructed from retained metadata,
all four learners completed, and the daemon logged zero `no golden layout`, CUDA,
module-reload, or graph-replay failures. The run reached 190.184 exact aggregate tok/s,
146.103 tail tok/s, and 2.292 learner-steps/s. Its no-reclamation control reached 189.972,
145.843, and 2.289 respectively: indistinguishable throughput at this precision while
retaining the golden at 4,072 MiB.

Final rollout hashes are not stable enough to gate this experiment. The reclamation and
control runs have identical dataset, initial-parameter, and initial-model-output hashes,
but their rollout sequences first diverge at steps 8--62 depending on learner. More
importantly, the no-reclamation control also diverges from the prior no-cache run for all
four learners at steps 8--56, while the reclamation run exactly matches that prior run
for two learners. This is pre-existing vLLM/GRPO run-to-run nondeterminism, not evidence
of a reclamation regression; completion, initial-state identity, finite training output,
and CUDA error counts remain the valid gate.

This is not production-ready reclamation yet. The test terminated the exact golden VMM
manually; retained helper metadata currently has daemon lifetime; and after memory-bearing
golden state is released, a crashed primary worker cannot be reconstructed. A production
default needs a sealed-pool lifecycle that reclaims only after every declared primary is
resident, releases cached helper metadata when the pool is destroyed, and either retains
or durably stages enough primary state for worker replacement. The current vLLM run also
reports `shared=0 private=33`, so this recovers the inactive golden's 2.56 GiB but does not
solve vLLM's separate lack of shareable base-weight ranges.

The cache-lifetime and daemon-lifetime pieces are now hardened independently. Each retained
helper layout tracks the local clone VMMs still waiting to start that process; it is released
after the last helper connects or a waiting VMM dies. A source-identical N=4, 20-step repeat
completed 4/4 at 200.553 exact / 197.074 tail tok/s and logged all four late helper spawns,
followed immediately by one metadata-layout release. The daemon idle watchdog also counts
routed clone workers, so releasing the golden connection cannot terminate a long-running
pool after the default five-minute idle interval. Full local validation passes 425 core,
71 CLI, 111 guest-agent, and 41 CUDA tests plus strict clippy. The remaining production
blocker is automatic, replenishment-safe golden-state ownership, not helper layout safety
or an unbounded metadata cache.

That remaining blocker is now resolved by a host-backed golden snapshot prototype. Before
the first primary clone starts, every private VMM range and ordinary `cudaMalloc` allocation
is copied from the frozen golden into a sealed Linux `memfd`; verified read-only weight ranges
remain backed by one exported CUDA allocation. The daemon retains the process-scoped module,
function, graph, stream, event, library-handle, and pointer-layout metadata alongside duplicate
snapshot descriptors. Later workers restore private bytes H2D at the original VMM addresses or
into translated ordinary allocations. The first unchanged vLLM GRPO clone reconstructed 33 VMM
ranges plus 64 ordinary allocations (97 mappings) and completed with the expected initial/final
state hashes.

Durable replenishment passes after real eviction. The frozen golden's two CUDA channels were
closed, its daemon allocation fell from 4,068 to 1,504 MiB, and a replacement fork created
afterward logged `reusing retained golden host snapshot`, reconstructed all 97 mappings, and
completed GRPO. Its initial parameter, final parameter, rollout, and model-output hashes are
byte-identical to the first clone. This distinguishes the design from manual golden teardown:
the golden VM remains frozen and forkable while its inactive CUDA context no longer occupies
VRAM. A later N=4 replacement also reached successful CUDA reconstruction from the retained
snapshot; that `machine fork` was then rejected by the separate clone-rejuvenation path with a
virtiofs `Stale file handle`, so it is not counted as an end-to-end replacement result.

The transparent N=4, 20-step paired gate is also complete. No snapshot opt-in was present;
declaring `--fork-pool-size 4` enabled the policy automatically. The snapshot was built once
(1,254,096,896 private bytes) and reused for the other three primaries, then the daemon closed
both golden CUDA channels when all four primaries were resident. All four learners completed
with exact setup hashes and independent final parameters:

| arm | exact aggregate tok/s | tail tok/s | sampled peak | completion |
|---|---:|---:|---:|---:|
| automatic golden eviction | 204.560 | 203.015 | **29,256 MiB** | 4/4 |
| `SMOLVM_CUDA_GOLDEN_EVICT=off` | 210.666 | 208.898 | 34,702 MiB | 4/4 |

The paired result therefore shows a clear **5,446 MiB (15.7%) peak-VRAM reduction** and a
2.9% throughput difference in favor of the control, which is inside the surrounding N=4 run
spread but must not be described as a speedup. Golden eviction is a memory optimization with
approximately preserved throughput. It still does not make vLLM's clone-private state shared:
the worker verdict remains `shared=0 private=33`. True one-copy weight sharing remains validated
for DPO and other frozen-base layouts whose upload-backed chunks pass fork-time verification.

That true-sharing case now has its own source-identical paired gate. With expandable
segments enabled, every one of four unchanged DPO learners reconstructed `shared=260
private=160` mappings and completed 20 updates:

| arm | exact aggregate tok/s | aggregate step/s | sampled peak | completion |
|---|---:|---:|---:|---:|
| automatic golden eviction | **566** | 1.083 | **12,936 MiB** | 4/4 |
| `SMOLVM_CUDA_GOLDEN_EVICT=off` | 554 | 1.018 | 14,596 MiB | 4/4 |

Eviction saves **1,660 MiB (11.4%)** in addition to the shared-weight density and is
2.2% faster in this pair; the throughput difference is small enough to treat as
preserved rather than claim a new speedup. A post-eviction replacement also reused the
sealed snapshot and reconstructed all 420 mappings with the same 260/160 ownership
split before the separate virtiofs rejuvenation path rejected the VM with `ESTALE`.

One unsafe boundary is now explicit in policy. An ordinary-allocation-only DPO control
failed because a device pointer embedded inside another device allocation cannot be
rewritten by RPC-boundary pointer translation. Automatic eviction therefore requires at
least one address-preserved VMM mapping; ordinary-only layouts keep their live golden.
This is a conservative exclusion, while mixed VMM/ordinary vLLM remains covered by the
successful 97-mapping replacement gate above.

The ordinary-allocation boundary is now resolved for the fast non-Standby vLLM path.
The unchanged workload consistently restored 117 ordinary allocations (3.825 GiB) and
412 inherited graph records, then failed with CUDA 700 on its first post-resume kernel.
Stage-by-stage synchronization proved reconstruction itself was clean. The first failing
launch used grid `[192,3,2]`, block `[128,1,1]`, 24 KiB dynamic shared memory, and 18
arguments; the later TMA/module messages were consequences of the already-poisoned
context, not the initiating fault.

The root cause is structural: top-level pointer translation cannot rewrite device pointers
or TMA descriptors embedded inside device-resident ordinary allocations. Clone workers now
reserve the golden allocation envelopes before context creation and recreate ordinary
regions with private CUDA VMM backing at their exact golden virtual addresses. Snapshot
bytes are restored into those same addresses, identity translations are registered, and
inherited `cudaFree` calls on region suballocations are deferred until worker teardown.
This is transparent to the workload and preserves clone-private physical memory.

Production diagnostics were removed before qualification. The exact candidate passed 438
core tests, strict all-target and no-default-feature CUDA clippy, formatting, and diff
checks, then passed these H100 gates with no CUDA 700, TMA, or module-reload failures:

| arm | exact aggregate tok/s | learner-steps/s | sampled peak | completion |
|---|---:|---:|---:|---:|
| address-preserved N=1, matched batch 1 | **75.740** | **1.183** | 14,655 MiB | 1/1 |
| Standby N=1, matched batch 1 | 62.899 | 0.983 | **10,590 MiB** | 1/1 |
| address-preserved N=4, batch 1 | **247.659** | **3.826** | 39,062 MiB | 4/4 |
| native N=4, batch 1 | 201.446 | 2.786 | **22,319 MiB** | 4/4 |

The fast fork path is therefore 20.4% faster than Standby at matched N=1 and 22.9%
faster than native in N=4 token throughput (37.3% in completed-step throughput). All four
learners had finite output and distinct parameter and rollout hashes. The tradeoff is
explicit: exact-address private restoration is throughput-first and uses 1.75x native
VRAM on this small 0.5B workload. Standby remains the lower-memory vLLM mode; true
physical sharing of immutable vLLM weights is the next independent optimization.

The production commit is `e655efc` on `cuda-clone-restore`. Supporting results are:

- `~/bench/results/fork_grpo-vllm-standby-current-n1_n1_s20_c4_20260729-083034_r1.json`
- `~/bench/results/fork_grpo-vllm-address-va-proto_n1_s20_c4_20260729-094033_r1.json`
- `~/bench/results/fork_grpo-vllm-address-va-b1_n1_s20_c4_20260729-094805_r1.json`
- `~/bench/results/fork_grpo-vllm-address-va-n4_n4_s20_c4_20260729-095454_r1.json`

A source-identical allocation census rules out region reclamation as the next material
lever. It reproduced the exact path at 71.628 tok/s and 14,655 MiB, then classified all
117 ordinary allocations and their inherited frees. The restored 3,825,205,248 bytes
contain 998,244,352 bytes in H2D-marked allocations. Fourteen wholly loaded regions total
532,676,608 bytes and are the only plausible immutable-sharing candidates. The resumed
workload frees 28 allocations totaling only 58,720,256 bytes, and no complete backing
region becomes free; suballocation-aware unmapping therefore cannot recover any VRAM.
Do not implement region reclamation for this workload. Validate ordinary-only golden
eviction first, then read-only-protect the 14 candidate regions before attempting physical
sharing. Evidence is in
`~/bench/results/fork_grpo-vllm-ordinary-census-exact_n1_s20_c4_20260729-103703_r1.json`
and its matching daemon log.

Ordinary-only golden eviction is now a release candidate. Once exact-address restore
made these allocations durable, the daemon could stage their 3.825 GiB in a sealed host
snapshot rather than a temporary GPU export, retain the process metadata, and close the
frozen golden CUDA connections after the declared pool became resident:

| arm | exact aggregate tok/s | aggregate step/s | sampled peak | completion |
|---|---:|---:|---:|---:|
| ordinary host snapshot, N=1 | 72.113 | 1.127 | **6,974 MiB** | 1/1 |
| live-golden GPU staging control, N=1 | 71.628 | 1.119 | 14,655 MiB | 1/1 |
| ordinary host snapshot, N=4 | **228.531** | **3.520** | **20,194 MiB** | 4/4 |
| live-golden GPU staging control, N=4 | 247.659 | 3.826 | 39,062 MiB | 4/4 |
| native N=4 | 201.446 | 2.786 | 22,319 MiB | 4/4 |

At N=1 this removes **7,681 MiB (52.4%)** with unchanged throughput. At N=4 it
removes **18,868 MiB (48.3%)**, uses 9.5% less memory than native, and remains 13.4%
faster in token throughput / 26.3% faster in completed-step throughput than native.
The N=4 candidate is 7.7--8.0% slower than the prior fork run, so eviction is not a
speed optimization; it converts the fast path into a simultaneously faster-than-native
and lower-memory result.

All N=1 and per-learner N=4 initial/final parameter, model-output, dataset, complete
rollout, per-step rollout, reward, and final loss fields are byte-identical to the
controls. Every value is finite, with four distinct final parameter hashes and four
distinct rollout hashes. The log shows eviction before ordinary restoration completed,
three later workers reusing the retained snapshot, 117/117 allocations restored in each,
and no CUDA/TMA/module failure. Host snapshots now fail closed if any golden address
cannot be reserved; they never fall back to pointer translation after evicting the source.

Supporting results:

- `~/bench/results/fork_grpo-vllm-ordinary-evict-n1_n1_s20_c4_20260729-104427_r1.json`
- `~/bench/results/fork_grpo-vllm-ordinary-evict-n4_n4_s20_c4_20260729-105141_r1.json`

#### Adaptive resident-clone admission

A larger logical pool no longer has to make every restored CUDA context resident at
once. The daemon can cap active clone VMMs within a declared fork pool, defer excess
real CUDA channels without blocking its accept loop, and admit a replacement only after
the prior VMM and all of its CUDA workers have exited. Queued warm dials are consumed
instead of occupying server threads. Replacement reconstruction is serialized only per
clone, so distinct admitted clones can restore their sealed host snapshots concurrently.
The machine's automatic vCPU sizing uses the resident cap rather than the total queued
pool size.

This path exposed and fixed two independent lifecycle bugs. First, after golden eviction,
a cached memory-bearing layout was incorrectly reclassified as metadata-only; completing
the second wave then released the module/function/handle metadata required by later
replacements. A frozen layout backed by a host snapshot now remains durable for the pool's
lifetime. Second, waiting for admission in the single listener accept loop deadlocked a
real GRPO run whenever a queued clone connected before an admitted clone. Admission waits
now own a duplicated connection on a detached route, leaving the listener available to
active workers.

The exact Qwen2.5-0.5B vLLM GRPO workload completed 16/16 with finite losses, one identical
initial adapter, and 16 distinct final adapters and rollout hashes. Pool throughput must
include admission and drain time; summing per-learner rates or dividing by the longest
individual `train_s` is invalid for staggered workers. The valid scheduled metrics are:

| resident cap | scheduled tok/s | scheduled steps/s | start-to-last span | sampled peak |
|---:|---:|---:|---:|---:|
| 4 | 111.178 | 1.741 | 183.786 s | **19,422 MiB** |
| 8 | **182.847** | **2.864** | **111.749 s** | 37,370 MiB |
| 12 | 176.183 | 2.759 | 115.976 s | 57,069 MiB |
| 16 control | approximately 209.0 | approximately 3.274 | 97.750 s from first ready | 81,075 MiB |

The eight-resident pool therefore retains about **87.5% of the full-pool token rate while
using 46.1% of its GPU memory**. Four residents maximize density but are not a throughput
default: deterministic slow learners 7, 12, and 15 also take roughly twice as long in the
full-concurrency controls, and a four-slot schedule exposes their drain tail instead of
hiding it behind other jobs. The active cap is a real throughput/memory policy knob, not
a universal constant. Twelve residents consumed another 19,699 MiB but were 3.6% slower
than eight, confirming that admitting more contexts can add GPU contention without hiding
this workload's deterministic stragglers. Eight is the measured Pareto point on this host,
but selecting it automatically still requires runtime pressure and progress signals rather
than a fixed global constant.

Supporting results:

- `~/bench/results/fork_grpo-vllm-adaptive-strict-parallel_n16_a4_s20_c4_20260729-203015_r1.json`
- `~/bench/results/fork_grpo-vllm-adaptive-strict-a8_n16_a8_s20_c4_20260729-204000_r1.json`
- `~/bench/results/fork_grpo-vllm-adaptive-strict-a12_n16_a12_s20_c4_20260729-204851_r1.json`
- `~/bench/results/fork_grpo-vllm-ordinary-evict-n16-vcpu1_n16_s20_c1_20260729-114616_r1.json`

Coarse ordinary-weight sharing is a **no-go**, including every candidate considered
individually. A strict N=1 probe kept every region private but changed all 14 wholly
loaded regions (532,676,608 bytes) to read-only after restore. The clone resumed and
reached training, then failed in the first training phase with `cudaErrorInvalidValue`.
A follow-up used one frozen golden to launch 14 clones, with a different one of those
regions read-only in each clone. All 14 reached the same correct ready-state parameter
and model-output hashes. None completed the first real update: 13 emitted explicit CUDA
tracebacks and the remaining workload exited without a final record. Thus this allocator
has mutable state in every candidate backing region; there is no safe region-granularity
subset to share transparently on this workload.

Therefore “every member allocation received an H2D” does not imply future immutability.
Do not share ordinary regions from the existing loaded bit, do not use read-write sharing
as a workaround, and do not continue the region-level physical-sharing prototype. A later
attempt would need allocation/suballocation boundaries that independently prove the bytes
remain read-only, which is no longer a transparent smolvm-only optimization for this
ordinary allocator layout. The excluded probes are
`~/bench_run/fork_grpo-vllm-ordinary-ro-n1_n1_s20_c4_20260729-110514_r1/` and
`~/bench_run/fork_grpo-vllm-ro-shards_n14_s1_c2_20260729-112104_r1/`. The first N=14
setup attempt at `20260729-111957` is not evidence: CPU auto-sizing reduced the VM to one
vCPU and the golden failed before boot.

Production guards are part of the candidate rather than benchmark convention: fork pools enable
eviction by default; `SMOLVM_CUDA_GOLDEN_EVICT=off` is the rollback switch; `/proc/meminfo`
capacity must cover the required private bytes plus a host reserve; incomplete D2H staging,
sealing, descriptor duplication, or metadata retention leaves the golden resident; restored
host bytes fail the worker immediately on read/H2D errors; ambiguous multi-VM token ownership
fails closed; and the idle watchdog retains the daemon while a live frozen golden owns a cached
snapshot. The cache releases its descriptors, alias-owner records, and retained layout after
the golden VMM and its workers exit. Local validation currently passes 431 core, 113 guest-agent, and 42 CUDA tests,
plus formatting and strict clippy.

H100 evidence:

- `~/bench/results/fork_grpo-vllm-golden-eviction-n1_n1_s20_c4_20260729-021320_r1.json`
  (real eviction, 2,564 MiB direct daemon reclamation, 20-step correctness).
- `~/bench/results/fork_grpo-vllm-golden-eviction-durable-n1_n1_s3_c4_20260729-022941_r1.json`
  plus its replacement clone (post-eviction durable replenishment and exact hashes).
- `~/bench/results/fork_grpo-vllm-golden-eviction-auto-n4_n4_s20_c4_20260729-024807_r1.json`
  (transparent default) and
  `~/bench/results/fork_grpo-vllm-golden-eviction-off-n4_n4_s20_c4_20260729-025707_r1.json`
  (source-identical rollback control).
- `~/bench/results/fork_dpo-golden-eviction-sharing_n4_s20_c4_20260729-031636_r1.json`
  (transparent true-sharing DPO) and
  `~/bench/results/fork_dpo-golden-eviction-off_n4_s20_c4_20260729-032210_r1.json`
  (source-identical rollback control).

Do not count primary-worker pre-warming as a performance win. The first 20-step run's
post-golden wall interval appeared to improve from the prior 51.88 to 43.38 seconds, but an
untouched source-identical repeat returned to 52.05 seconds. It reached 195.439 exact /
194.601 tail tok/s versus the prior 198.558 / 196.998, within the normal run spread. Its
28,140 MiB peak also matches the prior 28,166 MiB, proving that the first run's 34,718 MiB
sample was not a repeatable memory cost. Classification/pre-warm is safe infrastructure for
the handoff, not a measured throughput optimization.

The N=12 result is still useful product evidence: smolvm runs more independent GRPO
learners in 28.4 GiB than native N=10 uses in 74.3 GiB, but the extra learners do not
increase one-H100 throughput. **N=8 remains the measured performance optimum.** The
remaining same-source route to native-MPS parity is the previously identified
single-context/per-clone-translation architecture; the existing legacy mode is not a
safe proxy because it neither protects shared weights nor gives every clone a complete
process-scoped pointer/stream/library-handle namespace.

A GRPO operation census explains why it benefits from concurrency: an N=1 profile
(one golden warmup plus 20 learner steps) emitted 2.68 million proxy operations,
dominated by 721,806 quiet kernel launches and at least 1.62 million library calls. The
production 32-page request ring was observed full at least 8,192 times, motivating
the capacity-only A/B recorded below.

The operation-log profile is never used as a performance result: logging grows the
daemon log to about 90 MiB and perturbs decode time. The temporary ring-full counter
also adds atomics after every observed full condition, so it was removed before the
capacity run rather than being left as dormant production instrumentation. Partial
N=4 setup-dominated screening attempts are retained on the H100 but excluded from the
capacity verdict.

#### Request-ring capacity gate — **no-go; keep 32 pages**

The final capacity arm changed only the libcudart request ring from 32 to 128 pages;
the host binary (`18d33faa…`), driver shim, workload hash, N=8 shape, batch 1, 200
steps, two vCPUs, and managed-MPS policy match the production runs. It completed 8/8,
reported `shared=260 private=162`, had no CUDA operation errors, and used 21,463 MiB:

| request pages | aggregate learner-steps/s | tail aggregate tok/s | tail train | quality gate |
|---:|---:|---:|---:|---|
| 32 (two completed release runs) | **2.967–3.075** | trajectory-dependent | 520–539 s | one pass, one low-reward failure |
| 128 | **2.981** | 77.370 | 536.68 s | fail: two reward-mean deltas exceed 0.02 |

The 128-page step rate is inside the 32-page run-to-run band and 3.0% below the
passing 32-page run. Its token rate is 3.5% higher only because two sampled
trajectories generated more tokens; completed-step rate correctly shows no speedup.
All deterministic setup fields match and the maximum adapter-L2 relative delta is
only 0.000173, but reward-mean deltas of 0.0272 and 0.0362 exceed the release limit.
There is no performance or quality basis to ship the deeper ring, so the override and
all hot-loop diagnostics were removed and the H100 was restored byte-for-byte to the
32-page production bundle.

An earlier 128-page N=8 run used the harness default batch 2 by mistake. It completed
8/8 with healthy rewards, but `compare_grpo.py` rejected it because the production
reference is batch 1; its timings are retained only as a different-shape compatibility
result, never as capacity evidence.
The durable machine-readable evidence and exclusions are in
`bench/results/grpo-h100-20260726.json`.

### Shared-VMM SFT correctness and performance requalification

The earlier concurrent shared-weight SFT measurements are invalid as performance
evidence. Exact pre-training model-output fingerprints showed one divergent learner at
N=4 and only 2/8 exact learners at N=8, while the matching all-private controls were
exact. Serialized shared clones were also exact, isolating the failure to concurrent
reconstruction and mutation of shared CUDA VMM state rather than the workload, model,
or guest CPU allocation.

Two independent runtime defects caused the corruption:

1. Worker launch handed hundreds of exported CUDA allocation descriptors through a
   left-to-right `dup2` shuffle whose destination range overlapped later source
   descriptors. An early copy could therefore replace a later source and make one
   worker import the wrong physical allocation. Every source and the control socket are
   now lifted above the complete destination range before the child shuffle.
2. An allocator VMM chunk classified as loaded at the fork boundary could later be
   reused for explicit host-uploaded training data. Read/write sharing let that update
   mutate siblings. Clone workers now preserve shared ranges as read-only across
   replayed `cuMemSetAccess` calls and transparently replace every overlapping shared
   chunk with address-preserving private physical memory before the first explicit
   H2D, D2D, or memset write.

The combined fix passed exact concurrent and serialized reprobes and full 200-step
Unsloth SFT at both qualified scales:

| shape | corrected fork tok/s | fork peak | native tok/s | native peak | verdict |
|---:|---:|---:|---:|---:|---|
| N=4 | 2,570 | 17,266 MiB | 3,266 | 30,150 MiB | 4/4 exact; 78.7% throughput at 57.3% memory |
| N=8 | 3,036 | 28,280 MiB | 3,393 | 60,236 MiB | 8/8 exact; 89.5% throughput at 47.0% memory |

All learners had finite decreasing loss and distinct final adapter hashes. The N=8
qualification is `fork_sft-long200-cow_n8_a8_s200_c2_20260730-000725_r1`; the N=4
qualification is `fork_sft-long200-accessfix_n4_a4_s200_c2_20260729-231629_r1`.
The all-private N=8 isolation control is
`fork_sft-modelprobe-private-n8_n8_a8_s1_c2_20260729-233924_r1`. These corrected runs,
not the earlier nominal 2,572/2,948 tok/s runs, are the SFT release evidence.

The stripped production source was then rebuilt separately, without the adaptive
admission/routing prototype, as SHA-256 `923afdfe1d72b49991b4607e97962e15bb5bfe2b0595c6cd341a727d698df7f5`.
Its final adversarial N=8 gate passed 8/8 with both the first concurrent fingerprint
and the ordered reprobe equal to the golden hash `47fdd260…`, eight distinct dataset
hashes, `shared=260 private=169`, and no CUDA errors. That result is
`fork_sft-prod-clean-probe_n8_a8_s1_c2_20260730-003858_r1`.

Cross-workload regressions on the integrated candidate also pass. DPO completed 4/4
for 20 updates at 633 aggregate tok/s, 12,937 MiB, and `shared=260 private=160` in
`fork_dpo-vmm-cow-regression_n4_a4_s20_c4_20260730-002413_r1`. GRPO completed 4/4
for 20 stochastic updates with exact initial model/adapter hashes, four distinct
datasets and final adapters, healthy reward ranges, 19.145 exact aggregate tok/s,
12,907 MiB, and `shared=260 private=162` in
`fork_grpo-vmm-cow-regression_n4_a4_s20_c2_20260730-002859_r1`.

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
5. **Treat the remaining SFT gap as workload-specific.** Correctness and density are
   now requalified: N=8 reaches 89.5% of native throughput with 53.0% less VRAM.
   Concurrency amortizes the remoting tax, but the N=1 50-step continuation remains
   537 versus 1,103 tok/s native, so the aggregate N=8 result must not be generalized
   to latency-sensitive single jobs. The current Unsloth backward remains
   non-capturable; adaptive concurrency is the safe transparent lever while graphable
   framework regions remain opt-in.
6. **GRPO beats ordinary native but not an equal-MPS hot-worker pool.** The default
   long run reaches 75.5% of native-MPS tail token throughput and 74.3% of completed
   learner-step throughput while using 63.8% less VRAM. A representative warm
   snapshot reaches 82.6%/83.1%, but its extra preparation pays back only when the
   snapshot serves at least two job waves. Do not accept token rate by itself: short
   diagnostics and one earlier long fork sampled low-reward trajectories and are
   excluded by `compare_grpo.py`.

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

### Phase 1b — framework auto-graph policy: **validated candidate, bounded go**

Expose graph activation only through framework-supported compiled regions. The CLI,
Smolfile, API, named-machine, and packed-run paths are implemented and the fixed-shape
H100 A/B improves by about 39% median with identical output. Keep it opt-in because the
installed Unsloth training paths remain non-graphable and broader compiled-workload
memory/compatibility coverage is not complete.

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

Branch: `cuda-graph-capture-investigation` at `2070318`, forked from `a31810e` and
merged with `origin/main` through `4ba25f7`. The SFT qualification/fork-safety change described
below passed its final local and H100 release gates on immutable binary `18d33faa…`.
The GRPO qualification reuses that same binary and changes only benchmark and
documentation artifacts.

The clean production branch is `fix-vllm-fork-compatibility`, rebased onto
`origin/main` at `cb10202`; it contains only the core runtime fixes listed below.
Benchmark code, probes, results, and this document remain outside that branch.

The current high-fanout production worktree is
`/home/binsquare/smolvm-clone-restore` on `cuda-clone-restore`, based on the
dependent auto-graph stack. It contains only the bounded CUDA-map retry/fail-closed
restore, automatic fork-pool CPU sizing, bounded one-vCPU KVM recovery, the
libkrun submodule bump to `86daef5`, exact-address ordinary-allocation restore,
ordinary-only golden eviction, and their tests. The product commits are `c4eca0c`,
`2258677`, `e655efc`, `f4cdf8e`, and `c6929cf`, pushed in
PR #777. The separate libkrun worktree
is `/home/binsquare/libkrun-enomem-retry` on `enomem-retry-probe`; that branch is
committed and pushed in PR #49. The benchmark harness, raw result JSON, CPU traces, and this
document are intentionally not part of either product commit.

The shared-VMM safety follow-up is `/home/binsquare/smolvm-vmm-cow` on
`cuda-vmm-cow-safety`. PR #779 is stacked on PR #778 and contains only two product
files. Commit `7712d84` keeps frozen CUDA metadata alive across pool replenishment;
commit `43e1881` fixes exported-fd source clobbering, preserves shared mappings as
read-only across resume, and performs address-preserving VMM COW before explicit
post-fork writes. Benchmark code, raw results, and this document are excluded from
that PR.

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
| `900414c` | multi-process lineage routing; atomic late-attach live-RAM metadata; fail-closed tokenless routing; sharing kill-switch fix; generalized real-workload harness; SFT qualification and in-Trainer placement probe |
| `05c0e37` | initial real Qwen2.5-7B GRPO qualification, precision-contract diagnosis, exact stochastic-state checks, density controls, and the SFT parity roadmap |
| `824f378` | long N=8 GRPO qualification and quality-gated throughput evidence |
| `b3e47a4` | resident-base queue versus concurrent-fork comparison |
| `30c2aa0` | fresh-host GRPO reproduction entry point |
| `2070318` | worker-disabled warm-dial routing guard and regression test |
| `8c33033` in `smol-machines/libkrun` | opt-in one-time delay before the first KVM entry, preserving the legacy every-entry workaround |
| `9cb89fe` + `86daef5` in `smol-machines/libkrun` | bounded retry window for transient `KVM_RUN ENOMEM` after the one-time delay |
| `c4eca0c` on `cuda-clone-restore` | bounded imported-map retry and fail-closed CUDA clone reconstruction |
| `2258677` on `cuda-clone-restore` | transparent fork-pool CPU sizing and libkrun first-entry activation |
| `e655efc` on `cuda-clone-restore` | preserve ordinary CUDA allocation addresses across fork restoration |
| `f4cdf8e` on `cuda-clone-restore` | evict address-preserved ordinary CUDA goldens through sealed host snapshots |
| `c6929cf` on `cuda-clone-restore` | stabilize automatically one-vCPU CUDA fork pools on affected KVM hosts |
| `7712d84` on `cuda-vmm-cow-safety` | preserve frozen CUDA metadata across pool replenishment |
| `43e1881` on `cuda-vmm-cow-safety` | keep shared CUDA VMM writes clone-private and make worker fd handoff collision-free |
| `38aba8e` on `fix-vllm-fork-compatibility` | rebased worker-disabled warm-dial routing guard |
| `07ba737` on `fix-vllm-fork-compatibility` | resolve locally loaded CUDA for truthful NVML hardware queries |
| `eb8c950` on `fix-vllm-fork-compatibility` | safe lazy inherited-graph replay across clone channels |
| `2cf2e25` on `fix-vllm-fork-compatibility` | stop deferred CUDA batches after errors and reject unresolved virtual handles |
| `52c8d57` on `fix-vllm-fork-compatibility` | runtime-probed pre-context exact-address reservation |
| `bfa048f` on `fix-vllm-fork-compatibility` | report enforced CUDA allocation limits through total/free memory queries |
| current uncommitted evidence | equal-MPS controls, warm-snapshot root cause, transparent no-go screens, vLLM GRPO qualification, and diagnostic benchmark fixes |

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
| `bench/run_native_mps_matrix.sh` | source-identical ordinary versus private uncapped-MPS native-worker control with required GRPO reference warmup |
| `bench/results/mps-h100-20260725.json` | durable machine-readable MPS performance, correctness, cap, and lifecycle summary |
| `bench/results/mps-managed-h100-20260726.json` | managed-candidate N=4/N=8/native comparison and compiler/explicit-graph verdicts |
| `bench/bench.sh` | native/fork/resident-queue harness; benchmark manifests record MPS mode; fork errors are retained/retried before synchronized release; GRPO rollout/reward/RNG/model fields are retained |
| `bench/workload_dpo.py` | opt-in forced-compiler and explicit fixed-region graph diagnostics |
| `bench/workload_sft.py` | real Unsloth/TRL SFT qualification with exact adapter, model-output, dataset, and RNG fingerprints |
| `bench/workload_sft_resume.py` | diagnostic fork-from-live-Trainer placement upper bound |
| `bench/workload_grpo.py` | real sampled Unsloth/TRL GRPO qualification plus a resident-base queue control with exact adapter/RNG reset and per-step rollout/reward fingerprints |
| `bench/compare_grpo.py` | source-identity, deterministic-setup, reward/adapter-quality, queue-aware effective throughput, completed-step, and density release gate |
| `bench/results/grpo-h100-20260726.json` | durable H100 GRPO precision, correctness, density, isolation-control, exclusion, and next-gate summary |
| `bench/results/grpo-queue-vs-fork-h100-20260726.json` | source-identical resident-base queue versus concurrent-fork quality and performance verdict |
| `bench/results/grpo-native-mps-rootcause-h100-20260726.json` | equal-scheduling native-MPS control, phase attribution, representative-warmup gate, exclusions, and product decision |
| `src/cuda_daemon.rs` | managed private MPS policy, ownership supervisor, bounded path cleanup/collision refusal, fallback, and tests |
| `src/main.rs` | hidden MPS supervisor process entry point |
| `crates/smolvm-cuda/src/client.rs` | synchronous-call profiler now retains enough ranked classes to expose low-count barriers hidden by one-time module loads |
| `src/cuda_host.rs` | opt-in clone-RAM advert trace that exposed the warm-dial/proc-mem ordering bug |
| `crates/smolvm-nvml-shim/src/lib.rs` | explicit staged-driver resolution when CUDA was loaded locally by the framework |

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
  `18d33faae5fa996822e07cf1d407576f`. Real-GRPO evidence uses that same binary.
  The BF16 precision-contract failures are
  `fork_grpo_05b_bf16_fork_n1_s1_c2_20260726-062321_r1` and
  `fork_grpo_05b_nocompile_fork_n1_s1_c2_20260726-062640_r1`; dtype probes are
  `native_grpo_dtype_native_n1_s1_c2_20260726-063046_r1` and
  `fork_grpo_dtype_error_fork_n1_s1_c2_20260726-063756_r1`. The exact N=1
  two-step pair is `native_grpo_7b_ref_n1_s2_c2_20260726-064646_r1` /
  `fork_grpo_7b_fork_n1_s2_c2_20260726-064808_r1`. The final-source,
  compiler-cache-scoped four-step pair is
  `native_grpo_7b_scoped_ref_n1_s4_c2_20260726-072220_r1` /
  `fork_grpo_7b_scoped_fork_n1_s4_c2_20260726-072309_r1`. The N=4 density gate is
  `native_grpo_7b_n4_ref_n4_s4_c2_20260726-071458_r1` /
  `fork_grpo_7b_n4_fork_n4_s4_c2_20260726-071610_r1`; native-MPS and all-private
  controls are `native_grpo_7b_steps_native_mps_n1_s4_c2_20260726-070826_r1` and
  `fork_grpo_7b_steps_private_n1_s4_c2_20260726-070958_r1`. Matching raw JSON is
  under `~/bench/results/`. Long N=8 results are
  `native_grpo-long200-ref_n8_s200_c2_20260726-080310_r1`, excluded quality run
  `fork_grpo-long200-vcpu2-release_n8_s200_c2_20260726-083001_r1`, and passing run
  `fork_grpo-long200-vcpu2-repeat2_n8_s200_c2_20260726-084655_r1`. The diagnostic
  operation/ring profile is
  `fork_grpo-ring32-profile-final_n1_s20_c2_20260726-092231_r1`; its `daemon.log`
  contains the operation census and its `g.err` contains the exponentially sampled
  ring-full events. The source-identical 128-page capacity run is
  `fork_grpo-ring128-n8-b1-release_n8_s200_c2_20260726-101846_r1`; the excluded batch-2
  compatibility run is `fork_grpo-ring128-n8-release_n8_s200_c2_20260726-095756_r1`.
  The production hot-base comparison is
  `queue_grpo-queue-rngfixed-long200_n8_s200_c16_20260726-184329_r1` /
  `fork_grpo-fork-vsqueue-final-long200_n8_s200_c2_20260726-190515_r1`.
  The equal-MPS long control is
  `native_grpo-native-mps-rootcause-long200-20260726_n8_s200_c2_20260726-195442_r1`;
  the representative-warmup pair is
  `native_grpo-native-mps-rootcause-warm20-long200-n8-20260726_n8_s200_c2_20260726-210326_r1` /
  `fork_grpo-fork-mps-rootcause-warm20-long200-n8-20260726_n8_s200_c2_20260726-211201_r1`.
  Shared-context correctness evidence is in
  `fork_grpo-path2-namespace-gated-n2-s5-20260726_n2_s5_c2_20260726-235018_r1`,
  passing N=1 control
  `fork_grpo-path2-namespace-n1-s5-20260726_n1_s5_c2_20260726-235507_r1`,
  stream-zero screen
  `fork_grpo-path2-libstream0-n2-s5-20260726_n2_s5_c2_20260727-000256_r1`,
  module-isolation screen
  `fork_grpo-path2-private-modules-n2-s5-20260726_n2_s5_c2_20260727-001025_r1`,
  and epoch-scheduler screen
  `fork_grpo-path2-epoch-scheduler-n2-s5-20260726_n2_s5_c2_20260727-002412_r1`.
  Unchanged vLLM GRPO compatibility evidence is in
  `fork_grpo-vllm-sharedctx_n1_s1_c4_20260727-082225_r1` (stale-daemon routing
  failure) and `fork_grpo-vllm-routingfix_n1_s1_c4_20260727-083505_r1` (passing
  post-fork workload; the harness-only missing-sharing-verdict failure is excluded).
  The isolated no-MPS control is
  `fork_grpo-vllm-isolated-nomps_n1_s1_c4_20260727-084119_r1`; it retains the
  fixed-VA map failures and therefore excludes MPS as their root cause.
  Layout validation and no-pre-replay evidence is in
  `fork_grpo-vllm-layoutdebug-noprereplay_n1_s1_c4_20260727-085352_r1`.
  Individual pre-context reservations improved reconstruction to 22/33 mappings in
  `fork_grpo-vllm-prereserve-noprereplay_n1_s1_c4_20260727-090849_r1`; the first
  complete isolated clone is
  `fork_grpo-vllm-envelope-noprereplay_n1_s1_c4_20260727-092401_r1`.
  The portable alignment replay and ordinary-allocation candidate census is
  `fork_grpo-vllm-nonvmm-candidates_n1_s1_c4_20260727-094140_r1`.
  Whole-allocation verification and the late-channel graph-replay failure are in
  `fork_grpo-vllm-nonvmm-verified_n1_s1_c4_20260727-172631_r1`.
  The page-level no-go is
  `fork_grpo-vllm-nonvmm-pages_n1_s1_c4_20260727-174015_r1`.
  The passing all-private lazy-replay control is
  `fork_grpo-vllm-graph-optrace_n1_s1_c4_20260727-174938_r1`.
  The eager-pre-replay no-go is
  `fork_grpo-vllm-default-prereplay_n1_s1_c4_20260727-175656_r1`.
  The first fixed-default N=1 pass is
  `fork_grpo-vllm-latechannel-fix_n1_s1_c4_20260727-181046_r1`.
  The fixed-default N=2 concurrent-worker pass is
  `fork_grpo-vllm-latechannel-fix-n2_n2_s1_c4_20260727-181733_r1`.
  The managed-MPS N=1 pass is
  `fork_grpo-vllm-lazy-mps-default-n1_n1_s1_c4_20260727-182656_r1`.
  The managed-MPS N=2 parameter-update failure is
  `fork_grpo-vllm-lazy-mps-default-n2-update_n2_s3_c4_20260727-183332_r1`;
  the global launch-blocking control pass is
  `fork_grpo-vllm-mps-launch-blocking-n2_n2_s1_c4_20260727-184424_r1`.
  The first-replayed-launch-only synchronization no-go is
  `fork_grpo-vllm-mps-firstlaunchsync-n2-update_n2_s3_c4_20260727-185835_r1`.
  The every-replayed-graph-launch synchronization no-go is
  `fork_grpo-vllm-mps-graphsync-n2-update_n2_s3_c4_20260727-191033_r1`.
  The operation-ring root-cause run is
  `fork_grpo-vllm-mps-optrace-n2-update_n2_s3_c4_20260727-191827_r1`.
  The guarded clean-OOM validation is
  `fork_grpo-vllm-mps-vmm-oom-guard-n2_n2_s3_c4_20260727-193746_r1`.
  The standby-clamped 0.10 exclusion is
  `fork_grpo-vllm-mps-fit010-n2-update_n2_s3_c4_20260727-194504_r1`.
  The fitting N=2 managed-MPS parameter-update pass is
  `fork_grpo-vllm-mps-fit010-override-n2-update_n2_s3_c4_20260727-195440_r1`.
  Its unchanged repeat is
  `fork_grpo-vllm-mps-fit010-override-n2-repeat_n2_s3_c4_20260727-201340_r1`.
  The aligned native controls are
  `native_grpo-vllm-native-fit010-n2-update-parity_n2_s3_c4_20260727-200935_r1`
  (ordinary contexts),
  `native_grpo-vllm-native-mps-fit010-n2-update-parity_n2_s3_c4_20260727-201103_r1`
  (excluded partial MPS initialization), and
  `native_grpo-vllm-native-mps-fit012-n2-update-parity_n2_s3_c4_20260727-201220_r1`
  (passing MPS screen). The 20-step steady pair is
  `native_grpo-vllm-native-mps-fit012-n2-steady20_n2_s20_c4_20260727-202009_r1`
  versus
  `fork_grpo-vllm-mps-fit010-n2-steady20_n2_s20_c4_20260727-202118_r1`; the final
  equal-0.12 fork control is
  `fork_grpo-vllm-mps-fit012-n2-steady20-parity_n2_s20_c4_20260727-202926_r1`.
  Exact pushed-build qualification is
  `fork_grpo-vllm-production-52c8d57-fit012-n2-smoke_n2_s3_c4_20260727-210135_r1`;
  the generic 10 GiB/no-workload-override capacity probe is
  `fork_grpo-vllm-quota10g-no-workload-override-n2_n2_s3_c4_20260727-211144_r1`;
  its 20-update repeat is
  `fork_grpo-vllm-quota10g-no-override-n2-steady20_n2_s20_c4_20260727-212621_r1`.
  The fixed high-fanout DPO gate is
  `~/bench_run/fork_dpo-capacity-fix-n24_n24_s1_c4_20260729-045022_r1/` with result
  `~/bench/results/fork_dpo-capacity-fix-n24_n24_s1_c4_20260729-045022_r1.json`.
  Shared-VMM requalification is in
  `fork_sft-long200-accessfix_n4_a4_s200_c2_20260729-231629_r1`,
  `fork_sft-long200-cow_n8_a8_s200_c2_20260730-000725_r1`, and final stripped-source
  gate `fork_sft-prod-clean-probe_n8_a8_s1_c2_20260730-003858_r1`. Matching DPO and
  GRPO regressions are `fork_dpo-vmm-cow-regression_n4_a4_s20_c4_20260730-002413_r1`
  and `fork_grpo-vmm-cow-regression_n4_a4_s20_c2_20260730-002859_r1`.
- **In this repo**: `bench/results/*.json` (11 committed in #742), `bench/README.md`
  (harness usage), `bench/RESULTS.md` (generated summary table), and
  `bench/results/mps-h100-20260725.json` plus
  `bench/results/mps-managed-h100-20260726.json` (durable prototype and managed
  candidate summaries), and `bench/results/grpo-h100-20260726.json` (durable real
  GRPO qualification and exclusions).
- **Session memory** (cross-conversation continuity, not in this repo):
  `fork-vs-native-benchmark.md`, `cuda-fork-weight-sharing-fix.md`,
  `cuda-clone-procmem-transport.md`.
