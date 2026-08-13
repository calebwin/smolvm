# SmolGPU process-pool backend

## Scope

SmolGPU is an opt-in execution substrate for very large populations of similar,
independent CPU-style processes. SmolVM owns the artifact and pool lifecycle while the
separately distributed `smolgpu-host` process owns JIT preparation, GPU runtime state,
COW memory, deadlines, and shutdown.

This is intentionally separate from both existing GPU paths:

- `--gpu` remotes Vulkan through virtio-gpu/Venus into a complete libkrun VM;
- `--cuda` remotes CUDA APIs from a complete libkrun VM;
- `SmolGpuPool` executes an explicit packed RV64 process artifact through the bounded
  SmolGPU Linux ABI without booting a guest kernel.

It is not currently a `machine run` backend and does not accept an OCI image, rootfs,
arbitrary shell command, or unrestricted Linux process.

## Library API

```rust,no_run
use smolvm::smolgpu::{SmolGpuPool, SmolGpuPoolConfig};

let config = SmolGpuPoolConfig::new(
    "/usr/libexec/smolgpu-host",
    "/usr/libexec/smolgpu-gpu",
    "worker.sgpu",
    100_000,
)
.persistent(true)
.jit_cache_dir("/var/cache/smolgpu");

let mut pool = SmolGpuPool::start(config)?;
let batch = pool.execute_batch(&requests, true)?;
for output in batch.outputs.iter() {
    consume(output);
}
drop(batch);
pool.shutdown()?;
# fn consume(_: &[u8]) {}
# Ok::<(), smolvm::Error>(())
```

The output view borrows the shared response mapping. This prevents a subsequent
admission from overwriting results while the caller is consuming them and avoids one
allocation per logical process. `to_owned()` is available when convenience matters more
than allocation count.

One mutable pool serializes admission into one fixed resident population. Callers can
own multiple pools for independent host admission. Persistent mode retains each context's
registers, heap, COW pages, file descriptors, and application state between calls;
refork mode starts each admission from the shared golden snapshot. Optional request-size
grouping may reorder execution but never context identity or output order.

## Failure and isolation semantics

The adapter creates an anonymous shared-memory file with `CLOEXEC`, makes the descriptor
inheritable only in the child immediately before `exec`, and uses stdio only for a
versioned readiness record and eight-byte frame notifications. Every readiness and
execution wait has a deadline. Stderr is drained concurrently.

Input arithmetic, protocol versions, context counts, response lengths, output sizes,
status, and nonzero execution time are validated. After a timeout, malformed frame, or
runtime error, the pool is poisoned and its child is reaped; an orchestrator must replace
it rather than reuse uncertain process state. User-side shape errors detected before
notification do not poison a healthy pool.

Isolation here is one logical register/COW address space per context inside one host GPU
runtime. It is not a guest-kernel or hardware tenancy boundary. Untrusted multi-tenant
workloads still require an appropriate outer process/host isolation policy.

## End-to-end validation

The adapter was built from current SmolVM main and tested through the ordinary Rust
epoll TCP-server artifact. Each population handled a fixed request, distinct
independently sized requests, and transparently grouped requests across three persistent
connections. Every returned connection number, rolling history, score, and output
position was independently verified.

| GPU | contexts | admission | grouping | GPU tasks/s | complete adapter tasks/s | active bytes/context | dirty pages/context |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 3070 8 GB | 10,000 | 1 | no | 834,288 | 607,521 | 29,236 | 6.000 |
| RTX 3070 8 GB | 10,000 | 2 | no | 925,413 | 763,442 | 29,236 | 6.000 |
| RTX 3070 8 GB | 10,000 | 3 | yes | 980,740 | 832,187 | 29,236 | 6.000 |
| H100 80 GB | 100,000 | 1 | no | 2,931,927 | 1,276,581 | 29,236 | 6.000 |
| H100 80 GB | 100,000 | 2 | no | 3,544,253 | 1,488,385 | 29,236 | 6.000 |
| H100 80 GB | 100,000 | 3 | yes | 2,634,048 | 1,548,854 | 29,236 | 6.000 |

These are integration-smoke observations, not a native CPU comparison or latency
distribution. The meaningful result is that the public SmolVM adapter preserves the
validated 10k/100k population, exact process identity, and COW density across two GPU
generations without linking the private runtime crate.

The initial copied-pipe prototype retained only 58–62% of inner broker throughput on
H100. The implemented shared-memory transport removes payload pipes; the remaining gap
comes from the explicit host-process framing and mapping-to-mapping copies. A future
direct-frame handoff can reduce that copy without changing this public lifecycle.

## Remaining product work

Before this can become a normal SmolVM workload type, the following remain explicit:

1. versioned distribution and discovery of `smolgpu-host`, `smolgpu-gpu`, and `.sgpu`
   artifacts;
2. a build/pack flow that compiles eligible ordinary C or Rust programs to RV64 and
   reports unsupported ABI requirements before launch;
3. lease/admission integration above `SmolGpuPool`, including unhealthy-pool replacement;
4. broader Linux ABI coverage and representative workload compatibility;
5. a security model for mutually untrusted GPU contexts.

Until those exist, the library API remains explicit and opt-in rather than silently
changing the semantics of existing SmolVM machines.
