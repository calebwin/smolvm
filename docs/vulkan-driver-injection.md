# Vulkan driver injection for `--gpu` workload containers

Status: design approved 2026-08-17 (agent-side injection chosen over a VMM
size-shim); this document specifies the implementation. Validation results
that shaped it are in the appendix (2026-08-20 experiments).

## Problem

A `--gpu` VM exposes `/dev/dri` into every workload container
(`add_gpu_devices_if_available` in `crates/smolvm-agent/src/oci.rs`), but the
container can only *use* Vulkan if its image ships a Mesa build with the
virtio (Venus) ICD:

- On **macOS hosts** (16 KiB pages), every stock guest Mesa fails Venus blob
  negotiation — only the patched build from the `slp/mesa-krunkit` COPR works
  (validated: `Virtio-GPU Venus (Apple M4 Max)` enumerates, llama.cpp Vulkan
  runs at ~4,200 t/s pp128).
- On **any host**, common images simply lack the driver: el9 ships no virtio
  ICD on x86_64, alpine ships none at all, debian/ubuntu depend on version.

So `--gpu` "works" only for users willing to hand-install a distro-specific
Mesa. The CUDA path solved the same shape of problem with agent-side shim
injection (`crates/smolvm-agent/src/cuda.rs`); this mirrors it.

## Non-goals

- Not a VMM-side page-size shim: lying about alignment across the trust
  boundary fights the upstream negotiation work and was rejected.
- Not GPU context survival across fork (see appendix — the failure mode is
  already benign).
- Not WSI/presentation: Venus in this stack has no `VK_KHR_swapchain`;
  injection cannot add one. Compute and offscreen rendering only.

## Design

Mirror `cuda.rs` end to end: bundled blobs in the agent rootfs, a small
injection module, a bind mount plus environment in the OCI spec, graceful
no-op everywhere the pieces are missing.

### Blobs

Ship in the agent rootfs at `/usr/local/lib/smolvm-vulkan/`:

| file | source |
|---|---|
| `libvulkan_virtio.so` | extracted from the pinned COPR `mesa-vulkan-drivers` rpm (currently 24.2.8-104.el9), per arch (aarch64 + x86_64), glibc |
| `libvulkan.so.1` | Vulkan loader from the matching el9 `vulkan-loader` rpm, for images that lack a loader entirely |
| `virtio_icd.json` | written by us, `library_path` pointing at the **container** path below |

Build integration: `scripts/rebuild-agent.sh` (and the CI agent job) fetch the
rpms by pinned URL + sha256 and extract just these files — same provenance
discipline as the libkrun blobs, and a natural candidate for the
content-addressed dependency direction (#193). A musl build (for alpine
workloads) is a follow-up: until it exists, injection skips musl images.

### Injection module

New `crates/smolvm-agent/src/vulkan.rs`:

```
pub fn inject_into_container(spec: &mut OciSpec, rootfs: &Path)
```

Called next to `cuda::inject_into_container` on the fresh-container path, and
an `exec_env`-style helper on the `crun exec` path (same split as CUDA).

Gates, in order — every one degrades to a silent no-op:

1. `/dev/dri` exists in the VM (same signal `add_gpu_devices_if_available`
   uses — GPU VMs only).
2. The blob dir is present in the agent rootfs (builds without bundling keep
   today's manual-setup behavior).
3. The image is not musl (`<rootfs>/lib/ld-musl-*` absent) — until the musl
   driver build ships.
4. Opt-out env `SMOLVM_NO_VULKAN_INJECT=1` is not set on the request.

Actions when all gates pass:

1. Read-only bind mount `/usr/local/lib/smolvm-vulkan` → `/opt/smolvm-vulkan`.
2. Set `VK_DRIVER_FILES=/opt/smolvm-vulkan/virtio_icd.json` **only if the
   image/request did not already set `VK_DRIVER_FILES` or
   `VK_ICD_FILENAMES`** — a user-provided driver choice always wins.
   Pinning a single ICD also avoids the multi-ICD probe traps (fedora's full
   probe can segfault in unrelated ICDs; ANGLE picks the wrong device when
   llvmpipe is visible — both observed).
3. Append `/opt/smolvm-vulkan` to `LD_LIBRARY_PATH` (same helper as CUDA), so
   `libvulkan.so.1` resolves in loader-less images without shadowing an
   image-provided loader earlier on the path.

Why env-based pinning rather than overlay-mounting into
`/usr/share/vulkan/icd.d/`: the ICD directory varies by distro, the image may
legitimately carry other ICDs, and `VK_DRIVER_FILES` is the loader's own
override mechanism — no filesystem surgery, trivially reversible, and the
opt-out is just "don't set the env".

### Interactions

- **Fork**: orthogonal and safe — validated that clones open fresh Venus
  contexts post-fork, including after a live-context freeze (appendix).
- **Pack**: `pack run` has two launch paths; the dynamic launcher must apply
  the same injection the static path gets (this is the known parity footgun —
  add a regression test on both paths).
- **Version skew**: old agents simply never call the module; new agents on
  blob-less rootfs no-op. No protocol change at all.
- **Linux hosts**: the patched driver is upstream Mesa plus the alignment
  patch; running it on 4 KiB-page hosts is harmless (t480 validated stock
  behavior; injected driver is the same code path). One driver everywhere
  beats host-conditional injection.

### Sunset

Delete the module and blobs when upstream Mesa ships virtio blob-alignment
negotiation and the mainstream distros carry it — the gates make the removal
behavior-invisible.

### Test plan

- debian/ubuntu (glibc, no ICD): `vulkaninfo --summary` shows Venus with zero
  setup, on macOS and Linux hosts.
- almalinux 9 (glibc, no virtio ICD on x86_64): same.
- alpine (musl): injection skips; behavior unchanged.
- Image with its own working Mesa + `VK_DRIVER_FILES` set: untouched.
- `SMOLVM_NO_VULKAN_INJECT=1`: untouched.
- Fork: golden runs a compute job, clones re-run it (existing pattern).
- Pack: static and dynamic launch paths both inject.

## Appendix: validation experiments (2026-08-20, M4 Max, engine 1.8.3 local)

Golden `vkgold`: almalinux 9 + COPR mesa 24.2.8-104 + llama.cpp Vulkan +
SmolLM2-360M Q8. All zero engine changes.

**Fork + fresh contexts (positive).** Golden benches 2,493 t/s pp64 on Venus;
three clones forked in 107–181 ms each; a clone benches identically
(2,486 t/s). All three clones ran Venus compute **concurrently** at
~4,000–4,500 t/s pp128 each — independent live GPU contexts sharing the host
GPU.

**Fork during live GPU dispatch (negative control, key finding).** With
`llama-bench` mid-dispatch at freeze: the fork succeeds; in the clone the
inherited process self-terminates cleanly — guest Mesa's ring watchdog logs
`MESA-VIRTIO: aborting on expired ring alive status` — and a **fresh Vulkan
context in that same clone works perfectly** (242 t/s tg8). The old "live
contexts wedge the clone" constraint is softer than believed: stale contexts
die like a GPU-process crash, the device remains usable. This is exactly the
recovery model applications with GPU-crash handling (Chromium's GPU process,
most game engines) already implement.

**Browser GPU status (why the fan-out demo stays SwiftShader for now).**
Chromium 151 on the same guest cannot use Venus at all, independent of fork:

- ANGLE-Vulkan hard-requires `VK_KHR_swapchain`; Venus (virglrenderer, this
  stack) exposes **no** WSI extensions (`vulkaninfo` with the ICD pinned:
  zero matches). Instance-level `VK_KHR_surface`/`VK_EXT_headless_surface`
  seen earlier belong to lavapipe.
- Chromium 151 allows only `gl=egl-angle` — native EGL (`--use-gl=egl`,
  virgl-backed) is compiled out of the allowed implementations.
- `--use-angle=swiftshader` + `--use-vulkan=native --enable-features=Vulkan`
  boots the GPU process but Skia-Vulkan does not engage when GL is software.

Unblock paths, cheapest first: (a) Venus WSI — implement/enable
`VK_EXT_headless_surface` + swapchain in virglrenderer's venus, or (b) carry
an ANGLE patch relaxing the swapchain requirement for surfaceless displays,
or (c) a Chromium build with native-EGL allowed (virgl GL). Until one lands,
browser fan-out ships SwiftShader; GPU fan-out is real today for compute
(llama.cpp et al.) and for any app that opens its contexts post-fork or
tolerates a GPU reset.

**Also hit**: second-generation forking (clone → golden) fails in clone
rejuvenation — the persistent overlay is not re-keyed to the new golden name
(`missing /storage/overlays/persistent-<clone>/merged`); tracked as the
existing overlay re-key work.
