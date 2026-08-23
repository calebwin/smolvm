# A Linux desktop in a machine

Runs a real DRM Wayland compositor (Hyprland) inside a smolvm machine and
serves it over VNC from the **host**, so the guest needs no capture tool and no
compositor-specific screencopy protocol.

```sh
SMOLVM_DISPLAY=1280x800 SMOLVM_VNC=127.0.0.1:5900 \
  smolvm machine run --net --gpu --cpus 4 --mem 6144 \
    -v "$PWD:/in" --image archlinux:latest -- bash /in/run.sh
```

Then point any VNC client at `127.0.0.1:5900`.

## The two environment variables

`SMOLVM_DISPLAY=WIDTHxHEIGHT` adds a virtio-gpu scanout. Without it `--gpu`
gives the guest GPU *rendering* only: `/dev/dri/card0` is a render node with no
connector, and every DRM compositor refuses to start with "not a KMS device".

`SMOLVM_VNC=[host:]port` serves the resulting framebuffer over RFB. A bare port
binds loopback only. Output is one-way today — smolvm does not yet attach a
virtio-input device, so keyboard and pointer events from the client are
discarded rather than pretending to work.

Both are opt-in. A connector changes guest topology, and existing GPU workloads
(CUDA remoting, headless Vulkan) neither need nor want one.

## The one thing the guest must do

Run `seatd` with **`SEATD_VTBOUND=0`**:

```sh
SEATD_VTBOUND=0 seatd -g wheel &
```

seatd defaults to a *VT-bound* seat, which needs `/dev/tty0`. A workload
container has no VT, so a VT-bound seat cannot open a session — and because
seatd itself starts fine either way, the failure surfaces much later and in the
client, as libseat's misleading "Failed to open a session". This costs an hour
if you have not seen it before.

## Verifying it actually renders

A listening socket proves a server and an RFB banner proves a handshake;
neither proves a frame was ever presented. `rfb_probe.py` speaks the protocol,
pulls two full framebuffer updates a few seconds apart, and reports whether the
pixels are non-uniform (something was drawn) and whether they changed (it is
live):

```sh
python3 rfb_probe.py 127.0.0.1 5900
```

## Not Omarchy yet

Omarchy ships as an ISO that pacstraps `install/omarchy-base.packages`. 126 of
those 148 packages resolve from the official Arch repos and install cleanly
here; the remaining 22 are AUR or Omarchy-published (`omarchy-nvim`, `omacut`,
`aether`, …) and are ordinary Arch packaging work, not a smolvm limitation.
`run.sh` installs the Arch-resolvable base, which is the same compositor and
the same session — just without Omarchy's own packages and theming.
