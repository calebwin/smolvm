"""Bisect WHERE CUDA graph capture fails in a fork clone.

Capture gives 3.7x on native (4.55 -> 1.22 us/op) and is the only lever against
the op-count tax, but in a clone it dies with "CUDA error: unknown error". Walk
the capture sequence one step at a time so the failing operation is named
instead of inferred.
"""
import os, time, json, traceback

COORD = os.environ.get("COORD", "/tmp")
FORK = os.environ.get("FORK", "0") == "1"
LID = os.environ.get("LEARNER_ID", "0")

import torch
dev = torch.device("cuda")
small = torch.ones(256, device=dev)
steps = []


def step(name, fn):
    try:
        fn()
        torch.cuda.synchronize()
        steps.append({"step": name, "ok": True})
        return True
    except Exception as e:
        steps.append({"step": name, "ok": False, "err": str(e).split("\n")[0][:140]})
        return False


# Pre-fork: prove the sequence works in the golden itself.
step("pre_fork_eager_launch", lambda: small.add_(1.0))

if FORK:
    with open(f"{COORD}/golden_ready", "w") as f:
        f.write("1")
    while not os.path.exists(f"{COORD}/go"):
        time.sleep(0.05)
    for k in range(int(os.environ.get("NSLOTS", "64"))):
        try:
            fd = os.open(f"{COORD}/claim_{k}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd); LID = str(k); break
        except FileExistsError:
            continue

# Post-fork, one operation at a time.
step("post_fork_eager_launch", lambda: small.add_(1.0))
side = None
def mk_stream():
    global side
    side = torch.cuda.Stream()
step("stream_create", mk_stream)
if side is not None:
    step("warm_on_side_stream", lambda: [small.add_(1.0) for _ in range(3)] and None)
g = None
def mk_graph():
    global g
    g = torch.cuda.CUDAGraph()
step("cudagraph_alloc", mk_graph)

# Enter capture, launch inside it, exit — separately, so a mid-capture failure
# (e.g. lazy module/function resolution, which is not capture-safe) is visible.
cap_ok = False
try:
    torch.cuda.synchronize()
    g2 = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        small.add_(1.0)                      # resolve BEFORE capture
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    steps.append({"step": "pre_resolve_kernel", "ok": True})
    with torch.cuda.graph(g2):
        small.add_(1.0)
    steps.append({"step": "capture_with_preresolved_kernel", "ok": True})
    g2.replay()
    torch.cuda.synchronize()
    steps.append({"step": "replay", "ok": True})
    cap_ok = True
except Exception as e:
    steps.append({"step": "capture_sequence", "ok": False,
                  "err": str(e).split("\n")[0][:140],
                  "tb": traceback.format_exc().splitlines()[-1][:120]})

res = {"lid": LID, "capture_ok": cap_ok, "steps": steps}
with open(f"{COORD}/gp_{LID}.json", "w") as f:
    json.dump(res, f, indent=1)
print("GRAPHPROBE " + json.dumps(res))
