"""Run exactly one CUDA-graph capture in a fresh fork clone.

The existing graph_len.py sweep tries several graph sizes in one CUDA context.
That is useful as a smoke test, but a prior capture can change or poison the
state seen by every later size.  This probe reads one case assigned by the host
harness, performs one capture, validates the result, and writes a stage-by-stage
record.  Repeating it across newly forked clones separates graph length from
clone initialization and preceding-state effects.
"""

import json
import os
import time
import traceback

COORD = os.environ.get("COORD", "/tmp")
NSLOTS = int(os.environ.get("NSLOTS", "64"))

import torch

small = torch.zeros(256, device="cuda")
# Resolve the elementwise kernel before the fork/capture.
small.add_(1.0)
torch.cuda.synchronize()

with open(f"{COORD}/golden_ready", "w") as f:
    f.write("1")
while not os.path.exists(f"{COORD}/go"):
    time.sleep(0.02)

lid = None
for candidate in range(NSLOTS):
    try:
        fd = os.open(
            f"{COORD}/claim_{candidate}",
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
        os.close(fd)
        lid = candidate
        break
    except FileExistsError:
        continue
if lid is None:
    raise RuntimeError("no unclaimed graph trial slot")

with open(f"{COORD}/case_{lid}.json") as f:
    case = json.load(f)
with open(f"{COORD}/started_{lid}", "w") as f:
    f.write(str(os.getpid()))

k = int(case["k"])
state = case["state"]
replays = int(case.get("replays", 1))
steps = []


def record(name, fn):
    try:
        value = fn()
        torch.cuda.synchronize()
        steps.append({"stage": name, "ok": True})
        return value
    except Exception as exc:
        steps.append(
            {
                "stage": name,
                "ok": False,
                "error": str(exc).splitlines()[0][:200],
                "traceback_tail": traceback.format_exc().splitlines()[-1][:200],
            }
        )
        raise


result = {
    "lid": lid,
    "k": k,
    "state": state,
    "replays": replays,
    "ok": False,
    "steps": steps,
}

try:
    record("post_fork_eager", lambda: small.add_(1.0))

    if state in ("prealloc", "doublewarm"):
        record("preallocate_graph_object", torch.cuda.CUDAGraph)

    warm_repetitions = 2 if state == "doublewarm" else 1
    for warm_index in range(warm_repetitions):
        side = record(f"create_warm_stream_{warm_index}", torch.cuda.Stream)

        def warm():
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    small.add_(1.0)
            torch.cuda.current_stream().wait_stream(side)

        record(f"warm_side_stream_{warm_index}", warm)

    record("reset_value", lambda: small.zero_())
    graph = record("allocate_graph", torch.cuda.CUDAGraph)

    def capture():
        with torch.cuda.graph(graph):
            for _ in range(k):
                small.add_(1.0)

    record("capture", capture)
    replay_start = time.perf_counter()

    def replay():
        for _ in range(replays):
            graph.replay()

    record("replay", replay)
    replay_s = time.perf_counter() - replay_start
    got = float(record("readback", lambda: small[0].item()))
    result["graph_us_per_op"] = replay_s / (k * replays) * 1e6
    result["value"] = got
    result["expected"] = float(k * replays)
    result["ok"] = got == float(k * replays)
    if not result["ok"]:
        result["validation_error"] = f"expected {k * replays}, got {got}"
except Exception as exc:
    result["error"] = str(exc).splitlines()[0][:200]

with open(f"{COORD}/trial_{lid}.json", "w") as f:
    json.dump(result, f, indent=2)
print("GRAPH_FRESH_TRIAL " + json.dumps(result))
