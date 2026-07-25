"""Does CUDA-graph replay collapse the remoting tax? Mechanism test.

The tax is op-count x ~3 us of per-op boundary cost (per-op WORK is already at
native parity: 4.38 vs 4.47 us/launch). Graphs are the only lever that attacks
the count: capture K launches once, then each replay issues ONE op instead of K.

torch.compile could not deliver this in the unsloth+TRL workload (flags accepted
but no compile ran, graphs=0), so measure the mechanism directly with explicit
capture. If the clone's per-op cost collapses here, the payoff is real and the
remaining work is workload integration, not transport.
"""
import os, time, json

K = int(os.environ.get("GRAPH_K", "500"))        # launches captured per graph
REPLAYS = int(os.environ.get("GRAPH_REPLAYS", "400"))
COORD = os.environ.get("COORD", "/tmp")
FORK = os.environ.get("FORK", "0") == "1"
LID = os.environ.get("LEARNER_ID", "0")

import torch
dev = torch.device("cuda")
small = torch.ones(256, device=dev)


def eager_per_op(n):
    small.add_(1.0); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        small.add_(1.0)
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e6


def graph_per_op(k, replays):
    # Warm on a side stream, as CUDA graph capture requires.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            small.add_(1.0)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    t0 = time.time()
    with torch.cuda.graph(g):
        for _ in range(k):
            small.add_(1.0)
    capture_s = time.time() - t0
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(replays):
        g.replay()
    torch.cuda.synchronize()
    dt = time.time() - t0
    return (dt / (k * replays) * 1e6), capture_s


res = {}
try:
    res["eager_us_per_op"] = round(eager_per_op(20000), 3)
except Exception as e:
    res["eager_error"] = str(e)[:120]

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

try:
    per_op, cap = graph_per_op(K, REPLAYS)
    res["graph_us_per_op"] = round(per_op, 4)
    res["graph_capture_s"] = round(cap, 2)
    if "eager_us_per_op" in res:
        res["speedup"] = round(res["eager_us_per_op"] / max(per_op, 1e-9), 1)
except Exception as e:
    res["graph_error"] = str(e)[:160]

res["lid"] = LID; res["k"] = K; res["replays"] = REPLAYS
with open(f"{COORD}/gr_{LID}.json", "w") as f:
    json.dump(res, f)
print("GRAPHRATE " + json.dumps(res))
