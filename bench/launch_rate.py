"""LAUNCH RATE with real-workload op size — the measurement my earlier
microbenchmark got wrong.

The 4096^3 matmul benchmark showed per-op parity because each op carried 0.18 ms
of GPU work, hiding microseconds of guest overhead. Real training issues ~55.7k
ops/step at ~11 us of GPU work each, so the feed rate is what matters. Launch a
trivial kernel in a tight loop with ONE sync at the end: that measures ops/sec
the guest can push, isolated from GPU throughput.
"""
import os, time, json

ITERS = int(os.environ.get("LR_ITERS", "200000"))
COORD = os.environ.get("COORD", "/tmp")
FORK = os.environ.get("FORK", "0") == "1"
LID = os.environ.get("LEARNER_ID", "0")

import torch
dev = torch.device("cuda")
small = torch.ones(256, device=dev)          # tiny: kernel is ~2-3 us of GPU work


def launch_rate(iters):
    small.add_(1.0)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        small.add_(1.0)                       # one kernel launch, no sync
    enq = time.time() - t0                    # pure enqueue time (feed rate)
    torch.cuda.synchronize()
    tot = time.time() - t0                    # enqueue + drain
    return enq, tot


launch_rate(2000)

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

enq, tot = launch_rate(ITERS)
res = {
    "lid": LID, "iters": ITERS,
    "enqueue_us_per_launch": round(enq / ITERS * 1e6, 2),
    "total_us_per_launch": round(tot / ITERS * 1e6, 2),
    "launches_per_sec": int(ITERS / tot),
}
with open(f"{COORD}/lr_{LID}.json", "w") as f:
    json.dump(res, f)
print("LAUNCHRATE " + json.dumps(res))
