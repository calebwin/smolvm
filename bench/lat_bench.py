"""Per-op LATENCY, not throughput. The 1.57x remoting tax survives even though a
tight enqueue-then-sync loop showed per-op parity -- so the cost is round-trip
latency on blocking ops, which real training hits thousands of times per step.

  sync_empty   - cudaStreamSynchronize with nothing outstanding (pure transport RTT)
  sync_after_k - launch one tiny kernel then synchronize (RTT + completion)
  event_query  - cudaEventQuery (cheapest possible blocking round-trip)
  d2h_4b       - 4-byte device->host copy (what .item() does every step)
"""
import os, time, json

ITERS = int(os.environ.get("LAT_ITERS", "2000"))
COORD = os.environ.get("COORD", "/tmp")
FORK = os.environ.get("FORK", "0") == "1"
LID = os.environ.get("LEARNER_ID", "0")

import torch
dev = torch.device("cuda")
a = torch.ones(64, device=dev)
ev = torch.cuda.Event()
host = torch.empty(1, device="cpu", pin_memory=True)
one = torch.ones(1, device=dev)


def per_op_us(fn, iters):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    dt = time.time() - t0
    return dt / iters * 1e6          # microseconds per op


def sync_empty():
    torch.cuda.synchronize()


def sync_after_k():
    a.add_(1.0)
    torch.cuda.synchronize()


def event_query():
    ev.record()
    ev.query()


def d2h_4b():
    host.copy_(one, non_blocking=False)


def run_all(iters):
    return {
        "sync_empty_us":   round(per_op_us(sync_empty, iters), 1),
        "sync_after_k_us": round(per_op_us(sync_after_k, iters), 1),
        "event_query_us":  round(per_op_us(event_query, iters), 1),
        "d2h_4b_us":       round(per_op_us(d2h_4b, iters), 1),
    }


run_all(50)

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

res = run_all(ITERS); res["lid"] = LID
with open(f"{COORD}/lat_{LID}.json", "w") as f:
    json.dump(res, f)
print("LATBENCH " + json.dumps(res))
