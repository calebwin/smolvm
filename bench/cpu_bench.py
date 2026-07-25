"""Single-thread CPU speed: guest vs host. NO CUDA involved.

Every CUDA-side measurement came out at parity (per-op throughput, per-op
latency), yet fork's per-op WALL time is ~2.4x native and the daemon idles 73%
waiting on the guest. Python/torch dispatch is single-threaded, which is also
why extra vCPUs changed nothing. So measure the guest's scalar speed directly:
if the VM is ~2x slower per thread, that alone explains the feed deficit and the
fix is VM/CPU configuration, not the CUDA transport.
"""
import os, time, json

COORD = os.environ.get("COORD", "/tmp")
LID = os.environ.get("LEARNER_ID", "0")
ITERS = int(os.environ.get("CPU_ITERS", "3000000"))


def int_loop(n):
    t0 = time.perf_counter()
    x = 0
    for i in range(n):
        x = (x * 31 + i) & 0xFFFFFFFF
    return time.perf_counter() - t0, x


def attr_call_loop(n):
    # Closest proxy for torch's dispatch cost: attribute lookup + call.
    class C:
        def m(self, v):
            return v + 1
    c = C()
    t0 = time.perf_counter()
    v = 0
    for _ in range(n):
        v = c.m(v)
    return time.perf_counter() - t0, v


def alloc_loop(n):
    t0 = time.perf_counter()
    for _ in range(n):
        b = bytearray(256)
    return time.perf_counter() - t0, len(b)


res = {}
dt, _ = int_loop(ITERS);              res["int_loop_s"] = round(dt, 3)
dt, _ = attr_call_loop(ITERS // 3);   res["method_call_s"] = round(dt, 3)
dt, _ = alloc_loop(ITERS // 3);       res["alloc_s"] = round(dt, 3)
res["lid"] = LID
res["nproc"] = os.cpu_count()
with open(f"{COORD}/cpu_{LID}.json", "w") as f:
    json.dump(res, f)
print("CPUBENCH " + json.dumps(res))
