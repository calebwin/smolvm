"""GPU-bound microbenchmark: is a fork clone's COMPUTE slower, or only its op path?

Large matmuls are ~all GPU and almost no CUDA calls (one launch per iteration,
one sync at the end), the opposite of the DPO step's thousands of small ops. If
a clone matches native here, remoting does not slow compute and the 2.3x gap
lives in the op/sync path. If it is slower here too, something throttles the
clone's kernels themselves.

Env: N=matrix dim, ITERS, COORD, FORK=1 (golden/clone protocol), NSLOTS.
"""
import os, time, json, glob

DIM = int(os.environ.get("DIM", "4096"))
ITERS = int(os.environ.get("ITERS", "200"))
COORD = os.environ.get("COORD", "/tmp")
FORK = os.environ.get("FORK", "0") == "1"
LID = os.environ.get("LEARNER_ID", "0")

import torch

dev = torch.device("cuda")
a = torch.randn(DIM, DIM, device=dev, dtype=torch.float16)
b = torch.randn(DIM, DIM, device=dev, dtype=torch.float16)


def timed(iters):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.time() - t0
    # 2*N^3 flops per matmul
    tflops = (2.0 * DIM ** 3 * iters) / dt / 1e12
    return dt, tflops


timed(10)  # warm autotune/cublas

if FORK:
    # Golden: warm the compute path, then park at the barrier so its writes
    # land pre-fork (same discipline the training workload uses).
    with open(f"{COORD}/golden_ready", "w") as f:
        f.write("1")
    while not os.path.exists(f"{COORD}/go"):
        time.sleep(0.05)
    claimed = None
    for k in range(int(os.environ.get("NSLOTS", "64"))):
        try:
            fd = os.open(f"{COORD}/claim_{k}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            claimed = k
            break
        except FileExistsError:
            continue
    LID = str(claimed)

dt, tflops = timed(ITERS)
with open(f"{COORD}/mm_{LID}.json", "w") as f:
    json.dump({"lid": LID, "dim": DIM, "iters": ITERS,
               "seconds": round(dt, 3), "tflops": round(tflops, 1)}, f)
print(f"MM lid={LID} dim={DIM} iters={ITERS} {dt:.2f}s {tflops:.1f} TFLOP/s")
