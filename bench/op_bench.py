"""Which op PATH is slow in a fork clone? Time three shapes of work separately.

  matmul_fp16 -> cuBLASLt   (LibCall path)
  matmul_fp32 -> cuBLAS     (LibCall path, different entry)
  elementwise -> plain CUDA kernels (LaunchKernel path)
  alloc_free  -> allocator churn (MemAlloc/VMM path)

Native vs clone per row localises the 157x microbenchmark gap to one path
instead of blaming "remoting" as a whole.
"""
import os, time, json

DIM = int(os.environ.get("DIM", "4096"))
ITERS = int(os.environ.get("ITERS", "100"))
COORD = os.environ.get("COORD", "/tmp")
FORK = os.environ.get("FORK", "0") == "1"
LID = os.environ.get("LEARNER_ID", "0")

import torch
dev = torch.device("cuda")

a16 = torch.randn(DIM, DIM, device=dev, dtype=torch.float16)
b16 = torch.randn(DIM, DIM, device=dev, dtype=torch.float16)
a32 = torch.randn(DIM, DIM, device=dev, dtype=torch.float32)
b32 = torch.randn(DIM, DIM, device=dev, dtype=torch.float32)


def bench(fn, iters):
    fn(); torch.cuda.synchronize()          # warm
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0   # ms per op


def run_all(iters):
    return {
        "matmul_fp16_cublasLt_ms": round(bench(lambda: a16 @ b16, iters), 3),
        "matmul_fp32_cublas_ms":   round(bench(lambda: a32 @ b32, iters), 3),
        "elementwise_kernel_ms":   round(bench(lambda: a16 + b16, iters), 3),
        "alloc_free_ms":           round(bench(lambda: torch.empty(DIM * DIM, device=dev, dtype=torch.float16), iters), 3),
    }


run_all(3)  # burn in autotune

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

res = run_all(ITERS)
res["lid"] = LID
res["dim"] = DIM
with open(f"{COORD}/op_{LID}.json", "w") as f:
    json.dump(res, f)
print("OPBENCH " + json.dumps(res))
