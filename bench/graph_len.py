"""Find the capture-LENGTH limit for CUDA graphs in a fork clone.

A 1-launch capture succeeds in a clone; a 500-launch capture dies with "CUDA
error: unknown error". Since graphs are the only lever against the op-count tax
(3.7x on native), the threshold tells us what to fix.
"""
import os, time, json

COORD = os.environ.get("COORD", "/tmp")
FORK = os.environ.get("FORK", "0") == "1"
LID = os.environ.get("LEARNER_ID", "0")

import torch
dev = torch.device("cuda")
small = torch.ones(256, device=dev)
small.add_(1.0)                       # resolve the kernel before any capture
torch.cuda.synchronize()

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

out = []
for k in (1, 4, 16, 64, 128, 256, 384, 500):
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            small.add_(1.0)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for _ in range(k):
                small.add_(1.0)
        g.replay()
        torch.cuda.synchronize()
        out.append({"k": k, "ok": True})
    except Exception as e:
        out.append({"k": k, "ok": False, "err": str(e).split("\n")[0][:120]})
        break                            # context is likely poisoned after a failure

res = {"lid": LID, "results": out}
with open(f"{COORD}/gl_{LID}.json", "w") as f:
    json.dump(res, f, indent=1)
print("GRAPHLEN " + json.dumps(res))
