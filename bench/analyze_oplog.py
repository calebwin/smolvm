"""Summarize SMOLVM_CUDA_HOST_OPLOG traffic by process and operation class."""

import collections
import re
import sys

OPS = {
    0x01: "Init",
    0x02: "DeviceGetCount",
    0x06: "DeviceGetAttribute",
    0x12: "PrimaryCtxRetain",
    0x20: "ModuleLoadData",
    0x21: "ModuleGetFunction",
    0x23: "FuncGetParamInfo",
    0x25: "FuncGetAttribute",
    0x30: "MemAlloc",
    0x31: "MemFree",
    0x32: "MemcpyHtoD",
    0x33: "MemcpyDtoH",
    0x34: "MemcpyDtoD",
    0x35: "MemsetD8",
    0x36: "MemGetInfo",
    0x40: "LaunchKernel",
    0x50: "CtxSynchronize",
    0x60: "StreamCreate",
    0x61: "StreamDestroy",
    0x62: "StreamSynchronize",
    0x63: "StreamQuery",
    0x70: "EventCreate",
    0x71: "EventDestroy",
    0x72: "EventRecord",
    0x73: "EventSynchronize",
    0x74: "EventElapsedTime",
    0x75: "StreamWaitEvent",
    0x76: "EventQuery",
    0xA0: "LibCall",
    0xB2: "MemcpyGpaHtoD",
    0xB3: "MemcpyGpaDtoH",
    0xC0: "StreamBeginCapture",
    0xC1: "StreamEndCapture",
    0xC2: "GraphInstantiate",
    0xC3: "GraphLaunch",
    0xC4: "GraphExecDestroy",
    0xC5: "GraphDestroy",
    0xC6: "StreamCaptureInfo",
    0xC7: "MemsetD8Async",
    0xC8: "MemcpyDtoDAsync",
    0xCA: "ThreadExchangeCaptureMode",
    0xD0: "RingSetup",
    0xD1: "RingSetupFile",
    0xE0: "MemAddressReserve",
    0xE1: "MemCreate",
    0xE2: "MemMap",
    0xE3: "MemSetAccess",
    0xE4: "MemUnmap",
    0xE5: "MemRelease",
    0xE6: "MemAddressFree",
    0xE7: "MemGetAllocationGranularity",
    0xE8: "MemCreateVh",
    0xE9: "SetDeviceBase",
}

LINE = re.compile(
    r"\[(op(?:~b|~)?)\]\s+p(\d+)\s+0x([0-9a-fA-F]{2})"
    r"(?:\s+len=\d+)?(?:\s+lib=(\d+)\s+func=(\d+))?"
)

path = sys.argv[1]
selected_pid = int(sys.argv[2]) if len(sys.argv) > 2 else None
counts = collections.Counter()
pids = collections.Counter()
quiet_counts = collections.Counter()

with open(path, errors="replace") as source:
    for line in source:
        match = LINE.search(line)
        if not match:
            continue
        wrapper, pid_text, opcode_text, lib, func = match.groups()
        pid = int(pid_text)
        pids[pid] += 1
        if selected_pid is not None and pid != selected_pid:
            continue
        opcode = int(opcode_text, 16)
        name = OPS.get(opcode, f"Op0x{opcode:02x}")
        if lib is not None:
            name += f"({lib}:{func})"
        quiet = wrapper != "op"
        counts[(name, quiet)] += 1
        quiet_counts[quiet] += 1

print("pids " + " ".join(f"{pid}:{count}" for pid, count in pids.most_common()))
print(
    f"selected_pid={selected_pid or 'all'} total={sum(counts.values())} "
    f"quiet={quiet_counts[True]} blocking={quiet_counts[False]}"
)
print("count mode operation")
for (name, quiet), count in sorted(counts.items(), key=lambda item: -item[1]):
    print(f"{count:8d} {'quiet' if quiet else 'blocking':8s} {name}")
