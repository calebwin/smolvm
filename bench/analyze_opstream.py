"""Summarize SMOLVM_CUDA_OPSTREAM_PROBE segment stability from a daemon log."""

import collections
import re
import sys

LINE = re.compile(r"\[opstream\]\s+(.*)")
FIELD = re.compile(r"([a-z_]+)=([0-9a-f]+)")

rows = []
for line in open(sys.argv[1], errors="replace"):
    match = LINE.search(line)
    if not match:
        continue
    fields = dict(FIELD.findall(match.group(1)))
    rows.append(fields)

groups = collections.defaultdict(list)
for row in rows:
    groups[(row.get("ops"), row.get("ops_hash"), row.get("shape_hash"))].append(row)

print(f"segments={len(rows)} structural_groups={len(groups)}")
print(
    "count ops ptr_unique nonptr_unique handle_unique full_unique "
    "ptr_words shape_hash verdict"
)
for (ops, _ops_hash, shape_hash), group in sorted(
    groups.items(), key=lambda item: (-len(item[1]), item[0])
):
    if len(group) < 2:
        continue
    unique = lambda key: len({row.get(key) for row in group})
    ptr_unique = unique("ptr_hash")
    nonptr_unique = unique("nonptr_hash")
    handle_unique = unique("handle_hash")
    full_unique = unique("full_hash")
    pointer_words = sorted({row.get("ptr_words", "?") for row in group})
    if ptr_unique == 1 and nonptr_unique == 1:
        verdict = "exact-repeat"
    elif ptr_unique == 1:
        verdict = "stable-pointers"
    else:
        verdict = "moving-pointers"
    print(
        f"{len(group):5d} {int(ops or 0):5d} {ptr_unique:10d} "
        f"{nonptr_unique:13d} {handle_unique:13d} {full_unique:11d} "
        f"{','.join(pointer_words):>9} {shape_hash or '-'} {verdict}"
    )
