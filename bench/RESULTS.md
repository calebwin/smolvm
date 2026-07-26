# Measured results

All rows produced by `bench.sh` on one exclusive GPU. Regenerate with
`./summarize.py`; raw per-run JSON (including a full environment manifest)
is in `results/`.

Host: NVIDIA H100 80GB HBM3 · driver 570.148.08 · torch 2.7.0+cu126 · 26 cores / 237 GB

This is the original pre-MPS density matrix, not the current performance
qualification. See `results/mps-managed-h100-20260726.json` and
`results/grpo-h100-20260726.json` for the release-qualified DPO and GRPO results.

## Resident-base GRPO queue control

This production-style control keeps one Qwen2.5-7B base and its compiled kernels
resident while training eight reset LoRA jobs sequentially. The source-identical
fork arm runs the same eight jobs concurrently:

| arm | done | wall s | jobs/hour | effective tok/s | steps/s | peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| resident-base queue | 8/8 | 1,279.61 | 22.507 | 31.433 | 1.286 | **7,443** |
| smolvm fork + managed MPS | 8/8 | **751.16** | **38.341** | **74.870** | **3.015** | 21,479 |

Smolvm completes the batch 41.3% sooner and provides 1.70x jobs/hour while using
2.89x the memory. Exact setup checks and reward/adapter quality gates pass. This
comparison does not cover a fused concurrent multi-adapter trainer.

| arm | N | steps | batch×seq | done | agg tok/s | peak GPU MiB | golden load s | shared ranges |
|---|---|---|---|---|---|---|---|---|
| fork | 4 | 20 | 2x256 | 4/4 | 546 | 14550 | 165.59 | - |
| fork | 8 | 10 | 2x256 | 8/8 | 432 | 21500 | 163.73 | - |
| fork | 8 | 10 | 2x256 | 8/8 | 432 | 21480 | 169.23 | - |
| fork | 8 | 10 | 2x256 | 8/8 | 436 | 21480 | 166.66 | - |
| fork | 8 | 20 | 2x256 | 8/8 | 736 | 63082 | 156.34 | - |
| fork | 16 | 10 | 2x256 | 16/16 (2 nan) | 465 | 35400 | 166.47 | - |
| fork | 16 | 10 | 2x256 | 16/16 | 452 | 35340 | 167.02 | - |
| fork | 16 | 10 | 2x256 | 16/16 | 487 | 35340 | 164.39 | - |
| native | 8 | 20 | 2x256 | 8/8 | 1462 | 59080 | - | - |
| native | 12 | 20 | 2x256 | 7/12 | 1145 | 81057 | - | - |
| native | 16 | 10 | 2x256 | 2/16 | 467 | 81087 | - | - |

## Headline

- Weight sharing was inert before the fix: the daemon logged `shared=0 private=420` even when explicitly enabled, because zero-copy H2D uploads recorded chunk coverage with crc 0 and share verification rejects any zero-crc segment.
- After the fix: `shared=260 private=160`, per-clone VRAM 6928 -> 1648 MiB, and N=4 final losses bit-identical to all-private mode.
- Native's ceiling on an 80 GiB card is ~11 learners (N=12 -> 7/12, N=16 -> 2/16). Fork with sharing ran 16/16 in 35.4 GiB.
- Throughput is unchanged by the fix (+6% at N=4) and fork remains ~2x slower per learner than native; this is a density change, not a speed change.
