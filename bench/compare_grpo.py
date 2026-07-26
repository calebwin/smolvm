#!/usr/bin/env python3
"""Compare source-identical native/fork GRPO qualification results.

Sampled RL should not be required to stay bit-identical after tiny numerical
differences cross a token boundary. This gate keeps deterministic setup exact,
then bounds reward and adapter-norm drift while reporting tail-aware throughput.
"""

import argparse
import json
import math
import statistics
import sys


def load(path):
    with open(path) as stream:
        return json.load(stream)


def by_lid(result):
    return {str(learner["lid"]): learner for learner in result["learners"]}


def performance(result):
    learners = result["learners"]
    if not learners or any(
        learner.get("train_s", 0) <= 0
        or learner.get("rollout_tokens") is None
        or learner["rollout_tokens"] < 0
        for learner in learners
    ):
        return None
    tail_s = max(learner["train_s"] for learner in learners)
    tokens = sum(learner["rollout_tokens"] for learner in learners)
    if tokens <= 0 or result.get("peak_gpu_mib", 0) <= 0:
        return None
    return {
        "sum_learner_tok_s": sum(
            learner["rollout_tokens"] / learner["train_s"] for learner in learners
        ),
        "tail_aggregate_tok_s": tokens / tail_s,
        "aggregate_step_s": sum(
            learner.get("steps") or result["steps"] for learner in learners
        )
        / tail_s,
        "tail_train_s": tail_s,
        "peak_gpu_mib": result["peak_gpu_mib"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("native")
    parser.add_argument("fork")
    parser.add_argument("--max-reward-mean-delta", type=float, default=0.02)
    parser.add_argument("--max-parameter-l2-relative", type=float, default=0.001)
    args = parser.parse_args()

    native = load(args.native)
    fork = load(args.fork)
    failures = []

    for key in ("n", "steps", "workload_md5", "batch", "maxseq"):
        if native.get(key) != fork.get(key):
            failures.append(f"{key} differs: {native.get(key)!r} != {fork.get(key)!r}")
    if native.get("arm") != "native" or fork.get("arm") != "fork":
        failures.append("inputs are not ordered native then fork")
    for label, result in (("native", native), ("fork", fork)):
        if result.get("learners_done") != result.get("learners_expected"):
            failures.append(
                f"{label} incomplete: {result.get('learners_done')}/"
                f"{result.get('learners_expected')}"
            )
        if len(result.get("learners", [])) != result.get("n"):
            failures.append(
                f"{label} learner records: {len(result.get('learners', []))}/"
                f"{result.get('n')}"
            )

    native_learners = by_lid(native)
    fork_learners = by_lid(fork)
    if native_learners.keys() != fork_learners.keys():
        failures.append("learner IDs differ")

    learner_summaries = []
    exact_setup_fields = (
        "model_snapshot",
        "model_output_sha256",
        "initial_parameter_sha256",
        "parameter_count",
        "dataset_sha256",
        "cpu_rng_sha256",
        "cuda_rng_sha256",
        "final_cpu_rng_sha256",
    )
    for lid in sorted(native_learners.keys() & fork_learners.keys(), key=int):
        reference = native_learners[lid]
        candidate = fork_learners[lid]
        for key in exact_setup_fields:
            if key not in reference or key not in candidate:
                failures.append(f"learner {lid}: exact field {key} missing")
                continue
            if reference.get(key) != candidate.get(key):
                failures.append(f"learner {lid}: exact field {key} differs")

        reference_rewards = reference.get("rollout_step_rewards", [])
        candidate_rewards = candidate.get("rollout_step_rewards", [])
        if len(reference_rewards) != native["steps"] or len(candidate_rewards) != fork["steps"]:
            failures.append(f"learner {lid}: incomplete per-step reward sequence")
            continue
        if not all(math.isfinite(value) for value in reference_rewards + candidate_rewards):
            failures.append(f"learner {lid}: non-finite reward")
            continue
        reference_mean = statistics.mean(reference_rewards)
        candidate_mean = statistics.mean(candidate_rewards)
        reward_mean_delta = abs(reference_mean - candidate_mean)
        if reward_mean_delta > args.max_reward_mean_delta:
            failures.append(
                f"learner {lid}: reward mean delta {reward_mean_delta:.6f} exceeds "
                f"{args.max_reward_mean_delta:.6f}"
            )

        reference_l2 = reference.get("parameter_l2")
        candidate_l2 = candidate.get("parameter_l2")
        if (
            reference_l2 is None
            or candidate_l2 is None
            or not math.isfinite(reference_l2)
            or not math.isfinite(candidate_l2)
            or reference_l2 <= 0
        ):
            failures.append(f"learner {lid}: invalid parameter L2 norm")
            continue
        parameter_l2_relative = abs(reference_l2 - candidate_l2) / reference_l2
        if not math.isfinite(parameter_l2_relative):
            failures.append(f"learner {lid}: non-finite parameter norm delta")
        elif parameter_l2_relative > args.max_parameter_l2_relative:
            failures.append(
                f"learner {lid}: parameter L2 relative delta "
                f"{parameter_l2_relative:.8f} exceeds "
                f"{args.max_parameter_l2_relative:.8f}"
            )

        learner_summaries.append(
            {
                "lid": lid,
                "exact_reward_steps": sum(
                    left == right
                    for left, right in zip(reference_rewards, candidate_rewards)
                ),
                "reward_mean_native": reference_mean,
                "reward_mean_fork": candidate_mean,
                "reward_mean_delta": reward_mean_delta,
                "parameter_l2_relative_delta": parameter_l2_relative,
                "final_cuda_rng_match": reference.get("final_cuda_rng_sha256")
                == candidate.get("final_cuda_rng_sha256"),
            }
        )

    native_performance = performance(native)
    fork_performance = performance(fork)
    if native_performance is None or fork_performance is None:
        failures.append("missing or invalid performance fields")
        ratios = None
    else:
        ratios = {
            "tail_aggregate_tok_s": fork_performance["tail_aggregate_tok_s"]
            / native_performance["tail_aggregate_tok_s"],
            "aggregate_step_s": fork_performance["aggregate_step_s"]
            / native_performance["aggregate_step_s"],
            "peak_gpu_memory": fork_performance["peak_gpu_mib"]
            / native_performance["peak_gpu_mib"],
        }
    summary = {
        "passed": not failures,
        "failures": failures,
        "thresholds": {
            "max_reward_mean_delta": args.max_reward_mean_delta,
            "max_parameter_l2_relative": args.max_parameter_l2_relative,
        },
        "native": native_performance,
        "fork": fork_performance,
        "ratios": ratios,
        "learners": learner_summaries,
    }
    json.dump(summary, sys.stdout, indent=2)
    print()
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
