"""Real Unsloth + TRL GRPO qualification for native/smolvm A/B runs.

The task samples short arithmetic answers and uses a dense correctness reward so
even a small base model produces within-group reward variation. Native and fork
arms run this exact file and record rollout plus trainable-parameter digests.
"""

import faulthandler

faulthandler.enable()

import glob
import gc
import hashlib
import json
import os
import random
import re
import struct
import time

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Unsloth 2026.7.3's generated GRPO loss chooses its *inner* autocast from this
# environment variable and otherwise defaults to FP16, even when GRPOConfig
# below requests BF16. Keep the framework's two precision controls aligned.
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "bf16")

STEPS = int(os.environ.get("STEPS", "10"))
MAXSEQ = int(os.environ.get("MAXSEQ", "256"))
PROMPTS = int(os.environ.get("BATCH", "1"))
NGEN = int(os.environ.get("NGEN", "4"))
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION", "16"))
COORD = os.environ.get("COORD", "/coord")
MODEL = os.environ.get("MODEL", "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
ARM = os.environ.get("ARM", "?")
FORK = os.environ.get("FORK", "0") == "1"
LID = os.environ.get("LEARNER_ID", "0")
OUTBASE = os.environ.get("OUTBASE", "/root")
QUEUE_JOBS = int(os.environ.get("QUEUE_JOBS", "0"))

# Never let one native arm inherit Inductor/Unsloth artifacts from an earlier
# benchmark while a new VM starts from its clean overlay. The golden builds its
# scoped caches before the snapshot; every clone inherits the same files in its
# own COW root, while each native learner builds an equivalent private cache.
compile_scope = os.path.join(OUTBASE, f"grpo-compile-{LID}")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", compile_scope + "-inductor")
os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", compile_scope + "-unsloth")


def emit(lid, **fields):
    fields.update(lid=str(lid), arm=ARM, method="grpo", t=round(time.time(), 3))
    with open(f"{COORD}/learner_{lid}.jsonl", "a") as stream:
        stream.write(json.dumps(fields) + "\n")


snapshots = sorted(
    glob.glob(
        os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/hf")),
            "hub",
            "models--" + MODEL.replace("/", "--"),
            "snapshots",
            "*",
        )
    )
)
if snapshots:
    MODEL = snapshots[-1]
    os.environ["HF_HUB_OFFLINE"] = "1"

# Unsloth must patch Transformers/TRL before either is imported. An earlier
# scratch prototype imported trl.import_utils first and failed in GRPO's LoRA
# forward with mixed FP16/FP32 operands; that run is not qualification evidence.
from unsloth import FastLanguageModel
import torch

load_started = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL,
    max_seq_length=MAXSEQ,
    load_in_4bit=True,
    # GRPOConfig below requests BF16. Keep the quantized model's compute dtype
    # explicit as well; dtype=None selected an FP16 forward in the VM while the
    # GRPO LoRA path retained FP32 adapters and failed before the fork point.
    dtype=torch.bfloat16,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=0,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
torch.cuda.synchronize()
load_ms = (time.time() - load_started) * 1000
trainable_dtypes = sorted(
    {str(parameter.dtype) for parameter in model.parameters() if parameter.requires_grad}
)
model_runtime = {
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "device_capability": list(torch.cuda.get_device_capability()),
    "model_dtype": str(model.dtype),
    "model_snapshot": os.path.basename(MODEL.rstrip("/")),
    "compile_cache_scoped": True,
    "trainable_dtypes": trainable_dtypes,
}


def model_output_fingerprint():
    """Hash a fixed frozen-policy forward before any GRPO update."""
    was_training = model.training
    model.eval()
    encoded = tokenizer(
        "Question: what is 137+284? Reply with only the integer.\nAnswer:",
        return_tensors="pt",
        add_special_tokens=True,
    )
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(**encoded).logits[:, -4:, :].detach().contiguous().cpu()
    if was_training:
        model.train()
    values = output.double()
    result = {
        "model_output_sha256": hashlib.sha256(
            output.view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "model_output_sum": values.sum().item(),
        "model_output_l2": (values * values).sum().item() ** 0.5,
    }
    del output, values, encoded
    return result


model_runtime.update(model_output_fingerprint())

if os.environ.get("GRPO_DTYPE_PROBE", "0") == "1":
    # Diagnostic only: record the first fused-LoRA call at the exact failing
    # boundary without changing its operands or result.
    import unsloth.kernels.fast_lora as _fast_lora
    import unsloth.kernels.utils as _kernel_utils

    _original_matmul_lora = _fast_lora.matmul_lora
    _dtype_probe_seen = False

    def _probed_matmul_lora(x, weight, quant, adapter_a, adapter_b, scale, out=None):
        global _dtype_probe_seen
        if not _dtype_probe_seen:
            _dtype_probe_seen = True
            dtype = x.dtype
            reshape = x.dim() == 3
            if reshape:
                batch, sequence, _ = x.shape
                flat_x = x.view(-1, x.shape[-1])
            else:
                flat_x = x
            dequantized = _kernel_utils.fast_dequantize(
                weight, quant, use_global_buffer=True
            )
            base = torch.matmul(flat_x, dequantized.t(), out=out)
            cast_a = adapter_a.t().to(dtype)
            cast_b = adapter_b.t().to(dtype)
            projected = torch.matmul(flat_x, cast_a)
            record = {
                "event": "matmul_lora",
                "arm": ARM,
                "x": str(x.dtype),
                "weight": str(weight.dtype),
                "dequantized": str(dequantized.dtype),
                "adapter_a": str(adapter_a.dtype),
                "adapter_b": str(adapter_b.dtype),
                "cast_a": str(cast_a.dtype),
                "cast_b": str(cast_b.dtype),
                "base": str(base.dtype),
                "projected": str(projected.dtype),
                "out": None if out is None else str(out.dtype),
                "autocast": torch.is_autocast_enabled("cuda"),
                "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
            }
            with open(f"{COORD}/grpo_dtype_{os.getpid()}.jsonl", "a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            base.addmm_(projected, cast_b, alpha=scale)
            if reshape:
                return base.view(batch, sequence, -1)
            return base
        try:
            return _original_matmul_lora(
                x, weight, quant, adapter_a, adapter_b, scale, out=out
            )
        except Exception as error:
            record = {
                "event": "matmul_lora_error",
                "arm": ARM,
                "error": str(error),
                "x": str(x.dtype),
                "weight": str(weight.dtype),
                "adapter_a": str(adapter_a.dtype),
                "adapter_b": str(adapter_b.dtype),
                "out": None if out is None else str(out.dtype),
                "autocast": torch.is_autocast_enabled("cuda"),
                "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
            }
            with open(f"{COORD}/grpo_dtype_{os.getpid()}.jsonl", "a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            raise

    _fast_lora.matmul_lora = _probed_matmul_lora

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


def make_prompts(seed, count):
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        a = rng.randint(100, 999)
        b = rng.randint(100, 999)
        rows.append(
            {
                "prompt": (
                    f"Question: what is {a}+{b}? "
                    "Reply with only the integer.\nAnswer:"
                ),
                "answer": str(a + b),
            }
        )
    return Dataset.from_list(rows)


def trainable_fingerprint():
    """Exact digest plus stable CPU-side summaries of every trainable value."""
    digest = hashlib.sha256()
    count = 0
    total = 0.0
    absolute = 0.0
    squared = 0.0
    maximum = 0.0
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            if not parameter.requires_grad:
                continue
            original = parameter.detach().contiguous().cpu()
            values = original.double().reshape(-1)
            count += values.numel()
            total += values.sum().item()
            absolute += values.abs().sum().item()
            squared += (values * values).sum().item()
            if values.numel():
                maximum = max(maximum, values.abs().max().item())
            name_bytes = name.encode()
            digest.update(struct.pack("<I", len(name_bytes)))
            digest.update(name_bytes)
            digest.update(str(original.dtype).encode())
            digest.update(struct.pack("<I", original.ndim))
            for dimension in original.shape:
                digest.update(struct.pack("<Q", dimension))
            digest.update(original.view(torch.uint8).numpy().tobytes())
    return {
        "parameter_sha256": digest.hexdigest(),
        "parameter_count": count,
        "parameter_sum": total,
        "parameter_abs_sum": absolute,
        "parameter_l2": squared**0.5,
        "parameter_max_abs": maximum,
    }


def run_grpo(lid, steps, warm_only=False):
    seed_lid = int(lid) if str(lid).isdigit() else 0
    batch_size = NGEN * PROMPTS
    dataset = make_prompts(seed_lid + 100, max(64, batch_size * steps))
    dataset_sha256 = hashlib.sha256(
        json.dumps(dataset[: min(16, len(dataset))], sort_keys=True).encode()
    ).hexdigest()
    cpu_rng_sha256 = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    cuda_rng_sha256 = hashlib.sha256(
        b"".join(state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all())
    ).hexdigest()
    rollout_digest = hashlib.sha256()
    rollout_step_sha256 = []
    rollout_step_rewards = []
    rollout_tokens = 0
    reward_values = []

    def arithmetic_reward(completions, answer, **_kwargs):
        nonlocal rollout_tokens
        rewards = []
        step_digest = hashlib.sha256()
        for completion, expected in zip(completions, answer):
            # Conversational datasets may return a one-message completion.
            if isinstance(completion, list):
                completion = completion[0].get("content", "")
            completion = str(completion)
            rollout_digest.update(struct.pack("<I", len(completion.encode())))
            rollout_digest.update(completion.encode())
            rollout_digest.update(b"\0" + str(expected).encode() + b"\0")
            step_digest.update(struct.pack("<I", len(completion.encode())))
            step_digest.update(completion.encode())
            step_digest.update(b"\0" + str(expected).encode() + b"\0")
            rollout_tokens += len(
                tokenizer.encode(completion, add_special_tokens=False)
            )
            match = re.search(r"-?\d+", completion)
            if match is None:
                reward = 0.0
            else:
                error = abs(int(match.group()) - int(expected))
                # Dense arithmetic correctness in [0, 1]. Exact is 1; answers
                # at least 1000 away are 0. This avoids all-zero reward groups.
                reward = max(0.0, 1.0 - error / 1000.0)
            rewards.append(reward)
            reward_values.append(reward)
        rollout_step_sha256.append(step_digest.hexdigest())
        rollout_step_rewards.append(round(sum(rewards) / max(1, len(rewards)), 9))
        return rewards

    config = GRPOConfig(
        output_dir=OUTBASE + f"/grpo{lid}",
        per_device_train_batch_size=batch_size,
        num_generations=NGEN,
        max_steps=steps,
        learning_rate=0.0 if warm_only else 1e-5,
        logging_steps=1,
        optim="adamw_8bit",
        seed=42,
        data_seed=42,
        max_completion_length=MAX_COMPLETION,
        max_prompt_length=min(64, MAXSEQ // 2),
        temperature=0.9,
        beta=0.04,
        use_vllm=False,
        report_to=[],
        bf16=True,
        gradient_accumulation_steps=1,
        warmup_steps=1,
        shuffle_dataset=False,
        disable_dropout=True,
        save_strategy="no",
    )
    FastLanguageModel.for_training(model)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=arithmetic_reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    if os.environ.get("GRPO_DTYPE_PROBE", "0") == "1":
        record = {
            "event": "trainer",
            "arm": ARM,
            "mixed_precision": trainer.accelerator.mixed_precision,
            "bf16": trainer.args.bf16,
            "fp16": trainer.args.fp16,
        }
        with open(f"{COORD}/grpo_dtype_{os.getpid()}.jsonl", "a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    result = trainer.train()
    final_cpu_rng_sha256 = hashlib.sha256(
        torch.get_rng_state().numpy().tobytes()
    ).hexdigest()
    final_cuda_rng_sha256 = hashlib.sha256(
        b"".join(state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all())
    ).hexdigest()
    logs = trainer.state.log_history
    logged_rewards = [entry["reward"] for entry in logs if "reward" in entry]
    losses = [entry["loss"] for entry in logs if "loss" in entry]
    if not losses:
        losses = [result.training_loss]
    return {
        "rewards": logged_rewards or reward_values,
        "losses": losses,
        "rollout_sha256": rollout_digest.hexdigest(),
        "rollout_step_sha256": rollout_step_sha256,
        "rollout_step_rewards": rollout_step_rewards,
        "rollout_tokens": rollout_tokens,
        "dataset_sha256": dataset_sha256,
        "cpu_rng_sha256": cpu_rng_sha256,
        "cuda_rng_sha256": cuda_rng_sha256,
        "final_cpu_rng_sha256": final_cpu_rng_sha256,
        "final_cuda_rng_sha256": final_cuda_rng_sha256,
    }


if FORK or QUEUE_JOBS or os.environ.get("NATIVE_REFERENCE_WARMUP", "0") == "1":
    before_warm = trainable_fingerprint()
    run_grpo("warm", 1, warm_only=True)
    torch.cuda.synchronize()
    after_warm = trainable_fingerprint()
    if before_warm["parameter_sha256"] != after_warm["parameter_sha256"]:
        raise RuntimeError("zero-learning-rate GRPO warmup changed trainable parameters")

if FORK:
    with open(f"{COORD}/golden_ready", "w") as stream:
        stream.write(str(round(load_ms)))
    while not os.path.exists(f"{COORD}/go"):
        time.sleep(0.2)
    claimed = None
    for slot in range(int(os.environ.get("NSLOTS", "64"))):
        try:
            fd = os.open(
                f"{COORD}/claim_{slot}",
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            os.close(fd)
            claimed = slot
            break
        except FileExistsError:
            continue
    if claimed is None:
        raise RuntimeError("no learner slot available")
    LID = str(claimed)

def run_one_learner(lid):
    lid = str(lid)
    initial_parameters = trainable_fingerprint()
    emit(
        lid,
        event="ready",
        load_ms=round(load_ms),
        initial_parameter_sha256=initial_parameters["parameter_sha256"],
        **model_runtime,
    )
    torch.cuda.reset_peak_memory_stats()
    train_started = time.time()
    training = run_grpo(lid, STEPS)
    torch.cuda.synchronize()
    duration = time.time() - train_started

    parameters = trainable_fingerprint()
    rewards = training["rewards"]
    losses = training["losses"]
    reward0 = sum(rewards[: min(2, len(rewards))]) / max(1, min(2, len(rewards)))
    rewardN = sum(rewards[-min(2, len(rewards)) :]) / max(1, min(2, len(rewards)))
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    emit(
        lid,
        event="done",
        train_s=round(duration, 2),
        tok_s=round(training["rollout_tokens"] / duration),
        step_ms=round(duration / STEPS * 1000),
        steps=STEPS,
        loss0=round(losses[0], 6),
        lossN=round(losses[-1], 6),
        reward0=round(reward0, 6),
        rewardN=round(rewardN, 6),
        reward_max=round(max(rewards), 6),
        reward_min=round(min(rewards), 6),
        rollout_tokens=training["rollout_tokens"],
        rollout_sha256=training["rollout_sha256"],
        rollout_step_sha256=training["rollout_step_sha256"],
        rollout_step_rewards=training["rollout_step_rewards"],
        peak_gb=round(peak_gb, 2),
        initial_parameter_sha256=initial_parameters["parameter_sha256"],
        parameter_sha256=parameters["parameter_sha256"],
        parameter_count=parameters["parameter_count"],
        parameter_sum=round(parameters["parameter_sum"], 9),
        parameter_abs_sum=round(parameters["parameter_abs_sum"], 9),
        parameter_l2=round(parameters["parameter_l2"], 9),
        parameter_max_abs=round(parameters["parameter_max_abs"], 9),
        dataset_sha256=training["dataset_sha256"],
        cpu_rng_sha256=training["cpu_rng_sha256"],
        cuda_rng_sha256=training["cuda_rng_sha256"],
        final_cpu_rng_sha256=training["final_cpu_rng_sha256"],
        final_cuda_rng_sha256=training["final_cuda_rng_sha256"],
        **model_runtime,
    )
    print(
        f"GRPO LEARNER {lid} [{ARM}] DONE load={load_ms:.0f}ms "
        f"loss {losses[0]:.6f}->{losses[-1]:.6f} "
        f"reward {reward0:.3f}->{rewardN:.3f} "
        f"rollout_tok/s={training['rollout_tokens'] / duration:.0f} "
        f"peak={peak_gb:.1f}GB",
        flush=True,
    )


if QUEUE_JOBS:
    if FORK:
        raise RuntimeError("QUEUE_JOBS is a native resident-base control")
    # This is the strongest homogeneous queued baseline: the quantized base,
    # compiled kernels, and fixed-shape LoRA allocation stay resident. Each job
    # gets the exact same initial adapter values and a fresh Trainer/optimizer;
    # its final adapter is fingerprinted before the slot is reset for the next
    # queued job.
    initial_trainable = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected_initial = trainable_fingerprint()["parameter_sha256"]
    initial_cpu_rng = torch.get_rng_state().clone()
    initial_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
    for queued_lid in range(QUEUE_JOBS):
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(initial_trainable[name])
        if trainable_fingerprint()["parameter_sha256"] != expected_initial:
            raise RuntimeError(f"queued learner {queued_lid} adapter reset failed")
        torch.set_rng_state(initial_cpu_rng)
        torch.cuda.set_rng_state_all(initial_cuda_rng)
        run_one_learner(queued_lid)
        gc.collect()
        torch.cuda.empty_cache()
else:
    run_one_learner(LID)
