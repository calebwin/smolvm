"""Offline Unsloth SFT qualification workload for native/smolvm A/B runs."""

import faulthandler

faulthandler.enable()

import glob
import hashlib
import json
import os
import random
import struct
import time

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

STEPS = int(os.environ.get("STEPS", "20"))
MAXSEQ = int(os.environ.get("MAXSEQ", "512"))
BATCH = int(os.environ.get("BATCH", "2"))
COORD = os.environ.get("COORD", "/coord")
MODEL = os.environ.get("MODEL", "unsloth/Qwen2.5-7B-bnb-4bit")
ARM = os.environ.get("ARM", "?")
FORK = os.environ.get("FORK", "0") == "1"
LID = os.environ.get("LEARNER_ID", "0")


def emit(lid, **fields):
    fields.update(
        lid=str(lid),
        arm=ARM,
        method="sft",
        t=round(time.time(), 3),
    )
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

from unsloth import FastLanguageModel
import torch

load_started = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL,
    max_seq_length=MAXSEQ,
    load_in_4bit=True,
    dtype=None,
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

from datasets import Dataset
from trl import SFTConfig, SFTTrainer


def make_examples(seed, count):
    """Produce deterministic, near-MAXSEQ instruction/answer sequences."""
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        a = rng.randint(10, 999)
        b = rng.randint(10, 999)
        fact = (
            f"The sum of {a} and {b} is {a + b}. "
            f"The difference is {a - b}. "
        )
        # Repetition makes all batches exercise the configured sequence-length
        # regime instead of benchmarking a handful of short arithmetic tokens.
        body = (fact * (MAXSEQ // 12 + 4)).strip()
        rows.append(
            {
                "text": (
                    "### Instruction:\n"
                    "Explain the arithmetic facts carefully.\n"
                    "### Response:\n"
                    f"{body}{tokenizer.eos_token}"
                )
            }
        )
    return Dataset.from_list(rows)


def run_sft(lid, steps):
    seed = (int(lid) if str(lid).isdigit() else 0) + 300
    dataset = make_examples(seed, max(64, BATCH * steps))
    args = SFTConfig(
        per_device_train_batch_size=BATCH,
        max_steps=steps,
        learning_rate=5e-5,
        logging_steps=max(1, steps // 4),
        optim="adamw_8bit",
        seed=42,
        output_dir=os.environ.get("OUTBASE", "/root") + f"/sft{lid}",
        report_to=[],
        save_strategy="no",
        dataset_text_field="text",
        dataset_num_proc=1,
        max_length=MAXSEQ,
        packing=False,
        warmup_steps=1,
    )
    FastLanguageModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    dataset_sample = trainer.train_dataset[: min(8, len(trainer.train_dataset))]
    dataset_sha256 = hashlib.sha256(
        json.dumps(dataset_sample, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cpu_rng_sha256 = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    cuda_rng_sha256 = hashlib.sha256(
        b"".join(state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all())
    ).hexdigest()
    trainable_grad_tensors = sum(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    result = trainer.train()
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    if not losses:
        losses = [result.training_loss]
    return losses, {
        "dataset_sha256": dataset_sha256,
        "cpu_rng_sha256": cpu_rng_sha256,
        "cuda_rng_sha256": cuda_rng_sha256,
        "trainable_grad_tensors": trainable_grad_tensors,
    }


def trainable_fingerprint():
    """Stable CPU-side metrics and an exact byte digest of the LoRA state."""
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
            # Copy one adapter tensor at a time so qualification does not keep a
            # second full trainable state resident on either the GPU or CPU.
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


def model_output_fingerprint():
    """Hash a fixed eval forward to check frozen-weight reconstruction."""
    FastLanguageModel.for_training(model)
    was_training = model.training
    model.eval()
    encoded = tokenizer(
        "### Instruction:\nAdd 137 and 284.\n### Response:\n",
        return_tensors="pt",
        add_special_tokens=True,
    )
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    with torch.no_grad():
        output = model(**encoded).logits[:, -8:, :].detach().contiguous().cpu()
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


if FORK or os.environ.get("NATIVE_REFERENCE_WARMUP", "0") == "1":
    if os.environ.get("GOLDEN_WARMUP", "1") == "1":
        # warmup_steps=1 makes this first optimizer step a zero-LR path warmup,
        # so it establishes private training state without changing the LoRA.
        run_sft("warm", 1)
        torch.cuda.synchronize()
if FORK:
    if os.environ.get("GOLDEN_CUBLAS_PRIME", "0") == "1":
        # Diagnostic only: distinguish a clone worker that cannot initialize
        # cuBLAS after fork from a general SFT failure. Production smolvm must
        # perform any required library priming internally; workloads should not
        # need this branch.
        prime = torch.ones((16, 16), device="cuda")
        _ = prime @ prime
        torch.cuda.synchronize()
        del prime
    if os.environ.get("SFT_MODEL_PROBE", "0") == "1":
        with open(f"{COORD}/golden_model_probe.json", "w") as stream:
            json.dump(model_output_fingerprint(), stream, sort_keys=True)
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

LID = str(LID)
initial_parameters = trainable_fingerprint()
model_probe = (
    model_output_fingerprint()
    if os.environ.get("SFT_MODEL_PROBE", "0") == "1"
    else {}
)
emit(
    LID,
    event="ready",
    load_ms=round(load_ms),
    initial_parameter_sha256=initial_parameters["parameter_sha256"],
)
train_started = time.time()
losses, training_probe = run_sft(LID, STEPS)
torch.cuda.synchronize()
duration = time.time() - train_started

parameters = trainable_fingerprint()
nominal_tokens = STEPS * BATCH * MAXSEQ
emit(
    LID,
    event="done",
    train_s=round(duration, 2),
    tok_s=round(nominal_tokens / duration),
    examples_s=round(STEPS * BATCH / duration, 4),
    step_ms=round(duration / STEPS * 1000),
    loss0=round(losses[0], 4),
    lossN=round(losses[-1], 4),
    peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
    initial_parameter_sha256=initial_parameters["parameter_sha256"],
    parameter_sha256=parameters["parameter_sha256"],
    parameter_count=parameters["parameter_count"],
    parameter_sum=round(parameters["parameter_sum"], 9),
    parameter_abs_sum=round(parameters["parameter_abs_sum"], 9),
    parameter_l2=round(parameters["parameter_l2"], 9),
    parameter_max_abs=round(parameters["parameter_max_abs"], 9),
    **model_probe,
    **training_probe,
)
print(
    f"SFT LEARNER {LID} [{ARM}] DONE load={load_ms:.0f}ms "
    f"loss {losses[0]:.4f}->{losses[-1]:.4f} "
    f"tok/s={nominal_tokens / duration:.0f} "
    f"peak={torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
    flush=True,
)
