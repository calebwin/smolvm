"""Unsloth SFT prototype that forks from inside a live Trainer.

This is a placement experiment, not a proposed user-facing requirement. The
golden executes one zero-learning-rate step, blocks in a Trainer callback while
smolvm forks it, and every clone continues the already-built trainer. It
measures the fixed-cost ceiling that the ordinary workload pays when each clone
constructs a second SFTTrainer after the snapshot.
"""

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
    fields.update(lid=str(lid), arm=ARM, method="sft-resume", t=round(time.time(), 3))
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
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer


def make_examples(seed, count):
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        a = rng.randint(10, 999)
        b = rng.randint(10, 999)
        fact = (
            f"The sum of {a} and {b} is {a + b}. "
            f"The difference is {a - b}. "
        )
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


def trainable_fingerprint():
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


def claim_slot():
    for slot in range(int(os.environ.get("NSLOTS", "64"))):
        try:
            fd = os.open(
                f"{COORD}/claim_{slot}",
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            os.close(fd)
            return str(slot)
        except FileExistsError:
            continue
    raise RuntimeError("no learner slot available")


class ForkAfterWarmStep(TrainerCallback):
    def __init__(self):
        self.released_at = None
        self.lid = str(LID)
        self.initial = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step != 1 or self.released_at is not None:
            return control
        torch.cuda.synchronize()
        self.initial = trainable_fingerprint()
        if FORK:
            with open(f"{COORD}/golden_ready", "w") as stream:
                stream.write(str(round(load_ms)))
            while not os.path.exists(f"{COORD}/go"):
                time.sleep(0.2)
            self.lid = claim_slot()
        self.released_at = time.time()
        emit(
            self.lid,
            event="ready",
            load_ms=round(load_ms),
            initial_parameter_sha256=self.initial["parameter_sha256"],
        )
        return control


dataset = make_examples(300, max(64, BATCH * (STEPS + 1)))
args = SFTConfig(
    per_device_train_batch_size=BATCH,
    max_steps=STEPS + 1,
    learning_rate=5e-5,
    logging_steps=1,
    optim="adamw_8bit",
    seed=42,
    output_dir=os.environ.get("OUTBASE", "/root") + "/sft-resume",
    report_to=[],
    save_strategy="no",
    dataset_text_field="text",
    dataset_num_proc=1,
    max_length=MAXSEQ,
    packing=False,
    warmup_steps=1,
)
FastLanguageModel.for_training(model)
gate = ForkAfterWarmStep()
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
    callbacks=[gate],
)
result = trainer.train()
torch.cuda.synchronize()
if gate.released_at is None:
    raise RuntimeError("trainer never reached the fork gate")
duration = time.time() - gate.released_at

losses = [
    entry["loss"]
    for entry in trainer.state.log_history
    if "loss" in entry and float(entry.get("step", 0)) > 1
]
if not losses:
    losses = [result.training_loss]
parameters = trainable_fingerprint()
nominal_tokens = STEPS * BATCH * MAXSEQ
emit(
    gate.lid,
    event="done",
    train_s=round(duration, 2),
    tok_s=round(nominal_tokens / duration),
    examples_s=round(STEPS * BATCH / duration, 4),
    step_ms=round(duration / STEPS * 1000),
    loss0=round(losses[0], 4),
    lossN=round(losses[-1], 4),
    peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
    initial_parameter_sha256=gate.initial["parameter_sha256"],
    parameter_sha256=parameters["parameter_sha256"],
    parameter_count=parameters["parameter_count"],
    parameter_sum=round(parameters["parameter_sum"], 9),
    parameter_abs_sum=round(parameters["parameter_abs_sum"], 9),
    parameter_l2=round(parameters["parameter_l2"], 9),
    parameter_max_abs=round(parameters["parameter_max_abs"], 9),
)
print(
    f"SFT-RESUME LEARNER {gate.lid} [{ARM}] DONE "
    f"loss {losses[0]:.4f}->{losses[-1]:.4f} "
    f"tok/s={nominal_tokens / duration:.0f}",
    flush=True,
)
