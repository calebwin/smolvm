"""Run real GRPO updates while delegating generation to a shared LoRA pool."""

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "bf16")

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from peft import get_peft_model_state_dict
from safetensors.torch import save_file as save_safetensors
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import maybe_apply_chat_template


ROOT = Path(os.environ.get("POOL_ROOT", "/tmp/grpo-pool"))
MODEL = os.environ.get("MODEL", "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
MAX_SEQUENCE = int(os.environ.get("MAXSEQ", "256"))
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION", "32"))
STEPS = int(os.environ.get("STEPS", "5"))
BATCH = int(os.environ.get("BATCH", "4"))
GENERATIONS = int(os.environ.get("NGEN", "4"))
LEARNER_ID = int(os.environ.get("LEARNER_ID", "0"))
COORD = Path(os.environ.get("COORD", str(ROOT)))
TIMEOUT = float(os.environ.get("POOL_TIMEOUT", "300"))
FORK = os.environ.get("FORK", "0") == "1"
WAIT_FOR_GO = os.environ.get("WAIT_FOR_GO", "0") == "1"
ADAPTER_EXPORT_MODE = os.environ.get("ADAPTER_EXPORT_MODE", "peft")
REWARD_VALUES = []
REWARD_STDS = []


def save_adapter_flat(model, path):
    """Write a standard PEFT checkpoint with bulk device-to-host copies."""
    state = get_peft_model_state_dict(model, save_embedding_layers=False)
    groups = {}
    for name, tensor in sorted(state.items()):
        value = tensor.detach().contiguous()
        groups.setdefault((value.device, value.dtype), []).append((name, value))

    host_state = {}
    for items in groups.values():
        packed = torch.cat([tensor.reshape(-1) for _, tensor in items])
        host = packed.cpu()
        offset = 0
        for name, tensor in items:
            elements = tensor.numel()
            host_state[name] = host[offset : offset + elements].reshape(tensor.shape).clone()
            offset += elements

    config = model.peft_config[model.active_adapter]
    config.save_pretrained(path)
    save_safetensors(host_state, path / "adapter_model.safetensors", metadata={"format": "pt"})


class PooledGRPOTrainer(GRPOTrainer):
    def _generate_single_turn(self, prompts, images):
        if images is not None:
            raise NotImplementedError("the rollout-pool probe supports text prompts only")

        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"]
            for prompt in prompts
        ]
        call_index = getattr(self, "_pool_call_index", 0)
        self._pool_call_index = call_index + 1
        request_id = f"l{LEARNER_ID}-c{call_index}-{time.time_ns()}"
        adapter_id = LEARNER_ID * 1_000_000 + call_index + 1
        relative_adapter = Path("adapters") / request_id
        adapter_path = ROOT / relative_adapter
        adapter_path.mkdir(parents=True, exist_ok=False)

        export_started = time.time()
        unwrapped = self.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False)
        if ADAPTER_EXPORT_MODE == "flat":
            save_adapter_flat(unwrapped, adapter_path)
        else:
            unwrapped.save_pretrained(
                adapter_path,
                safe_serialization=True,
                save_embedding_layers=False,
            )
        export_s = time.time() - export_started

        request = {
            "request_id": request_id,
            "adapter_id": adapter_id,
            "adapter_path": str(relative_adapter),
            "prompts": prompts_text,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": -1 if self.top_k is None else self.top_k,
            "min_p": 0.0 if self.min_p is None else self.min_p,
            "max_tokens": self.max_completion_length,
            "seed": 100_000 + LEARNER_ID * 10_000 + call_index,
        }
        request_path = ROOT / "requests" / f"{request_id}.json"
        temporary = request_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(request, separators=(",", ":")))
        os.replace(temporary, request_path)

        response_path = ROOT / "responses" / f"{request_id}.json"
        wait_started = time.time()
        while not response_path.exists():
            if time.time() - wait_started > TIMEOUT:
                raise TimeoutError(f"rollout response timed out: {request_id}")
            time.sleep(0.002)
        response = json.loads(response_path.read_text())
        response_path.unlink()
        if "error" in response:
            raise RuntimeError(response["error"])

        metrics = getattr(self, "_pool_metrics", [])
        metrics.append(
            {
                "adapter_export_s": export_s,
                "roundtrip_s": time.time() - wait_started,
                "batch_requests": response["batch_requests"],
                "batch_prompts": response["batch_prompts"],
                "completion_tokens": sum(len(ids) for ids in response["completion_ids"]),
                "server_batch_s": response["batch_s"],
            }
        )
        self._pool_metrics = metrics
        return response["prompt_ids"], response["completion_ids"], None, {}


def build_dataset():
    rng = random.Random(20_000 + LEARNER_ID)
    rows = []
    for _index in range(max(64, STEPS * BATCH * 2)):
        left = rng.randint(100, 999)
        right = rng.randint(100, 999)
        rows.append(
            {
                "prompt": (
                    f"Question: what is {left}+{right}? "
                    "Reason briefly, then put the integer after Answer:."
                ),
                "expected": left + right,
            }
        )
    return Dataset.from_list(rows)


def arithmetic_reward(prompts, completions, expected, **_kwargs):
    rewards = []
    for completion, target in zip(completions, expected):
        match = re.search(r"-?\d+", completion)
        if match is None:
            rewards.append(0.0)
        else:
            error = abs(int(match.group()) - int(target))
            rewards.append(max(0.0, 1.0 - error / 1000.0))
    REWARD_VALUES.extend(rewards)
    if len(rewards) > 1:
        REWARD_STDS.append(float(torch.tensor(rewards).std(unbiased=False)))
    return rewards


started = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL,
    max_seq_length=MAX_SEQUENCE,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    fast_inference=False,
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


def adapter_hash():
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" in name:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# Hash while the golden still owns a live context. This is probe bookkeeping, not
# part of GRPO, and issuing a diagnostic D2H copy immediately after restore can
# obscure whether the workload itself resumed correctly.
initial_adapter_digest = adapter_hash()

if FORK:
    (COORD / "golden_ready").write_text(str(time.time() - started))
    while not (COORD / "go").exists():
        time.sleep(0.05)

    claimed = None
    fork_env = Path("/etc/smolvm/fork-env")
    if fork_env.exists():
        for line in fork_env.read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator and key == "LEARNER_ID":
                claimed = int(value)
                break
    if claimed is None:
        for slot in range(int(os.environ.get("NSLOTS", "64"))):
            claim = COORD / f"pool_claim_{slot}"
            try:
                descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(descriptor)
                claimed = slot
                break
            except FileExistsError:
                continue
    if claimed is None:
        raise RuntimeError("no pooled-GRPO learner slot available")
    LEARNER_ID = claimed
elif WAIT_FOR_GO:
    (COORD / f"native_ready_{LEARNER_ID}.json").write_text(str(time.time() - started))
    while not (COORD / "go").exists():
        time.sleep(0.05)

learner_started = time.time()
FastLanguageModel.for_training(model)

config = GRPOConfig(
    output_dir=str(COORD / f"pool-grpo-{LEARNER_ID}"),
    per_device_train_batch_size=BATCH,
    num_generations=GENERATIONS,
    max_steps=STEPS,
    learning_rate=1e-5,
    logging_steps=1,
    optim="adamw_8bit",
    seed=42 + LEARNER_ID,
    data_seed=42 + LEARNER_ID,
    max_completion_length=MAX_COMPLETION,
    max_prompt_length=min(64, MAX_SEQUENCE // 2),
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
trainer = PooledGRPOTrainer(
    model=model,
    reward_funcs=arithmetic_reward,
    args=config,
    train_dataset=build_dataset(),
    processing_class=tokenizer,
)
train_result = trainer.train()
torch.cuda.synchronize()

metrics = getattr(trainer, "_pool_metrics", [])
record = {
    "event": "done",
    "learner_id": LEARNER_ID,
    "steps": STEPS,
    "wall_s": time.time() - learner_started,
    "total_wall_s": time.time() - started,
    "train_runtime": train_result.metrics.get("train_runtime"),
    "initial_adapter_sha256": initial_adapter_digest,
    "final_adapter_sha256": adapter_hash(),
    "pool_calls": len(metrics),
    "adapter_export_s": sum(item["adapter_export_s"] for item in metrics),
    "pool_roundtrip_s": sum(item["roundtrip_s"] for item in metrics),
    "server_batch_s": sum(item["server_batch_s"] for item in metrics),
    "rollout_tokens": sum(item["completion_tokens"] for item in metrics),
    "max_batch_requests": max((item["batch_requests"] for item in metrics), default=0),
    "max_batch_prompts": max((item["batch_prompts"] for item in metrics), default=0),
    "reward_min": min(REWARD_VALUES) if REWARD_VALUES else None,
    "reward_max": max(REWARD_VALUES) if REWARD_VALUES else None,
    "reward_std_max": max(REWARD_STDS) if REWARD_STDS else None,
}
(COORD / f"pool_trainer_{LEARNER_ID}.json").write_text(json.dumps(record, sort_keys=True))
print(json.dumps(record, sort_keys=True))
