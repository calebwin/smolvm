"""Validate isolated multi-adapter rollout batching in one vLLM engine."""

import hashlib
import json
import os
import random
import time

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "bf16")

from unsloth import FastLanguageModel
import torch
from vllm import SamplingParams


ROUNDS = int(os.environ.get("STEPS", "20"))
PROMPTS_PER_JOB = int(os.environ.get("BATCH", "4"))
GENERATIONS = int(os.environ.get("NGEN", "4"))
MAX_SEQUENCE = int(os.environ.get("MAXSEQ", "256"))
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION", "32"))
JOBS = int(os.environ.get("QUEUE_JOBS", "4"))
REFRESH_ADAPTERS = os.environ.get("REFRESH_ADAPTERS", "0") == "1"
MODEL = os.environ.get("MODEL", "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
COORD = os.environ.get("COORD", "/tmp")
OUTBASE = os.environ.get("OUTBASE", "/tmp")

def emit(**fields):
    fields.update(lid="fused", arm="queue", method="grpo_rollout_multiadapter", t=time.time())
    with open(f"{COORD}/learner_fused.jsonl", "a") as stream:
        stream.write(json.dumps(fields, sort_keys=True) + "\n")


def prompts_for(job_index, round_index):
    rng = random.Random(10_000 + job_index * 1_000 + round_index)
    return [
        f"Question: what is {a}+{b}? Reason briefly, then put the integer after Answer:."
        for a, b in (
            (rng.randint(100, 999), rng.randint(100, 999))
            for _ in range(PROMPTS_PER_JOB)
        )
    ]


def request_digest(request):
    digest = hashlib.sha256()
    for name, tensor in sorted(request.tensors.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


load_started = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL,
    max_seq_length=MAX_SEQUENCE,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    fast_inference=True,
    max_lora_rank=16,
    max_loras=JOBS,
    gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.6")),
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
torch.cuda.synchronize()
load_s = time.time() - load_started


def make_adapter(job_index):
    """Snapshot a genuinely different policy adapter under a stable vLLM id."""
    with torch.no_grad():
        torch.manual_seed(50_000 + job_index)
        for name, parameter in model.named_parameters():
            if ".lora_B." in name:
                parameter.normal_(mean=0.0, std=0.05)

    request = model.load_lora(
        os.path.join(OUTBASE, "multiadapter-policy-lora"),
        load_tensors=True,
        lora_request_id=job_index + 1,
    )
    tensors = {
        name: tensor.detach().clone()
        for name, tensor in request.tensors.items()
    }
    return type(request)(
        str(job_index + 1),
        job_index + 1,
        lora_tensors=tensors,
        lora_config=request.config,
    )


adapter_started = time.time()
adapters = [make_adapter(job_index) for job_index in range(JOBS)]
adapter_digests = [request_digest(request) for request in adapters]
assert len(set(adapter_digests)) == JOBS, "adapter tensors are not isolated"

probe_prompts = ["Question: what is 377+488? Answer:"] * JOBS
probe_sampling = SamplingParams(n=1, max_tokens=24, temperature=0.0, seed=7)
probe_outputs = model.fast_generate(
    probe_prompts,
    sampling_params=probe_sampling,
    lora_request=adapters,
    use_tqdm=False,
)
torch.cuda.synchronize()
probe_texts = [output.outputs[0].text for output in probe_outputs]
probe_digests = [hashlib.sha256(text.encode()).hexdigest() for text in probe_texts]
adapter_setup_s = time.time() - adapter_started

torch.cuda.reset_peak_memory_stats()
digest = hashlib.sha256()
token_count = 0
latencies = []
refresh_s = 0.0
started = time.time()
for round_index in range(ROUNDS):
    if REFRESH_ADAPTERS:
        refresh_started = time.time()
        refreshed = []
        for job_index, adapter in enumerate(adapters):
            request_id = JOBS + round_index * JOBS + job_index + 1
            tensors = {}
            for name, tensor in adapter.tensors.items():
                updated = tensor.detach().clone()
                if ".lora_B." in name:
                    updated.add_((round_index + 1) * 1e-4)
                tensors[name] = updated
            refreshed.append(
                type(adapter)(
                    str(request_id),
                    request_id,
                    lora_tensors=tensors,
                    lora_config=adapter.config,
                )
            )
        round_adapters = refreshed
        refresh_s += time.time() - refresh_started
    else:
        round_adapters = adapters

    prompts = []
    requests = []
    for job_index in range(JOBS):
        job_prompts = prompts_for(job_index, round_index)
        prompts.extend(job_prompts)
        requests.extend([round_adapters[job_index]] * len(job_prompts))

    sampling = SamplingParams(
        n=GENERATIONS,
        max_tokens=MAX_COMPLETION,
        temperature=0.9,
        top_p=0.95,
        seed=900_000 + round_index,
    )
    call_started = time.time()
    outputs = model.fast_generate(
        prompts,
        sampling_params=sampling,
        lora_request=requests,
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    latencies.append(time.time() - call_started)
    for request_output in outputs:
        for completion in request_output.outputs:
            token_count += len(completion.token_ids)
            digest.update(completion.text.encode())
            digest.update(b"\0")

duration = time.time() - started
emit(
    event="done",
    jobs=JOBS,
    steps=ROUNDS * JOBS,
    train_s=round(duration, 6),
    rollout_tokens=token_count,
    tok_s=round(token_count / duration, 3),
    rollout_sha256=digest.hexdigest(),
    load_s=round(load_s, 6),
    adapter_setup_s=round(adapter_setup_s, 6),
    adapter_refresh=REFRESH_ADAPTERS,
    adapter_refresh_s=round(refresh_s, 6),
    unique_adapter_digests=len(set(adapter_digests)),
    unique_probe_outputs=len(set(probe_digests)),
    adapter_digests=adapter_digests,
    probe_digests=probe_digests,
    first_batch_s=round(latencies[0], 6),
    median_batch_s=round(sorted(latencies)[len(latencies) // 2], 6),
    peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3),
)
