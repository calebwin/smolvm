"""Batch versioned LoRA rollout requests through one vLLM engine."""

import json
import os
import signal
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from unsloth import FastLanguageModel
import torch
from vllm import SamplingParams


ROOT = Path(os.environ.get("POOL_ROOT", "/tmp/grpo-pool"))
MODEL = os.environ.get("MODEL", "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
MAX_SEQUENCE = int(os.environ.get("MAXSEQ", "256"))
MAX_POLICIES = int(os.environ.get("MAX_POLICIES", "4"))
BATCH_WINDOW_MS = float(os.environ.get("BATCH_WINDOW_MS", "20"))
GPU_UTILIZATION = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.14"))
STOP = False


def stop(_signum, _frame):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "requests").mkdir(exist_ok=True)
(ROOT / "responses").mkdir(exist_ok=True)

load_started = time.time()
model, _tokenizer = FastLanguageModel.from_pretrained(
    MODEL,
    max_seq_length=MAX_SEQUENCE,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    fast_inference=True,
    max_lora_rank=16,
    max_loras=MAX_POLICIES,
    gpu_memory_utilization=GPU_UTILIZATION,
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
(ROOT / "ready.json").write_text(
    json.dumps({"load_s": time.time() - load_started, "pid": os.getpid()})
)


def write_response(request_id, payload):
    destination = ROOT / "responses" / f"{request_id}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(temporary, destination)


while not STOP:
    request_paths = sorted((ROOT / "requests").glob("*.json"))
    if not request_paths:
        time.sleep(0.002)
        continue

    deadline = time.monotonic() + BATCH_WINDOW_MS / 1000.0
    known = {path.name: path for path in request_paths}
    while time.monotonic() < deadline and not STOP:
        for path in (ROOT / "requests").glob("*.json"):
            known[path.name] = path
        time.sleep(0.001)
    request_paths = [known[name] for name in sorted(known)]

    requests = []
    for path in request_paths:
        try:
            requests.append((path, json.loads(path.read_text())))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    if not requests:
        continue

    started = time.time()
    prompts = []
    sampling_params = []
    lora_requests = []
    slices = []
    try:
        for _path, request in requests:
            adapter_path = ROOT / request["adapter_path"]
            lora_request = model.load_lora(
                str(adapter_path),
                load_tensors=False,
                lora_request_id=int(request["adapter_id"]),
            )
            begin = len(prompts)
            request_prompts = request["prompts"]
            prompts.extend(request_prompts)
            lora_requests.extend([lora_request] * len(request_prompts))
            for prompt_index in range(len(request_prompts)):
                sampling_params.append(
                    SamplingParams(
                        n=1,
                        repetition_penalty=float(request["repetition_penalty"]),
                        temperature=float(request["temperature"]),
                        top_p=float(request["top_p"]),
                        top_k=int(request["top_k"]),
                        min_p=float(request["min_p"]),
                        max_tokens=int(request["max_tokens"]),
                        seed=int(request["seed"]) + prompt_index,
                    )
                )
            slices.append((begin, len(prompts)))

        outputs = model.fast_generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_requests,
            use_tqdm=False,
        )
        torch.cuda.synchronize()
        batch_s = time.time() - started
        batch_tokens = sum(
            len(completion.token_ids)
            for output in outputs
            for completion in output.outputs
        )
        with (ROOT / "batches.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "requests": len(requests),
                        "prompts": len(prompts),
                        "tokens": batch_tokens,
                        "batch_s": batch_s,
                        "tok_s": batch_tokens / batch_s,
                        "adapter_ids": [request["adapter_id"] for _path, request in requests],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        for (_path, request), (begin, end) in zip(requests, slices):
            selected = outputs[begin:end]
            write_response(
                request["request_id"],
                {
                    "prompt_ids": [output.prompt_token_ids for output in selected],
                    "completion_ids": [list(output.outputs[0].token_ids) for output in selected],
                    "batch_requests": len(requests),
                    "batch_prompts": len(prompts),
                    "batch_tokens": batch_tokens,
                    "batch_s": batch_s,
                },
            )
    except Exception as error:
        for _path, request in requests:
            write_response(request["request_id"], {"error": repr(error)})
    finally:
        for path, _request in requests:
            path.unlink(missing_ok=True)

(ROOT / "stopped").write_text(str(time.time()))
