import faulthandler; faulthandler.enable()
import os, time, json, random, glob
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
STEPS = int(os.environ.get("STEPS", "20"))
MAXSEQ = int(os.environ.get("MAXSEQ", "256"))
BATCH = int(os.environ.get("BATCH", "2"))
COORD = os.environ.get("COORD", "/coord")
MODEL = os.environ.get("MODEL", "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
ARM = os.environ.get("ARM", "?")
FORK = os.environ.get("FORK", "0") == "1"
LID = os.environ.get("LEARNER_ID", "0")
BETA = float(os.environ.get("DPO_BETA", "0.1"))

def emit(lid, **kw):
    kw.update(lid=str(lid), arm=ARM, method="dpo", t=round(time.time(), 3))
    with open(f"{COORD}/learner_{lid}.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")

# Resolve the model to its local snapshot so loading never hits the hub.
_snaps = sorted(glob.glob(os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/hf")), "hub",
    "models--" + MODEL.replace("/", "--"), "snapshots", "*")))
if _snaps:
    MODEL = _snaps[-1]
    os.environ["HF_HUB_OFFLINE"] = "1"

from unsloth import FastLanguageModel
import torch
t0 = time.time()
model, tok = FastLanguageModel.from_pretrained(
    MODEL, max_seq_length=MAXSEQ, load_in_4bit=True, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=0)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
torch.cuda.synchronize()
load_ms = (time.time() - t0) * 1000

from datasets import Dataset
from trl import DPOTrainer, DPOConfig

# Diagnostic workload-visible graph arm. Unsloth replaces Accelerate's
# TorchDynamoPlugin kwargs and explicitly sets `triton.cudagraphs=False`, so
# `torch_compile=True` alone can silently remain an eager boundary sequence.
# This opt-in override is intentionally separate from the unchanged-workload
# MPS path: it tests whether fixed-shape compilation can make the framework
# issue real CUDA graph launches through smolvm.
if os.environ.get("FORCE_CUDAGRAPHS") == "1":
    import accelerate
    import torch._dynamo
    import torch._inductor
    from unsloth.models import _utils as unsloth_utils

    graph_options = dict(unsloth_utils.torch_compile_options)
    graph_options.update({
        "max_autotune": False,
        "triton.cudagraphs": True,
    })

    def graph_compile_kwargs(*args, **kwargs):
        fullgraph = os.environ.get("FORCE_CUDAGRAPHS_FULLGRAPH") == "1"
        print(
            "SMOLVM_GRAPH_PROBE compile kwargs: "
            f"static + cudagraphs + fullgraph={fullgraph}",
            flush=True,
        )
        return {
            "dynamic": False,
            "fullgraph": fullgraph,
            "options": graph_options,
        }

    accelerate.utils.dataclasses.TorchDynamoPlugin.to_kwargs = graph_compile_kwargs
    accelerate.utils.TorchDynamoPlugin.to_kwargs = graph_compile_kwargs
    accelerate.accelerator.TorchDynamoPlugin.to_kwargs = graph_compile_kwargs
    torch._dynamo.config.suppress_errors = False
    torch._inductor.config.triton.cudagraphs = True


def make_prefs(seed, n):
    """Synthetic arithmetic preference pairs: 'chosen' is the correct answer,
    'rejected' is a plausible-but-wrong one. A real DPO signal (prefer correct)
    with a distinct per-learner shard via the seed."""
    r = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = r.randint(1, 20), r.randint(1, 20)
        prompt = f"### Q: what is {a}+{b}?\n### A:"
        chosen = f" {a + b}"
        rejected = f" {a + b + r.choice([-2, -1, 1, 2, 3])}"
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return Dataset.from_list(rows)


def run_dpo(lid, steps):
    if (
        os.environ.get("EXPLICIT_DPO_GRAPH_MODE") in {"eager", "graph"}
        and str(lid) != "warm"
    ):
        return run_explicit_dpo_graph_probe(lid, steps)

    seed = (int(lid) if str(lid).isdigit() else 0) + 100
    ds = make_prefs(seed, max(64, BATCH * steps))
    cfg = DPOConfig(
        per_device_train_batch_size=BATCH, max_steps=steps, learning_rate=5e-5,
        logging_steps=max(1, steps // 4), optim="adamw_8bit", seed=42,
        output_dir=os.environ.get("OUTBASE", "/root") + f"/dpo{lid}", report_to=[], beta=BETA,
        max_length=MAXSEQ, max_prompt_length=MAXSEQ // 2,
        remove_unused_columns=False, warmup_steps=1,
        # GRAPHS=1 is a diagnostic framework-compile arm. Installed Unsloth
        # overrides Inductor cudagraphs off, so FORCE_CUDAGRAPHS above is
        # required to prove whether real captures occur. It is not a production
        # performance option.
        **({"torch_compile": True, "torch_compile_mode": "reduce-overhead"}
           if os.environ.get("GRAPHS") == "1" else {}),
    )
    FastLanguageModel.for_training(model)
    # ref_model=None: with a PEFT/LoRA policy, DPO uses the adapter-disabled
    # base as the implicit frozen reference — no second model copy. This is the
    # smolvm --share-weights fit: the frozen base (=reference) is shared, each
    # fork trains only its own LoRA policy.
    tr = DPOTrainer(model=model, ref_model=None, args=cfg,
                    train_dataset=ds, processing_class=tok)
    tr.train()
    losses = [h["loss"] for h in tr.state.log_history if "loss" in h]
    return losses


def _prepare_capture_batch(trainer, batch):
    """Normalize padding outside capture and retain only fixed-address tensors."""
    from trl.trainer.utils import flush_left
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    num_examples = batch["prompt_input_ids"].shape[0]
    combined = trainer.concatenated_inputs(batch, padding_value=trainer.pad_token_id)
    input_ids = torch.cat(
        (combined["prompt_input_ids"], combined["completion_input_ids"]), dim=1
    )
    attention_mask = torch.cat(
        (
            combined["prompt_attention_mask"],
            combined["completion_attention_mask"],
        ),
        dim=1,
    )
    loss_mask = torch.cat(
        (
            torch.zeros_like(combined["prompt_attention_mask"]),
            combined["completion_attention_mask"],
        ),
        dim=1,
    )
    attention_mask, input_ids, loss_mask = flush_left(
        attention_mask, input_ids, loss_mask
    )
    if trainer.max_length is not None and input_ids.shape[1] > trainer.max_length:
        if trainer.truncation_mode == "keep_start":
            input_ids = input_ids[:, : trainer.max_length]
            attention_mask = attention_mask[:, : trainer.max_length]
            loss_mask = loss_mask[:, : trainer.max_length]
        elif trainer.truncation_mode == "keep_end":
            input_ids = input_ids[:, -trainer.max_length :]
            attention_mask = attention_mask[:, -trainer.max_length :]
            loss_mask = loss_mask[:, -trainer.max_length :]
            attention_mask, input_ids, loss_mask = flush_left(
                attention_mask, input_ids, loss_mask
            )
        else:
            raise ValueError(f"unsupported truncation mode: {trainer.truncation_mode}")

    labels = torch.roll(input_ids, shifts=-1, dims=1)
    shifted_loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()
    labels = torch.where(shifted_loss_mask, labels, torch.zeros_like(labels))
    mask_dtype = model.get_input_embeddings().weight.dtype
    converter = AttentionMaskConverter(is_causal=True)
    attention_mask_4d = converter.to_4d(
        attention_mask,
        input_ids.shape[1],
        dtype=mask_dtype,
        key_value_length=input_ids.shape[1],
    )
    attention_mask_4d = converter._unmask_unattended(
        attention_mask_4d, min_dtype=torch.finfo(mask_dtype).min
    )
    return {
        "input_ids": input_ids,
        # Pre-expanding the padding/causal mask avoids Transformers'
        # capture-incompatible `torch.all(mask == 1)` host decision.
        "attention_mask": attention_mask_4d,
        "labels": labels,
        "loss_mask": shifted_loss_mask,
        "num_examples": num_examples,
        "ref_chosen_logps": batch["ref_chosen_logps"],
        "ref_rejected_logps": batch["ref_rejected_logps"],
    }


def _dpo_loss_without_metrics(trainer, policy, batch):
    """Capture-safe DPO loss without padding or metric host synchronizations."""
    from trl.trainer.dpo_trainer import selective_log_softmax

    outputs = policy(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        output_hidden_states=True,
    )
    per_token_logps = selective_log_softmax(outputs.logits, batch["labels"])
    per_token_logps = torch.where(
        batch["loss_mask"], per_token_logps, torch.zeros_like(per_token_logps)
    )
    per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)
    all_logps = per_token_logps[:, 1:].sum(-1)
    n = batch["num_examples"]
    model_output = {
        "chosen_logps": all_logps[:n],
        "rejected_logps": all_logps[n:],
    }
    ref_chosen_logps = batch["ref_chosen_logps"]
    ref_rejected_logps = batch["ref_rejected_logps"]

    losses = 0
    for idx, loss_type in enumerate(trainer.loss_type):
        part, _, _ = trainer.dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
            loss_type,
            model_output,
        )
        weight = trainer.loss_weights[idx] if trainer.loss_weights else 1.0
        losses = losses + part * weight
    if trainer.args.rpo_alpha is not None:
        losses = losses + trainer.args.rpo_alpha * model_output["nll_loss"]
    if trainer.use_weighting:
        losses = losses * model_output["policy_weights"]
    if trainer.aux_loss_enabled:
        losses = losses + trainer.aux_loss_coef * model_output["aux_loss"]
    return losses.mean()


class _DpoLossModule(torch.nn.Module):
    """Tensor-only DPO region for torch.cuda.make_graphed_callables."""

    def __init__(self, trainer, policy, num_examples):
        super().__init__()
        self.trainer = trainer
        self.policy = policy
        self.num_examples = num_examples

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        loss_mask,
        ref_chosen_logps,
        ref_rejected_logps,
    ):
        return _dpo_loss_without_metrics(
            self.trainer,
            self.policy,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "loss_mask": loss_mask,
                "num_examples": self.num_examples,
                "ref_chosen_logps": ref_chosen_logps,
                "ref_rejected_logps": ref_rejected_logps,
            },
        )


def _capture_batch_args(batch):
    return (
        batch["input_ids"],
        batch["attention_mask"],
        batch["labels"],
        batch["loss_mask"],
        batch["ref_chosen_logps"],
        batch["ref_rejected_logps"],
    )


def run_explicit_dpo_graph_probe(lid, steps):
    """Compare an explicit fixed-address DPO region with its eager equivalent.

    This is deliberately an opt-in graphability probe, not a production trainer:
    it repeats two fixed microbatches so input addresses and shapes stay stable.
    The frozen-reference log probabilities are cached, as they can be in a real
    static-buffer adapter. PyTorch's training-specific wrapper captures policy
    forward and backward as separate graphs; gradient zeroing and bitsandbytes
    Adam remain eager. Adam's step counter is a host scalar and therefore cannot
    be captured correctly.
    """
    mode = os.environ["EXPLICIT_DPO_GRAPH_MODE"]
    warmup_steps = int(os.environ.get("EXPLICIT_DPO_WARMUP", "2"))
    seed = (int(lid) if str(lid).isdigit() else 0) + 100
    ds = make_prefs(seed, max(64, BATCH * 4))
    cfg = DPOConfig(
        per_device_train_batch_size=BATCH,
        max_steps=max(steps, 1),
        learning_rate=5e-5,
        logging_steps=max(1, steps),
        optim="adamw_8bit",
        seed=42,
        output_dir=os.environ.get("OUTBASE", "/root") + f"/dpo{lid}",
        report_to=[],
        beta=BETA,
        max_length=MAXSEQ,
        max_prompt_length=MAXSEQ // 2,
        remove_unused_columns=False,
        warmup_steps=0,
    )
    FastLanguageModel.for_training(model)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )

    # Unsloth's offloaded checkpoint implementation coordinates a legacy
    # default stream with an extra CUDA stream. CUDA rejects that cross-stream
    # dependency during capture, so this arm measures the memory/performance
    # tradeoff of capturing without activation checkpointing.
    FastLanguageModel.for_training(model, use_gradient_checkpointing=False)

    # The optimization normally computes this value via a GPU .item() inside
    # concatenated_forward. Disabling it removes that host synchronization and
    # keeps the semantically equivalent full logits needed by this probe.
    trainer.use_logits_to_keep = False
    data_iter = iter(trainer.get_train_dataloader())
    batches = [trainer._prepare_inputs(next(data_iter)) for _ in range(2)]
    for batch in batches:
        with torch.no_grad():
            ref_chosen, ref_rejected = trainer.compute_ref_log_probs(batch)
        batch["ref_chosen_logps"] = ref_chosen.detach()
        batch["ref_rejected_logps"] = ref_rejected.detach()
    batches = [_prepare_capture_batch(trainer, batch) for batch in batches]
    # Reuse one fixed-shape/static-address microbatch twice. A production
    # adapter would copy new same-bucket values into these buffers before each
    # replay; this probe isolates graph execution and numerical equivalence.
    batches = [batches[0], batches[0]]

    trainer.create_optimizer()
    optimizer = trainer.optimizer
    graph_runner = None

    def forward_backward():
        optimizer.zero_grad(set_to_none=False)
        step_losses = []
        for batch in batches:
            if graph_runner is None:
                loss = _dpo_loss_without_metrics(trainer, model, batch)
            else:
                loss = graph_runner(*_capture_batch_args(batch))
            trainer.accelerator.backward(loss / len(batches))
            step_losses.append(loss)
        return sum(step_losses) / len(step_losses)

    # Allocate all gradient and bitsandbytes optimizer state before capture.
    for _ in range(warmup_steps):
        warm_loss = forward_backward()
        optimizer.step()
    torch.cuda.synchronize()

    capture_ms = None
    if mode == "graph":
        capture_start = time.time()
        graph_module = _DpoLossModule(
            trainer, model, batches[0]["num_examples"]
        )
        graph_runner = torch.cuda.make_graphed_callables(
            graph_module,
            _capture_batch_args(batches[0]),
            num_warmup_iters=2,
        )
        torch.cuda.synchronize()
        capture_ms = round((time.time() - capture_start) * 1000, 2)

    losses = []
    started = time.time()
    for _ in range(steps):
        loss = forward_backward()
        # bitsandbytes uses a changing host-side Adam step counter, so keeping
        # this outside the graph is required for correct bias correction.
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    elapsed = time.time() - started

    trainable = [p.detach().float() for p in model.parameters() if p.requires_grad]
    parameter_sum = float(sum(p.sum() for p in trainable))
    parameter_l2 = float(torch.sqrt(sum((p * p).sum() for p in trainable)))
    record = {
        "mode": mode,
        "steps": steps,
        "microbatches_per_step": len(batches),
        "warmup_steps": warmup_steps,
        "capture_ms": capture_ms,
        "train_s": round(elapsed, 6),
        "losses": losses,
        "parameter_sum": parameter_sum,
        "parameter_l2": parameter_l2,
    }
    with open(f"{COORD}/explicit_graph_{lid}.json", "w") as f:
        json.dump(record, f, indent=2)
    print("SMOLVM_EXPLICIT_GRAPH_PROBE " + json.dumps(record), flush=True)
    return losses


# FORK mode: golden loads once, warms the DPO path, waits at a barrier; the host
# forks N share-weights clones; each resumes here and claims a distinct id.
if FORK:
    if os.environ.get("GOLDEN_WARMUP", "1") == "1":
        # One DPO step in the golden exercises the training write-path so the
        # daemon marks touched chunks private (see qlora_train GOLDEN_WARMUP).
        run_dpo("warm", 1)
        torch.cuda.synchronize()
    with open(f"{COORD}/golden_ready", "w") as f:
        f.write(str(round(load_ms)))
    while not os.path.exists(f"{COORD}/go"):
        time.sleep(0.2)
    claimed = None
    for k in range(int(os.environ.get("NSLOTS", "64"))):
        try:
            fd = os.open(f"{COORD}/claim_{k}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            claimed = k
            break
        except FileExistsError:
            continue
    LID = str(claimed)
LID = str(LID)

emit(LID, event="ready", load_ms=round(load_ms))
t = time.time()
losses = run_dpo(LID, STEPS)
dur = time.time() - t
toks = STEPS * BATCH * MAXSEQ
emit(LID, event="done", train_s=round(dur, 2), tok_s=round(toks / dur),
     step_ms=round(dur / STEPS * 1000), loss0=round(losses[0], 4),
     lossN=round(losses[-1], 4),
     peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
print(f"DPO LEARNER {LID} [{ARM}] DONE load={load_ms:.0f}ms "
      f"loss {losses[0]:.3f}->{losses[-1]:.3f} tok/s={toks/dur:.0f} "
      f"peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
