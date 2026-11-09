"""DDP on two T4 GPUs; native FP16 QLoRA; same distributed recipe in preflight."""
from modeling_agent import ResponseCollator, load_model, amp_mode
import argparse
import hashlib
import inspect
import json
import math
import os
import time
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
from agent_core import read_jsonl, training_signature, write_json
from distributed_utils import training_plan, smoke_dataset
from amp_health import AMPHealth, initialize_grad_scaler, AMP_INIT_SCALE


class TokenDataset(Dataset):
    def __init__(self, path):
        self.rows = read_jsonl(path)
        if not self.rows:
            raise ValueError("Empty token dataset")
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return self.rows[index]


class EpochSaveCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True


class HealthCallback(TrainerCallback):
    def __init__(self, accelerator, output_dir):
        self.accelerator = accelerator
        self.output_dir = Path(output_dir)
        self.health = AMPHealth(accelerator.mixed_precision == "fp16")

    def summary(self):
        scaler = getattr(self.accelerator, "scaler", None)
        scale = float(scaler.get_scale()) if scaler is not None and scaler.is_enabled() else None
        if scale is not None and (not math.isfinite(scale) or scale <= 0):
            raise FloatingPointError("Invalid GradScaler scale")
        return {**self.health.summary(), "scale":scale, "fresh_run_init_scale":AMP_INIT_SCALE}

    def save_report(self):
        rank = int(os.environ.get("RANK", "0"))
        write_json(self.output_dir / f"amp_health.rank{rank}.json", self.summary())

    def on_optimizer_step(self, args, state, control, **kwargs):
        # This hook runs AFTER AcceleratedOptimizer.step + scaler.update.
        # A non-finite pre-clip norm is expected when AMP has skipped this update.
        skipped = self.accelerator.optimizer_step_was_skipped
        try:
            self.health.optimizer_step(skipped)
        finally:
            if skipped:
                self.save_report()
        if skipped and self.accelerator.is_main_process:
            print(json.dumps({"amp_overflow":True, "step":state.global_step+1, **self.summary()}), flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.health.check_logs(logs or {})

    def on_train_end(self, args, state, control, **kwargs):
        self.save_report()


def build_args(cfg, smoke=False):
    params = inspect.signature(TrainingArguments).parameters
    kwargs = dict(output_dir=str(Path(cfg["output_dir"]) / ("smoke_tmp" if smoke else "checkpoints")),
                  per_device_train_batch_size=cfg["per_gpu_batch_size"], per_device_eval_batch_size=1,
                  gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
                  num_train_epochs=cfg["epochs"], learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
                  lr_scheduler_type="cosine", max_grad_norm=cfg["max_grad_norm"],
                  fp16=amp_mode()=="fp16", bf16=amp_mode()=="bf16", tf32=False,
                  gradient_checkpointing=False, optim=cfg.get("optimizer", "adamw_8bit"),
                  eval_strategy="no" if smoke else "epoch", save_strategy="no" if smoke else "steps", save_steps=cfg.get("save_steps", 100),
                  load_best_model_at_end=False, save_total_limit=None,
                  prediction_loss_only=True, logging_steps=1 if smoke else cfg["logging_steps"],
                  logging_nan_inf_filter=False, report_to="none", remove_unused_columns=False,
                  dataloader_num_workers=0, dataloader_pin_memory=True, seed=cfg["seed"], data_seed=cfg["seed"],
                  ddp_find_unused_parameters=False, ddp_broadcast_buffers=False)
    # Transformers 5 moved warmup ratio support to warmup_steps; 4.x expects integer steps.
    if "warmup_ratio" in params:
        kwargs["warmup_ratio"] = cfg["warmup_ratio"]
    else:
        kwargs["warmup_steps"] = cfg["warmup_ratio"]
    if "group_by_length" in params:
        kwargs["group_by_length"] = not smoke
    if smoke:
        kwargs.update(max_steps=cfg["smoke_steps"], disable_tqdm=True)
    unknown = set(kwargs)-set(params)
    if unknown:
        raise ValueError(f"Unsupported TrainingArguments: {unknown}; inspect package_versions.json")
    return TrainingArguments(**kwargs)


def make_trainer(cfg, model, tokenizer, training, validation=None, smoke=False):
    kwargs = dict(model=model, args=build_args(cfg, smoke), train_dataset=training,
                  eval_dataset=validation, data_collator=ResponseCollator(tokenizer.pad_token_id),
                  callbacks=[] if smoke else [EpochSaveCallback()])
    key = "processing_class" if "processing_class" in inspect.signature(Trainer).parameters else "tokenizer"
    kwargs[key] = tokenizer
    trainer = (SmokeTrainer if smoke else Trainer)(**kwargs)
    initialize_grad_scaler(trainer.accelerator)
    trainer.amp_health_callback = HealthCallback(trainer.accelerator, cfg["output_dir"])
    trainer.add_callback(trainer.amp_health_callback)
    return trainer


class SmokeTrainer(Trainer):
    def _get_train_sampler(self, *args, **kwargs):
        # Accelerate performs distributed sharding; do not shard the rows twice.
        return torch.utils.data.SequentialSampler(self.train_dataset)


def rank_info(cfg):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world != cfg["num_gpus"]:
        raise ValueError(f"Expected {cfg['num_gpus']} training ranks, WORLD_SIZE={world}. Launch through torchrun.")
    torch.cuda.set_device(local)
    return rank, world, local


def check_ddp(trainer, cfg):
    world = cfg["num_gpus"]
    if trainer.accelerator.num_processes != world:
        raise RuntimeError("Accelerate process count does not match the DDP recipe")
    if world > 1:
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() != world:
            raise RuntimeError("DDP process group was not initialized")
        if not isinstance(trainer.model_wrapped, torch.nn.parallel.DistributedDataParallel):
            raise RuntimeError("Expected a DistributedDataParallel model, not independent training replicas")


def adapter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def preflight(cfg):
    rank, world, local_rank = rank_info(cfg)
    set_seed(cfg["seed"])
    out = Path(cfg["output_dir"])
    model, tokenizer = load_model(cfg, adapter_path=cfg.get("initial_adapter") or None, training=True)
    rows = TokenDataset(out / "train_tokens.jsonl")
    ordered = sorted(rows.rows, key=lambda r:r["length"])
    cases = [ordered[len(ordered)//2], ordered[-1], max(rows.rows, key=lambda r:sum(t!=-100 for t in r["labels"]))]
    for part in ("validation", "test"):
        longest = max(read_jsonl(out / f"{part}_tokens.jsonl"), key=lambda r:r["length"])
        cases.append(longest)
    if cfg["smoke_steps"] != len(cases):
        raise ValueError("smoke_steps must cover all five stress cases")
    stress = smoke_dataset(cases, cfg)
    trainer = make_trainer(cfg, model, tokenizer, stress, smoke=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = trainer.train()
    check_ddp(trainer, cfg)
    torch.cuda.synchronize()
    total = torch.cuda.get_device_properties(model.device).total_memory
    peak = torch.cuda.max_memory_allocated()
    local = {"rank":rank, "local_rank":local_rank, "world_size":world, "allocated_fraction":peak/total,
             "peak_allocated_gib":peak/2**30, "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,
             "total_gib":total/2**30, "finite_loss":math.isfinite(result.metrics["train_loss"]),
             "adapter_sha256":adapter_digest(model), "seconds":time.monotonic()-started,
             "amp_health":trainer.amp_health_callback.summary()}
    gathered = [None]*world
    if world > 1:
        torch.distributed.all_gather_object(gathered, local)
    else:
        gathered = [local]
    synchronized = len({r["adapter_sha256"] for r in gathered}) == 1
    report = {"passed":synchronized and all(r["finite_loss"] and r["amp_health"]["successful_steps"] > 0
              and r["allocated_fraction"] <= cfg["vram_limit"] for r in gathered),
              "recipe":training_signature(cfg), "metrics":result.metrics,
              "seconds":max(r["seconds"] for r in gathered), "gpu_memory":gathered,
              "ddp_world_size":world, "adapter_weights_synchronized":synchronized,
              "effective_batch":training_plan(cfg, len(stress))["effective_batch"],
              "stress_lengths_per_rank":[r["length"] for r in cases],
              "note":"DDP backward, real accumulation, gradient synchronization and optimizer states checked on every GPU. Smoke weights are discarded."}
    write_json(out / f"preflight.rank{rank}.json", local)
    if rank == 0:
        write_json(out / "preflight_report.json", report)
        print(json.dumps(report, indent=2), flush=True)
    trainer.accelerator.wait_for_everyone()
    if not report["passed"]:
        raise RuntimeError("Preflight failed. No automatic batch/sequence reduction was performed.")


def train(cfg):
    rank, world, local_rank = rank_info(cfg)
    out = Path(cfg["output_dir"])
    recipe = training_signature(cfg)
    recipe_file = out / "training_recipe.json"
    if recipe_file.exists() and json.loads(recipe_file.read_text()) != recipe:
        raise ValueError("Recipe/data changed. Use a new OUTPUT_DIR")
    resume = cfg.get("resume_checkpoint", "")
    if resume == "auto":
        checkpoints = sorted((out / "checkpoints").glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[-1]))
        complete = [p for p in checkpoints if all((p / f).is_file() for f in
                    ("trainer_state.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt"))]
        resume = str(complete[-1]) if complete else ""
        if checkpoints and not complete:
            raise ValueError("No complete checkpoint. Inspect the interrupted save; do not resume partial weights.")
    if not resume and any((out / "checkpoints").glob("checkpoint-*")):
        raise ValueError("Existing checkpoints: set RESUME_CHECKPOINT or choose a new OUTPUT_DIR")
    if resume:
        old = Path(resume).parent.parent / "training_recipe.json"
        if not old.is_file() or json.loads(old.read_text()) != recipe:
            raise ValueError("Resume checkpoint requires identical recipe/data")
    set_seed(cfg["seed"])
    training, validation = TokenDataset(out / "train_tokens.jsonl"), TokenDataset(out / "validation_tokens.jsonl")
    plan = training_plan(cfg, len(training))
    if rank == 0:
        write_json(out / "training_plan.json", plan)
        write_json(recipe_file, recipe)
    # Validate the installed Trainer API before allocating the model.
    build_args(cfg)
    model, tokenizer = load_model(cfg, adapter_path=cfg.get("initial_adapter") or None, training=True)
    trainer = make_trainer(cfg, model, tokenizer, training, validation)
    if trainer.accelerator.num_processes != world:
        raise RuntimeError("Trainer did not initialize the requested distributed world")
    if rank == 0:
        print(json.dumps(plan, indent=2), flush=True)
    result = trainer.train(resume_from_checkpoint=resume or None)
    check_ddp(trainer, cfg)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        trainer.save_state()
        write_json(out / "training_complete.json", {"recipe":recipe, "metrics":result.metrics})
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    # NCCL is not fork-safe; keep 1 bounded-prefetch DataLoader worker per rank.
    torch.multiprocessing.set_start_method("spawn", force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    try:
        preflight(cfg) if a.smoke else train(cfg)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
