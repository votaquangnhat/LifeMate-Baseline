"""Shared launch and batch accounting for preprocessing, DDP train and rollout."""
import math
import sys
from pathlib import Path


def launch_spec(cfg, script, arguments):
    evaluation = script == "rollout_agent.py"
    rerank = script in ("prepare_agent.py", "prepare_public.py")
    training = script == "train_agent.py"
    count = int(cfg["eval_num_gpus"]) if evaluation else int(cfg["num_gpus"]) if training else 1
    if count < 1:
        raise ValueError("GPU process count must be positive")
    command = [sys.executable, "-u"]
    if count > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(count)]
    command += [str(Path(cfg["code_dir"]) / script), "--config", cfg["config_path"]]
    command += [str(a) for a in arguments]
    devices = str(cfg["rerank_gpu"]) if rerank else ",".join(str(i) for i in range(count))
    return command, devices


def training_plan(cfg, samples):
    world, batch, accumulation = (int(cfg[k]) for k in
                                  ("num_gpus", "per_gpu_batch_size", "gradient_accumulation_steps"))
    if min(world, batch, accumulation) < 1:
        raise ValueError("World size, batch and accumulation must be positive")
    effective = world * batch * accumulation
    # Accelerate may pad the last distributed batch; report the rounded update count.
    return {"actions":samples, "epochs":cfg["epochs"], "effective_batch":effective,
            "updates_per_epoch":math.ceil(math.ceil(samples/(world*batch))/accumulation),
            "selection":"full validation ToolEnv rollout score", "train_gpus":world,
            "per_gpu_batch_size":batch, "gradient_accumulation_steps":accumulation}


def smoke_dataset(cases, cfg):
    # Accelerate shards sequential batches round-robin across ranks. Each case
    # spans exactly one full accumulated optimizer update on EVERY rank.
    copies = int(cfg["num_gpus"]) * int(cfg["per_gpu_batch_size"]) * int(cfg["gradient_accumulation_steps"])
    return [row for row in cases for _ in range(copies)]
