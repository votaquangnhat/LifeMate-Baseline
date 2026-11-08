"""Monitor actual AMP skips, preserve scaler state, and reuse compatible prepared rows."""
import json
import math
import shutil
import time
from pathlib import Path
from agent_core import write_json

AMP_INIT_SCALE = 1024.0
MAX_CONSECUTIVE_AMP_SKIPS = 8


def initialize_grad_scaler(accelerator):
    """Set a conservative fresh-run scale via the public state API.

    Called before Trainer.train / optimizer preparation. Checkpoint loading in
    Trainer.train still restores the checkpoint's own scaler state afterwards.
    """
    scaler = getattr(accelerator, "scaler", None)
    if accelerator.mixed_precision != "fp16":
        return False
    if scaler is None or not scaler.is_enabled():
        raise RuntimeError("FP16 requires an enabled GradScaler")
    state = dict(scaler.state_dict())
    state.update(scale=AMP_INIT_SCALE, _growth_tracker=0)
    scaler.load_state_dict(state)
    return True


class AMPHealth:
    def __init__(self, enabled, max_consecutive_skips=MAX_CONSECUTIVE_AMP_SKIPS):
        self.enabled = bool(enabled)
        self.limit = max_consecutive_skips
        self.attempted_steps = self.successful_steps = self.skipped_steps = 0
        self.consecutive_skips = 0
        self.last_skipped = False

    def optimizer_step(self, skipped):
        self.last_skipped = bool(skipped)
        self.attempted_steps += 1
        if skipped:
            if not self.enabled:
                raise FloatingPointError("Optimizer skipped without FP16 GradScaler")
            self.skipped_steps += 1
            self.consecutive_skips += 1
            if self.consecutive_skips >= self.limit:
                raise FloatingPointError(f"Persistent AMP overflow: {self.consecutive_skips} consecutive skipped updates")
        else:
            self.successful_steps += 1
            self.consecutive_skips = 0

    def check_logs(self, logs):
        for key in ("loss", "eval_loss"):
            if key in logs and not math.isfinite(float(logs[key])):
                raise FloatingPointError("Non-finite " + key)
        if "grad_norm" in logs and not math.isfinite(float(logs["grad_norm"])):
            if not (self.enabled and self.last_skipped):
                raise FloatingPointError("Non-finite grad_norm without a confirmed AMP skip")

    def summary(self):
        return {"amp_enabled":self.enabled, "attempted_steps":self.attempted_steps,
                "successful_steps":self.successful_steps, "skipped_steps":self.skipped_steps,
                "consecutive_skips":self.consecutive_skips}


def recover_prepared_after_amp_fix(output_dir, current):
    """Retain prepared rows for this training-only fix; archive an uncheckpointed attempt.

    The input/config/package fingerprint and every preprocessing module must
    match. Never reuse/resume trained checkpoints under a different recipe.
    """
    out = Path(output_dir)
    marker = out / "prepared_fingerprint.json"
    if not marker.is_file():
        return False
    old = json.loads(marker.read_text())
    if old == current:
        return False
    if {k:v for k,v in old.items() if k != "code_sha256"} != {k:v for k,v in current.items() if k != "code_sha256"}:
        return False
    before, after = old.get("code_sha256"), current.get("code_sha256")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False
    changed = {k for k in set(before)|set(after) if before.get(k) != after.get(k)}
    allowed = {"train_agent.py", "run_pipeline.py", "amp_health.py", "selftest_amp_health.py"}
    if not changed or changed-allowed:
        return False
    if (out/"training_complete.json").exists() or any((out/"checkpoints").glob("checkpoint-*")):
        return False
    expected = [out/f"{p}_{k}.jsonl" for p in ("train","validation","test") for k in ("items","tokens","references")]
    if not all(p.is_file() for p in expected) or not (out/"length_report.json").is_file() or not (out/"tokenizer").is_dir():
        return False
    archive = out / "failed_runs" / ("before_ampfix_" + str(time.time_ns()))
    archive.mkdir(parents=True)
    shutil.copy2(marker, archive/marker.name)
    # A new preflight and a fresh training run are mandatory after code changes.
    for name in ("training_recipe.json", "training_plan.json", "preflight_report.json"):
        if (out/name).is_file():
            (out/name).rename(archive/name)
    for pattern in ("preflight.rank*.json", "amp_health.rank*.json"):
        for path in out.glob(pattern):
            path.rename(archive/path.name)
    write_json(marker, current)
    return True
