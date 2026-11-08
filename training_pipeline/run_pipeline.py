"""Notebook driver: subprocess GPU isolation, metric selection and exports."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from agent_core import metric_summary, read_jsonl, training_signature, write_json, write_jsonl
from distributed_utils import launch_spec
from report_metadata import refresh_prepared_report_fingerprint
from amp_health import recover_prepared_after_amp_fix


def run_gpu(cfg, script, arguments, log_name):
    out, code = Path(cfg["output_dir"]), Path(cfg["code_dir"])
    command, devices = launch_spec(cfg, script, arguments)
    env = os.environ.copy()
    env.update(TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1")
    env["CUDA_VISIBLE_DEVICES"] = devices
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # Every rerun retains its old log.
    path = logs / log_name
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND " + repr(command) + "\n")
        handle.flush()
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"GPU job exited {code}. See {path}; configuration was not automatically changed.")


def collect(cfg, part, directory):
    directory = Path(directory)
    rows = []
    for rank in range(cfg["eval_num_gpus"]):
        rows.extend(read_jsonl(directory / f"results.rank{rank}.jsonl"))
    by_id = {r["id"]: r for r in rows}
    if part == "public_test":
        source = Path(cfg["data_root"]) / "public_test" / "public_test.jsonl"
    else:
        source = Path(cfg["output_dir"]) / f"{part}_items.jsonl"
    ids = [str(r["id"]) for r in read_jsonl(source)]
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError("Predictions must cover each input exactly once")
    rows = [by_id[sid] for sid in ids]
    write_jsonl(directory / "predictions_with_trace.jsonl", rows)
    write_jsonl(directory / "submission.jsonl", [{k: row[k] for k in ("id", "tool_calls", "state_hash")} for row in rows])
    if part != "public_test":
        summary = metric_summary(rows)
        write_json(directory / "metrics.json", summary)
        write_jsonl(directory / "error_review.jsonl", [r for r in rows if r["score"] < 1])
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    return {"samples": len(rows), "aborted_rollouts": sum(bool(r["error"]) for r in rows)}


def train(cfg):
    marker = Path(cfg["output_dir"]) / "training_complete.json"
    if marker.exists():
        old = json.loads(marker.read_text())["recipe"]
        if old != training_signature(cfg):
            raise ValueError("Saved training recipe differs from current config/data")
        print("Training already completed; reusing its checkpoints.", flush=True)
        return
    if not (Path(cfg["output_dir"]) / "preflight_report.json").is_file():
        raise ValueError("Run the GPU preflight cell first")
    report = json.loads((Path(cfg["output_dir"]) / "preflight_report.json").read_text())
    if report["recipe"] != training_signature(cfg):
        raise ValueError("Configuration changed since preflight. Run the preflight cell again.")
    if not report.get("passed") or any(r["allocated_fraction"] > cfg["vram_limit"] for r in report["gpu_memory"]):
        raise ValueError("The GPU preflight did not pass its VRAM margin")
    run_gpu(cfg, "train_agent.py", [], "train.log")


def baseline(cfg):
    directory = Path(cfg["output_dir"]) / "baseline_validation"
    run_gpu(cfg, "rollout_agent.py", ["--adapter", "BASE", "--part", "validation", "--output", directory],
            "baseline_validation.log")
    summary = collect(cfg, "validation", directory)
    write_json(Path(cfg["output_dir"]) / "baseline_metrics.json", summary)


def select(cfg):
    out = Path(cfg["output_dir"])
    checkpoints = sorted((out / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    # Select only epoch boundaries and the final checkpoint, not every resume snapshot.
    candidates = []
    for checkpoint in checkpoints:
        state_path = checkpoint / "trainer_state.json"
        if not state_path.is_file() or not (checkpoint / "adapter_model.safetensors").is_file():
            continue
        epoch = float(json.loads(state_path.read_text()).get("epoch") or 0)
        if abs(epoch - round(epoch)) < 1e-6 or checkpoint == checkpoints[-1]:
            candidates.append(checkpoint)
    checkpoints = candidates
    if not checkpoints or not (out / "training_complete.json").exists():
        raise ValueError("Complete training before model selection")
    # Keep yesterday's model if none of today's checkpoints improves validation.
    if cfg.get('initial_adapter'):
        checkpoints.insert(0, Path(cfg['initial_adapter']))
    summaries = []
    for checkpoint in checkpoints:
        directory = out / "validation_rollouts" / checkpoint.name
        run_gpu(cfg, "rollout_agent.py", ["--adapter", checkpoint, "--part", "validation", "--output", directory],
                f"validation_{checkpoint.name}.log")
        summary = collect(cfg, "validation", directory)
        summaries.append({"checkpoint": str(checkpoint), **summary})
    # Stable tie-break: earlier checkpoint. Internal test is never consulted.
    winner = max(summaries, key=lambda row: row["official_score_100"])
    best = out / "best_adapter"
    best.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(Path(winner["checkpoint"]) / name, best / name)
    shutil.copytree(out / "tokenizer", best, dirs_exist_ok=True)
    write_json(out / "selection.json", {"criterion": "full validation official_score_100",
                "tie_break": "previous best first, then earliest checkpoint", "winner": winner, "all_checkpoints": summaries})
    write_json(best / "agent_config.json", cfg)
    print("BEST", json.dumps(winner, indent=2), flush=True)


def final_eval(cfg, public=False):
    out = Path(cfg["output_dir"])
    if not (out / "selection.json").is_file():
        raise ValueError("Select best by validation first")
    part = "public_test" if public else "test"
    if public:
        run_gpu(cfg, "prepare_public.py", [], "public_rerank.log")
    directory = out / (part + "_best")
    run_gpu(cfg, "rollout_agent.py", ["--adapter", out / "best_adapter", "--part", part, "--output", directory],
            part + ".log")
    summary = collect(cfg, part, directory)
    if public:
        # Official submission schema validation runs with PUBLIC's zaic_tools,
        # in a fresh process; there is no reset or replay of prediction sessions.
        env = os.environ.copy()
        subprocess.run([sys.executable, "-m", "zaic_tools", "--split", "public_test", "--root", cfg["data_root"],
                        "--validate", str(directory / "submission.jsonl")],
                       cwd=str(Path(cfg["data_root"]) / "public_test"), env=env, check=True)
    write_json(out / ("public_summary.json" if public else "test_metrics.json"), summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["prepare", "smoke", "baseline", "train", "select", "test", "public"], required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.stage == "prepare":
        from prepare_agent import fingerprint as prepared_fingerprint
        root = Path(cfg["data_root"])
        current = prepared_fingerprint(cfg, root/"train/train.jsonl", root/"train/train_tools_catalog.json")
        if recover_prepared_after_amp_fix(cfg["output_dir"], current):
            print("AMP training fix only: reused prepared rows; archived old attempt metadata. Run preflight, then train from the base model.", flush=True)
        if refresh_prepared_report_fingerprint(cfg["output_dir"], current):
            print("Prepared inputs and preprocessing are unchanged; reusing cached rows after the report serialization fix.", flush=True)
        run_gpu(cfg, "prepare_agent.py", [], "prepare_rerank.log")
    elif args.stage == "smoke":
        (Path(cfg["output_dir"]) / "preflight_report.json").unlink(missing_ok=True)
        run_gpu(cfg, "train_agent.py", ["--smoke"], "preflight.log")
    elif args.stage == "baseline":
        baseline(cfg)
    elif args.stage == "train":
        train(cfg)
    elif args.stage == "select":
        select(cfg)
    else:
        final_eval(cfg, public=args.stage == "public")

