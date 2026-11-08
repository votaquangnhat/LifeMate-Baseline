"""One model per GPU; deterministic sharded rollout, resumable without reset."""
import argparse
import json
import os
import time
from pathlib import Path

from agent_core import (append_jsonl, import_tool_env, load_catalog, read_jsonl, rollout,
                        score_sample, sha256_file, write_json, write_jsonl)
from modeling_agent import load_model, make_generator


def evaluate(cfg, adapter, part, output):
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if part == "public_test":
        items = read_jsonl(Path(cfg["output_dir"]) / "public_test_items.jsonl")
        refs, env_split = None, "public_test"
        input_sha = sha256_file(cfg["public_transcripts_path"])
    else:
        items = read_jsonl(Path(cfg["output_dir"]) / f"{part}_items.jsonl")
        refs = {r["id"]: r for r in read_jsonl(Path(cfg["output_dir"]) / f"{part}_references.jsonl")}
        env_split = "train"
        input_sha = sha256_file(Path(cfg["output_dir"]) / f"{part}_items.jsonl")
    # No sampling of validation or test: every sample contributes to the mean.
    items = items[rank::world]
    is_base = adapter == "BASE"
    fingerprint = {"adapter_weights_sha256": "BASE" if is_base else sha256_file(Path(adapter) / "adapter_model.safetensors"),
                   "engine": "unsloth", "method": "qlora",
                   "full_attention": "native_unsloth_qwen3",
                   "prepared_sha256": sha256_file(Path(cfg["output_dir"]) / "prepared_fingerprint.json"),
                   "compiled_input_sha256": sha256_file(Path(cfg["output_dir"]) / f"{part}_items.jsonl"),
                   "runtime_versions_sha256": sha256_file(Path(cfg["output_dir"]) / "package_versions.json"),
                   "part": part, "world_size": world, "input_sha256": input_sha,
                   "max_length": cfg["max_length"], "max_new_tokens": cfg["max_new_tokens"],
                   "max_agent_steps": cfg["max_agent_steps"], "max_repairs": cfg["max_repairs"],
                   "model_revision": cfg["model_revision"], "global_meta_tools": cfg["global_meta_tools"]}
    marker = output / f"fingerprint.rank{rank}.json"
    if marker.exists() and json.loads(marker.read_text()) != fingerprint:
        raise ValueError("Cannot reuse an execution journal with changed weights, inputs, sharding or config")
    write_json(marker, fingerprint)
    journal = output / f"journal.rank{rank}.jsonl"
    records = read_jsonl(journal) if journal.exists() else []
    starts = {r["id"] for r in records if r["phase"] == "started"}
    finished = {r["id"]: r["result"] for r in records if r["phase"] == "finished"}
    if starts - finished.keys():
        raise RuntimeError("A sample started but has no committed result. Refusing to reset ToolEnv: " +
                           str(sorted(starts - finished.keys())[:10]))
    pending = [item for item in items if str(item["id"]) not in finished]
    if pending:
        ToolEnv = import_tool_env(cfg["data_root"], env_split)
        tools = load_catalog(cfg["data_root"], env_split)
        model, tokenizer = load_model(cfg, adapter_path=None if is_base else adapter, training=False)
        generate = make_generator(model, tokenizer, cfg)
        for i, item in enumerate(pending):
            sid = str(item["id"])
            append_jsonl(journal, {"phase": "started", "id": sid})
            # EXACTLY ONE session per prediction, no get_state() or state.json.
            env = ToolEnv(env_split, item["id"], root=Path(cfg["data_root"]))
            started_at = time.monotonic()
            result = rollout(item, tools, env, generate, cfg)
            result['planner_seconds'] = time.monotonic() - started_at
            result['timing_includes_asr_and_rerank'] = False
            if refs is not None:
                ref = refs[sid]
                result.update(score_sample(result["tool_calls"], ref["tool_calls"],
                                           result["state_hash"], ref["state_hash"], tools))
            append_jsonl(journal, {"phase": "finished", "id": sid, "result": result})
            finished[sid] = result
            print(f"rank={rank} {part} {i+1}/{len(pending)} {sid} score={result.get('score')} error={result['error']}", flush=True)
    ordered = [finished[str(item["id"])] for item in items]
    write_jsonl(output / f"results.rank{rank}.jsonl", ordered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--part", choices=["validation", "test", "public_test"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluate(json.loads(Path(args.config).read_text()), args.adapter, args.part, args.output)

