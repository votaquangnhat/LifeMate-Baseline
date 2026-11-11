"""One model per GPU; deterministic sharded rollout, resumable at committed sample boundaries."""
import argparse
import json
import os
from pathlib import Path

from agent_core import (import_tool_env, load_catalog, read_jsonl, rollout,
                        score_sample, sha256_file, write_json, write_jsonl)
from modeling_agent import load_model, make_generator
from tqdm.auto import tqdm
from session_runtime import SessionPaused, stop_if_deadline
from thinking_policy import settings


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
    is_lora = cfg.get("model_family") == "qwen3_5"
    fingerprint = {"adapter_weights_sha256": "BASE" if is_base else sha256_file(Path(adapter) / "adapter_model.safetensors"),
                    "engine": "unsloth", "method": "lora" if is_lora else "qlora",
                   "full_attention": "model_default" if is_lora else "native_unsloth_qwen3",
                   "prepared_sha256": sha256_file(Path(cfg["output_dir"]) / "prepared_fingerprint.json"),
                   "compiled_input_sha256": sha256_file(Path(cfg["output_dir"]) / f"{part}_items.jsonl"),
                   "runtime_versions_sha256": sha256_file(Path(cfg["output_dir"]) / "package_versions.json"),
                   "part": part, "world_size": world, "input_sha256": input_sha,
                   "max_length": cfg["max_length"], "max_new_tokens": cfg["max_new_tokens"],
                   "max_agent_steps": cfg["max_agent_steps"], "max_repairs": cfg["max_repairs"],
                   "model_revision": cfg["model_revision"], "global_meta_tools": cfg["global_meta_tools"],
                   "thinking": settings(cfg)}
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
        for i, item in enumerate(tqdm(pending,desc=f"{part} rank {rank}",unit="sample",position=rank)):
            try:
                stop_if_deadline(cfg)
            except SessionPaused:
                break
            sid = str(item["id"])
            # An uncommitted attempt can be restarted from the sample's initial
            # simulator state. A completed sample is always reused verbatim.
            env = ToolEnv(env_split, item["id"], root=Path(cfg["data_root"]))
            result = rollout(item, tools, env, generate, cfg)
            if refs is not None:
                ref = refs[sid]
                result.update(score_sample(result["tool_calls"], ref["tool_calls"],
                                           result["state_hash"], ref["state_hash"], tools))
                if cfg.get('history_router'):
                    required=[c['name'].split('.',1)[0] for c in ref['tool_calls'] if not c['name'].startswith('sys.')]
                    for phase,key in (('initial','initial_apps'),('final','final_apps')):
                        visible=set(result['router'][key])
                        result['router'][phase+'_gold_call_app_coverage']=sum(a in visible for a in required)/len(required) if required else 1.
                    result['router']['final_all_required_apps_available']=all(a in result['router']['final_apps'] for a in required)
                    opened=set(result['router']['final_apps'])
                    executed={c['name'].split('.',1)[0] for c in result['tool_calls']}
                    result['router']['never_opened_gold_apps']=sorted(set(required)-opened)
                    result['router']['opened_but_not_executed_gold_apps']=sorted(set(required)&opened-executed)
            records.extend([{"phase":"started","id":sid},
                            {"phase":"finished","id":sid,"result":result}])
            # Atomic replacement prevents a truncated journal on a hard deadline.
            write_jsonl(journal, records)
            finished[sid] = result
            print(f"rank={rank} {part} {i+1}/{len(pending)} {sid} score={result.get('score')} error={result['error']}", flush=True)
    complete=all(str(item["id"]) in finished for item in items)
    write_json(output/f"rollout_status.rank{rank}.json",{"complete":complete,"total":len(items),
               "finished":sum(str(item["id"]) in finished for item in items)})
    if complete:
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

