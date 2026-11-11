import json
from pathlib import Path
from agent_core import read_jsonl,write_jsonl,write_json,metric_summary

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
        if cfg.get('history_router'):
            routed=[r['router'] for r in rows]
            summary['router']={'fallback_rate':sum((r['fallback_all_apps'] or r.get('fallback_menu_only',False)) for r in routed)/len(routed),
                'mean_qwen_calls':sum(r['qwen_calls'] for r in routed)/len(routed),
                'initial_gold_call_app_coverage':sum(r['initial_gold_call_app_coverage'] for r in routed)/len(routed),
                'final_gold_call_app_coverage':sum(r['final_gold_call_app_coverage'] for r in routed)/len(routed),
                'all_required_apps_available_rate':sum(r['final_all_required_apps_available'] for r in routed)/len(routed)}
        if cfg.get("thinking_mode"):
            events=[e for r in rows for e in r.get("thinking",[])]
            reasons=[e["reasoning_tokens"] for e in events if e["reasoning_tokens"] is not None]
            summary["thinking"]={"generations":len(events),
                "mean_reasoning_tokens":sum(reasons)/max(1,len(reasons)),
                "reasoning_cap_rate":sum(e["reasoning_cap_reached"] for e in events)/max(1,len(events)),
                "missing_eos":sum(not e["eos_seen"] for e in events)}
        write_json(directory / "metrics.json", summary)
        write_jsonl(directory / "error_review.jsonl", [r for r in rows if r["score"] < 1])
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    return {"samples": len(rows), "aborted_rollouts": sum(bool(r["error"]) for r in rows)}
