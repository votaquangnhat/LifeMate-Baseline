"""Replay references, compile label-free input once, then tokenize response-only SFT."""
import argparse
import collections
import copy
import json
from pathlib import Path

from agent_core import (add_observation, canonical_call, dumps, gold_replay, import_tool_env,
                        load_catalog, make_splits, read_jsonl, sha256_file, write_json, write_jsonl,
                        tokenize_sft, token_length_record, resolve_token_limits, apply_token_limits,
                        check_sft_limits)
from context_rerank import CachedReranker, compile_context, retrieval_signature


def fingerprint(cfg, source, catalog):
    return {"source":sha256_file(source), "catalog":sha256_file(catalog), "seed":cfg["seed"],
            "model_id":cfg["model_id"], "model_revision":cfg["model_revision"],
            "tokenizer_id":cfg["tokenizer_id"], "tokenizer_revision":cfg["tokenizer_revision"],
            "retrieval":retrieval_signature(cfg), "meta_tools":cfg["global_meta_tools"],
            "asr_manifest_sha":sha256_file(cfg["asr_manifest_path"]) if cfg.get("asr_manifest_path") else None,
            "hash_audio_for_split":cfg.get("hash_audio_for_split", True),
            "lengths":{k:cfg[k] for k in ("requested_max_length", "requested_max_new_tokens", "max_length_cap",
                                         "max_new_tokens_cap", "length_safety_margin", "auto_token_limits", "length_round_to")},
            "code_sha256": {p.name:sha256_file(p) for p in sorted(Path(cfg["code_dir"]).glob("*.py"))},
            "runtime_versions_sha256":sha256_file(Path(cfg["output_dir"]) / "package_versions.json"),
            "minimum_train_schema_coverage":cfg.get("minimum_train_schema_coverage", cfg.get("minimum_train_tool_recall", 1.0))}


def prepare(cfg):
    from transformers import AutoTokenizer
    root, out = Path(cfg["data_root"]), Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    source, catalog = root / "train/train.jsonl", root / "train/train_tools_catalog.json"
    mark = out / "prepared_fingerprint.json"
    sig = fingerprint(cfg, source, catalog)
    if mark.exists():
        if json.loads(mark.read_text()) != sig:
            raise ValueError("Input, retrieval or code changed. Choose a new OUTPUT_DIR; rerank cache may be copied separately.")
        expected = [out / f"{part}_{kind}.jsonl" for part in ("train", "validation", "test")
                    for kind in ("items", "tokens", "references")]
        if not all(p.is_file() for p in expected):
            raise ValueError("Prepared output is incomplete; use a new OUTPUT_DIR")
        report = json.loads((out / "length_report.json").read_text())
        apply_token_limits(cfg, report)
        print("Reusing identical compiled/tokenized data", flush=True)
        return
    items, tools = read_jsonl(source), load_catalog(root, "train")
    if len({str(x["id"]) for x in items}) != len(items):
        raise ValueError("Duplicate sample IDs")
    splits, method = make_splits(items, cfg)
    if set(splits.values()) != {"train", "validation", "test"}:
        raise ValueError("Empty split; inspect grouping")
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_id"], revision=cfg["tokenizer_revision"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    probe = tokenizer.apply_chat_template([{"role":"user", "content":"test"}], tokenize=False, add_generation_prompt=True)
    if "<think>" in probe or not probe.endswith("<|im_start|>assistant\n"):
        raise ValueError("Expected the native Qwen3 Instruct-2507 non-thinking template")
    if tokenizer.convert_tokens_to_ids("<|im_end|>") != tokenizer.eos_token_id:
        raise ValueError("Unexpected EOS token")
    tokenizer.save_pretrained(out / "tokenizer")
    ranker = CachedReranker(cfg)
    ToolEnv = import_tool_env(root, "train")
    token_rows, input_rows, refs = (collections.defaultdict(list) for _ in range(3))
    audits, lengths, previews = [], [], []
    counters = {p:collections.Counter() for p in ("train", "validation", "test")}
    for index, item in enumerate(items):
        part = splits[str(item["id"])]
        clean = {k:v for k,v in item.items() if k not in ("label", "final_state", "tool_results", "_agent_messages")}
        messages, audit = compile_context(clean, tools, ranker, tokenizer, cfg)
        # Selection is complete BEFORE reading labels/replaying the final request.
        observations, state_hash = gold_replay(item, ToolEnv, root, tools, cfg["global_meta_tools"])
        calls = [canonical_call(c) for c in item["label"]["tool_calls"]]
        clean["_agent_messages"] = copy.deepcopy(messages)
        input_rows[part].append(clean)
        refs[part].append({"id":str(item["id"]), "state_hash":state_hash, "tool_calls":calls})
        selected = set(audit["selected_tools"])
        visible = set(audit["visible_tools"])
        missing = [c["name"] for c in calls if c["name"] not in visible]
        primary_missing = [c["name"] for c in calls if c["name"] not in selected]
        count = counters[part]
        count.update(samples=1, gold_calls=len(calls), visible_gold_calls=len(calls)-len(missing),
                     primary_gold_calls=len(calls)-len(primary_missing),
                     fallback_gold_calls=sum(c["name"] in visible-selected for c in calls),
                     samples_all_schemas_available=int(not missing),
                     samples_all_primary_tools_kept=int(not primary_missing))
        # Save only train/validation gold recall here; held-out test is scored after selection.
        audits.append({**audit, "split":part,
                       **({"missing_gold_tools":missing, "gold_tools_outside_primary_set":primary_missing}
                          if part != "test" else {})})
        for step, target in enumerate(calls + [{"done":True}]):
            row = {"id":str(item["id"]), "step":step, "messages":copy.deepcopy(messages), "target":dumps(target)}
            encoded = tokenize_sft(row, tokenizer)
            lengths.append(token_length_record(row, encoded, part))
            token_rows[part].append(encoded)
            if len(previews) < 3:
                previews.append({**row, "length":encoded["length"]})
            if step < len(calls):
                add_observation(messages, calls[step], observations[step])
        if (index+1) % 50 == 0:
            print(f"Rerank + replay {index+1}/{len(items)} prompt {audit['original_prompt_tokens']} -> {audit['compiled_prompt_tokens']}", flush=True)
    ranker.close()  # Process exits before Qwen is loaded; no reranker in the train loop.
    def stats(values):
        values = sorted(values)
        return {"mean":sum(values)/len(values), "median":values[len(values)//2],
                "p95":values[min(len(values)-1, int(len(values)*0.95))], "max":values[-1]}
    retrieval = {"config":retrieval_signature(cfg), "splits":{},
                 "prompt_before":stats([a["original_prompt_tokens"] for a in audits]),
                 "prompt_after":stats([a["compiled_prompt_tokens"] for a in audits]),
                 "note":"Primary reranker recall and visible-schema coverage are separate. Compact fallback is applied to every sample without consulting gold labels. No rows are discarded."}
    for part in ("train", "validation"):
        c = counters[part]
        retrieval["splits"][part] = {**dict(c),
            "primary_gold_call_tool_recall":c["primary_gold_calls"]/c["gold_calls"] if c["gold_calls"] else 1,
            "schema_tool_coverage":c["visible_gold_calls"]/c["gold_calls"] if c["gold_calls"] else 1,
            "all_schemas_available_rate":c["samples_all_schemas_available"]/c["samples"],
            "primary_all_tools_kept_rate":c["samples_all_primary_tools_kept"]/c["samples"]}
    write_json(out / "retrieval_report.json", retrieval)
    write_jsonl(out / "retrieval_audit.jsonl", audits)
    print(json.dumps(retrieval, ensure_ascii=False, indent=2), flush=True)
    report = resolve_token_limits(lengths, cfg)
    write_json(out / "length_report.json", report)
    write_jsonl(out / "token_lengths.jsonl", lengths)
    print("Trajectory token lengths:", json.dumps(report["observed"],ensure_ascii=False), flush=True)
    print("Requested -> resolved token limits:", report["requested"], "->", report["resolved"], flush=True)
    train_coverage = retrieval["splits"]["train"]["schema_tool_coverage"]
    minimum_coverage = cfg.get("minimum_train_schema_coverage", cfg.get("minimum_train_tool_recall", 1.0))
    if train_coverage < minimum_coverage:
        raise ValueError(f"Visible tool schema coverage {train_coverage:.3%} below minimum {minimum_coverage:.3%}. "
                         "Use tool_schema_policy='compact_all' to retain compact schemas for every installed tool. "
                         "See retrieval_audit.jsonl; no gold-conditioned rescue is performed.")
    if report["errors"]:
        raise ValueError("; ".join(report["errors"]) + ". Inspect length_report.json. No request/target/observation was truncated.")
    for rows in token_rows.values():
        for row in rows:
            check_sft_limits(row, report["resolved"]["max_length"], report["resolved"]["max_new_tokens"])
    summary = {"split_method":method, "token_limits":report["resolved"], "splits":{}, "truncated_rows":0, "discarded_rows":0}
    for part in ("train", "validation", "test"):
        write_jsonl(out / f"{part}_items.jsonl", input_rows[part])
        write_jsonl(out / f"{part}_references.jsonl", refs[part])
        write_jsonl(out / f"{part}_tokens.jsonl", token_rows[part])
        summary["splits"][part] = {"samples":len(input_rows[part]), "actions":len(token_rows[part])}
    write_json(out / "sft_preview.json", previews)
    write_json(out / "split_assignments.json", splits)
    write_json(out / "data_summary.json", summary)
    apply_token_limits(cfg, report)
    write_json(mark, sig)  # Commit last.
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    prepare(json.loads(Path(p.parse_args().config).read_text()))
