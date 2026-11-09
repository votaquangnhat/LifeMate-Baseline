"""Compile external public transcripts in a separate process before loading Qwen."""
import argparse
import json
from pathlib import Path
from agent_core import read_jsonl, write_jsonl, write_json, sha256_file, load_catalog
from context_rerank import CachedReranker, compile_context, retrieval_signature


def prepare_public(cfg):
    from transformers import AutoTokenizer
    root, out = Path(cfg["data_root"]), Path(cfg["output_dir"])
    if not cfg.get("public_transcripts_path"):
        raise ValueError("Public needs ASR transcripts; no gold text is assumed")
    source = root / "public_test/public_test.jsonl"
    transcripts = Path(cfg["public_transcripts_path"])
    catalog = root / "public_test/public_test_tools_catalog.json"
    sig = {"public_input":sha256_file(source), "transcripts":sha256_file(transcripts),
           "catalog":sha256_file(catalog), "retrieval":retrieval_signature(cfg),
           "global_meta_tools":cfg["global_meta_tools"],
           "tokenizer_id":cfg["tokenizer_id"], "tokenizer_revision":cfg["tokenizer_revision"],
           "code_sha256": {p.name:sha256_file(p) for p in sorted(Path(cfg["code_dir"]).glob("*.py"))}}
    marker, target = out / "public_rerank_fingerprint.json", out / "public_test_items.jsonl"
    if marker.exists():
        if json.loads(marker.read_text()) != sig:
            raise ValueError("Public inputs/retrieval changed; use a new output experiment")
        if target.is_file():
            return
    items, rows = read_jsonl(source), read_jsonl(transcripts)
    from asr_contract import merge_transcripts
    items = merge_transcripts(items, rows)
    tokenizer = AutoTokenizer.from_pretrained(out / "tokenizer")
    from qwen35_policy import configure_tokenizer
    configure_tokenizer(tokenizer, cfg)
    tools, ranker = load_catalog(root, "public_test"), CachedReranker(cfg)
    audits, compiled = [], []
    for index, original in enumerate(items):
        item = {k:v for k,v in original.items() if k not in ("label", "final_state", "tool_results", "_agent_messages")}
        messages, audit = compile_context(item, tools, ranker, tokenizer, cfg)
        item["_agent_messages"] = messages
        compiled.append(item)
        audits.append(audit)
        if (index+1)%50 == 0:
            print(f"Public rerank {index+1}/{len(items)}", flush=True)
    ranker.close()
    write_jsonl(target, compiled)
    write_jsonl(out / "public_retrieval_audit.jsonl", audits)
    write_json(marker, sig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    prepare_public(json.loads(Path(p.parse_args().config).read_text()))
