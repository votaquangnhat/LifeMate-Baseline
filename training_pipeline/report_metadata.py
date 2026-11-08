"""JSON conversion for runtime reports; prepared-data reuse for this report fix."""
import json
from pathlib import Path
from agent_core import write_json


def report_json_value(value):
    """Copy report metadata into JSON values without modifying live model config."""
    def encode_extra(obj):
        cls = type(obj)
        if cls.__module__ == "torch" and cls.__name__ in {"dtype", "device"}:
            return str(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Unsupported runtime report metadata: {cls.__module__}.{cls.__name__}")
    # No default=str for arbitrary objects; non-finite numbers still fail.
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, default=encode_extra))


def refresh_prepared_report_fingerprint(output_dir, current):
    """Reuse complete prepared rows when only reporting/runtime code changed.

    Input hashes, preprocessing code, tokenizer/model/reranker revisions, split,
    retrieval policy, length limits and package versions must match exactly.
    Never rewrite fingerprints of a run with a training recipe/checkpoints.
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
    old_hashes, new_hashes = old.get("code_sha256"), current.get("code_sha256")
    if not isinstance(old_hashes, dict) or not isinstance(new_hashes, dict):
        return False
    changed = {name for name in set(old_hashes)|set(new_hashes) if old_hashes.get(name) != new_hashes.get(name)}
    allowed = {"modeling_agent.py", "run_pipeline.py", "report_metadata.py", "selftest_report_metadata.py"}
    if not changed or changed-allowed:
        return False
    if (out/"training_recipe.json").exists() or (out/"training_complete.json").exists() or any((out/"checkpoints").glob("checkpoint-*")):
        return False
    expected = [out/f"{part}_{kind}.jsonl" for part in ("train","validation","test")
                for kind in ("items","tokens","references")]
    if not all(p.is_file() for p in expected) or not (out/"length_report.json").is_file() or not (out/"tokenizer").is_dir():
        return False
    write_json(marker, current)
    return True
