"""Label-free multilingual tool/history selection, shared by train and rollout."""
import copy
import hashlib
import json
import math
import re
import sqlite3
from itertools import islice
from pathlib import Path

from agent_core import dumps, initial_messages, installed_tool_names, prompt_ids

COMPILER_VERSION = "qwen3-rerank-v1.2-compact-all-schemas"
RETRIEVAL_KEYS = ("reranker_id", "reranker_revision", "rerank_max_length", "rerank_query_tokens",
                  "rerank_batch_size", "top_k_apps", "top_k_tools", "top_k_history",
                  "keep_recent_history", "schema_description_chars", "prompt_target_tokens",
                  "tool_dependencies")


def retrieval_signature(cfg):
    return {"compiler": COMPILER_VERSION, **{k: cfg[k] for k in RETRIEVAL_KEYS},
            "tool_schema_policy":cfg.get("tool_schema_policy", "compact_all")}


def encode_rerank_pairs(tokenizer, pairs, max_length, return_tensors=None):
    """Public string-pair API supported by both Transformers 4.x and 5.x."""
    if not pairs:
        raise ValueError("Cannot encode an empty pair batch")
    queries, documents = zip(*pairs)
    return tokenizer(text=list(queries), text_pair=list(documents), add_special_tokens=True,
                     padding=True, truncation=True, max_length=max_length,
                     return_attention_mask=True, return_tensors=return_tensors)


def text_windows(tokenizer, text, width):
    # Long input is intentional here; every window is bounded again for the model.
    ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    if len(ids) <= width:
        return [text]
    return [tokenizer.decode(window, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            for window in CachedReranker.windows(ids, width)]


def iter_rerank_pairs(tokenizer, query, documents, indices, max_length, query_tokens):
    document_tokens = max_length-query_tokens-tokenizer.num_special_tokens_to_add(pair=True)
    if min(query_tokens, document_tokens) < 64:
        raise ValueError("rerank_max_length leaves insufficient query/document space")
    queries = text_windows(tokenizer, query, query_tokens)
    for index in indices:
        windows = text_windows(tokenizer, documents[index], document_tokens)
        for q in queries:
            for d in windows:
                yield index, (q, d)


def rerank_pair_batches(tokenizer, query, documents, indices, cfg):
    size = int(cfg["rerank_batch_size"])
    if size < 1:
        raise ValueError("rerank_batch_size must be positive")
    iterator = iter_rerank_pairs(tokenizer, query, documents, indices,
                                 cfg["rerank_max_length"], cfg["rerank_query_tokens"])
    while True:
        records = list(islice(iterator, size))
        if not records:
            return
        yield [pair for _,pair in records], [owner for owner,_ in records]


def check_reranker_tokenizer(tokenizer, max_length):
    """Fail on tokenizer incompatibility before downloading/loading model weights."""
    pairs = [("Nhắc tôi gọi cho mẹ ngày mai", "Tạo nhắc việc với ngày giờ và tiêu đề"),
             ("Tìm số điện thoại của An", "Tìm liên hệ theo tên và trả về ID")]
    batch = encode_rerank_pairs(tokenizer, pairs, max_length)
    if len(batch["input_ids"]) != len(pairs) or "attention_mask" not in batch:
        raise ValueError("Reranker tokenizer must return model-ready pair input_ids and masks")
    if any(not ids or len(ids)>max_length for ids in batch["input_ids"]):
        raise ValueError("Unexpected reranker pair length")
    return {"tokenizer_class":type(tokenizer).__name__, "pair_api":"__call__(text, text_pair)",
            "pair_lengths":[len(ids) for ids in batch["input_ids"]], "passed":True}


class CachedReranker:
    """FP16 on T4; bounded batches; all windows of long documents are scored."""
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.cfg = torch, cfg
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["reranker_id"], revision=cfg["reranker_revision"])
        probe = check_reranker_tokenizer(self.tokenizer, cfg["rerank_max_length"])
        from agent_core import write_json
        write_json(Path(cfg["output_dir"]) / "reranker_tokenizer_probe.json", probe)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg["reranker_id"], revision=cfg["reranker_revision"],
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        cache = Path(cfg.get("local_cache_dir", cfg["output_dir"])) / "rerank_scores.sqlite"
        cache.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache)
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL NOT NULL)")
        self.signature = {k: cfg[k] for k in ("reranker_id", "reranker_revision", "rerank_max_length", "rerank_query_tokens")}
        self.signature["window_version"] = 2

    @staticmethod
    def windows(ids, width, overlap=32):
        if not ids:
            return [[]]
        return [ids[start:start+width] for start in range(0, len(ids), max(1, width-overlap))
                if start == 0 or start < len(ids)-overlap]

    def score(self, query, documents):
        if not documents:
            return []
        keys = [hashlib.sha256(dumps([self.signature, query, d]).encode()).hexdigest() for d in documents]
        values = [self.db.execute("SELECT score FROM scores WHERE key=?", (key,)).fetchone() for key in keys]
        result = [float(v[0]) if v else None for v in values]
        missing = [i for i, v in enumerate(result) if v is None]
        if missing:
            tok, torch = self.tokenizer, self.torch
            best = {i: -math.inf for i in missing}
            with torch.inference_mode():
                for pairs, owners in rerank_pair_batches(tok, query, documents, missing, self.cfg):
                    batch = encode_rerank_pairs(tok, pairs, self.cfg["rerank_max_length"], return_tensors="pt")
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    scores = self.model(**batch, return_dict=True).logits.reshape(-1).float().cpu().tolist()
                    if len(scores) != len(owners):
                        raise ValueError("Expected one reranker logit per pair")
                    if not all(math.isfinite(v) for v in scores):
                        raise FloatingPointError("Non-finite reranker scores")
                    for i, score in zip(owners, scores):
                        best[i] = max(best[i], score)
            for i, score in best.items():
                result[i] = score
                self.db.execute("INSERT OR REPLACE INTO scores VALUES (?,?)", (keys[i], score))
            self.db.commit()
        return result

    def close(self):
        self.db.close()


def compact_schema(value, description_chars):
    """Keep types, required, enum, formats, nested/conditional constraints and $defs."""
    if isinstance(value, list):
        return [compact_schema(v, description_chars) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, val in value.items():
        # These values are maps of user-defined names, not schema keywords.
        # An argument/definition named 'title', 'description' or 'examples'
        # must never be mistaken for removable schema metadata.
        if key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas") and isinstance(val, dict):
            result[key] = {name:compact_schema(schema, description_chars) for name,schema in val.items()}
            continue
        if key in ("enum", "const", "default", "dependentRequired"):
            result[key] = copy.deepcopy(val)  # Literal data, not nested schemas.
            continue
        if key in ("examples", "title", "$schema"):
            continue
        if key == "description" and isinstance(val, str):
            if description_chars > 0:
                result[key] = val if len(val) <= description_chars else val[:description_chars] + "…"
        else:
            result[key] = compact_schema(val, description_chars)
    return result


def spec_for(tool, cfg, detailed=True):
    return compact_schema({k: tool[k] for k in ("name", "description", "kind", "parameters") if k in tool},
                          cfg["schema_description_chars"] if detailed else 0)


def history_blocks(history):
    """A user turn and its assistant/tool responses are an indivisible unit."""
    blocks = []
    for turn in history:
        if turn["role"] == "user" or not blocks:
            blocks.append([])
        blocks[-1].append(turn)
    return blocks


def ranked_indices(scores):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def is_lookup(name, spec):
    kind = str(spec.get("kind", "")).lower()
    leaf = name.rsplit(".", 1)[-1].lower()
    return kind in {"read", "query", "search", "lookup", "read_only", "readonly"} or bool(
        re.match(r"^(get|list|search|find|lookup|query|resolve|fetch|check)(_|[A-Z]|$)", leaf, re.I))


def dependency_closure(selected, names, tools, cfg):
    """Conservative schema/name heuristics, with explicit edges for unusual catalogs."""
    selected = set(selected)
    while True:
        namespaces = {n.split(".", 1)[0] for n in selected}
        id_stems = set()
        for n in selected:
            id_stems.update(re.findall(r'"([A-Za-z][A-Za-z_]*?)_ids?"\s*:', dumps(tools[n].get("parameters", {}))))
        for n in names:
            app = n.split(".", 1)[0]
            app_words = set(app.lower().split("_")) | {app.lower()}
            if any(stem.lower().rstrip("s") in {w.rstrip("s") for w in app_words} for stem in id_stems):
                namespaces.add(app)
        expanded = selected | {n for n in names if n.split(".", 1)[0] in namespaces and is_lookup(n, tools[n])}
        expanded |= {dep for n in selected for dep in cfg["tool_dependencies"].get(n, []) if dep in names}
        if expanded == selected:
            return selected
        selected = expanded


def compile_context(item, tools, ranker, tokenizer, cfg):
    # Only initial_messages' explicit input whitelist is read. Labels never enter selection.
    source = initial_messages(item, tools, cfg["global_meta_tools"])
    request_marker = "\nFINAL_REQUEST="
    history_raw, request = source[1]["content"].split(request_marker, 1)
    history = json.loads(history_raw.removeprefix("HISTORY="))
    blocks = history_blocks(history)
    recent_n = int(cfg["keep_recent_history"])
    recent = set(range(max(0, len(blocks)-recent_n), len(blocks))) if recent_n else set()
    history_scores = ranker.score(request, [dumps(b) for b in blocks])
    retrieved = [i for i in ranked_indices(history_scores) if i not in recent][:cfg["top_k_history"]]
    chosen_history = set(retrieved) | recent
    # Include actual conversational references when selecting tools. Preserve the full request.
    context_users = [t["text"] for i in sorted(chosen_history) for t in blocks[i] if t["role"] == "user"]
    query = request + ("\nTham chiếu hội thoại: " + " | ".join(context_users[-2:]) if context_users else "")
    names = installed_tool_names(item, tools, cfg["global_meta_tools"])
    docs = [dumps({k: tools[n][k] for k in ("name", "description", "kind", "parameters") if k in tools[n]}) for n in names]
    queries = [query]
    clauses = [s.strip() for s in re.split(r";|\n+|\brồi\b|\bsau đó\b|\bvà\b", request, flags=re.I) if len(s.strip()) >= 10]
    if len(clauses) > 1:
        queries += clauses
    all_scores = [ranker.score(q, docs) for q in queries]
    scores = [max(row[i] for row in all_scores) for i in range(len(names))]
    order = ranked_indices(scores)
    selected = {names[i] for i in order[:cfg["top_k_tools"]]}
    # Per-intent diversity helps requests that mention several apps.
    for row in all_scores[1:]:
        selected.update(names[i] for i in ranked_indices(row)[:2])
    apps = sorted({n.split(".", 1)[0] for n in names if not n.startswith("sys.")},
                  key=lambda a: (-max(scores[i] for i,n in enumerate(names) if n.startswith(a+".")), a))
    for app in apps[:cfg["top_k_apps"]]:
        app_candidates = [i for i in order if names[i].startswith(app+".")]
        selected.update(names[i] for i in app_candidates[:2])
    # Keep lookup dependencies in every selected app; never add by gold action.
    selected.update(n for n in cfg["global_meta_tools"] if n in names)
    selected = dependency_closure(selected, names, tools, cfg)
    if not selected and names:
        selected = set(names)  # Empty candidate set is never silently substituted.
    policy = cfg.get("tool_schema_policy", "compact_all")
    if policy not in ("compact_all", "hard_filter"):
        raise ValueError("Unknown tool_schema_policy: " + str(policy))
    # The same rule applies to EVERY sample, before any gold call is read.
    # Reranking controls descriptions; compact schemas preserve the remaining
    # installed tool names, argument names, types and machine constraints.
    visible = set(names) if policy == "compact_all" else set(selected)
    detailed = set(selected)
    old_tools = json.loads(source[0]["content"].split("\nTOOLS=", 1)[1].split("\nCONTEXT=", 1)[0])
    system_prefix, rest = source[0]["content"].split("\nTOOLS=", 1)
    public_context = rest.split("\nCONTEXT=", 1)[1]
    def render(history_ids):
        specs = [spec_for(tools[n], cfg, detailed=n in detailed) for n in sorted(visible)]
        h = [t for i in sorted(history_ids) for t in blocks[i]]
        return [{"role":"system", "content":system_prefix + "\nTOOLS=" + dumps(specs) + "\nCONTEXT=" + public_context},
                {"role":"user", "content":"HISTORY=" + dumps(h) + "\nFINAL_REQUEST=" + request}]
    messages = render(chosen_history)
    removed_detail = []
    # Prefer removing prose to removing a relevant old conversation block.
    # Never remove a tool name, argument schema, current request or recent block.
    score_map = dict(zip(names, scores))
    for name in sorted(detailed, key=lambda n:(score_map[n], n)):
        if len(prompt_ids(tokenizer, messages)) <= cfg["prompt_target_tokens"]:
            break
        detailed.remove(name)
        removed_detail.append(name)
        messages = render(chosen_history)
    removed_history = []
    # Only optional old blocks can be removed to approach the target budget.
    # Recent blocks, request, environment context and all visible schemas stay intact.
    for i in sorted(retrieved, key=lambda i: (history_scores[i], i)):
        if len(prompt_ids(tokenizer, messages)) <= cfg["prompt_target_tokens"]:
            break
        chosen_history.remove(i)
        removed_history.append(i)
        messages = render(chosen_history)
    specs = [spec_for(tools[n], cfg, detailed=n in detailed) for n in sorted(visible)]
    audit = {"id":str(item["id"]), "available_tools":names, "selected_tools":sorted(selected),
             "visible_tools":sorted(visible), "compact_fallback_tools":sorted(visible-selected),
             "tool_schema_policy":policy, "detailed_tools_in_prompt":sorted(detailed),
             "descriptions_removed_for_budget":removed_detail,
             "history_blocks_before":len(blocks), "history_blocks_kept":sorted(chosen_history),
             "recent_history_blocks":sorted(recent), "old_history_removed_for_budget":removed_history,
             "original_prompt_tokens":len(prompt_ids(tokenizer, source)),
             "compiled_prompt_tokens":len(prompt_ids(tokenizer, messages)),
             "tool_schema_chars_before":len(dumps(old_tools)), "tool_schema_chars_after":len(dumps(specs)),
             "tool_scores":{n:round(s, 5) for n,s in zip(names, scores)}}
    return messages, audit
