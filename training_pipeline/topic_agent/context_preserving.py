"""Opt-in inference context policy preserving requests, schemas and observations."""
import copy
import json
from agent_core import dumps, prompt_ids


def fit_prompt_preserving(tokenizer, messages, cfg, prompt_limit=None):
    capacity = int(cfg['max_length'])
    reserve = max(int(cfg['max_new_tokens']), int(cfg['max_new_tokens_cap']))
    limit = min(capacity-reserve-int(cfg.get('length_safety_margin',128)),
                int(prompt_limit) if prompt_limit is not None else capacity)
    if messages and messages[0]['content'].startswith('MODE=ROUTER'):
        limit = min(limit, int(cfg['router_prompt_tokens']))
    elif messages and messages[0]['content'].startswith('MODE=EXECUTOR'):
        limit = min(limit, int(cfg['executor_prompt_tokens']))
    if limit <= 0:
        raise ValueError('No capacity reserved for prompt')
    result = copy.deepcopy(messages)
    before = len(prompt_ids(tokenizer, result))
    audit = dict(before_tokens=before, limit=limit, history_blocks_removed=0,
                 tool_results_compacted=0, old_messages_removed=0, schema_names_only=False,
                 descriptions_removed=False, system_content_shortened=False, emergency_token_clip=False)
    def size(): return len(prompt_ids(tokenizer, result))
    if size() > limit and result and '\nTOOLS=' in result[0]['content'] and not cfg.get('preserve_full_schemas'):
        from context_rerank import compact_schema
        prefix, rest = result[0]['content'].split('\nTOOLS=',1)
        specs, context = rest.split('\nCONTEXT=',1)
        result[0]['content'] = prefix+'\nTOOLS='+dumps(compact_schema(json.loads(specs),0))+'\nCONTEXT='+context
        audit['descriptions_removed'] = True
    for message in result:
        content = message.get('content','')
        if size() <= limit: break
        if message['role'] != 'user' or not content.startswith('HISTORY=') or '\nFINAL_REQUEST=' not in content:
            continue
        raw, request = content.split('\nFINAL_REQUEST=',1)
        from context_rerank import history_blocks
        blocks = history_blocks(json.loads(raw.removeprefix('HISTORY=')))
        keep = max(1,int(cfg.get('keep_recent_history',2)))
        while len(blocks) > keep and size() > limit:
            blocks.pop(0)
            audit['history_blocks_removed'] += 1
            message['content'] = 'HISTORY='+dumps([t for b in blocks for t in b])+'\nFINAL_REQUEST='+request
    ids = prompt_ids(tokenizer,result)
    if len(ids) > limit:
        raise ValueError(f'Protected context overflow: {len(ids)} > {limit}. Request, argument schemas and tool results were not cut.')
    audit.update(after_tokens=len(ids),truncated=before>len(ids))
    return result, ids, audit


def retrieve_history_preserving(history, bundle, request, ranker, tokenizer, cfg):
    from context_rerank import history_blocks
    from history_memory import bm25_candidates, block_text
    blocks=history_blocks(history);budget=int(cfg['history_token_budget'])
    recent=list(range(max(0,len(blocks)-int(cfg['keep_recent_history'])),len(blocks)))
    def cost(indices):return len(tokenizer.encode(dumps([t for i in sorted(indices) for t in blocks[i]]),add_special_tokens=False))
    full=cost(range(len(blocks)))
    audit=dict(full_history=full<=budget, recent_block_ids=recent,source_history_tokens=full,
               emergency_history_clip=False,candidate_segments=[],rerank_scores={})
    if full<=budget:
        return copy.deepcopy(history),dict(audit,selected_block_ids=list(range(len(blocks))),history_tokens=full)
    segments=bundle['segments']
    context=request+'\nRecent context:\n'+'\n'.join(block_text(blocks[i]) for i in recent)
    # Retain candidates matching FINAL_REQUEST even when long recent turns dominate BM25.
    direct,_=bm25_candidates(request,segments,cfg)
    contextual,_=bm25_candidates(context,segments,cfg)
    candidates=list(dict.fromkeys(direct+contextual+[i for i,s in enumerate(segments) if set(recent)&set(s['block_ids'])]))
    scores=ranker.score(request,[segments[i]['raw_text'] for i in candidates])
    selected=set(recent)
    # Keep at least the newest complete block. Do not turn old dialogue into a synthetic user request.
    while len(selected)>1 and cost(selected)>budget:selected.remove(min(selected))
    for j in sorted(range(len(candidates)),key=lambda j:(-scores[j],-candidates[j])):
        proposed=selected|set(segments[candidates[j]]['block_ids'])
        if cost(proposed)<=budget:selected=proposed
    kept=sorted(selected)
    return copy.deepcopy([t for i in kept for t in blocks[i]]),dict(audit,
        selected_block_ids=kept,candidate_segments=candidates,
        rerank_scores={str(i):scores[j] for j,i in enumerate(candidates)},history_tokens=cost(kept),
        oversized_recent_block=cost(kept)>budget)
