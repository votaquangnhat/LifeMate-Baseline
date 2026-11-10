"""Observable validation errors and unlabeled public diagnostics, after execution only."""
import collections
import json
from pathlib import Path
from agent_core import read_jsonl,write_json,write_jsonl,load_catalog


def signature(call):
    return json.dumps(call,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def call_differences(predicted,reference):
    # Exact differences are diagnostics, not the competition's fuzzy argument scorer.
    pred=collections.Counter(signature(c) for c in predicted)
    gold=collections.Counter(signature(c) for c in reference)
    extra=[json.loads(k) for k,n in (pred-gold).items() for _ in range(n)]
    missing=[json.loads(k) for k,n in (gold-pred).items() for _ in range(n)]
    used=set();fields=[]
    for wanted in missing:
        for i,actual in enumerate(extra):
            if i in used or actual['name']!=wanted['name']:continue
            used.add(i)
            a,b=actual['arguments'],wanted['arguments']
            fields.append({'tool':wanted['name'],'pairing':'first remaining call with same tool name',
                           'different_keys':[k for k in sorted(set(a)|set(b)) if k not in a or k not in b or a[k]!=b[k]],
                           'predicted':a,'reference':b})
            break
    return extra,missing,fields


def classify(row,reference,tools):
    pred=row['tool_calls'];gold=reference['tool_calls']
    pnames=collections.Counter(c['name'] for c in pred);gnames=collections.Counter(c['name'] for c in gold)
    if row.get('error'):return 'early_stop'
    if row.get('score',0)>=1-1e-9:return 'full_score'
    if not gold and pred:return 'calls_when_reference_empty'
    if gold and not pred:return 'no_executed_call'
    ask='sys.ask_missing_info'
    if gnames[ask] and not pnames[ask]:return 'missing_clarification_call'
    if pnames[ask] and not gnames[ask]:return 'extra_clarification_call'
    if row['state_hash']==reference['state_hash']:return 'correct_state_partial_call_score'
    missing=gnames-pnames;extra=pnames-gnames
    if missing and extra:return 'state_mismatch_different_tools_or_counts'
    if missing:return 'state_mismatch_missing_calls'
    if extra:return 'state_mismatch_extra_calls'
    return 'state_mismatch_same_tool_counts'


def analyze(cfg,part,rows):
    out=Path(cfg['output_dir']);root=Path(cfg['data_root'])
    tools=load_catalog(root,'public_test' if part=='public_test' else 'train')
    train_apps={n.split('.',1)[0] for n in load_catalog(root,'train')}
    repeated_reads=0;repeated_writes=0;novel_app_samples=0
    for r in rows:
        counts=collections.Counter(signature(c) for c in r['tool_calls'])
        for key,count in counts.items():
            kind=tools.get(json.loads(key)['name'],{}).get('kind')
            if kind=='read':repeated_reads+=max(0,count-1)
            elif kind=='write':repeated_writes+=max(0,count-1)
        novel_app_samples+=any(c['name'].split('.',1)[0] not in train_apps for c in r['tool_calls'])
    common=dict(samples=len(rows),executed_repeated_reads=repeated_reads,executed_repeated_writes=repeated_writes,
                samples_executing_apps_absent_from_train_catalog=novel_app_samples,
                early_stops=sum(bool(r.get('error')) for r in rows))
    if part=='public_test':
        write_json(out/'public_error_diagnostics.json',dict(common,
            note='Public labels are unavailable. These are observable events, not correctness or causal error rates. Repeated writes may be intentional.'))
        return
    refs={str(r['id']):r for r in read_jsonl(out/'validation_references.jsonl')}
    inputs={str(r['id']):r for r in read_jsonl(out/'validation_source.jsonl')}
    if set(refs)!={str(r['id']) for r in rows}:raise ValueError('Error analysis requires identical validation IDs')
    cases=[];groups=collections.defaultdict(lambda:{'samples':0,'lost_score_points':0.0})
    for r in rows:
        ref=refs[str(r['id'])];category=classify(r,ref,tools)
        groups[category]['samples']+=1
        groups[category]['lost_score_points']+=100*(1-float(r['score']))/len(rows)
        if category=='full_score':continue
        extra,missing,fields=call_differences(r['tool_calls'],ref['tool_calls'])
        cases.append(dict(id=r['id'],category=category,score=r['score'],state_match=r['state_hash']==ref['state_hash'],
            transcript=inputs[str(r['id'])].get('audio_text'),error=r.get('error'),
            extra_exact_calls=extra,missing_exact_calls=missing,argument_differences=fields,
            trace_file=str(out/'validation_best/predictions_with_trace.jsonl')))
    write_json(out/'validation_error_analysis.json',dict(common,categories=dict(groups),
        transcript_source='asr' if cfg.get('validation_transcripts_path') else 'gold',
        note='Groups describe outcomes, not proven causes. Exact call diffs and same-name argument pairing are review aids; original fuzzy competition scores are unchanged.'))
    write_jsonl(out/'validation_error_cases.jsonl',sorted(cases,key=lambda r:(r['score'],str(r['id']))))
