"""Inference-only preparation, rollout collection and diagnostics for imported adapters."""
import argparse
import collections
import json
from pathlib import Path
from agent_core import read_jsonl, write_jsonl, write_json, sha256_file, load_catalog


def validation_source(cfg):
    out=Path(cfg['output_dir']);export=Path(cfg['import_dir']);root=Path(cfg['data_root'])
    source=root/'train/train.jsonl'
    signature=json.loads((export/'prepared_fingerprint.json').read_text())
    if signature.get('source')!=sha256_file(source) or signature.get('catalog')!=sha256_file(root/'train/train_tools_catalog.json'):
        raise ValueError('Validation dataset differs from teammate training data; do not compare these scores.')
    refs=read_jsonl(export/'validation_references.jsonl')
    splits=json.loads((export/'split_assignments.json').read_text())
    wanted={str(r['id']) for r in refs}
    if len(wanted)!=len(refs) or wanted!={str(k) for k,v in splits.items() if v=='validation'}:
        raise ValueError('Validation references and saved split disagree')
    items=read_jsonl(source);selected=[]
    for item in items:
        if str(item['id']) in wanted:
            selected.append({k:v for k,v in item.items() if k not in ('label','final_state','tool_results','_agent_messages','_router_context')})
    if {str(i['id']) for i in selected}!=wanted:raise ValueError('Missing validation inputs')
    if cfg.get('validation_transcripts_path'):
        from asr_contract import merge_transcripts
        if sha256_file(cfg['validation_transcripts_path'])!=cfg['validation_transcripts_sha256']:
            raise ValueError('Validation ASR file changed. Use a new experiment name.')
        selected=merge_transcripts(selected,read_jsonl(cfg['validation_transcripts_path']))
    write_jsonl(out/'validation_source.jsonl',selected)
    write_jsonl(out/'validation_references.jsonl',refs)


def prepare_validation(cfg):
    from transformers import AutoTokenizer
    from thinking_policy import validate_template
    from context_rerank import CachedReranker, compile_context, context_retention_report
    out=Path(cfg['output_dir'])
    tokenizer=AutoTokenizer.from_pretrained(out/'tokenizer');validate_template(tokenizer,cfg['thinking_mode'])
    tools=load_catalog(cfg['data_root'],'train');ranker=CachedReranker(cfg)
    compiled=[];audits=[]
    for item in read_jsonl(out/'validation_source.jsonl'):
        messages,audit=compile_context(item,tools,ranker,tokenizer,cfg)
        item['_agent_messages']=messages;compiled.append(item);audits.append(dict(audit,split='validation'))
    ranker.close()
    write_jsonl(out/'validation_items.jsonl',compiled)
    write_jsonl(out/'validation_retrieval_audit.jsonl',audits)
    write_json(out/'validation_context_retention_report.json',context_retention_report(audits))


def finish(cfg,part):
    from collect_results import collect
    out=Path(cfg['output_dir']);directory=out/(part+'_best')
    status=directory/'rollout_status.rank0.json'
    if not status.is_file() or not json.loads(status.read_text()).get('complete'):
        raise ValueError('Rollout is incomplete. Rerun the same experiment to reuse completed samples.')
    summary=collect(cfg,part,directory)
    rows=read_jsonl(directory/'predictions_with_trace.jsonl')
    errors=[{'id':r['id'],'error':r['error']} for r in rows if r.get('error')]
    counts=collections.Counter()
    for r in rows:
        for trim in r.get('context_trims',[]):
            for key in ('schema_names_only','emergency_token_clip','system_content_shortened'):
                counts[key]+=bool(trim.get(key))
            counts['tool_results_compacted']+=trim.get('tool_results_compacted',0)
            counts['old_messages_removed']+=trim.get('old_messages_removed',0)
        counts['duplicate_reads_blocked']+=sum(t.get('phase')=='cached_read' for t in r.get('trace',[]))
        counts['router_fallbacks']+=bool(r.get('router',{}).get('fallback_all_apps'))
        counts['invalid_model_actions']+=r.get('invalid_model_actions',0)
    summary['agent_diagnostics']=dict(counts)
    summary['adapter_sha256']=sha256_file(Path(cfg['import_dir'])/'best_adapter/adapter_model.safetensors')
    summary['variant']=cfg['agent_variant']
    summary['validation_transcripts_sha256']=cfg.get('validation_transcripts_sha256')
    summary['input_sha256']=sha256_file(out/(part+'_items.jsonl'))
    write_json(out/('public_summary.json' if part=='public_test' else 'validation_metrics.json'),summary)
    write_json(out/(part+'_quality_report.json'),{'early_stops':errors,'diagnostics':dict(counts)})
    from error_analysis import analyze
    analyze(cfg,part,rows)
    print(json.dumps(summary,indent=2),flush=True)
    for error in errors:print('Early stop:',error['id'],error['error'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['validation-source','prepare-validation','finish'],required=True)
    p.add_argument('--part',default='validation',choices=['validation','public_test'])
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    if a.stage=='validation-source':validation_source(cfg)
    elif a.stage=='prepare-validation':prepare_validation(cfg)
    else:finish(cfg,a.part)
