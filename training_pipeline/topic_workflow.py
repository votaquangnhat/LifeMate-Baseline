"""Git-backed host workflow for the teammate's selected topical-router adapter."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from topic_artifacts import digest, import_export


def _name(value):
    if not value or Path(value).name!=value or value in ('.','..') or '\\' in value:
        raise ValueError('Use a single folder name: '+str(value))
    return value


def prepare(*, drive_root, import_name, experiment_name, adapter_zip, reports_zip='',
            work_root='/content/lifemate_topic_work'):
    root=Path(drive_root);work=Path(work_root);work.mkdir(parents=True,exist_ok=True)
    imported=import_export(adapter_zip,root/'imports'/_name(import_name),reports_zip)
    experiment=root/'runs'/_name(experiment_name);experiment.mkdir(parents=True,exist_ok=True)
    code=Path(__file__).resolve().parent/'topic_agent'
    signature={'import_manifest':digest(imported/'import_manifest.json'),
        'code':{str(p.relative_to(Path(__file__).parent)):digest(p) for p in sorted(code.glob('*.py'))}}
    for name in ('topic_workflow.py','topic_environment.py','topic_artifacts.py','colab_workflow.py','gipformer_asr.py'):
        signature['code'][name]=digest(Path(__file__).parent/name)
    for p in sorted((Path(__file__).parent/'gipformer').glob('*.py')):
        signature['code']['gipformer/'+p.name]=digest(p)
    marker=experiment/'experiment_source.json'
    if marker.exists() and json.loads(marker.read_text())!=signature:
        raise ValueError('Adapter/code changed; choose a new EXPERIMENT_NAME. Existing results are preserved.')
    marker.write_text(json.dumps(signature,indent=2))
    print('Imported selected adapter:',imported/'best_adapter',flush=True)
    from topic_environment import setup_environment
    hints=json.loads((imported/'package_versions.json').read_text())
    python=setup_environment(work,experiment,package_hints=hints)
    import colab_workflow
    data=colab_workflow.prepare_data(work)
    return dict(root=str(root),work=str(work),imported=str(imported),experiment=str(experiment),
                code=str(code),python=str(python),data=str(data))


VARIANTS = ('baseline','enhanced','read_guard')


def configure(session,variant,transcripts='',validation_transcripts=None):
    if variant not in VARIANTS:raise ValueError('MODE must be baseline, enhanced or read_guard')
    imported=Path(session['imported']);out=Path(session['experiment'])/variant;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((imported/'run_config.json').read_text())
    # Kaggle runtime deadlines/paths are not model parameters and must not cross into a new session.
    for key in list(cfg):
        if 'deadline' in key:cfg.pop(key)
    cfg.update(agent_variant=variant,import_dir=str(imported),output_dir=str(out),
        config_path=str(out/'run_config.json'),code_dir=session['code'],data_root=session['data'],
        tokenizer_path=str(out/'tokenizer'),local_cache_dir=str(Path(session['work'])/Path(session['experiment']).name/variant/'cache'),
        num_gpus=1,eval_num_gpus=1,rerank_gpu=0,public_transcripts_path=str(transcripts))
    if variant=='enhanced':
        cfg.update(max_length=8192,max_length_cap=8192,requested_max_length=8192,
                   history_token_budget=2048,router_prompt_tokens=4096,executor_prompt_tokens=7168)
    if variant=='read_guard':
        cfg.update(duplicate_read_guard=True,max_duplicate_read_blocks=3)
    config=out/'run_config.json'
    existing=json.loads(config.read_text()) if config.exists() else {}
    validation_path=str(validation_transcripts) if validation_transcripts is not None else existing.get('validation_transcripts_path','')
    cfg['validation_transcripts_path']=validation_path
    cfg['validation_transcripts_sha256']=digest(validation_path) if validation_path else None
    if config.exists():
        old=json.loads(config.read_text())
        # ASR is irrelevant to validation; public preparation has its own transcript fingerprint.
        changed=[k for k in set(old)|set(cfg) if k!='public_transcripts_path' and old.get(k)!=cfg.get(k)]
        if changed:raise ValueError('Configuration changed. Use a new experiment name: '+str(changed))
    config.write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    for name in ('prepared_fingerprint.json',):shutil.copy2(imported/name,out/name)
    shutil.copy2(Path(session['experiment'])/'package_versions.json',out/'package_versions.json')
    tokenizer=out/'tokenizer';tokenizer.mkdir(exist_ok=True)
    for source in (imported/'best_adapter').iterdir():
        if source.suffix in ('.json','.jinja','.txt') and source.name not in ('adapter_config.json','agent_config.json'):
            shutil.copy2(source,tokenizer/source.name)
    return cfg


def command(session,cfg,script,*args):
    argv=[session['python'],'-u',str(Path(session['code'])/script),'--config',cfg['config_path'],*map(str,args)]
    env=os.environ.copy();env['TOKENIZERS_PARALLELISM']='false'
    print('Running:',script,' '.join(map(str,args)),flush=True)
    subprocess.run(argv,check=True,env=env)


def run_part(session,cfg,part):
    out=Path(cfg['output_dir'])
    if part=='validation':command(session,cfg,'infer.py','--stage','validation-source')
    for phase in ('segment','compress'):
        command(session,cfg,'prepare_history.py','--phase',phase,'--part',part)
    if part=='validation':command(session,cfg,'infer.py','--stage','prepare-validation')
    else:command(session,cfg,'prepare_public.py')
    command(session,cfg,'rollout_agent.py','--adapter',Path(session['imported'])/'best_adapter',
            '--part',part,'--output',out/(part+'_best'))
    command(session,cfg,'infer.py','--stage','finish','--part',part)


def compare_validation(session,validation_transcripts=''):
    for name in ('validation_references.jsonl','split_assignments.json'):
        if not (Path(session['imported'])/name).is_file():
            raise FileNotFoundError('Upload reports.zip from the same teammate run; missing '+name)
    metrics={}
    for variant in VARIANTS:
        cfg=configure(session,variant,validation_transcripts=validation_transcripts);run_part(session,cfg,'validation')
        metrics[variant]=json.loads((Path(cfg['output_dir'])/'validation_metrics.json').read_text())
    for variant in VARIANTS[1:]:
        if any(metrics['baseline'].get(k)!=metrics[variant].get(k) for k in ('adapter_sha256','samples','validation_transcripts_sha256')):
            raise ValueError('Comparison requires the same adapter, validation samples and transcripts')
    chosen=max(VARIANTS,key=lambda v:metrics[v]['official_score_100'])
    report={'selection':'full original validation split; baseline wins ties','chosen':chosen,'metrics':metrics,
            'validation_input':'asr' if validation_transcripts else 'gold',
            'validation_transcripts_sha256':digest(validation_transcripts) if validation_transcripts else None,
            'experiment_source_sha256':digest(Path(session['experiment'])/'experiment_source.json')}
    (Path(session['experiment'])/'agent_comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report


def choose_mode(session,mode):
    if mode in VARIANTS:return mode
    if mode!='auto':raise ValueError('MODE must be auto, baseline, enhanced or read_guard')
    path=Path(session['experiment'])/'agent_comparison.json'
    if not path.is_file():
        raise FileNotFoundError('Run the validation comparison first, or explicitly set MODE="baseline" / "enhanced".')
    report=json.loads(path.read_text())
    if report['experiment_source_sha256']!=digest(Path(session['experiment'])/'experiment_source.json'):
        raise ValueError('Comparison belongs to different code or weights')
    return report['chosen']


def public(session,*,mode='auto',asr_mode='gipformer',asr_checkpoint='',transcripts_path='',asr_decoder='auto'):
    variant=choose_mode(session,mode)
    experiment=Path(session['experiment']);transcripts=Path(transcripts_path) if transcripts_path else experiment/'asr/public_asr.jsonl'
    if asr_mode=='gipformer':
        from gipformer_asr import resolve_local_checkpoint,transcribe
        checkpoint=resolve_local_checkpoint(asr_checkpoint,Path(session['work'])/'asr_checkpoint_cache')
        transcribe(checkpoint=checkpoint,data_root=Path(session['data']),output=experiment/'asr/gipformer',
            destination=transcripts,work_root=Path(session['work']),decoder=asr_decoder)
    elif asr_mode=='transcripts':
        if not transcripts.is_file():raise FileNotFoundError(transcripts)
    else:raise ValueError('Use gipformer or transcripts; no ASR training is needed')
    cfg=configure(session,variant,str(transcripts));run_part(session,cfg,'public_test')
    import colab_workflow
    exported=colab_workflow.publish_submission(Path(cfg['output_dir']),Path(session['root']),Path(session['data']),
        Path(session['experiment']).name+'/'+variant,Path(session['imported'])/'best_adapter',transcripts)
    return exported
