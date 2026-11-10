"""Import the teammate's data-only adapter export; never execute bundled code."""
import hashlib
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def zip_entries(z):
    entries={}
    for item in z.infolist():
        name=item.filename
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or stat.S_ISLNK(item.external_attr>>16):
            raise ValueError('Unsafe archive member: '+name)
        if name in entries:raise ValueError('Duplicate archive member: '+name)
        entries[name]=item
    return entries


def validated_config(cfg, revisions, adapter):
    if cfg.get('model_family')!='qwen3_5' or not cfg.get('history_router') or cfg.get('thinking_mode') is not True:
        raise ValueError('Expected the teammate Qwen3.5 Thinking topical-router export, not our older compact agent.')
    if cfg.get('model_id')!='unsloth/Qwen3.5-4B' or cfg.get('tokenizer_id')!='Qwen/Qwen3.5-4B':
        raise ValueError('Unexpected base model/tokenizer in teammate export')
    for prefix in ('model','tokenizer','reranker','history_compressor','segmenter_model','segmenter_tokenizer'):
        key=prefix+'_id';revision=cfg.get(prefix+'_revision')
        if not isinstance(revision,str) or not re.fullmatch('[0-9a-f]{40}',revision):
            raise ValueError('Export must contain pinned '+prefix+' revision')
        if revisions.get(cfg[key])!=revision:raise ValueError('Model revision manifest mismatch: '+key)
    for key,configkey in [('r','lora_rank'),('lora_alpha','lora_alpha'),('lora_dropout','lora_dropout')]:
        if adapter.get(key)!=cfg.get(configkey):raise ValueError('Adapter configuration mismatch: '+key)
    if adapter.get('peft_type')!='LORA':raise ValueError('Expected a LoRA adapter')
    return cfg


def import_export(archive, destination, reports_archive=''):
    archive=Path(archive);destination=Path(destination)
    if not archive.is_file():raise FileNotFoundError(f'Upload the teammate Step 12 best_adapter.zip to {archive}')
    identity={'best_adapter_zip_sha256':digest(archive),
              'reports_zip_sha256':digest(reports_archive) if reports_archive else None}
    marker=destination/'import_manifest.json'
    if marker.exists():
        if json.loads(marker.read_text())['source']!=identity:
            raise ValueError('Imported archive changed. Use a new IMPORT_NAME; existing artifacts are preserved.')
        manifest=json.loads(marker.read_text())
        for name,sha in manifest['files'].items():
            if not (destination/name).is_file() or digest(destination/name)!=sha:
                raise ValueError('Imported artifact changed or incomplete: '+name)
        return destination
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        entries=zip_entries(z)
        required=['run_config.json','model_revisions.json','package_versions.json','prepared_fingerprint.json',
                  'best_adapter/adapter_config.json','best_adapter/adapter_model.safetensors',
                  'best_adapter/tokenizer.json','best_adapter/tokenizer_config.json']
        missing=[p for p in required if p not in entries or entries[p].file_size==0]
        if missing:raise ValueError('Incomplete best_adapter.zip: '+', '.join(missing))
        cfg=json.loads(z.read('run_config.json'));revisions=json.loads(z.read('model_revisions.json'))
        validated_config(cfg,revisions,json.loads(z.read('best_adapter/adapter_config.json')))
        token_cfg=json.loads(z.read('best_adapter/tokenizer_config.json'))
        if 'best_adapter/chat_template.jinja' not in entries and not token_cfg.get('chat_template'):
            raise ValueError('Missing exported tokenizer chat template')
        selected=[n for n in entries if n in required or n in ('training_recipe.json','requirements-agent.txt')
                  or (PurePosixPath(n).parent==PurePosixPath('best_adapter') and PurePosixPath(n).suffix in ('.json','.jinja','.txt'))]
        for name in selected:
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(entries[name]) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
    if reports_archive:
        with zipfile.ZipFile(reports_archive) as z:
            entries=zip_entries(z)
            for name in ('validation_references.jsonl','split_assignments.json','selection.json','test_metrics.json','epoch_validation.json'):
                if name in entries:(destination/name).write_bytes(z.read(name))
        if not all((destination/name).is_file() for name in ('validation_references.jsonl','split_assignments.json')):
            raise ValueError('reports.zip lacks validation references or split assignments; export reports from the same completed teammate run.')
    files={str(p.relative_to(destination)):digest(p) for p in sorted(destination.rglob('*')) if p.is_file() and p!=marker}
    marker.write_text(json.dumps({'source':identity,'files':files},indent=2))
    return destination
