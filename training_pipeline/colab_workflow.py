"""Small host-side helpers for the Git-backed Colab notebook.

Importing this module does not install packages, download data or initialize CUDA.
GPU work runs through the existing stage scripts in fresh subprocesses.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

def setup_environment(WORK_ROOT, OUTPUT_DIR):
    import os
    import sys
    import json
    import subprocess
    from pathlib import Path


    def prepare_agent_python(env_dir, host_python):
        env_dir = Path(env_dir)
        env_dir.mkdir(parents=True, exist_ok=True)
        # Always repair/revalidate: a failed ensurepip run can leave bin/python behind.
        # --without-pip avoids invoking ensurepip at all; no apt/get-pip.py required.
        subprocess.run([str(host_python), '-m', 'venv', '--without-pip',
                        '--system-site-packages', str(env_dir)], check=True)
        agent_python = env_dir / 'bin/python'
        prefix = json.loads(subprocess.check_output(
            [str(agent_python), '-c', 'import sys,json; print(json.dumps({"prefix":sys.prefix,"base":sys.base_prefix}))'],
            text=True))
        if Path(prefix['prefix']).resolve() != env_dir.resolve() or prefix['prefix'] == prefix['base']:
            raise RuntimeError('The agent interpreter is not inside the intended environment.')
        pip_env = os.environ.copy()
        pip_command = [str(host_python), '-m', 'pip']
        help_text = subprocess.check_output(pip_command + ['--help'], text=True, env=pip_env)
        if '--python' not in help_text:
            # Compatibility for a very old host pip; keep the upgrade local.
            bootstrap = env_dir / 'pip-bootstrap'
            subprocess.run(pip_command + ['install', '--upgrade', '--target', str(bootstrap), 'pip>=22.3'],
                           check=True, env=pip_env)
            pip_env['PYTHONPATH'] = str(bootstrap) + os.pathsep + pip_env.get('PYTHONPATH', '')
        # Host pip can manage a target environment even when it has no pip installed.
        pip_command += ['--python', str(agent_python)]
        subprocess.run(pip_command + ['--version'], check=True, env=pip_env)
        print('Verified agent interpreter:', agent_python, flush=True)
        return agent_python, pip_command, pip_env

    # WORK_ROOT defined above
    ENV_DIR = WORK_ROOT / 'agent_env'
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_PYTHON = ENV_DIR / 'bin/python'
    probe_code = dedent('''
    import torch, json, importlib.metadata as m
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 1, 'Select a Colab GPU runtime.' 
    pinned={}
    for name in ('torch','torchvision','torchaudio','triton'):
     try: pinned[name]=m.version(name)
     except m.PackageNotFoundError: pass
    print(json.dumps({'pinned': pinned, 'gpus': [{'name':torch.cuda.get_device_name(i),'gib':torch.cuda.get_device_properties(i).total_memory/2**30} for i in range(torch.cuda.device_count())]}))
    ''')
    host = json.loads(subprocess.check_output([sys.executable, '-c', probe_code], text=True))
    print('Existing CUDA stack:', json.dumps(host, indent=2))
    AGENT_PYTHON, AGENT_PIP, AGENT_PIP_ENV = prepare_agent_python(ENV_DIR, sys.executable)
    constraints = ENV_DIR / 'cuda-constraints.txt'
    constraints.write_text(''.join(f'{name}=={version}\n' for name, version in host['pinned'].items()))
    PACKAGES = [line.strip() for line in (Path(__file__).parent / 'requirements.txt').read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#')]
    # Reproduce package versions/Unsloth commits when resuming this run.
    saved_path = OUTPUT_DIR / 'package_versions.json'
    saved_versions = json.loads(saved_path.read_text()) if saved_path.is_file() else None
    if saved_versions:
        if saved_versions['torch'] != host['pinned']['torch']:
            raise RuntimeError('Colab CUDA stack changed; use a new run or restore the original runtime.')
        PACKAGES = [f'{name}=={saved_versions[name]}' for name in
                    ('transformers','peft','accelerate','bitsandbytes','datasets','trl','tokenizers',
                     'jinja2','huggingface_hub','safetensors','soundfile','librosa','sentencepiece')]
        for name, url in [('unsloth','https://github.com/unslothai/unsloth.git'),
                          ('unsloth_zoo','https://github.com/unslothai/unsloth-zoo.git')]:
            commit = saved_versions.get('source_revisions',{}).get(name,{}).get('vcs_info',{}).get('commit_id')
            PACKAGES.append(f'{name} @ git+{url}@{commit}' if commit else f'{name}=={saved_versions[name]}')
    # Resolve compatible library versions; preserve the existing torch/CUDA builds.
    subprocess.run(AGENT_PIP + ['install', '--upgrade',
                    '--upgrade-strategy', 'only-if-needed', '-c', str(constraints), *PACKAGES], check=True, env=AGENT_PIP_ENV)
    version_code = dedent('''
    import importlib.metadata as m, json, torch
    names=['torch','transformers','peft','accelerate','bitsandbytes','datasets','trl','unsloth','unsloth_zoo','tokenizers','jinja2','huggingface_hub','safetensors','soundfile','librosa','sentencepiece']
    result={name:m.version(name) for name in names}
    result['source_revisions']={}
    for name in ['unsloth','unsloth_zoo']:
     raw=m.distribution(name).read_text('direct_url.json')
     if raw:
      info=json.loads(raw)
      result['source_revisions'][name]={'url':info.get('url'),'vcs_info':{'commit_id':info.get('vcs_info',{}).get('commit_id')}}
    print(json.dumps(result))
    ''')
    versions = json.loads(subprocess.check_output([str(AGENT_PYTHON), '-c', version_code], text=True))
    if versions['torch'] != host['pinned']['torch']:
        raise RuntimeError('Unexpected CUDA stack change. Inspect install output before loading a model.')
    print('Agent environment:', json.dumps(versions, indent=2))
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    if saved_versions and versions != saved_versions:
        raise RuntimeError('Package versions differ from the saved run. Inspect before resuming.')
    (OUTPUT_DIR / 'package_versions.json').write_text(json.dumps(versions, indent=2))

    return AGENT_PYTHON


def prepare_data(WORK_ROOT, LOCAL_ARCHIVE=Path("/content/train+public_test.zip")):
    DATA_DIR = WORK_ROOT / 'data'
    if not LOCAL_ARCHIVE.is_file():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
        subprocess.run([sys.executable, '-m', 'gdown', '1f5Vdnm3fhLdBYZFmRPPYvKscmFmbozmQ',
                        '-O', str(LOCAL_ARCHIVE)], check=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not list(DATA_DIR.rglob('public_test.jsonl')):
        shutil.unpack_archive(str(LOCAL_ARCHIVE), str(DATA_DIR))
    candidates = [p.parent.parent for p in DATA_DIR.rglob('train.jsonl')
                  if (p.parent / 'train_tools_catalog.json').is_file()]
    assert len(candidates) == 1, f'Expected one dataset root, got {candidates}'
    DATA_ROOT = candidates[0]
    assert (DATA_ROOT / 'public_test/public_test.jsonl').is_file()
    print(DATA_ROOT)

    return DATA_ROOT


def configure(*, DATA_ROOT, OUTPUT_DIR, CODE_DIR, CONFIG_PATH, WORK_ROOT, RUN_NAME, EPOCHS, TRAIN_START, PREVIOUS_RUN_NAME, RESUME_CHECKPOINT, ASR_TRANSCRIPTS, DRIVE_ROOT):
    CONFIG = dict(
        train_start=TRAIN_START, previous_run_name=PREVIOUS_RUN_NAME if TRAIN_START == "best" else None,
        data_root=str(DATA_ROOT), output_dir=str(OUTPUT_DIR), code_dir=str(CODE_DIR), config_path=str(CONFIG_PATH),
        local_cache_dir=str(WORK_ROOT / RUN_NAME / 'cache'),
        model_id='unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit', model_revision='7744afa8566e264af1a92a806d8d9aae00cc7c78',
        tokenizer_id='Qwen/Qwen3-4B-Instruct-2507', tokenizer_revision='cdbee75f17c01a7cc42f958dc650907174af0554',
        reranker_id='BAAI/bge-reranker-v2-m3', reranker_revision='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e',
        seed=42, num_gpus=1, eval_num_gpus=1, rerank_gpu=0,
        epochs=EPOCHS, learning_rate=2e-4, per_gpu_batch_size=1, gradient_accumulation_steps=8,
        lora_rank=8, lora_alpha=16, lora_dropout=0.0, gradient_checkpointing='unsloth',
        weight_decay=0.01, warmup_ratio=0.05, max_grad_norm=1.0, logging_steps=10, save_steps=100,
        top_k_apps=2, top_k_tools=8, top_k_history=2, keep_recent_history=2,
        schema_description_chars=140, tool_schema_policy='compact_all',
        rerank_batch_size=8, rerank_max_length=512, rerank_query_tokens=160,
        prompt_target_tokens=1536, minimum_train_schema_coverage=1.0, tool_dependencies={},
        requested_max_length=2048, requested_max_new_tokens=512, max_length=2048, max_new_tokens=512,
        auto_token_limits=True, max_length_cap=8192, max_new_tokens_cap=1024,
        length_round_to=256, length_safety_margin=128,
        max_agent_steps=24, max_repairs=2, vram_limit=0.95, smoke_steps=5,
        asr_manifest_path='', hash_audio_for_split=True, global_meta_tools=['sys.ask_missing_info'],
        resume_checkpoint=RESUME_CHECKPOINT, public_transcripts_path=str(ASR_TRANSCRIPTS),
    )
    assert TRAIN_START in {'fresh', 'best'}
    CONFIG['initial_adapter'] = ''
    CONFIG['initial_adapter_sha256'] = None
    if TRAIN_START == 'best':
        assert PREVIOUS_RUN_NAME != RUN_NAME, 'Use a NEW RUN_NAME for each additional training session.'
        previous_run = DRIVE_ROOT / 'runs' / PREVIOUS_RUN_NAME
        prior_config = json.loads((previous_run / 'run_config.json').read_text())
        for key in ('model_id','model_revision','tokenizer_id','tokenizer_revision','lora_rank','lora_alpha',
                    'lora_dropout','seed','hash_audio_for_split','asr_manifest_path','tool_schema_policy',
                    'top_k_apps','top_k_tools','top_k_history','keep_recent_history','prompt_target_tokens'):
            assert prior_config.get(key) == CONFIG[key], f'Previous best has incompatible {key}'
        source_adapter = previous_run / 'best_adapter'
        assert (previous_run / 'selection.json').is_file(), 'Run best-checkpoint selection in the previous run first.'
        for name in ('adapter_model.safetensors', 'adapter_config.json'):
            assert (source_adapter / name).is_file(), f'Missing previous adapter file: {name}'
        initial = OUTPUT_DIR / 'initial_adapter'
        source_hash = hashlib.sha256((source_adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()
        if initial.exists():
            assert (initial / 'adapter_model.safetensors').is_file(), 'Incomplete initial_adapter snapshot.'
            assert hashlib.sha256((initial / 'adapter_model.safetensors').read_bytes()).hexdigest() == source_hash, 'Previous best changed; choose a new RUN_NAME.'
        else:
            staging = OUTPUT_DIR / 'initial_adapter.tmp'
            shutil.copytree(source_adapter, staging, dirs_exist_ok=True)
            staging.rename(initial)
        CONFIG['initial_adapter'] = str(initial)
        CONFIG['initial_adapter_sha256'] = source_hash

    if CONFIG_PATH.is_file():
        previous = json.loads(CONFIG_PATH.read_text())
        mutable = {'resume_checkpoint', 'public_transcripts_path', 'max_length', 'max_new_tokens'}
        changed = [k for k, v in CONFIG.items() if k not in mutable and previous.get(k) != v]
        assert not changed, f'Use a new RUN_NAME for changed configuration: {changed}'
        CONFIG['max_length'] = previous['max_length']
        CONFIG['max_new_tokens'] = previous['max_new_tokens']
    CONFIG_PATH.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2))
    (OUTPUT_DIR / 'model_revisions.json').write_text(json.dumps({CONFIG[p+'_id']:CONFIG[p+'_revision']
                                                              for p in ('model','tokenizer','reranker')}, indent=2))
    print('One GPU, effective batch:', CONFIG['per_gpu_batch_size'] * CONFIG['gradient_accumulation_steps'])

    return CONFIG


def check_runtime(AGENT_PYTHON, CODE_DIR, OUTPUT_DIR):
    # Keep a reviewable source copy alongside the model and checkpoints.
    SOURCE_DIR = OUTPUT_DIR / 'source'
    SOURCE_DIR.mkdir(exist_ok=True)
    for path in CODE_DIR.glob('*.py'):
        shutil.copy2(path, SOURCE_DIR / path.name)
    tests = sorted(p.name for p in CODE_DIR.glob('selftest_*.py'))
    for test in tests:
        subprocess.run([str(AGENT_PYTHON), str(CODE_DIR / test)], check=True)


def run_stage(AGENT_PYTHON, CODE_DIR, CONFIG_PATH, stage):
    subprocess.run([str(AGENT_PYTHON), '-u', str(CODE_DIR / 'run_pipeline.py'),
                    '--config', str(CONFIG_PATH), '--stage', stage], check=True)


def export_model(AGENT_PYTHON, OUTPUT_DIR, CODE_DIR, CONFIG_PATH, EXPORT_BASE_WEIGHTS=True):
    CONFIG = json.loads(CONFIG_PATH.read_text())
    BEST_MODEL = OUTPUT_DIR / 'best_model'
    BEST_MODEL.mkdir(exist_ok=True)
    shutil.copytree(OUTPUT_DIR / 'best_adapter', BEST_MODEL / 'adapter', dirs_exist_ok=True)
    shutil.copytree(OUTPUT_DIR / 'tokenizer', BEST_MODEL / 'tokenizer', dirs_exist_ok=True)
    shutil.copytree(CODE_DIR, BEST_MODEL / 'source', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('run_config.json','selection.json','package_versions.json','model_revisions.json',
                 'training_recipe.json','split_assignments.json','data_summary.json'):
        shutil.copy2(OUTPUT_DIR / name, BEST_MODEL / name)
    exported = json.loads(CONFIG_PATH.read_text())
    if EXPORT_BASE_WEIGHTS:
        download_code = "from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], ignore_patterns=['*.gguf','*.onnx','*.h5','*.msgpack'])"
        for prefix, folder in [('model','base'), ('reranker','reranker')]:
            target = BEST_MODEL / folder
            subprocess.run([str(AGENT_PYTHON), '-c', download_code,
                            CONFIG[prefix+'_id'], CONFIG[prefix+'_revision'], str(target)], check=True)
            exported[prefix+'_id'] = str(target)
            exported[prefix+'_revision'] = None
    exported.update(output_dir=str(BEST_MODEL), code_dir=str(BEST_MODEL / 'source'),
                    config_path=str(BEST_MODEL / 'deployment_config.json'))
    exported['tokenizer_id'] = str(BEST_MODEL / 'tokenizer')
    exported['tokenizer_revision'] = None
    (BEST_MODEL / 'deployment_config.json').write_text(json.dumps(exported, indent=2))
    (BEST_MODEL / 'LOAD_MODEL.txt').write_text(
        'Use source/modeling_agent.py: load_model(deployment_config, adapter_path=best_model/adapter, training=False).\n'
        'Call prepare_public.py with this config to compile your ASR transcripts; use its public catalog and ToolEnv.\n'
        'The full staged driver for this completed training run still uses runs/<RUN_NAME>/run_config.json.\n'
        'The exported adapter is not a standalone full-precision model.\n')
    (BEST_MODEL / 'EXPORT_COMPLETE.json').write_text(json.dumps({'base_weights_included':EXPORT_BASE_WEIGHTS,
        'selected_checkpoint':json.loads((OUTPUT_DIR/'selection.json').read_text())['winner']['checkpoint']}, indent=2))
    print('Best model saved:', BEST_MODEL)

    return BEST_MODEL


def write_asr_manifest(DATA_ROOT, DRIVE_ROOT, ASR_TRANSCRIPTS):
    public_rows = [json.loads(line) for line in (DATA_ROOT / 'public_test/public_test.jsonl').read_text().splitlines() if line.strip()]
    manifest = [{'id': r['id'], 'audio': r['audio'],
                 'history_audio': [h['audio'] for h in r.get('history', []) if h['role']=='user']}
                for r in public_rows]
    manifest_path = DRIVE_ROOT / 'inputs/public_asr_requests.jsonl'
    manifest_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in manifest))
    print('ASR request manifest:', manifest_path)
    print('Expected transcript output:', ASR_TRANSCRIPTS)

    return manifest_path


def prepare_asr(AGENT_PYTHON, CODE_DIR, DATA_ROOT, CONFIG_PATH, ASR_MODE, ASR_TRANSCRIPTS, ASR_MODEL):
    assert ASR_MODE in {'transcripts','whisper'}
    if ASR_MODE == 'whisper' and not ASR_TRANSCRIPTS.is_file():
        assert ASR_MODEL, 'Set ASR_MODEL to the teammate’s merged Whisper checkpoint.'
        subprocess.run([str(AGENT_PYTHON), '-u', str(CODE_DIR / 'export_whisper_asr.py'),
                        '--root', str(DATA_ROOT), '--model', ASR_MODEL, '--output', str(ASR_TRANSCRIPTS)], check=True)
    assert ASR_TRANSCRIPTS.is_file(), f'ASR handoff required: place the complete teammate transcript JSONL at {ASR_TRANSCRIPTS}. Training/model export are already complete.'
    check = "from asr_contract import merge_transcripts; from agent_core import read_jsonl; import sys; x=merge_transcripts(read_jsonl(sys.argv[1]),read_jsonl(sys.argv[2])); print('Validated ASR for',len(x),'samples')"
    subprocess.run([str(AGENT_PYTHON), '-c', check, str(DATA_ROOT / 'public_test/public_test.jsonl'),
                    str(ASR_TRANSCRIPTS)], cwd=CODE_DIR, check=True)
    CONFIG = json.loads(CONFIG_PATH.read_text())
    CONFIG['public_transcripts_path'] = str(ASR_TRANSCRIPTS)
    CONFIG_PATH.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2))


def publish_submission(OUTPUT_DIR, DRIVE_ROOT, DATA_ROOT, RUN_NAME, BEST_MODEL, ASR_TRANSCRIPTS):
    public_rows = [json.loads(line) for line in (DATA_ROOT / 'public_test/public_test.jsonl').read_text().splitlines() if line.strip()]
    summary = json.loads((OUTPUT_DIR / 'public_summary.json').read_text())
    print(json.dumps(summary, indent=2))
    assert summary['aborted_rollouts'] == 0, 'Public rollouts aborted: inspect public_test_best/predictions_with_trace.jsonl before exporting.'
    SUBMISSION = OUTPUT_DIR / 'public_test_best/submission.jsonl'
    rows = [json.loads(line) for line in SUBMISSION.read_text().splitlines() if line.strip()]
    assert [r['id'] for r in rows] == [r['id'] for r in public_rows]
    assert all(set(r) == {'id','tool_calls','state_hash'} for r in rows)
    latest = DRIVE_ROOT / 'submission.jsonl'
    temporary = latest.with_suffix('.jsonl.tmp')
    shutil.copy2(SUBMISSION, temporary)
    temporary.replace(latest)
    provenance = {'run':RUN_NAME, 'samples':len(rows), 'model':str(BEST_MODEL),
                  'submission_sha256':hashlib.sha256(latest.read_bytes()).hexdigest(),
                  'asr_sha256':hashlib.sha256(ASR_TRANSCRIPTS.read_bytes()).hexdigest(),
                  'validation':summary}
    (DRIVE_ROOT / 'submission_provenance.json').write_text(json.dumps(provenance, indent=2))
    print('Validated submission saved:', latest)
    print('Run-specific original:', SUBMISSION)

    return latest

