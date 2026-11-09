"""Restore a selected planner and run public inference without training cells."""
import hashlib
import importlib
import json
import shutil
from pathlib import Path

import colab_workflow


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def restore_run(training_run, output, data_root, code_dir, work_root, asr_mode, asr_model, transcripts,
                inference_settings=None):
    """Snapshot only inference artifacts; never rewrite the completed training run."""
    training_run, output = Path(training_run), Path(output)
    required = ['run_config.json', 'selection.json', 'prepared_fingerprint.json',
                'package_versions.json', 'best_adapter/adapter_config.json',
                'best_adapter/adapter_model.safetensors', 'tokenizer/tokenizer_config.json']
    missing = [name for name in required if not (training_run / name).is_file()]
    if missing:
        raise FileNotFoundError('Saved training run is missing: ' + ', '.join(missing) +
                                '. Check TRAINED_RUN_NAME; run Step 9 once if best selection is missing. No retraining is needed.')
    cfg = json.loads((training_run / 'run_config.json').read_text())
    signature = {'training_config': digest(training_run / 'run_config.json'),
                 'adapter': digest(training_run / 'best_adapter/adapter_model.safetensors'),
                 'asr_mode': asr_mode, 'asr_model': asr_model if asr_mode == 'whisper' else None,
                 'transcripts': str(transcripts),
                 'inference_settings': inference_settings or {},
                 'runtime_code': {str(p.relative_to(code_dir)): digest(p)
                                  for p in sorted(Path(code_dir).rglob('*.py'))}}
    marker = output / 'inference_source.json'
    if marker.exists() and json.loads(marker.read_text()) != signature:
        raise ValueError('Inference model/settings/code changed. Use a new INFERENCE_NAME; existing journals are preserved.')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('selection.json', 'prepared_fingerprint.json', 'package_versions.json'):
        if not (output / name).exists():
            shutil.copy2(training_run / name, output / name)
    for name in ('best_adapter', 'tokenizer'):
        shutil.copytree(training_run / name, output / name, dirs_exist_ok=True)
    cfg.update(data_root=str(data_root), output_dir=str(output), code_dir=str(code_dir),
               config_path=str(output / 'run_config.json'),
               local_cache_dir=str(Path(work_root) / 'cache' / output.name),
               eval_num_gpus=1, rerank_gpu=0, public_transcripts_path=str(transcripts))
    (output / 'run_config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    marker.write_text(json.dumps(signature, indent=2))
    return cfg


def run(*, drive_root, trained_run_name, inference_name, code_dir,
        asr_mode='whisper', asr_model='vinai/PhoWhisper-small', transcripts_path='',
        asr_checkpoint='', asr_decoder='auto', inference_max_length=8192,
        work_root='/content/lifemate_inference_work'):
    workflow = importlib.reload(colab_workflow)
    drive_root, code_dir, work_root = Path(drive_root), Path(code_dir), Path(work_root)
    for name in (trained_run_name, inference_name):
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Run names must be single directory names')
    if asr_mode not in ('whisper', 'transcripts', 'gipformer'):
        raise ValueError('ASR_MODE must be gipformer, whisper or transcripts')
    if type(inference_max_length) is not int or inference_max_length < 1024:
        raise ValueError('inference_max_length must be an integer >= 1024')
    training_run = drive_root / 'runs' / trained_run_name
    output = training_run / 'inference' / inference_name
    transcripts = Path(transcripts_path) if transcripts_path else output / 'public_asr.jsonl'
    if asr_mode == 'transcripts' and not transcripts.is_file():
        raise FileNotFoundError(f'Missing teammate transcripts: {transcripts}. Use whisper mode to infer now.')
    work_root.mkdir(parents=True, exist_ok=True)
    settings = {'max_length': inference_max_length}
    checkpoint = None
    if asr_mode == 'gipformer':
        from gipformer_asr import resolve_local_checkpoint, checkpoint_identity
        if asr_decoder not in ('auto', 'greedy', 'beam', 'beam_context'):
            raise ValueError('Unknown ASR_DECODER')
        checkpoint = resolve_local_checkpoint(asr_checkpoint, work_root / 'asr_checkpoint_cache')
        settings.update(asr_checkpoint=checkpoint_identity(checkpoint), asr_decoder=asr_decoder)
    # Validate saved model before downloading data or installing packages.
    cfg = restore_run(training_run, output, work_root / 'data', code_dir,
                      work_root, asr_mode, asr_model, transcripts, settings)
    print('Restoring selected planner:', training_run / 'best_adapter', flush=True)
    print('ASR:', checkpoint if asr_mode == 'gipformer' else asr_model if asr_mode == 'whisper' else transcripts, flush=True)
    agent_python = workflow.setup_environment(work_root, output, MODEL_PROFILE=cfg.get("model_profile", "qwen3"))
    data_root = workflow.prepare_data(work_root)
    cfg['data_root'] = str(data_root)
    cfg['max_length'] = max(cfg.get('max_length', 0), inference_max_length)
    config_path = Path(cfg['config_path'])
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    if asr_mode == 'gipformer':
        from gipformer_asr import transcribe
        transcribe(checkpoint=checkpoint, data_root=data_root, output=output / 'gipformer',
                   destination=transcripts, work_root=work_root, decoder=asr_decoder)
    else:
        workflow.prepare_asr(agent_python, code_dir, data_root, config_path,
                             asr_mode, transcripts, asr_model)
    workflow.run_stage(agent_python, code_dir, config_path, 'public')
    return workflow.publish_submission(output, drive_root, data_root,
                                       f'{trained_run_name}/inference/{inference_name}',
                                       output / 'best_adapter', transcripts)
