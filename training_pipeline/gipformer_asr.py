"""Host-side Gipformer checkpoint handoff; the ASR runs in its own CUDA runtime."""
import os
from pathlib import Path

from gipformer.submission_utils import (read_jsonl, write_json, write_jsonl, sha256_file,
                                       signature, guard, safe_extract, pick_asr, nearby)

SOURCE_COMMIT = '9c966762e058bd7ba79d9da71e42fc06b64b703e'
DEFAULTS = dict(variant='full', model_family='gipformer1.5_rnnt',
                model_id='g-group-ai-lab/gipformer1.5-68M-rnnt', model_revision='dd9227d',
                tokenizer_version='gipformer_sp_upper_v2', normalizer_version='spoken_vi_v4.1',
                zipformer_schedule_start=20000, beam_width=4, context_weight=0.6,
                max_context_phrases=128, use_initial_state_context=False, cudnn_plan_cache_limit=128)


def resolve_local_checkpoint(source, cache):
    path = Path(source).expanduser()
    if not source or not path.exists():
        raise FileNotFoundError(f'Gipformer checkpoint not found: {source}. Upload best_export/ (or its ZIP) to Drive and set ASR_CHECKPOINT in Step 13.')
    if path.is_file() and path.suffix not in ('.pt', '.pth'):
        target = Path(cache) / sha256_file(path)
        marker = target / '.extracted'
        if not marker.is_file():
            safe_extract(path, target)
            marker.write_text('complete\n')
        path = target
    return pick_asr(path).resolve()


def checkpoint_identity(checkpoint):
    checkpoint = Path(checkpoint)
    metadata = {}
    for name in ('bpe.model', 'tokens.txt', 'config.json', 'run_config.json', 'decode_config.json',
                 'decoder_validation_comparison.json', 'metrics.json'):
        path = nearby(checkpoint, name)
        if path:
            metadata[name] = sha256_file(path)
    return {'weights_sha256': sha256_file(checkpoint), 'metadata': metadata}


def transcribe(*, checkpoint, data_root, output, destination, work_root, decoder='auto'):
    from gipformer.runtime import setup_asr
    from gipformer.processes import run_jobs
    from asr_contract import merge_transcripts
    if decoder not in ('auto', 'greedy', 'beam', 'beam_context'):
        raise ValueError('Unknown Gipformer decoder')
    code = Path(__file__).parent / 'gipformer'
    output, destination, data_root, work_root = map(Path, (output, destination, data_root, work_root))
    source = data_root / 'public_test/public_test.jsonl'
    identity = checkpoint_identity(checkpoint)
    fingerprint = {'checkpoint': identity, 'decoder': decoder, 'input': sha256_file(source),
                   'source_commit': SOURCE_COMMIT, 'defaults': DEFAULTS,
                   'code': {p.name: sha256_file(p) for p in sorted(code.glob('*.py'))}}
    guard(output / 'gipformer_fingerprint.json', fingerprint)
    items = read_jsonl(source)
    if destination.is_file():
        merge_transcripts(items, read_jsonl(destination))
        print('Reusing complete Gipformer transcripts:', destination, flush=True)
        return destination
    cfg = dict(work_root=str(work_root), output_dir=str(output), code_dir=str(code),
               asr_code_dir=str(code), data_root=str(data_root), split='public_test',
               input_jsonl=str(source), asr_checkpoint=str(checkpoint), asr_decoder=decoder,
               asr_defaults=DEFAULTS, num_gpus=1, source_commit=SOURCE_COMMIT,
               source_dir=str(work_root / ('gipformer_icefall_' + SOURCE_COMMIT[:8])),
               checkpoint_sha256=identity['weights_sha256'], run_fingerprint=signature(fingerprint))
    config_path = output / 'gipformer_config.json'
    write_json(config_path, cfg)
    python = setup_asr(cfg)
    env = os.environ.copy()
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(name, None)
    env.update(CUDA_VISIBLE_DEVICES='0', INFER_RANK='0', PYTHONPATH=os.pathsep.join((str(code.parent), str(code))),
               OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1', TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='128')
    run_jobs([('gipformer', [python, code / 'infer_asr.py', '--config', config_path], env, None)], output / 'logs')
    rows = read_jsonl(output / 'asr/transcripts.rank0.jsonl')
    merged = merge_transcripts(items, rows)
    # Keep the original rich ASR log separately; publish the planner's simple contract.
    write_jsonl(destination, [{'id': r['id'], 'audio_text': r['audio_text'],
                              'history_audio_texts': [h['audio_text'] for h in r.get('history', [])
                                                     if h['role'] == 'user']} for r in merged])
    print('Gipformer transcripts ready:', destination, flush=True)
    return destination
