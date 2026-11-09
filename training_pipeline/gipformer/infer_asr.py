"""Load the fine-tuned Gipformer once per GPU; decode main and history audio."""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

from gipformer.submission_utils import (audio_jobs, guard, nearby, read_json, read_jsonl,
                              sha256_file, strict_ids, write_json, write_jsonl)


def unpack_weights(state):
    if isinstance(state, dict):
        for key in ('model', 'state_dict'):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    if not isinstance(state, dict) or not state:
        raise ValueError('Missing ASR state dict')
    result = {}
    for key, value in state.items():
        while key.startswith(('module.', 'model.', 'asr.')):
            key = key.split('.', 1)[1]
        if key in result:
            raise ValueError('Duplicate ASR key after removing wrapper prefixes: ' + key)
        result[key] = value
    return result


def main(cfg):
    import torch
    from huggingface_hub import snapshot_download
    rank = int(os.environ.get('INFER_RANK', '0'))
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES isolates the physical GPU.
    torch.set_num_threads(2)
    out = Path(cfg['output_dir']) / 'asr'
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(cfg['asr_checkpoint'])
    if sha256_file(checkpoint) != cfg['checkpoint_sha256']:
        raise ValueError('ASR checkpoint changed after configuration')
    # mmap avoids materializing optimizer tensors from a full best.pt/last.pt.
    try:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError as exc:
        if 'mmap can only be used' not in str(exc):
            raise
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    asr_cfg = dict(cfg['asr_defaults'])
    if isinstance(state.get('config'), dict):
        asr_cfg.update(state['config'])
    for filename in ('run_config.json', 'decode_config.json'):
        path = nearby(checkpoint, filename)
        if path:
            asr_cfg.update(read_json(path))
    # Checkpoint metadata must never enable hidden-state reads.
    asr_cfg['use_initial_state_context'] = False
    if asr_cfg.get('variant') != 'full' or asr_cfg.get('model_family') != 'gipformer1.5_rnnt':
        raise ValueError('This notebook requires the attached Gipformer 1.5 full-finetune recipe')
    if asr_cfg.get('source_commit', cfg['source_commit']) != cfg['source_commit']:
        raise ValueError('ASR checkpoint expects a different icefall source commit')
    assets = checkpoint.parent
    if not all((assets / name).is_file() for name in ('bpe.model','tokens.txt','config.json')):
        assets = Path(snapshot_download(asr_cfg.get('model_id', 'g-group-ai-lab/gipformer1.5-68M-rnnt'),
            revision=asr_cfg.get('model_revision') or 'dd9227d',
            allow_patterns=['bpe.model','tokens.txt','config.json']))
    asr_cfg.update(model_dir=str(assets), source_dir=cfg['source_dir'],
                   vocab_path=str(assets / 'tokens.txt'), code_dir=cfg['asr_code_dir'])
    if asr_cfg.get('bpe_sha256') and sha256_file(assets / 'bpe.model') != asr_cfg['bpe_sha256']:
        raise ValueError('ASR tokenizer hash differs from the fine-tune checkpoint')
    mode = cfg['asr_decoder']
    if mode == 'auto':
        mode = asr_cfg.get('selected_decoder')
        if not mode:
            for filename, key in [('decoder_validation_comparison.json','selected_decoder'), ('metrics.json','decoder')]:
                metadata = nearby(checkpoint, filename)
                if metadata:
                    mode = read_json(metadata).get(key)
                    if mode:
                        break
        if not mode:
            mode = 'beam_context'
            print('No saved decoder selection; using beam_context (beam=4, saved context weight).', flush=True)
    if mode not in ('greedy','beam','beam_context'):
        raise ValueError('Unsupported ASR decoder: ' + str(mode))
    os.environ['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT'] = str(asr_cfg.get('cudnn_plan_cache_limit',128))
    torch.backends.cudnn.benchmark = False
    weights = unpack_weights(state)
    # Instantiate architecture ONLY; never call build_model/load_base here.
    # Those helpers initialize from pretrained model.pt; this path loads the submitted fine-tune directly.
    from worker import SpeechRNNT, VocabTokenizer
    from speech_features import extract_feature
    from rnnt_decode import greedy_ids, beam_ids
    from text_processing import context_phrases, spoken_text
    model = SpeechRNNT(asr_cfg)
    model.asr.load_state_dict(weights, strict=True)
    del weights, state
    gc.collect()
    model.requires_grad_(False)
    model = model.to('cuda:0').eval()
    tokenizer = VocabTokenizer(assets / 'tokens.txt')
    fingerprint = {'run':cfg['run_fingerprint'], 'decoder':mode, 'world':cfg['num_gpus'],
                   'rank':rank, 'asr_cfg':asr_cfg,
                   'assets':{n:sha256_file(assets/n) for n in ('config.json','tokens.txt','bpe.model')}}
    guard(out / f'fingerprint.rank{rank}.json', fingerprint)
    target = out / f'transcripts.rank{rank}.jsonl'
    previous = read_jsonl(target) if target.exists() else []
    saved = {str(row['id']):row for row in previous}
    if len(saved) != len(previous):
        raise ValueError('Duplicate ASR resume IDs')
    items = read_jsonl(cfg['input_jsonl'])[rank::cfg['num_gpus']]
    allowed = {str(item['id']) for item in items}
    if set(saved) - allowed:
        raise ValueError('ASR resume contains IDs outside this shard')
    write_json(out / f'decoder.rank{rank}.json', {'decoder':mode, 'config':asr_cfg})
    for i, item in enumerate(items):
        sid = str(item['id'])
        if sid in saved:
            continue
        phrases = context_phrases(item,
            Path(cfg['data_root']) / cfg['split'] if asr_cfg.get('use_initial_state_context',True) else None,
            cfg['split'], int(asr_cfg.get('max_context_phrases',128))) if mode == 'beam_context' else []
        row = {'id':sid, 'history_audio_texts':{}, 'asr_audio':[]}
        # Decode every audio once within a sample, including history u1/u2/... .
        decoded = {}
        for kind, index, relative, audio in audio_jobs(item,cfg):
            if str(audio) not in decoded:
                features, duration = extract_feature(audio)
                with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
                    encoded, _, lengths = model.encode(features[None].to('cuda:0'),
                        torch.tensor([len(features)],dtype=torch.long,device='cuda:0'))
                    ids = (greedy_ids(model.asr,encoded,lengths)[0] if mode == 'greedy' else
                           beam_ids(model.asr,encoded,int(lengths[0]),tokenizer,
                                    phrases if mode == 'beam_context' else (), asr_cfg))
                text = spoken_text(tokenizer.decode(ids),asr_cfg)
                if not text.strip():
                    raise ValueError(f'ASR returned empty text: {sid} {relative}; inspect the audio/decoder')
                decoded[str(audio)] = (text,duration)
            text,duration = decoded[str(audio)]
            if kind == 'main':
                row['audio_text'] = text
            else:
                row['history_audio_texts'].update({f'u{index}':text, relative:text, Path(relative).name:text})
            row['asr_audio'].append({'kind':kind, 'user_index':index, 'audio':relative,
                                     'seconds':duration, 'text':text})
        if not row.get('audio_text'):
            raise ValueError('Missing final-request ASR for ' + sid)
        saved[sid] = row
        write_jsonl(target, [saved[str(x['id'])] for x in items if str(x['id']) in saved])
        print(f'ASR GPU {rank} {i+1}/{len(items)} {sid}: {row["audio_text"]}', flush=True)
    # Empty shards still produce a real, empty JSONL file.
    write_jsonl(target, [saved[str(x['id'])] for x in items])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    main(read_json(parser.parse_args().config))
