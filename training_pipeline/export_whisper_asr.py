"""Optional inference only for a teammate's merged Whisper-compatible checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from transformers import pipeline
from agent_core import read_jsonl, write_jsonl

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.root) / 'public_test'
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError('Use the existing transcript file or choose a new output path')
    recognizer = pipeline('automatic-speech-recognition', model=args.model, device=0,
                          torch_dtype=torch.float16, chunk_length_s=30)
    def text(turn):
        path = (root / turn['audio']).resolve()
        if not path.is_relative_to((root / 'audio').resolve()):
            raise ValueError('Audio path escapes audio directory')
        value = recognizer(str(path), generate_kwargs={'language':'vi', 'task':'transcribe'})['text'].strip()
        if not value:
            raise ValueError('Empty ASR transcript: ' + str(path))
        return value
    rows = []
    for i, item in enumerate(read_jsonl(root / 'public_test.jsonl')):
        rows.append({'id': item['id'], 'audio_text': text(item),
                     'history_audio_texts': [text(t) for t in item.get('history', []) if t['role'] == 'user']})
        if (i + 1) % 20 == 0:
            print('ASR samples:', i + 1, flush=True)
    write_jsonl(destination, rows)
