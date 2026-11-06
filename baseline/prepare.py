"""Create a common train/dev split plus ASR and native tool SFT handoff files."""
import argparse
import hashlib
from pathlib import Path
import random

from .agent import available_tools
from .data import DEFAULT_ROOT, load_split, utterances, write_json, write_jsonl
from .prompts import PROMPT_VERSION, build_tools
from .training import sft_conversation, tokenize_sft
from .vnnum import itn_time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--output', type=Path, default=Path('artifacts'))
    p.add_argument('--dev-size', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tokenizer', help='Optional local tokenizer path; also exports mentor-compatible sft_data.jsonl')
    p.add_argument('--max-length', type=int, default=12288)
    args = p.parse_args()
    rows, catalog = load_split(args.data_root, 'train')
    if not 0 < args.dev_size < len(rows):
        p.error('dev-size must leave both train and dev nonempty')
    if not 1 <= args.max_length <= 16384:
        p.error('max-length must be in [1, 16384]')
    dev_ids = set(random.Random(args.seed).sample([r['id'] for r in rows], args.dev_size))
    pool, dev = [r for r in rows if r['id'] not in dev_ids], [r for r in rows if r['id'] in dev_ids]
    # Same seed/sample rule as the mentor notebook. Keep all history from each row together.
    for name, selected in [('pool', pool), ('dev', dev)]:
        write_json(args.output / 'splits' / f'{name}_ids.json', [r['id'] for r in selected])
        write_jsonl(args.output / f'asr_{name}.jsonl', [
            {'id': sid, 'audio': turn['audio'], 'text': turn['audio_text'], 'split': 'train'}
            for sid, turn in utterances(selected)])
        # Direct input shape expected by the mentor's ASR training script.
        write_json(args.output / f'asr_{name}_pairs.json', [
            [turn['audio'], turn['audio_text']] for _, turn in utterances(selected)])
    exports = []
    for row in pool:
        messages, first = sft_conversation(row, lambda t: itn_time(t['audio_text']))
        exports.append({'id': row['id'], 'prompt_version': PROMPT_VERSION,
            'messages': messages, 'tools': build_tools(available_tools(catalog, row['context']), row),
            'first_target_message': first})
    write_jsonl(args.output / 'sft_pool.jsonl', exports)
    write_jsonl(args.output / 'fewshot_pool.jsonl', [
        {'id': r['id'], 'audio_text': r['audio_text'], 'label': {
            'tool_calls': r['label']['tool_calls'], 'tool_results': r['label']['tool_results']}}
        for r in pool])
    skipped = 0
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        tokenized = []
        for item in exports:
            record = tokenize_sft(tokenizer, item['messages'], item['tools'], item['first_target_message'])
            if len(record['input_ids']) > args.max_length:
                skipped += 1
            else:
                tokenized.append(record)
        write_jsonl(args.output / 'sft_data.jsonl', tokenized)
    manifest = {'seed': args.seed, 'dev_size': len(dev), 'pool_size': len(pool),
        'prompt_version': PROMPT_VERSION, 'itn': True, 'max_length': args.max_length,
        'tokenizer': args.tokenizer, 'skipped_long': skipped,
        'train_manifest_sha256': hashlib.sha256((args.data_root / 'train/train.jsonl').read_bytes()).hexdigest()}
    write_json(args.output / 'data_manifest.json', manifest)
    print(f'Pool={len(pool)}, dev={len(dev)}, SFT={len(exports)}, skipped_long={skipped} -> {args.output}')


if __name__ == '__main__':
    main()
