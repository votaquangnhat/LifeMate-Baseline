"""Validate submission structure, complete coverage, and schemas without opening any environment."""
import argparse
from pathlib import Path
import re
from .agent import available_tools
from .data import DEFAULT_ROOT, load_runtime, load_split, read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('submission', type=Path)
    p.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--split', choices=['train', 'public_test', 'private_test'], default='public_test')
    args = p.parse_args()
    rows, catalog = load_split(args.data_root, args.split)
    rows = {r['id']: r for r in rows}
    predictions = read_jsonl(args.submission)
    ids = [r.get('id') for r in predictions]
    if len(ids) != len(set(ids)) or set(ids) != set(rows):
        raise ValueError('Submission IDs must cover the split exactly once')
    runtime = load_runtime(args.data_root, args.split)
    errors = []
    for pred in predictions:
        if set(pred) != {'id', 'tool_calls', 'state_hash'} or not isinstance(pred['tool_calls'], list):
            raise ValueError(f'Invalid submission fields for {pred["id"]}')
        if not isinstance(pred['state_hash'], str) or not re.fullmatch(r'[0-9a-f]{64}', pred['state_hash']):
            raise ValueError(f'Invalid SHA256 state hash for {pred["id"]}')
        specs = available_tools(catalog, rows[pred['id']]['context'])
        for call in pred['tool_calls']:
            if not isinstance(call, dict) or set(call) != {'name', 'arguments'} or not isinstance(call['arguments'], dict):
                errors.append((pred['id'], 'Malformed call'))
                continue
            error = runtime.schema_error(call['name'], call['arguments'], specs, set(specs))
            if error:
                errors.append((pred['id'], error))
    if errors:
        raise ValueError(f'{len(errors)} invalid calls: {errors[:10]}')
    print(f'Validated {len(predictions)} samples: coverage and schemas OK. Hash values are not replay-verified.')


if __name__ == '__main__':
    main()
