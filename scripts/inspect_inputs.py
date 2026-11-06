import json
from pathlib import Path
root = Path('train+public_test/train+public_test')
for split in ('train', 'public_test'):
    with (root / split / f'{split}.jsonl').open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    roles = sorted({t.get('role', '<missing>') for x in rows for t in x.get('history', [])})
    with (root / split / f'{split}_tools_catalog.json').open() as f:
        catalog = json.load(f)
    from baseline.agent import available_tools
    sizes = [len(available_tools(catalog, x['context'])) for x in rows]
    print(split, 'samples=', len(rows), 'history_roles=', roles, 'max_tools=', max(sizes))
    if split == 'train':
        print('train history example:', next((x['history'][:1] for x in rows if x['history']), []))
