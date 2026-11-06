"""Input-only dataset helpers and isolated official SDK loading."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

DEFAULT_ROOT = Path('train+public_test/train+public_test')


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_split(root, split):
    directory = Path(root).resolve() / split
    rows = read_jsonl(directory / f'{split}.jsonl')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs')
    catalog = read_json(directory / f'{split}_tools_catalog.json')
    return rows, catalog


def select_rows(rows, ids_path=None, limit=0, offset=0):
    if ids_path:
        ids = read_json(ids_path)
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise ValueError('IDs file must be a JSON array of strings')
        missing = set(ids) - {r['id'] for r in rows}
        if missing or len(set(ids)) != len(ids):
            raise ValueError(f'Duplicate or unknown requested IDs: {sorted(missing)}')
        wanted = set(ids)
        rows = [r for r in rows if r['id'] in wanted]
    rows = rows[offset:]
    return rows[:limit] if limit else rows


def load_runtime(root, split):
    directory = Path(root).resolve() / split / 'zaic_tools'
    name = 'lifemate_sdk_' + hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, directory / '__init__.py',
            submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def audio_path(split_dir, relative):
    path = (Path(split_dir) / relative).resolve()
    if not path.is_relative_to((Path(split_dir) / 'audio').resolve()):
        raise ValueError('Audio path must remain inside the split audio directory')
    return path


def input_only(row):
    """Whitelist inputs: neither labels nor gold audio_text enter normal inference."""
    history = [{k: h[k] for k in ('role', 'audio', 'text', 'tool_calls', 'tool_results') if k in h}
               for h in row.get('history', [])]
    return {k: row[k] for k in ('id', 'audio', 'context')} | {'history': history}


def utterances(rows):
    seen = set()
    for row in rows:
        for turn in [row, *(h for h in row.get('history', []) if h['role'] == 'user')]:
            if turn['audio'] not in seen:
                seen.add(turn['audio'])
                yield row['id'], turn


def prepare_input(item, split_dir, recognizer, oracle, deadline):
    """Compatibility helper for the original baseline's model-free tests."""
    def text(turn):
        return turn['audio_text'] if oracle else recognizer.transcribe(audio_path(split_dir, turn['audio']), deadline)
    history = []
    for turn in item.get('history', []):
        history.append({'role': 'user', 'text': text(turn)} if turn['role'] == 'user' else
                       {k: turn[k] for k in ('role', 'text', 'tool_calls', 'tool_results') if k in turn})
    return text(item), history
