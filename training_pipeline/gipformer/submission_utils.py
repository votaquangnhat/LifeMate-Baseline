import hashlib,json,tarfile,zipfile
from pathlib import Path

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        for row in values:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def signature(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def guard(path, fingerprint):
    path = Path(path)
    if path.exists() and read_json(path) != fingerprint:
        raise RuntimeError(f'Inputs/checkpoint/config changed: {path}. Set a new RUN_NAME to create a fresh run.')
    write_json(path, fingerprint)

def strict_ids(rows):
    ids = [str(row['id']) for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Expected nonempty input with unique IDs')
    return ids

def safe_extract(source, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    def target(name):
        path = (destination / name).resolve()
        if not path.is_relative_to(destination):
            raise ValueError('Archive path escapes extraction directory: ' + name)
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                target(info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Symlinks are not accepted in checkpoint archives')
            archive.extractall(destination)
    elif tarfile.is_tarfile(source):
        with tarfile.open(source) as archive:
            for info in archive.getmembers():
                target(info.name)
                if not (info.isfile() or info.isdir()):
                    raise ValueError('Only regular files/directories are accepted in checkpoint archives')
            archive.extractall(destination)
    else:
        raise ValueError('Expected a ZIP or TAR checkpoint archive: ' + str(source))
    return destination

def pick_asr(path):
    path = Path(path)
    if path.is_file():
        if path.suffix not in ('.pt','.pth'):
            raise ValueError('ASR checkpoint must be best.pt/last.pt/model.pt, a directory, or an archive')
        return path
    for relative in ('best_export/model.pt','model.pt','best.pt'):
        if (path / relative).is_file():
            return path / relative
    options = sorted(path.rglob('best_export/model.pt')) or sorted(path.rglob('best.pt')) or sorted(path.rglob('model.pt'))
    if len(options) == 1:
        return options[0]
    raise ValueError(f'Choose an exact ASR checkpoint path; found {len(options)} candidates under {path}')

def nearby(path, filename, levels=3):
    root = Path(path)
    if root.is_file():
        root = root.parent
    for _ in range(levels):
        candidate = root / filename
        if candidate.is_file():
            return candidate
        root = root.parent
    return None

def resolve_audio(data_root, split, relative):
    data_root = Path(data_root).resolve()
    split_root = data_root / split
    value = Path(str(relative))
    candidates = [value] if value.is_absolute() else [split_root / value, data_root / value]
    for path in candidates:
        path = path.resolve()
        if path.is_relative_to(split_root.resolve()) and path.is_file():
            return path
    raise FileNotFoundError(f'Audio missing under {split_root}: {relative}')

def audio_jobs(item, cfg):
    split_root = Path(cfg['data_root']) / cfg['split']
    main = item.get('audio') or f"audio/{item['id']}/{item['id']}.wav"
    yield 'main', 0, str(main), resolve_audio(cfg['data_root'], cfg['split'], main)
    index = 0
    for turn in item.get('history', []):
        if turn.get('role') != 'user':
            continue
        index += 1
        audio = turn.get('audio')
        text = next((turn.get(k) for k in ('audio_text','transcript','text')
                     if isinstance(turn.get(k),str) and turn[k].strip()), None)
        if not audio:
            fallback = f"audio/{item['id']}/{item['id']}_u{index}.wav"
            if (split_root / fallback).is_file():
                audio = fallback
            elif text:
                continue
            else:
                raise ValueError(f"{item['id']}: user history {index} has neither audio nor provided text")
        yield 'history', index, str(audio), resolve_audio(cfg['data_root'], cfg['split'], audio)
