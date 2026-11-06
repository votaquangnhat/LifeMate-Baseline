"""Check the environment, data catalogs, and checkpoint files without loading weights."""
import argparse
import importlib.metadata
from pathlib import Path
import sys

from .config import load_config
from .data import load_split, read_json
from .agent import available_tools


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/baseline.json')
    p.add_argument('--split', default='train', choices=['train', 'public_test', 'private_test'])
    args = p.parse_args()
    cfg = load_config(args.config)
    failures = []
    print(f'Python {sys.version.split()[0]} | executable: {sys.executable}')
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        failures.append('Use Python 3.11 or 3.12 for the pinned dependencies')
    for package in ('torch', 'transformers', 'peft', 'soundfile', 'scikit-learn'):
        try:
            print(f'{package}: {importlib.metadata.version(package)}')
        except importlib.metadata.PackageNotFoundError:
            failures.append(f'Missing package: {package}')
    rows, catalog = load_split(cfg['data_root'], args.split)
    for row in rows:
        available_tools(catalog, row['context'])
    print(f'{args.split}: {len(rows)} samples; catalogs consistent')
    try:
        import torch
        print('CUDA:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('GPU:', torch.cuda.get_device_name(0))
    except ImportError:
        torch = None
    for name in ('llm', 'asr'):
        spec = cfg[name]
        path = Path(spec['path'])
        weights = list(path.glob('*.safetensors')) + list(path.glob('pytorch_model*.bin'))
        if not (path / 'config.json').is_file() or not weights:
            failures.append(f'{name}: full weights/config missing at {path}')
        if spec['device'].startswith('cuda') and (torch is None or not torch.cuda.is_available()):
            failures.append(f'{name}: CUDA requested but unavailable in this Python environment')
        if spec['adapter'] and not (Path(spec['adapter']) / 'adapter_config.json').is_file():
            failures.append(f'{name}: missing adapter_config.json in {spec["adapter"]}')
        print(f'{name}: {path}; adapter={spec["adapter"]}')
    if cfg['agent']['fewshot']:
        for key in ('pool_ids', 'retrieval_path'):
            if not Path(cfg[key]).is_file():
                failures.append(f'Missing {cfg[key]}; run python -m baseline.prepare')
    if failures:
        raise SystemExit('\n'.join(f'FAIL: {message}' for message in failures))
    print('Preflight passed. This does not certify accuracy, GPU memory fit, or the 30s limit.')


if __name__ == '__main__':
    main()
