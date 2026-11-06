"""Small JSON configuration; all relative paths resolve from the project working directory."""
from .data import read_json

DEFAULT = {
    'data_root': 'train+public_test/train+public_test',
    'llm': {'path': 'models/Qwen3-4B-Instruct-2507', 'adapter': None,
            'tokenizer': None, 'device': 'cuda:0', 'dtype': 'float16'},
    'asr': {'path': 'models/PhoWhisper-small', 'adapter': None,
            'processor': None, 'device': 'cuda:0', 'dtype': 'float16'},
    'agent': {'seconds': 28, 'max_steps': 14, 'max_calls': 20,
              'max_context': 16384, 'max_new_tokens': 256, 'fewshot': 2, 'itn': True},
    'pool_ids': 'artifacts/splits/pool_ids.json',
    'retrieval_path': 'artifacts/fewshot_pool.jsonl', 'seed': 42,
}


def load_config(path):
    import copy
    cfg = copy.deepcopy(DEFAULT)
    if path:
        raw = read_json(path)
        for key, value in raw.items():
            if key not in cfg:
                raise ValueError(f'Unknown configuration key: {key}')
            if isinstance(cfg[key], dict):
                if not isinstance(value, dict) or set(value) - set(cfg[key]):
                    raise ValueError(f'Unknown or invalid keys in {key}')
                cfg[key].update(value)
            else:
                cfg[key] = value
    a = cfg['agent']
    if not 512 <= a['max_context'] <= 16384 or not 1 <= a['max_new_tokens'] < a['max_context']:
        raise ValueError('Invalid context/output token limits')
    if min(a['seconds'], a['max_steps'], a['max_calls']) <= 0 or a['fewshot'] < 0:
        raise ValueError('Invalid agent limits')
    return cfg
