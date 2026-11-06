"""Train-only transcript cache for fast planner experiments. Never touches ToolEnv."""
import argparse
from pathlib import Path
import time
from .config import load_config
from .data import audio_path, load_split, select_rows, utterances, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/baseline.json')
    p.add_argument('--ids', default='artifacts/splits/dev_ids.json')
    p.add_argument('--output', type=Path, default=Path('artifacts/asr_dev_predictions.jsonl'))
    p.add_argument('--seconds-per-audio', type=float, default=120)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists; select a fresh filename')
    cfg = load_config(args.config)
    from .models import SpeechRecognizer, setup_runtime
    setup_runtime(cfg['seed'])
    a = cfg['asr']
    model = SpeechRecognizer(a['path'], a['device'], adapter=a['adapter'], processor=a['processor'], dtype=a['dtype'])
    rows, _ = load_split(cfg['data_root'], 'train')
    rows = select_rows(rows, args.ids)
    records = []
    for _, turn in utterances(rows):
        start = time.monotonic()
        text = model.transcribe(audio_path(Path(cfg['data_root']) / 'train', turn['audio']), start + args.seconds_per_audio)
        records.append({'audio': turn['audio'], 'text': text, 'seconds': time.monotonic() - start})
        print(f'{len(records)}: {turn["audio"]}', flush=True)
    write_jsonl(args.output, records)


if __name__ == '__main__':
    main()
