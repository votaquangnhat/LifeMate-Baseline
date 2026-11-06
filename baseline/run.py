"""Run the shared native-tool pipeline. Use --help for checkpoint/config options."""
import argparse
import json
from pathlib import Path
import time

from .agent import available_tools
from .config import load_config
from .data import (audio_path, input_only, load_runtime, load_split, prepare_input,
                   read_jsonl, select_rows, utterances, write_json)
from .prompts import PROMPT_VERSION, FewShot, build_messages, build_tools
from .tool_agent import execute
from .vnnum import itn_time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/baseline.json')
    p.add_argument('--data-root')
    p.add_argument('--split', choices=['train', 'public_test', 'private_test'], default='train')
    p.add_argument('--ids', help='JSON array of sample IDs, e.g. artifacts/splits/dev_ids.json')
    p.add_argument('--sample-id', help='Run one specific input ID, e.g. train_0004')
    p.add_argument('--verbose', action='store_true', help='Print ASR text and tool execution progress')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--oracle-transcripts', action='store_true', help='Train-only planner diagnosis')
    p.add_argument('--transcripts', type=Path, help='Train-only ASR cache {audio,text} JSONL; timing excludes ASR')
    p.add_argument('--seconds', type=float, help='Override per-sample cooperative budget')
    args = p.parse_args()
    if args.split != 'train' and (args.oracle_transcripts or args.transcripts):
        p.error('Oracle/cached transcripts are train-only diagnostics. Test inference must process audio.')
    if args.oracle_transcripts and args.transcripts:
        p.error('Choose oracle OR cached transcripts')
    if args.limit < 0 or args.offset < 0:
        p.error('limit and offset must be nonnegative')
    cfg = load_config(args.config)
    if args.data_root:
        cfg['data_root'] = args.data_root
    if args.seconds is not None:
        if args.seconds <= 0:
            p.error('--seconds must be positive')
        cfg['agent']['seconds'] = args.seconds
    root = Path(cfg['data_root']).resolve()
    rows, catalog = load_split(root, args.split)
    rows = select_rows(rows, args.ids, args.limit, args.offset)
    if args.sample_id:
        rows = [r for r in rows if r['id'] == args.sample_id]
    if not rows:
        p.error('No selected samples')
    sidecar = args.output.with_suffix('.diagnostics.jsonl')
    manifest_path = args.output.with_suffix('.run.json')
    if any(path.exists() for path in (args.output, sidecar, manifest_path)):
        p.error('Output already exists: choose a new filename to preserve previous results')
    cache = None
    if args.transcripts:
        records = read_jsonl(args.transcripts)
        cache = {r['audio']: r['text'] for r in records}
        if len(cache) != len(records) or not all(isinstance(v, str) for v in cache.values()):
            p.error('Cache needs unique audio keys and string text values')
        missing = {t['audio'] for _, t in utterances(rows)} - cache.keys()
        if missing:
            p.error(f'Cache is missing {len(missing)} audio files, including history')
    fewshot = None
    if cfg['agent']['fewshot']:
        # Prepared training-only retrieval travels with the package, independent of /data.
        pool = select_rows(read_jsonl(cfg['retrieval_path']), cfg['pool_ids'])
        if args.split == 'train' and {r['id'] for r in pool} & {r['id'] for r in rows}:
            p.error('Selected train samples overlap retrieval pool. Use --ids artifacts/splits/dev_ids.json or set fewshot=0.')
        fewshot = FewShot(pool)
    # Heavy imports only after input/config validation. Help and tests need no ML stack.
    from .models import Planner, SpeechRecognizer, setup_runtime
    torch = setup_runtime(cfg['seed'])
    l, a, agent = cfg['llm'], cfg['asr'], cfg['agent']
    print(f"Loading planner: {l['path']} on {l['device']} ...", flush=True)
    planner = Planner(l['path'], device=l['device'], dtype=l['dtype'], adapter=l['adapter'],
        tokenizer=l['tokenizer'], max_context=agent['max_context'], max_new_tokens=agent['max_new_tokens'])
    asr = None
    if not args.oracle_transcripts and cache is None:
        print(f"Loading ASR: {a['path']} on {a['device']} ...", flush=True)
        asr = SpeechRecognizer(a['path'], device=a['device'], dtype=a['dtype'],
                               adapter=a['adapter'], processor=a['processor'])
    parameters = planner.parameter_count + (asr.parameter_count if asr else 0)
    if parameters >= 6_000_000_000:
        raise ValueError(f'Total parameters {parameters} exceeds the <6B limit')
    print(f'Models ready. Processing {len(rows)} sample(s).', flush=True)
    runtime = load_runtime(root, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'config': cfg, 'prompt_version': PROMPT_VERSION, 'split': args.split,
        'ids': [r['id'] for r in rows], 'oracle_transcripts': args.oracle_transcripts,
        'transcripts': str(args.transcripts) if args.transcripts else None,
        'loaded_parameters': parameters, 'timing_includes_asr': asr is not None,
        'torch': torch.__version__}
    write_json(manifest_path, manifest)
    errors = 0
    with args.output.open('x', encoding='utf-8') as output, sidecar.open('x', encoding='utf-8') as logs:
        for original in rows:
            start = time.monotonic()
            deadline = start + agent['seconds']
            row = original if args.oracle_transcripts else input_only(original)
            env = runtime.ToolEnv(args.split, row['id'], root=root)
            transcripts = {}
            diagnostics = {}
            asr_seconds = 0.0
            if args.verbose:
                print(f"[{row['id']}] Starting audio processing ...", flush=True)
            try:
                asr_start = time.monotonic()
                for _, turn in utterances([row]):
                    if args.oracle_transcripts:
                        text = turn['audio_text']
                    elif cache is not None:
                        text = cache[turn['audio']]
                    else:
                        text = asr.transcribe(audio_path(root / args.split, turn['audio']), deadline)
                    transcripts[turn['audio']] = itn_time(text) if agent['itn'] else text
                    if args.verbose:
                        print(f"  ASR {turn['audio']}: {text}", flush=True)
                asr_seconds = time.monotonic() - asr_start
                text_fn = lambda turn: transcripts[turn['audio']]
                demos = fewshot.topk(text_fn(row), agent['fewshot'], exclude=row['id']) if fewshot else []
                specs = available_tools(catalog, row['context'])
                tools = build_tools(specs, row)
                messages = build_messages(row, demos, text_fn)
                # Drop demonstrations first if the prompt is too long; never drop tool schemas or current input.
                while demos and not planner.fits(messages, tools):
                    demos.pop()
                    messages = build_messages(row, demos, text_fn)
                diagnostics = execute(env, planner, messages, tools, specs, row['context'],
                    deadline, agent['max_steps'], agent['max_calls'],
                    progress=(lambda message: print(message, flush=True)) if args.verbose else None)
            except Exception as exc:
                diagnostics = {'stop_reason': 'timeout' if isinstance(exc, TimeoutError) else 'error',
                    'error': f'{type(exc).__name__}: {exc}'}
                errors += 1
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
            prediction = {'id': row['id'], 'tool_calls': env.get_tool_calls(), 'state_hash': env.get_state_hash()}
            seconds = time.monotonic() - start
            diagnostics.update(id=row['id'], seconds=seconds, exceeded_30s=seconds > 30,
                transcripts=transcripts, timing_includes_asr=asr is not None, asr_seconds=asr_seconds)
            output.write(json.dumps(prediction, ensure_ascii=False) + '\n')
            logs.write(json.dumps(diagnostics, ensure_ascii=False) + '\n')
            output.flush()
            logs.flush()
            print(f"{row['id']}: {len(prediction['tool_calls'])} calls, {seconds:.2f}s, {diagnostics['stop_reason']}", flush=True)
    print(f'{len(rows)} predictions, {errors} exceptions -> {args.output}')
    if errors:
        raise SystemExit('Some samples failed; inspect diagnostics before submitting.')


if __name__ == '__main__':
    main()
