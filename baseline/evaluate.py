"""Train-only reference preparation and a pure file-based reconstruction of mentor metric."""
import argparse
from collections import Counter
from pathlib import Path
import re
import unicodedata

from .data import DEFAULT_ROOT, load_runtime, load_split, read_jsonl, select_rows, write_json, write_jsonl


def token_f1(a, b):
    def tokens(s):
        return re.findall(r'\w+', unicodedata.normalize('NFC', str(s).lower()))
    left, right = tokens(a), tokens(b)
    if not left and not right:
        return 1.0
    common = sum((Counter(left) & Counter(right)).values())
    return 2 * common / (len(left) + len(right)) if common else 0.0


def call_match(submitted, gold, specs):
    if not isinstance(submitted, dict) or submitted.get('name') != gold['name']:
        return False
    sa, ga = submitted.get('arguments'), gold['arguments']
    if not isinstance(sa, dict) or set(sa) != set(ga):
        return False
    props = specs[gold['name']].get('parameters', {}).get('properties', {})
    return all(token_f1(sa[key], value) > 0.6 if props.get(key, {}).get('x-match') == 'fuzzy'
               else sa[key] == value and (isinstance(sa[key], bool) == isinstance(value, bool))
               for key, value in ga.items())


def n_matched(submitted, gold, specs):
    edges = [[j for j, g in enumerate(gold) if call_match(s, g, specs)] for s in submitted]
    owners = [-1] * len(gold)
    def augment(i, seen):
        for j in edges[i]:
            if j not in seen:
                seen.add(j)
                if owners[j] == -1 or augment(owners[j], seen):
                    owners[j] = i
                    return True
        return False
    return sum(augment(i, set()) for i in range(len(submitted)))


def score_one(prediction, reference, specs):
    sub, gold = prediction['tool_calls'], reference['tool_calls']
    matched = n_matched(sub, gold, specs)
    f1 = 2 * matched / (len(sub) + len(gold)) if sub or gold else 1.0
    state = prediction['state_hash'] == reference['state_hash']
    score = float(not sub) if not gold else (f1 if state else 0.0)
    return {'score': score, 'state_match': state, 'call_f1': f1, 'matched': matched}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['references', 'score'])
    p.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--ids', default='artifacts/splits/dev_ids.json')
    p.add_argument('--references', type=Path, default=Path('artifacts/dev_references.jsonl'))
    p.add_argument('--predictions', type=Path)
    p.add_argument('--output', type=Path, default=Path('result/evaluation.json'))
    p.add_argument('--allow-partial', action='store_true', help='Explicitly score only predicted reference IDs; report coverage')
    p.add_argument('--diagnostics', type=Path)
    p.add_argument('--enforce-timeouts', action='store_true')
    args = p.parse_args()
    rows, catalog = load_split(args.data_root, 'train')
    rows = select_rows(rows, args.ids)
    if args.mode == 'references':
        if args.references.exists():
            p.error('References already exist; choose a fresh reference filename')
        runtime = load_runtime(args.data_root, 'train')
        refs = []
        for row in rows:
            # Separate evaluation command, train only, one official environment per reference.
            env = runtime.ToolEnv('train', row['id'], root=args.data_root)
            for call in row['label']['tool_calls']:
                result = env.call(call['name'], call['arguments'])
                if not result.get('ok'):
                    raise ValueError(f"Invalid gold call for {row['id']}: {result}")
            refs.append({'id': row['id'], 'tool_calls': env.get_tool_calls(), 'state_hash': env.get_state_hash()})
        write_jsonl(args.references, refs)
        print(f'Prepared {len(refs)} TRAIN references -> {args.references}')
        return
    if not args.predictions:
        p.error('--predictions is required for score')
    references = {r['id']: r for r in read_jsonl(args.references)}
    if set(references) != {r['id'] for r in rows}:
        p.error('References do not match selected train IDs')
    predictions = read_jsonl(args.predictions)
    ids = [r['id'] for r in predictions]
    if len(set(ids)) != len(ids) or set(ids) - references.keys() or not ids:
        p.error('Duplicate, unknown, or empty prediction IDs')
    if not args.allow_partial and set(ids) != set(references):
        p.error('Missing predictions; use --allow-partial only for explicit smoke tests')
    diagnostics = {r['id']: r for r in read_jsonl(args.diagnostics)} if args.diagnostics else {}
    if args.enforce_timeouts and (not set(ids) <= diagnostics.keys() or
            not all(diagnostics[i].get('timing_includes_asr') for i in ids)):
        p.error('Timeout enforcement needs complete end-to-end audio diagnostics')
    specs = {t['name']: t for app in catalog['apps'] for t in app['tools']}
    scores = []
    for pred in predictions:
        record = {'id': pred['id'], **score_one(pred, references[pred['id']], specs)}
        if pred['id'] in diagnostics:
            record['seconds'] = diagnostics[pred['id']]['seconds']
            if args.enforce_timeouts and record['seconds'] > 30:
                record['score'] = 0.0
        scores.append(record)
    report = {'metric': 'mentor reconstruction; supplied hashes trusted, no prediction replay',
        'count': len(scores), 'reference_count': len(references),
        'score': 100 * sum(r['score'] for r in scores) / len(scores),
        'state_match_rate': sum(r['state_match'] for r in scores) / len(scores),
        'mean_call_f1': sum(r['call_f1'] for r in scores) / len(scores), 'samples': scores}
    write_json(args.output, report)
    print(f"Score={report['score']:.2f}; coverage={len(scores)}/{len(references)} -> {args.output}")


if __name__ == '__main__':
    main()
