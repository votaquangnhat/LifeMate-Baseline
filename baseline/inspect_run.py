"""Display what happened, distinguishing a saved prediction from a completed request."""
import argparse
from collections import Counter
from pathlib import Path
from .data import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('predictions', type=Path)
    p.add_argument('--require-complete', action='store_true')
    args = p.parse_args()
    predictions = read_jsonl(args.predictions)
    logs = read_jsonl(args.predictions.with_suffix('.diagnostics.jsonl'))
    diagnostics = {r['id']: r for r in logs}
    if len(diagnostics) != len(logs) or {r['id'] for r in predictions} != set(diagnostics):
        raise SystemExit('Predictions and diagnostics have inconsistent IDs')
    states = Counter(r['stop_reason'] for r in logs)
    print(f'Samples: {len(predictions)} | stop reasons: {dict(states)}')
    print(f"Above 30 seconds: {sum(r['seconds'] > 30 for r in logs)}")
    failed_outcomes = sum(1 for diag in logs for step in diag.get('trace', [])
        for call in step.get('calls', []) if call.get('executed') and
        (call.get('result', {}).get('ok') is False or call.get('result', {}).get('done') is False))
    print(f'Unsuccessful tool outcomes: {failed_outcomes}')
    if len(predictions) <= 5:
        for pred in predictions:
            diag = diagnostics[pred['id']]
            print(f"\n{pred['id']}: {diag['stop_reason']}, {diag['seconds']:.2f}s")
            for audio, transcript in diag.get('transcripts', {}).items():
                print(f'  Heard ({audio}): {transcript}')
            executed = [call for step in diag.get('trace', []) for call in step.get('calls', []) if call.get('executed')]
            for index, call in enumerate(pred['tool_calls']):
                print(f"  Call: {call['name']} {call['arguments']}")
                if index < len(executed):
                    print(f"    Result: {executed[index]['result']}")
            print(f"  State hash: {pred['state_hash']}")
            if diag.get('error'):
                print(f"  ERROR: {diag['error']}")
    incomplete = sum(n for status, n in states.items() if status not in ('finished', 'asked_missing_info'))
    if incomplete and args.require_complete:
        raise SystemExit(f'{incomplete} sample(s) did not complete. Inspect diagnostics; do not count this as a passed smoke test.')
    print('Completion does not prove that the model selected the correct actions.')


if __name__ == '__main__':
    main()
