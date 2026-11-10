"""An ASR provider supplies text, never replacement context/history/tool data."""
import copy
from agent_core import read_jsonl

def merge_transcripts(items, rows):
    mapping = {str(r['id']): r for r in rows}
    if len(mapping) != len(rows) or set(mapping) != {str(r['id']) for r in items}:
        raise ValueError('ASR IDs must cover the public split exactly once')
    merged = []
    for original in items:
        sid = str(original['id'])
        transcript = mapping[sid]
        text = transcript.get('audio_text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'{sid}: missing final ASR audio_text')
        item = {k: copy.deepcopy(original[k]) for k in ('id', 'audio', 'context', 'history') if k in original}
        item['audio_text'] = text.strip()
        users = [t for t in item.get('history', []) if t['role'] == 'user']
        history = transcript.get('history_audio_texts', [])
        if isinstance(history, list):
            if len(history) != len(users):
                raise ValueError(f'{sid}: expected {len(users)} user-history transcripts, got {len(history)}')
            values = history
        elif isinstance(history, dict):
            # Exact relative audio paths, not ambiguous basenames or turn indices.
            values = [history.get(t['audio']) for t in users]
        else:
            raise ValueError(f'{sid}: history_audio_texts must be list or audio-path dictionary')
        for turn, value in zip(users, values):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{sid}: missing history ASR for {turn.get("audio")}')
            for key in ('audio_text', 'text', 'transcript'):
                turn.pop(key, None)
            turn['audio_text'] = value.strip()
        merged.append(item)
    return merged
