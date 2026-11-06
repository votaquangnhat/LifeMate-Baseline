"""Native Qwen tool-call loop; the same message format is exported for SFT."""
from datetime import datetime
import json
import re
import time
from zoneinfo import ZoneInfo

from .prompts import ASK, MAX_RESULT_CHARS, from_fn, to_fn


def parse_tool_calls(raw):
    blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', raw, re.S)
    if raw.count('<tool_call>') != len(blocks) or raw.count('</tool_call>') != len(blocks):
        raise ValueError('Incomplete tool_call block; emit a complete JSON object inside tool_call tags.')
    if not blocks:
        # Plain text means finished. Bare JSON that looks like a tool is a format error.
        if '"arguments"' in raw or '"name"' in raw:
            raise ValueError('Tool calls must use <tool_call>{"name":...,"arguments":{...}}</tool_call>.')
        return []
    calls = []
    for block in blocks:
        obj = json.loads(block)
        if not isinstance(obj, dict) or not isinstance(obj.get('name'), str):
            raise ValueError('Tool call needs a string name')
        args = obj.get('arguments')
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict):
            raise ValueError('Tool arguments must be a JSON object')
        calls.append({'name': from_fn(obj['name']), 'arguments': args})
    return calls


def normalize_arguments(args, spec, context):
    """Only unambiguous schema-driven formatting. Never prune optional intent values."""
    props = spec.get('parameters', {}).get('properties', {})
    result = dict(args)
    for key, value in args.items():
        prop = props.get(key, {})
        if prop.get('type') == 'integer' and isinstance(value, str) and re.fullmatch(r'-?\d+', value.strip()):
            result[key] = int(value)
        elif prop.get('type') == 'integer' and isinstance(value, float) and value.is_integer():
            result[key] = int(value)
        elif isinstance(value, str) and prop.get('enum'):
            matches = [v for v in prop['enum'] if isinstance(v, str) and v.casefold() == value.strip().casefold()]
            if len(matches) == 1:
                result[key] = matches[0]
        elif isinstance(value, str) and prop.get('format') == 'date-time':
            # Do not turn dates alone or arbitrary free text into timestamps.
            if re.match(r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}', value):
                try:
                    timezone = ZoneInfo(context['timezone']) if context.get('timezone') else datetime.fromisoformat(context['current_time']).tzinfo
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    dt = dt.replace(tzinfo=timezone) if dt.tzinfo is None else dt.astimezone(timezone)
                    result[key] = dt.isoformat(timespec='seconds')
                except (ValueError, KeyError):
                    pass
    return result


def execute(env, planner, messages, tools, specs, context, deadline, max_steps=14, max_calls=20, progress=None):
    trace, successful_writes = [], set()
    stop_reason = 'step_limit'
    for step in range(max_steps):
        if time.monotonic() >= deadline:
            stop_reason = 'timeout'
            break
        if not planner.fits(messages, tools):
            stop_reason = 'context_limit'
            break
        if progress:
            progress(f'  Planner step {step + 1}/{max_steps} ...')
        raw = planner.generate(messages, tools, deadline)
        trace.append({'raw': raw, 'calls': []})
        if time.monotonic() >= deadline:
            stop_reason = 'timeout'
            break
        try:
            calls = parse_tool_calls(raw)
        except ValueError as exc:
            messages.extend([{'role': 'assistant', 'content': raw},
                {'role': 'user', 'content': f'Chưa thực thi tool. Lỗi định dạng: {exc}. Hãy sửa.'}])
            trace[-1]['error'] = str(exc)
            continue
        if not calls:
            stop_reason = 'finished'
            break
        ids = [f's{step}_{i}' for i in range(len(calls))]
        messages.append({'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': cid, 'type': 'function', 'function': {'name': to_fn(c['name']), 'arguments': c['arguments']}}
            for cid, c in zip(ids, calls)]})
        stop = False
        for cid, call in zip(ids, calls):
            name, args = call['name'], call['arguments']
            executed = False
            if time.monotonic() >= deadline:
                stop, stop_reason = True, 'timeout'
            if stop:
                result = {'ok': False, 'error': 'Đã dừng, chưa thực thi.'}
            elif name not in specs:
                result = {'ok': False, 'error': 'Tool không có trong danh sách được phép; chưa thực thi.'}
            else:
                args = normalize_arguments(args, specs[name], context)
                signature = json.dumps([name, args], ensure_ascii=False, sort_keys=True)
                validation = env.validate(name, args)
                if not validation['ok']:
                    result = {**validation, 'note': 'Chưa thực thi; sửa tham số.'}
                elif signature in successful_writes:
                    result = {'ok': False, 'error': 'Thao tác ghi giống hệt đã thành công; không lặp lại.'}
                else:
                    result = env.call(name, args)
                    executed = True
                    if specs[name].get('kind') == 'write' and result.get('ok') and result.get('done') is not False:
                        successful_writes.add(signature)
                    if name == ASK and result.get('ok'):
                        stop, stop_reason = True, 'asked_missing_info'
                    elif len(env.get_tool_calls()) >= max_calls:
                        stop, stop_reason = True, 'call_limit'
            trace[-1]['calls'].append({'name': name, 'arguments': args, 'result': result, 'executed': executed})
            if progress:
                progress(f"  {'Executed' if executed else 'Not executed'}: {name} {json.dumps(args, ensure_ascii=False)}")
            messages.append({'role': 'tool', 'tool_call_id': cid,
                'content': json.dumps(result, ensure_ascii=False)[:MAX_RESULT_CHARS]})
        if stop:
            break
    return {'stop_reason': stop_reason, 'trace': trace}
