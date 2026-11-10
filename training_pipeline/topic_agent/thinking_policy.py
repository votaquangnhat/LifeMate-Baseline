"""Short grounded SFT targets and native Qwen3 Thinking framing."""
import json
import re

VERSION = 'causal_step_rationale_v1'
OPEN = '<think>'
CLOSE = '</think>'
ACTION_RULE = ('Trước action, suy nghĩ ngắn về yêu cầu, dữ kiện đã có và bước kế tiếp. '
               'Chỉ suy ra điều kiện từ dữ kiện thật. Sau phần suy luận, trả đúng một JSON theo giao thức.')


def settings(cfg):
    if not cfg.get('thinking_mode'):
        return None
    return {'version': VERSION, **{k: cfg[k] for k in
            ('router_thinking_tokens', 'executor_thinking_tokens', 'action_json_tokens')}}


def validate_template(tokenizer, enabled):
    tokenizer.topic_thinking = bool(enabled)
    probe = tokenizer.apply_chat_template([{'role': 'user', 'content': 'test'}],
                                           tokenize=False, add_generation_prompt=True, **({"enable_thinking":True} if enabled else {}))
    expected = '<|im_start|>assistant\n' + (OPEN + '\n' if enabled else '')
    if not probe.endswith(expected):
        raise ValueError('Tokenizer does not use the expected native assistant prefix for the configured model')
    if enabled:
        think_open, think_close = thought_ids(tokenizer)
        if tokenizer.encode(OPEN, add_special_tokens=False) != [think_open] or tokenizer.encode(
                CLOSE, add_special_tokens=False) != [think_close]:
            raise ValueError('Thinking delimiters must be single native tokenizer tokens')


def thought_ids(tokenizer):
    ids = tokenizer.convert_tokens_to_ids(OPEN), tokenizer.convert_tokens_to_ids(CLOSE)
    if len(set((*ids, tokenizer.eos_token_id))) != 3 or not all(isinstance(i, int) and i >= 0 for i in ids):
        raise ValueError('Invalid native Thinking delimiter IDs')
    return ids


def budget_for(messages, cfg):
    key = 'router_thinking_tokens' if messages[0]['content'].startswith('MODE=ROUTER') else 'executor_thinking_tokens'
    return int(cfg[key]), int(cfg['action_json_tokens'])


def _values(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _values(child)
    elif isinstance(value, str):
        yield value


def sft_rationale(row):
    """Procedural supervision from the current target and only prior visible observations."""
    target = json.loads(row['target'])
    task = row['task']
    if task == 'router':
        apps = ', '.join(target['apps'])
        return f'Nhóm phù hợp với bước đầu là {apps}. Chọn nhóm rồi mới đọc schema.' if apps else 'Yêu cầu này chưa cần mở nhóm công cụ.'
    if task == 'open_apps':
        return 'Bước kế tiếp cần schema của nhóm ' + ', '.join(target['open_apps']) + '. Mở schema trước khi gọi.'
    if task == 'done':
        return 'Dừng ở trạng thái hiện tại, tránh tạo thêm thao tác ngoài yêu cầu.'
    if task != 'tool_call':
        raise ValueError(f'Unsupported Thinking SFT task: {task}')
    if target['name'] == 'sys.ask_missing_info':
        return 'Thiếu dữ kiện bắt buộc cho thao tác này. Yêu cầu thông tin bổ sung trước khi thực hiện.'

    previous = [json.loads(m['content']) for m in row['messages'] if m['role'] == 'tool']
    values = {s for obs in previous for s in _values(obs)}
    grounded_id = any(k.endswith('_id') and isinstance(v, str) and v in values
                      for k, v in target['arguments'].items())
    if grounded_id:
        return 'Dùng ID có trong kết quả tra cứu trước để gọi công cụ với đúng đối tượng.'
    user = next(m['content'] for m in row['messages'] if m['role'] == 'user')
    request = user.partition('\nFINAL_REQUEST=')[2]
    if re.search(r'\b(?:nếu|trừ khi|miễn là)\b', request.casefold()):
        return ('Đối chiếu điều kiện với kết quả đã nhận rồi chọn bước tiếp theo.'
                if previous else 'Cần kiểm tra dữ kiện cho điều kiện trước khi quyết định thao tác tiếp theo.')
    return 'Chọn ' + target['name'] + ' và điền các tham số từ yêu cầu và dữ kiện hiện có.'


def assistant_suffix(row, tokenizer, cfg):
    reasoning = sft_rationale(row)
    think_cap, action_cap = budget_for(row['messages'], cfg)
    rationale = tokenizer.encode(reasoning + '\n', add_special_tokens=False)
    # Tokenize together to preserve the native boundary after the delimiter.
    action = tokenizer.encode(CLOSE + '\n\n' + row['target'] + '<|im_end|>', add_special_tokens=False)
    _, close_id = thought_ids(tokenizer)
    if not action or action[0] != close_id or action[-1] != tokenizer.eos_token_id:
        raise ValueError('Invalid native boundary in Thinking training target')
    if len(rationale) > think_cap or len(action) - 1 > action_cap:
        raise ValueError(f'{row["id"]} step {row["step"]}: full Thinking target exceeds stage budget')
    return rationale + action, len(rationale)


def split_completion(ids, open_id, close_id, eos_id, think_cap, action_cap):
    """Accept exactly one closed reasoning block and a completed action suffix."""
    if eos_id not in ids:
        raise ValueError('Action exhausted its token budget without EOS')
    ids = ids[:ids.index(eos_id)]
    if ids.count(close_id) != 1 or open_id in ids:
        raise ValueError('Expected one native </think> boundary before the action')
    boundary = ids.index(close_id)
    action = ids[boundary + 1:]
    if not action or boundary > think_cap or len(action) + 1 > action_cap:
        raise ValueError('Missing action or invalid Thinking/JSON token budget')
    return action, boundary
