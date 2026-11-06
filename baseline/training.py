"""Shared SFT export only: actual fine-tuning belongs to the teammate's trainer."""
import json
from .prompts import ASK, MAX_RESULT_CHARS, build_messages, to_fn

FINAL_REPLY = {'act': 'Dạ, mình đã xử lý xong.', 'ask': 'Bạn cho mình thêm thông tin để làm tiếp nhé.',
               'none': 'Dạ, việc này mình không cần thao tác gì.'}


def sft_conversation(row, text_fn):
    messages = build_messages(row, [], text_fn)
    first = len(messages)
    calls, results = row['label']['tool_calls'], row['label']['tool_results']
    if len(calls) != len(results):
        raise ValueError(f"Misaligned train trajectory: {row['id']}")
    for i, (call, result) in enumerate(zip(calls, results)):
        messages.append({'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': f't{i}', 'type': 'function', 'function': {'name': to_fn(call['name']),
                'arguments': json.dumps(call['arguments'], ensure_ascii=False)}}]})
        messages.append({'role': 'tool', 'tool_call_id': f't{i}',
            'content': json.dumps(result, ensure_ascii=False)[:MAX_RESULT_CHARS]})
    kind = 'none' if not calls else ('ask' if calls[-1]['name'] == ASK else 'act')
    messages.append({'role': 'assistant', 'content': FINAL_REPLY[kind]})
    return messages, first


def tokenize_sft(tokenizer, messages, tools, first):
    """Tokenize once, masking everything except target assistant spans.

    Using offsets avoids changing BPE segmentation by separately tokenizing substrings.
    Reject incompatible chat templates rather than silently learning the wrong spans.
    """
    full = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, enable_thinking=False)
    spans = []
    for i in range(first, len(messages)):
        if messages[i]['role'] != 'assistant':
            continue
        prefix = tokenizer.apply_chat_template(messages[:i], tools=tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        upto = tokenizer.apply_chat_template(messages[:i + 1], tools=tools, tokenize=False, enable_thinking=False)
        if not upto.startswith(prefix) or not full.startswith(upto):
            raise ValueError('Chat template does not preserve assistant prefix; cannot safely build loss mask')
        spans.append((len(prefix), len(upto)))
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    labels = [token if end > start and any(start >= a and end <= b for a, b in spans) else -100
              for token, (start, end) in zip(encoded['input_ids'], encoded['offset_mapping'])]
    if not any(x != -100 for x in labels):
        raise ValueError('No assistant target tokens')
    return {'input_ids': encoded['input_ids'], 'labels': labels}
