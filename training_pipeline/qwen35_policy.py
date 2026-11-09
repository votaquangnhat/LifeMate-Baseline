"""Explicit Qwen3.5 thinking protocol; no generated rationale is executed."""
import json


def configure_tokenizer(tokenizer, cfg):
    tokenizer.lifemate_thinking = bool(cfg.get("thinking_mode"))
    if not tokenizer.lifemate_thinking:
        return
    prompt = tokenizer.apply_chat_template([{"role":"user", "content":"test"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if not prompt.endswith("<|im_start|>assistant\n<think>\n"):
        raise ValueError("Expected Qwen3.5 thinking template")
    for token in ("<think>", "</think>", "<|im_end|>"):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or ids[0] != tokenizer.convert_tokens_to_ids(token):
            raise ValueError("Invalid special token: " + token)
    if tokenizer.convert_tokens_to_ids("<|im_end|>") != tokenizer.eos_token_id:
        raise ValueError("Unexpected Qwen3.5 EOS")


def training_suffix(row):
    # These are short procedural scaffolds, NOT teacher-generated reasoning.
    # Only the current supervised action is consulted, never future observations.
    target = json.loads(row["target"])
    if target.get("done") is True:
        reason = "I should end this turn without issuing an additional tool action."
    else:
        reason = "I will issue the next tool action using its JSON schema, preserving the requested values and available tool results."
    return reason + "\n</think>\n\n" + row["target"] + "<|im_end|>"


def action_from_tokens(tokens, tokenizer):
    end = tokenizer.eos_token_id
    closing = tokenizer.convert_tokens_to_ids("</think>")
    opening = tokenizer.convert_tokens_to_ids("<think>")
    if end not in tokens:
        raise ValueError("Action exhausted token budget without EOS; partial call was not executed")
    tokens = tokens[:tokens.index(end)]
    if tokens.count(closing) != 1 or opening in tokens:
        raise ValueError("Malformed thinking completion; no tool call executed")
    action = tokens[tokens.index(closing)+1:]
    text = tokenizer.decode(action, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
    if not text:
        raise ValueError("Empty action after thinking")
    return text


class ThinkingBudget:
    def __init__(self, prefix_length, tokenizer, limit):
        self.start, self.limit = prefix_length, limit
        self.open = tokenizer.convert_tokens_to_ids("<think>")
        self.close = tokenizer.convert_tokens_to_ids("</think>")
        self.eos = tokenizer.eos_token_id

    def __call__(self, input_ids, scores):
        for i, row in enumerate(input_ids):
            generated = row[self.start:].tolist()
            scores[i, self.open] = -float("inf")
            if self.close in generated:
                scores[i, self.close] = -float("inf")
            elif len(generated) >= self.limit:
                scores[i, :] = -float("inf")
                scores[i, self.close] = 0
            else:
                scores[i, self.eos] = -float("inf")
        return scores

