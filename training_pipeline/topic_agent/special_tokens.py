"""Validate the pinned Qwen vocabularies and align the model configs."""


def align_qwen_special_tokens(model, tokenizer, family="qwen3"):
    eos, pad = {"qwen3":(151645,151643),"qwen3_5":(248046,248044)}[family]
    expected = {"bos_token_id":None, "eos_token_id":eos, "pad_token_id":pad}
    actual = {key:getattr(tokenizer,key,None) for key in expected}
    if actual != expected:
        raise ValueError(f"Expected native {family} special tokens {expected}, got {actual}")
    if tokenizer.convert_tokens_to_ids("<|im_end|>") != actual["eos_token_id"]:
        raise ValueError("Tokenizer EOS must be <|im_end|>")
    if tokenizer.convert_tokens_to_ids("<|endoftext|>") != actual["pad_token_id"]:
        raise ValueError("Tokenizer PAD must be <|endoftext|>")
    changes = {}
    configs = [("model",model.config)]
    text_config = getattr(model.config,"text_config",None)
    if text_config is not None:
        configs.append(("text_config",text_config))
    for prefix, config in configs:
        for key, value in expected.items():
            previous = getattr(config,key,None)
            if previous != value:
                changes[prefix+"."+key] = {"before":previous, "after":value}
            setattr(config,key,value)
    generation = getattr(model,"generation_config",None)
    if generation is not None:
        for key in ("bos_token_id","pad_token_id"):
            previous = getattr(generation,key,None)
            if previous != expected[key]:
                changes["generation."+key] = {"before":previous, "after":expected[key]}
            setattr(generation,key,expected[key])
        # Preserve native extra stop tokens, as Trainer does. Generation in our
        # single-action protocol still explicitly stops on tokenizer.eos_token_id.
        previous = getattr(generation,"eos_token_id",None)
        stops = [] if previous is None else list(previous) if isinstance(previous,(list,tuple)) else [previous]
        if any(not isinstance(v,int) for v in stops):
            raise ValueError("Generation EOS IDs must be integers")
        if expected["eos_token_id"] not in stops:
            stops.insert(0,expected["eos_token_id"])
        stops = list(dict.fromkeys(stops))
        if previous != stops:
            changes["generation.eos_token_id"] = {"before":previous, "after":stops}
        generation.eos_token_id = stops
    return {"tokenizer":actual, "model":{key:getattr(model.config,key) for key in expected},
            "generation_eos_token_id":getattr(generation,"eos_token_id",None), "aligned_fields":changes}
