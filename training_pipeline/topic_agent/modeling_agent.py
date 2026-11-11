"""Unsloth LoRA for the Qwen3.5 agent and a separate frozen history segmenter."""
from unsloth import FastLanguageModel, FastModel
import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList
from agent_core import prompt_ids, write_json
from report_metadata import report_json_value
from special_tokens import align_qwen_special_tokens
from thinking_policy import budget_for, split_completion, thought_ids, validate_template

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def amp_mode(model=None):
    native_bf16 = torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    if model is None:
        return "bf16" if native_bf16 else "fp16"
    weights_dtype = model.get_input_embeddings().weight.dtype
    if (getattr(model, "_unsloth_forced_float32", False)
            or os.environ.get("UNSLOTH_FORCE_FLOAT32") == "1"
            or weights_dtype == torch.float32):
        return "fp32"
    if weights_dtype == torch.bfloat16:
        if not native_bf16:
            raise RuntimeError("Qwen3.5 loaded BF16 weights on a GPU without native BF16 support.")
        return "bf16"
    if weights_dtype != torch.float16:
        raise ValueError(f"Unexpected LoRA base dtype: {weights_dtype}")
    return "fp16"


def load_model(cfg, adapter_path=None, training=False):
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    qwen35 = cfg.get("model_family") == "qwen3_5"
    loader = FastModel if qwen35 else FastLanguageModel
    dtype = torch.bfloat16 if amp_mode()=="bf16" else torch.float16
    precision = dict(load_in_4bit=False, load_in_8bit=False, load_in_16bit=True) if qwen35 else dict(
        load_in_4bit=True, load_in_8bit=False)
    if not qwen35 and (training or adapter_path):
        raise ValueError("The 2507 branch is a frozen history segmenter.")
    model, _ = loader.from_pretrained(
        model_name=cfg["model_id"], revision=cfg["model_revision"], use_exact_model_name=True,
        max_seq_length=cfg["max_length"], dtype=dtype, **precision, full_finetuning=False,
        fast_inference=False, device_map={"":rank}, use_gradient_checkpointing=cfg["gradient_checkpointing"],
    )
    # Reuse exactly the native tokenizer prepared for this component.
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("tokenizer_path", Path(cfg["output_dir"]) / "tokenizer"))
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right" if training else "left"
    validate_template(tokenizer, cfg.get("thinking_mode", False))
    qconfig = getattr(model.config, "quantization_config", None) or {}
    qconfig = qconfig.to_dict() if hasattr(qconfig, "to_dict") else qconfig
    if qwen35:
        if model.config.model_type not in ("qwen3_5", "qwen3_5_text"):
            raise ValueError(f"Expected Qwen3.5 architecture, got {model.config.model_type}")
        if qconfig or getattr(model,"is_quantized",False) or getattr(model,"is_loaded_in_4bit",False) or getattr(model,"is_loaded_in_8bit",False):
            raise ValueError("Qwen3.5 LoRA requires an unquantized backbone.")
        if training:
            amp_mode(model)  # Validate the precision before creating the optimizer.
    elif not qconfig.get("load_in_4bit",qconfig.get("_load_in_4bit",False)) or qconfig.get("bnb_4bit_quant_type") != "nf4":
        raise ValueError("Expected the frozen NF4 history segmenter.")
    if training or adapter_path:
        model = loader.get_peft_model(
            model, r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            target_modules=TARGET_MODULES, bias="none", use_gradient_checkpointing=cfg["gradient_checkpointing"],
            random_state=cfg["seed"],
        )
        if adapter_path:
            from peft import load_peft_weights, set_peft_model_state_dict
            adapter_cfg = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
            for k,v in (("r",cfg["lora_rank"]), ("lora_alpha",cfg["lora_alpha"]), ("lora_dropout",cfg["lora_dropout"])):
                if adapter_cfg.get(k) != v:
                    raise ValueError("Adapter recipe mismatch: " + k)
            result = set_peft_model_state_dict(model, load_peft_weights(str(adapter_path), device="cpu"), adapter_name="default")
            missing = [k for k in result.missing_keys if "lora_" in k]
            if missing or result.unexpected_keys:
                raise ValueError(f"Adapter state mismatch: {missing[:4]} / {result.unexpected_keys[:4]}")
        active = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not active or any("lora_" not in n or {"visual","vision_tower","vision_model"} & set(n.split(".")) for n,_ in active):
            raise ValueError("Only language LoRA parameters may be trainable.")
        if training:
            # AMP unscales FP32 adapter gradients, while frozen weights retain loader dtype.
            for _, param in active:
                if param.dtype != torch.float32:
                    param.data = param.data.float()
    if training:
        loader.for_training(model, use_gradient_checkpointing=cfg["gradient_checkpointing"])
        model.print_trainable_parameters()
    else:
        loader.for_inference(model)
        model.eval()
        model.requires_grad_(False)
    model.config.use_cache = not training
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None:
        text_config.use_cache = not training
    special_tokens = align_qwen_special_tokens(model, tokenizer, family=cfg.get("model_family","qwen3"))
    parameter_devices = {str(p.device) for p in model.parameters()}
    if parameter_devices != {f"cuda:{rank}"}:
        raise ValueError(f"Rank {rank}: expected one complete model on cuda:{rank}, got {parameter_devices}")
    report = {"model_id":cfg["model_id"], "thinking_mode":cfg.get("thinking_mode",False), "engine":"unsloth",
              "method":"LoRA" if qwen35 else "frozen_segmenter", "quantization":qconfig,
              "requested_weight_dtype":str(dtype), "actual_weight_dtype":str(model.get_input_embeddings().weight.dtype),
              "amp":amp_mode(model) if qwen35 and training else "inference",
              "attention":"model_default", "gpu":torch.cuda.get_device_name(rank),
              "rank":int(os.environ.get("RANK","0")), "world_size":int(os.environ.get("WORLD_SIZE","1")),
              "local_rank":rank, "parameter_cuda_devices":[rank], "special_tokens":special_tokens,
              "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad)}
    # BitsAndBytes/Unsloth can leave torch.dtype values in quantization_config.
    # Convert the REPORT copy; do not change model dtypes/quantization settings.
    report = report_json_value(report)
    report_prefix = "model_report" if qwen35 else "segmenter_model_report"
    write_json(Path(cfg["output_dir"]) / f"{report_prefix}.rank{rank}.json", report)
    print(json.dumps(report), flush=True)
    return model, tokenizer


class ResponseCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        length = ((max(len(r["input_ids"]) for r in rows)+7)//8)*8
        batch = {k:[] for k in ("input_ids", "attention_mask", "labels")}
        for row in rows:
            pad = length-len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad_id]*pad)
            batch["attention_mask"].append(row["attention_mask"] + [0]*pad)
            batch["labels"].append(row["labels"] + [-100]*pad)
        return {k:torch.tensor(v, dtype=torch.long) for k,v in batch.items()}


class ThinkBudget(LogitsProcessor):
    """Force a delimiter after the reasoning allowance, leaving JSON budget intact."""
    def __init__(self, prompt_length, cap, open_id, close_id, eos_id):
        self.prompt_length = prompt_length
        self.cap = cap
        self.open_id, self.close_id, self.eos_id = open_id, close_id, eos_id

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length:]
        closed = (generated == self.close_id).any(dim=1)
        scores[:, self.open_id] = -torch.inf
        scores[:, self.eos_id].masked_fill_(~closed, -torch.inf)
        scores[:, self.close_id].masked_fill_(closed, -torch.inf)
        if generated.shape[1] >= self.cap:
            forced = ~closed
            scores.masked_fill_(forced[:, None], -torch.inf)
            scores[:, self.close_id].masked_fill_(forced, 0.0)
        return scores


def make_generator(model, tokenizer, cfg):
    thinking = cfg.get("thinking_mode", False)
    eos = tokenizer.eos_token_id
    open_id, close_id = thought_ids(tokenizer) if thinking else (None, None)

    @torch.inference_mode()
    def generate(messages):
        from context_window import fit_prompt
        generate.last_thinking = None
        _,ids,trim=fit_prompt(tokenizer,messages,cfg)
        generate.last_context_trim=trim
        budget = min(cfg["max_new_tokens"], cfg["max_length"]-len(ids))
        extra = {}
        if thinking:
            think_cap, action_cap = budget_for(messages, cfg)
            required = think_cap + 1 + action_cap
            if budget < required:
                raise ValueError(f"Context must reserve {required} generation tokens, got {budget}")
            budget = required
            extra["logits_processor"] = LogitsProcessorList([
                ThinkBudget(len(ids), think_cap, open_id, close_id, eos)])
        inputs = torch.tensor([ids], device=model.device, dtype=torch.long)
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                                max_new_tokens=budget, use_cache=True, eos_token_id=eos,
                                pad_token_id=tokenizer.pad_token_id, **extra)
        generated = output[0, len(ids):].tolist()
        if thinking:
            reason_count = generated.index(close_id) if close_id in generated else None
            generate.last_thinking = {"reasoning_tokens":reason_count, "generated_tokens":len(generated),
                "reasoning_cap_reached":reason_count == think_cap, "eos_seen":eos in generated}
            generated, _ = split_completion(generated, open_id, close_id, eos, think_cap, action_cap)
        elif eos not in generated:
            raise ValueError("Action exhausted token budget without EOS; partial call was not executed")
        return tokenizer.decode(generated, skip_special_tokens=not thinking, clean_up_tokenization_spaces=False).strip()
    generate.tokenizer = tokenizer
    return generate
