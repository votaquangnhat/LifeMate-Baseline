"""Qwen3 Instruct-2507, Unsloth NF4 QLoRA, native attention and masked loss."""
from unsloth import FastLanguageModel
import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer
from agent_core import prompt_ids, write_json
from report_metadata import report_json_value

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def amp_mode():
    # T4 (SM75) must use FP16 even if a torch build reports emulated BF16 support.
    return "bf16" if torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported() else "fp16"


def load_model(cfg, adapter_path=None, training=False):
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    model, _ = FastLanguageModel.from_pretrained(
        model_name=cfg["model_id"], revision=cfg["model_revision"], use_exact_model_name=True,
        max_seq_length=cfg["max_length"], dtype=torch.bfloat16 if amp_mode()=="bf16" else torch.float16,
        load_in_4bit=True, load_in_8bit=False, full_finetuning=False,
        fast_inference=False, device_map={"":rank},
        use_gradient_checkpointing=cfg["gradient_checkpointing"],
    )
    # Use the official non-thinking template, identically to preprocessing.
    tokenizer = AutoTokenizer.from_pretrained(Path(cfg["output_dir"]) / "tokenizer")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right" if training else "left"
    probe = tokenizer.apply_chat_template([{"role":"user", "content":"test"}], tokenize=False, add_generation_prompt=True)
    if "<think>" in probe:
        raise ValueError("Wrong tokenizer: Instruct-2507 must not insert thinking blocks")
    qconfig = getattr(model.config, "quantization_config", {})
    qconfig = qconfig.to_dict() if hasattr(qconfig, "to_dict") else qconfig
    if not qconfig.get("load_in_4bit", qconfig.get("_load_in_4bit", False)) or qconfig.get("bnb_4bit_quant_type") != "nf4":
        raise ValueError(f"Expected bitsandbytes NF4 4-bit, got {qconfig}")
    if training or adapter_path:
        model = FastLanguageModel.get_peft_model(
            model, r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules=TARGET_MODULES, bias="none", use_gradient_checkpointing=cfg["gradient_checkpointing"],
            random_state=cfg["seed"], max_seq_length=cfg["max_length"],
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
        active = [n for n,p in model.named_parameters() if p.requires_grad]
        if not active or any("lora_" not in n for n in active):
            raise ValueError("Only LoRA parameters may be trainable")
    if training:
        FastLanguageModel.for_training(model, use_gradient_checkpointing=cfg["gradient_checkpointing"])
        model.config.use_cache = False
        model.print_trainable_parameters()
    else:
        FastLanguageModel.for_inference(model)
        model.eval()
        model.config.use_cache = True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    parameter_devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    if parameter_devices != [rank]:
        raise ValueError(f"Rank {rank}: expected all weights on cuda:{rank}, got {parameter_devices}")
    report = {"model_id":cfg["model_id"], "engine":"unsloth", "method":"QLoRA", "quantization":qconfig,
              "amp":amp_mode(), "attention":"native Unsloth Qwen3", "gpu":torch.cuda.get_device_name(rank),
              "rank":int(os.environ.get("RANK", "0")), "world_size":int(os.environ.get("WORLD_SIZE", "1")),
              "local_rank":rank, "parameter_cuda_devices":parameter_devices,
              "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad)}
    # BitsAndBytes/Unsloth can leave torch.dtype values in quantization_config.
    # Convert the REPORT copy; do not change model dtypes/quantization settings.
    report = report_json_value(report)
    write_json(Path(cfg["output_dir"]) / f"model_report.rank{rank}.json", report)
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


def make_generator(model, tokenizer, cfg):
    @torch.inference_mode()
    def generate(messages):
        ids = prompt_ids(tokenizer, messages)
        budget = min(cfg["max_new_tokens"], cfg["max_length"]-len(ids))
        if budget <= 0:
            raise ValueError(f"Context overflow: {len(ids)} prompt tokens. No request or observations were truncated.")
        inputs = torch.tensor([ids], device=model.device, dtype=torch.long)
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                                max_new_tokens=budget, use_cache=True,
                                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        generated = output[0, len(ids):].tolist()
        if tokenizer.eos_token_id not in generated:
            raise ValueError("Action exhausted token budget without EOS; partial call was not executed")
        return tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    return generate
