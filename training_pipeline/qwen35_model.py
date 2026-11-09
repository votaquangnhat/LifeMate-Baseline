"""BF16 Qwen3.5 language-only LoRA and bounded thinking inference."""
from unsloth import FastModel
import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, LogitsProcessorList
from agent_core import prompt_ids, write_json
from qwen35_policy import configure_tokenizer, action_from_tokens, ThinkingBudget




def load_model(cfg, adapter_path=None, training=False):
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability(rank)[0] < 8 or not torch.cuda.is_bf16_supported():
        raise ValueError("Qwen3.5 BF16 LoRA requires a native BF16 GPU: choose L4/A100, not T4.")
    model, _ = FastModel.from_pretrained(
        model_name=cfg["model_id"], revision=cfg["model_revision"], use_exact_model_name=True, fast_inference=False,
        max_seq_length=cfg["max_length"], dtype=torch.bfloat16,
        load_in_4bit=False, load_in_8bit=False, load_in_16bit=True,
        full_finetuning=False, device_map={"":rank},
        use_gradient_checkpointing=cfg["gradient_checkpointing"],
    )
    if getattr(model.config, "model_type", "") not in ("qwen3_5", "qwen3_5_text"):
        raise ValueError("Expected Qwen3.5 architecture")
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise ValueError("Qwen3.5 profile requires unquantized BF16 base weights")
    if model.get_input_embeddings().weight.dtype != torch.bfloat16:
        raise ValueError("Loader did not retain BF16 base weights; inspect the Unsloth runtime before training.")
    tokenizer = AutoTokenizer.from_pretrained(Path(cfg["output_dir"]) / "tokenizer")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right" if training else "left"
    configure_tokenizer(tokenizer, cfg)
    for config in (model.config, getattr(model.config, "text_config", None), model.generation_config):
        if config is not None:
            config.eos_token_id = tokenizer.eos_token_id
            config.pad_token_id = tokenizer.pad_token_id
    if training or adapter_path:
        model = FastModel.get_peft_model(model,
            finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            bias="none", use_gradient_checkpointing=cfg["gradient_checkpointing"], random_state=cfg["seed"])
        if adapter_path:
            from peft import load_peft_weights, set_peft_model_state_dict
            ac = json.loads((Path(adapter_path)/"adapter_config.json").read_text())
            for key, value in (("r",cfg["lora_rank"]),("lora_alpha",cfg["lora_alpha"]),("lora_dropout",cfg["lora_dropout"])):
                if ac.get(key) != value:
                    raise ValueError("Adapter recipe mismatch: " + key)
            result = set_peft_model_state_dict(model, load_peft_weights(str(adapter_path), device="cpu"), adapter_name="default")
            missing = [k for k in result.missing_keys if "lora_" in k]
            if missing or result.unexpected_keys:
                raise ValueError(f"Adapter state mismatch: {missing[:4]} / {result.unexpected_keys[:4]}")
        active = [n for n,p in model.named_parameters() if p.requires_grad]
        if not active or any("lora_" not in n or "visual" in n or "vision" in n for n in active):
            raise ValueError("Only language LoRA parameters may be trainable")
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    if training:
        FastModel.for_training(model, use_gradient_checkpointing=cfg["gradient_checkpointing"])
        model.config.use_cache = False
        model.print_trainable_parameters()
    else:
        FastModel.for_inference(model)
        model.eval()
        model.config.use_cache = True
        for p in model.parameters():
            p.requires_grad_(False)
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = not training
    devices = {p.device for p in model.parameters()}
    if devices != {torch.device("cuda",rank)}:
        raise ValueError(f"Expected all model parameters on cuda:{rank}, got {devices}")
    report = dict(model_id=cfg["model_id"], method="BF16 LoRA", thinking=True,
        gpu=torch.cuda.get_device_name(rank), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    write_json(Path(cfg["output_dir"])/f"model_report.rank{rank}.json",report)
    print(json.dumps(report),flush=True)
    return model, tokenizer


def make_generator(model, tokenizer, cfg):
    @torch.inference_mode()
    def generate(messages):
        ids = prompt_ids(tokenizer,messages)
        thought = cfg["thinking_tokens"]
        budget = thought + 1 + cfg["action_json_tokens"]
        if len(ids) + budget > cfg["max_length"]:
            raise ValueError(f"Context overflow: {len(ids)} prompt tokens plus {budget} completion tokens. No observations truncated.")
        inputs = torch.tensor([ids],device=model.device,dtype=torch.long)
        output = model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),
            do_sample=False, max_new_tokens=budget, use_cache=True,
            logits_processor=LogitsProcessorList([ThinkingBudget(len(ids),tokenizer,thought)]),
            eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
        return action_from_tokens(output[0,len(ids):].tolist(),tokenizer)
    return generate
