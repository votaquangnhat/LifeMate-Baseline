"""CPU contracts for unquantized loading, adapter isolation and native tokens."""
import contextlib
import io
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from session_runtime import commit_checkpoint, resolve_resume

NS = types.SimpleNamespace


class Tensor:
    def __init__(self, dtype="fp16"):
        self.dtype = dtype
        self.device = "cuda:0"
        self.requires_grad = False
        self.data = self

    def float(self):
        self.dtype = "fp32"
        return self

    def numel(self):
        return 4


class Tokenizer:
    bos_token_id = None
    def __init__(self, family):
        self.family = family
        self.eos_token_id, self.pad_token_id = (248046,248044) if family=="qwen3_5" else (151645,151643)
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|im_end|>"

    def convert_tokens_to_ids(self, name):
        return {"<|im_end|>":self.eos_token_id,"<|endoftext|>":self.pad_token_id,
                "<think>":200,"</think>":201}[name]

    def encode(self, text, **kwargs):
        return [self.convert_tokens_to_ids(text)]

    def apply_chat_template(self, *args, **kwargs):
        return "<|im_start|>assistant\n" + ("<think>\n" if self.family=="qwen3_5" else "")


class Model:
    def __init__(self, family, dtype, forced=False):
        self.config = NS(model_type=family, quantization_config=None if family=="qwen3_5" else
                         {"load_in_4bit":True,"bnb_4bit_quant_type":"nf4"})
        if family=="qwen3_5":
            self.config.text_config=NS(eos_token_id=248044)
        self.generation_config = NS(eos_token_id=248044 if family=="qwen3_5" else 151645)
        self._unsloth_forced_float32 = forced
        self.weight=Tensor(dtype)
        self.adapter=None

    def get_input_embeddings(self):
        return NS(weight=self.weight)

    def named_parameters(self):
        yield "model.language_model.embed_tokens.weight", self.weight
        if self.adapter is not None:
            yield "model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight", self.adapter

    def parameters(self):
        return (p for _,p in self.named_parameters())

    def print_trainable_parameters(self):
        pass

    def eval(self):
        pass

    def requires_grad_(self, value):
        for p in self.parameters():
            p.requires_grad=value


def load_fixture(root, family, dtype="fp16", forced=False):
    calls=[]
    tok=Tokenizer(family)
    def loader(name):
        def from_pretrained(**kwargs):
            calls.append(("load",name,kwargs))
            return Model(family,dtype,forced),tok
        def attach(model, **kwargs):
            calls.append(("peft",name,kwargs))
            model.adapter=Tensor()
            model.adapter.requires_grad=True
            return model
        return NS(from_pretrained=from_pretrained,get_peft_model=attach,
                  for_training=lambda model,**kwargs:None,for_inference=lambda model:None)
    torch=NS(float16="fp16",bfloat16="bf16",float32="fp32",
             cuda=NS(get_device_capability=lambda:(7,5),is_bf16_supported=lambda:False,
                     set_device=lambda rank:None,get_device_name=lambda rank:"Tesla T4"))
    modules={"unsloth":NS(FastModel=loader("FastModel"),FastLanguageModel=loader("FastLanguageModel")),
             "torch":torch,"transformers":NS(AutoTokenizer=NS(from_pretrained=lambda p:tok),
                                              LogitsProcessor=object,LogitsProcessorList=list)}
    with patch.dict(sys.modules, modules),patch.dict(os.environ,{"LOCAL_RANK":"0","UNSLOTH_FORCE_FLOAT32":"0"}):
        runtime=runpy.run_path(str(Path(__file__).with_name("modeling_agent.py")))
        cfg={"model_id":"fake","model_revision":"fixed","model_family":family,
             "output_dir":str(root),"max_length":4096,"thinking_mode":family=="qwen3_5",
             "lora_rank":8,"lora_alpha":16,"lora_dropout":0.,"seed":42,"gradient_checkpointing":"unsloth"}
        with contextlib.redirect_stdout(io.StringIO()):
            model,tok=runtime["load_model"](cfg,training=family=="qwen3_5")
        return model,tok,calls,runtime["amp_mode"](model)


class LoRATests(unittest.TestCase):
    def test_agent_uses_unquantized_language_lora_and_native_eos(self):
        with tempfile.TemporaryDirectory() as tmp:
            model,tok,calls,amp=load_fixture(tmp,"qwen3_5")
            self.assertEqual([c[1] for c in calls],["FastModel","FastModel"])
            flags=calls[0][2]
            self.assertTrue(flags["load_in_16bit"])
            self.assertFalse(flags["load_in_4bit"] or flags["load_in_8bit"] or flags["full_finetuning"])
            self.assertNotIn("attn_implementation",flags)
            self.assertFalse(calls[1][2]["finetune_vision_layers"])
            self.assertEqual(model.config.text_config.eos_token_id,tok.eos_token_id)
            self.assertEqual(model.config.pad_token_id,248044)
            self.assertEqual(amp,"fp16")
            self.assertEqual(model.adapter.dtype,"fp32")
            self.assertEqual(model.weight.dtype,"fp16")
            self.assertFalse(model.weight.requires_grad)

    def test_forced_fp32_is_reported_and_segmenter_is_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,amp=load_fixture(tmp,"qwen3_5",dtype="fp32",forced=True)
            self.assertEqual(amp,"fp32")
            model,_,calls,_=load_fixture(tmp,"qwen3")
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][1],"FastLanguageModel")
            self.assertFalse(any(p.requires_grad for p in model.parameters()))



if __name__=="__main__":
    unittest.main()
