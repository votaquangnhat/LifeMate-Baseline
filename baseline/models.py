"""Offline Transformers backends with merged-model and PEFT adapter support."""
import math
from pathlib import Path
import time


def configure_sdpa_compatibility():
    """Avoid Torch 2.6 GQA math fallback on CPUs and pre-Ampere CUDA devices.

    Transformers 4.57 enables native GQA by Torch version alone. Torch 2.6's fused
    GQA path requires Flash Attention, which is unavailable on GTX 1650/T4.
    Repeating KV heads instead lets SDPA choose the efficient attention kernel.
    Patch only this backend-selection predicate in memory, not installed files.
    """
    import torch
    from transformers.integrations import sdpa_attention
    if getattr(sdpa_attention, '_lifemate_compatibility', False):
        return
    original = sdpa_attention.use_gqa_in_sdpa

    def compatible_gqa(attention_mask, key):
        if key.device.type != 'cuda':
            return False
        if torch.cuda.get_device_capability(key.device)[0] < 8:
            return False
        return original(attention_mask, key)

    sdpa_attention.use_gqa_in_sdpa = compatible_gqa
    sdpa_attention._lifemate_compatibility = True


def local_checkpoint(path):
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'Missing local checkpoint: {path}. Run scripts/download_models.py first.')
    return str(path)


def model_kwargs(device, dtype):
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable. Use CPU config for debugging or install CUDA PyTorch on a GPU machine.')
    if dtype not in ('float32', 'float16', 'bfloat16'):
        raise ValueError('dtype must be float32, float16 or bfloat16')
    return dict(local_files_only=True, torch_dtype=getattr(torch, dtype),
                device_map={'': device}, low_cpu_mem_usage=True)


def attach_adapter(model, path):
    if not path:
        return model
    from peft import PeftModel
    path = local_checkpoint(path)
    if not (Path(path) / 'adapter_config.json').is_file():
        raise ValueError(f'{path} is not a PEFT adapter. Put a merged model in the model path instead.')
    # Explicit base model supplied by caller: never fetch the training machine's base path.
    return PeftModel.from_pretrained(model, path, local_files_only=True,
        is_trainable=False).merge_and_unload(safe_merge=True)


def stopping(deadline):
    from transformers import StoppingCriteria, StoppingCriteriaList
    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return time.monotonic() >= deadline
    return StoppingCriteriaList([Deadline()])


class Planner:
    def __init__(self, model_path, device='cuda:0', max_context=16384, max_new_tokens=256,
                 adapter=None, tokenizer=None, dtype='float16'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        configure_sdpa_compatibility()
        self.torch, self.max_context, self.max_new_tokens = torch, max_context, max_new_tokens
        model_path = local_checkpoint(model_path)
        if not tokenizer and adapter and (Path(adapter) / 'tokenizer_config.json').is_file():
            tokenizer = adapter
        self.tokenizer = AutoTokenizer.from_pretrained(local_checkpoint(tokenizer or model_path), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
            attn_implementation='sdpa', **model_kwargs(device, dtype))
        self.model = attach_adapter(self.model, adapter).eval()
        self.parameter_count = self.model.num_parameters()
        if self.parameter_count >= 5_000_000_000:
            raise ValueError('Planner must have fewer than 5B parameters')
        if len(self.tokenizer) > self.model.get_input_embeddings().num_embeddings:
            raise ValueError('Tokenizer vocabulary exceeds model embeddings; use matching tokenizer and checkpoint')
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    def encode(self, messages, tools):
        text = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return self.tokenizer(text, return_tensors='pt', add_special_tokens=False)

    def fits(self, messages, tools):
        return self.encode(messages, tools)['input_ids'].shape[1] + self.max_new_tokens <= self.max_context

    def generate(self, messages, tools, deadline):
        encoded = self.encode(messages, tools).to(self.model.device)
        if encoded['input_ids'].shape[1] + self.max_new_tokens > self.max_context:
            raise ValueError('Context limit exceeded')
        if time.monotonic() >= deadline:
            raise TimeoutError('Deadline before LLM generation')
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, do_sample=False, num_beams=1,
                max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping(deadline))
        # Qwen's tool delimiters are ordinary text; retain them for parsing.
        return self.tokenizer.decode(output[0, encoded['input_ids'].shape[1]:], skip_special_tokens=True)


class SpeechRecognizer:
    def __init__(self, model_path, device='cpu', adapter=None, processor=None, dtype='float32'):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        self.torch = torch
        model_path = local_checkpoint(model_path)
        if not processor and adapter and (Path(adapter) / 'preprocessor_config.json').is_file():
            processor = adapter
        self.processor = AutoProcessor.from_pretrained(local_checkpoint(processor or model_path), local_files_only=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, **model_kwargs(device, dtype))
        self.model = attach_adapter(self.model, adapter).eval()
        if self.model.config.model_type != 'whisper':
            raise ValueError('This backend supports Whisper/PhoWhisper checkpoints. Implement transcribe() for other ASR families.')
        self.model.config.forced_decoder_ids = None
        self.model.generation_config.forced_decoder_ids = None
        self.parameter_count = self.model.num_parameters()

    def transcribe(self, path, deadline):
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        audio, sr = sf.read(Path(path), dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
        if not len(audio):
            raise ValueError(f'Empty audio: {path}')
        if sr != 16000:
            divisor = math.gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // divisor, sr // divisor).astype(np.float32)
        parts = []
        for start in range(0, len(audio), 30 * 16000):
            if time.monotonic() >= deadline:
                raise TimeoutError('ASR time budget exceeded')
            inputs = self.processor(audio[start:start + 30 * 16000], sampling_rate=16000,
                return_tensors='pt', return_attention_mask=True)
            with self.torch.inference_mode():
                tokens = self.model.generate(
                    input_features=inputs.input_features.to(self.model.device, dtype=self.model.dtype),
                    attention_mask=inputs.attention_mask.to(self.model.device), language='vi',
                    task='transcribe', do_sample=False, num_beams=1, max_new_tokens=256,
                    stopping_criteria=stopping(deadline))
            if time.monotonic() >= deadline:
                raise TimeoutError('ASR time budget exceeded')
            parts.append(self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip())
        return ' '.join(parts)


def setup_runtime(seed):
    import os
    import random
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch
