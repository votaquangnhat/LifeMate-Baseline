import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from baseline.agent import available_tools
from baseline.data import DEFAULT_ROOT, input_only, load_runtime, load_split
from baseline.evaluate import n_matched, score_one, token_f1
from baseline.prompts import build_messages, build_tools
from baseline.tool_agent import execute, normalize_arguments, parse_tool_calls
from baseline.training import sft_conversation, tokenize_sft
from baseline.vnnum import itn_time


class Scripted:
    def __init__(self, responses):
        self.responses = iter(responses)
    def fits(self, messages, tools):
        return True
    def generate(self, messages, tools, deadline):
        return next(self.responses)


def native(call):
    return '<tool_call>' + json.dumps({'name': call['name'].replace('.', '__', 1),
                                     'arguments': call['arguments']}) + '</tool_call>'


class ProtocolTests(unittest.TestCase):
    def test_parse_multiple_and_malformed(self):
        c = {'name': 'sms.send', 'arguments': {'content': 'hello'}}
        self.assertEqual(parse_tool_calls(native(c) + native(c)), [c, c])
        self.assertEqual(parse_tool_calls('Đã xong.'), [])
        for raw in ('<tool_call>{', '<tool_call>[]</tool_call>', '<tool_call>{"name":"x","arguments":[]}</tool_call>'):
            with self.assertRaises(ValueError):
                parse_tool_calls(raw)

    def test_normalization_preserves_values_and_optional_intent(self):
        spec = {'parameters': {'properties': {'when': {'format': 'date-time'},
            'amount': {'type': 'integer'}, 'content': {'type': 'string'}}}}
        args = {'when': '2026-11-05T08:00', 'amount': '1.5', 'content': '2026-11-05T08:00', 'note': 'keep me'}
        fixed = normalize_arguments(args, spec, {'timezone': 'Asia/Ho_Chi_Minh'})
        self.assertEqual(fixed['when'], '2026-11-05T08:00:00+07:00')
        self.assertEqual(fixed['amount'], '1.5')
        self.assertEqual(fixed['content'], args['content'])
        self.assertEqual(fixed['note'], 'keep me')

    def test_itn_distinguishes_current_time_from_seven(self):
        self.assertEqual(itn_time('bây giờ bật đèn'), 'bây giờ bật đèn')
        self.assertIn('19:00', itn_time('bảy giờ tối'))

    def test_inputs_do_not_expose_labels(self):
        row = {'id': 'x', 'audio': 'a', 'context': {}, 'audio_text': 'SECRET', 'label': {'x': 1},
            'history': [{'role': 'user', 'audio': 'b', 'audio_text': 'SECRET'}]}
        self.assertNotIn('SECRET', json.dumps(input_only(row)))
        self.assertNotIn('label', input_only(row))


class MetricTests(unittest.TestCase):
    specs = {'sms.send': {'parameters': {'properties': {'content': {'x-match': 'fuzzy'}}}}}
    def test_hash_gate_and_extra_calls(self):
        c = {'name': 'sms.send', 'arguments': {'content': 'hello world'}}
        ref = {'tool_calls': [c], 'state_hash': 'gold'}
        self.assertEqual(score_one({'tool_calls': [c], 'state_hash': 'wrong'}, ref, self.specs)['score'], 0)
        self.assertAlmostEqual(score_one({'tool_calls': [c, c], 'state_hash': 'gold'}, ref, self.specs)['score'], 2 / 3)

    def test_maximum_matching_not_greedy(self):
        def call(s):
            return {'name': 'sms.send', 'arguments': {'content': s}}
        # First prediction matches both, second only the first: a greedy matching can lose one.
        gold = [call('a b c'), call('a b d')]
        submitted = [call('a b'), call('a b c e')]
        self.assertEqual(n_matched(submitted, gold, self.specs), 2)
        self.assertEqual(token_f1('a b c', 'a b c'), 1)


@unittest.skipUnless((DEFAULT_ROOT / 'train/train.jsonl').exists(), 'Dataset not present')
class SDKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.catalog = load_split(DEFAULT_ROOT, 'train')

    def test_native_gold_trajectory_and_history(self):
        row = next(r for r in self.rows if r['history'] and len(r['label']['tool_calls']) >= 2)
        env = load_runtime(DEFAULT_ROOT, 'train').ToolEnv('train', row['id'], root=DEFAULT_ROOT)
        specs = available_tools(self.catalog, row['context'])
        messages = build_messages(row, [], lambda t: t['audio_text'])
        original = copy.deepcopy(messages)
        diag = execute(env, Scripted([*(native(c) for c in row['label']['tool_calls']), 'Xong']),
            messages, build_tools(specs, row), specs, row['context'], time.monotonic() + 10)
        self.assertIn(diag['stop_reason'], ('finished', 'asked_missing_info'))
        self.assertEqual(env.get_tool_calls(), row['label']['tool_calls'])
        self.assertEqual(messages[:len(original)], original)

    def test_training_targets_use_real_chat_template(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest('Optional model stack absent')
        path = Path('models/Qwen3-4B-Instruct-2507')
        if not path.is_dir():
            self.skipTest('Tokenizer metadata absent')
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        row = self.rows[0]
        messages, first = sft_conversation(row, lambda t: itn_time(t['audio_text']))
        tools = build_tools(available_tools(self.catalog, row['context']), row)
        record = tokenize_sft(tokenizer, messages, tools, first)
        full = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, enable_thinking=False)
        self.assertEqual(record['input_ids'], tokenizer(full, add_special_tokens=False)['input_ids'])
        learned = tokenizer.decode([t for t in record['labels'] if t != -100])
        self.assertIn('<tool_call>', learned)
        self.assertNotIn('Thời gian hiện tại:', learned)
        self.assertNotIn('"forecast":', learned)

    def test_train_and_public_runtimes_are_isolated(self):
        train = load_runtime(DEFAULT_ROOT, 'train')
        public = load_runtime(DEFAULT_ROOT, 'public_test')
        self.assertIsNot(train, public)
        self.assertNotEqual(train.__name__, public.__name__)



class AdapterLoadingTests(unittest.TestCase):
    def test_local_peft_adapter_merges_into_planner(self):
        try:
            import torch
            from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest('Optional ML stack absent')
        tok_path = Path('models/Qwen3-0.6B')
        if not tok_path.is_dir():
            self.skipTest('Smoke tokenizer absent')
        from baseline.models import Planner
        with tempfile.TemporaryDirectory() as directory:
            base, adapter = Path(directory) / 'base', Path(directory) / 'adapter'
            tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, head_dim=8, tie_word_embeddings=True))
            model.save_pretrained(base)
            tokenizer.save_pretrained(base)
            peft = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=2,
                lora_alpha=4, target_modules=['q_proj']))
            peft.save_pretrained(adapter)
            loaded = Planner(str(base), device='cpu', dtype='float32', adapter=str(adapter))
            self.assertFalse(any('lora_' in n for n, _ in loaded.model.named_parameters()))
            self.assertTrue(loaded.fits([{'role': 'user', 'content': 'Xin chào'}], []))

    def test_local_peft_adapter_merges_into_asr(self):
        try:
            from transformers import WhisperConfig, WhisperForConditionalGeneration, AutoProcessor
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest('Optional ML stack absent')
        processor_path = Path('models/PhoWhisper-small')
        if not processor_path.is_dir():
            self.skipTest('Whisper processor absent')
        from baseline.models import SpeechRecognizer
        with tempfile.TemporaryDirectory() as directory:
            base, adapter = Path(directory) / 'base', Path(directory) / 'adapter'
            model = WhisperForConditionalGeneration(WhisperConfig(vocab_size=51865, d_model=16,
                encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32))
            model.save_pretrained(base)
            AutoProcessor.from_pretrained(processor_path, local_files_only=True).save_pretrained(base)
            wrapped = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj']))
            wrapped.save_pretrained(adapter)
            loaded = SpeechRecognizer(str(base), device='cpu', dtype='float32', adapter=str(adapter))
            self.assertEqual(loaded.model.config.model_type, 'whisper')
            self.assertFalse(any('lora_' in n for n, _ in loaded.model.named_parameters()))

if __name__ == '__main__':
    unittest.main()

