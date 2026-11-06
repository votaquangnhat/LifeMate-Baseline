"""Check the local/Colab launch assets without needing a Colab account."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from baseline.config import load_config
from baseline.tool_agent import execute


class WorkflowTests(unittest.TestCase):
    def test_attention_compatibility_keeps_gqa_numerically_equivalent(self):
        try:
            import torch
            from baseline.models import configure_sdpa_compatibility
            from transformers.integrations import sdpa_attention
        except ImportError:
            self.skipTest('Optional ML stack absent')
        from types import SimpleNamespace
        configure_sdpa_compatibility()
        query = torch.randn(1, 4, 5, 8)
        key, value = torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8)
        self.assertFalse(sdpa_attention.use_gqa_in_sdpa(None, key))
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value,
            is_causal=True, enable_gqa=True)
        module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
        actual, _ = sdpa_attention.sdpa_attention_forward(module, query, key, value, None)
        torch.testing.assert_close(actual.transpose(1, 2), expected, atol=1e-6, rtol=1e-5)
        fake_cuda_key = SimpleNamespace(device=SimpleNamespace(type='cuda'))
        with patch('torch.cuda.get_device_capability', return_value=(7, 5)):
            self.assertFalse(sdpa_attention.use_gqa_in_sdpa(None, fake_cuda_key))

    def test_colab_cells_compile_and_use_full_public_split(self):
        try:
            from IPython.core.inputtransformer2 import TransformerManager
        except ImportError:
            self.skipTest('Install requirements-dev.txt for notebook syntax checks')
        transformer = TransformerManager()
        notebook = json.loads(Path('notebooks/lifemate_colab.ipynb').read_text(encoding='utf-8'))
        source = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                text = ''.join(cell['source'])
                compile(transformer.transform_cell(text), 'lifemate_colab.ipynb', 'exec')
                source.append(text)
        full_run = next(s for s in source if '# 8.' in s)
        self.assertIn('--split public_test', full_run)
        self.assertNotIn('--limit', full_run)
        self.assertNotIn('--oracle-transcripts', full_run)
        self.assertIn('baseline.validate_submission', '\n'.join(source))
        self.assertNotIn('subprocess', '\n'.join(source))
        self.assertNotIn('drive.mount', '\n'.join(source))
        self.assertNotIn('.venv-colab', '\n'.join(source))

    def test_local_and_colab_use_same_parameter_contract(self):
        local = load_config('configs/local.json')
        colab = load_config('configs/colab.json')
        self.assertEqual(local['llm']['device'], 'cuda:0')
        self.assertEqual(local['asr']['device'], 'cpu')
        self.assertEqual(colab['llm']['device'], 'cuda:0')
        self.assertEqual(colab['asr']['device'], 'cuda:0')
        self.assertLessEqual(colab['agent']['max_context'], 16384)

    def test_late_native_generation_does_not_execute(self):
        class Planner:
            def fits(self, messages, tools):
                return True
            def generate(self, messages, tools, deadline):
                return '<tool_call>{"name":"sms__send","arguments":{}}</tool_call>'
        class Environment:
            def call(self, *args):
                raise AssertionError('A late call must never execute')
        events = []
        with patch('baseline.tool_agent.time.monotonic', side_effect=[0, 11]):
            diagnostics = execute(Environment(), Planner(), [], [], {}, {}, 10, progress=events.append)
        self.assertEqual(diagnostics['stop_reason'], 'timeout')
        self.assertEqual(len(events), 1)


if __name__ == '__main__':
    unittest.main()
