"""Thinking protocol and model migration tests; no weights or CUDA required."""
import json
import ast
import tempfile
import unittest
from pathlib import Path
import torch
import colab_workflow as workflow
from qwen35_policy import ThinkingBudget, action_from_tokens, training_suffix


class Tokenizer:
    eos_token_id = 0
    def convert_tokens_to_ids(self, token):
        return {'<think>':1, '</think>':2}[token]
    def decode(self, tokens, **kwargs):
        return ''.join({3:'{"done":true}',4:'{"name":"unsafe","arguments":{}}'}.get(t,'') for t in tokens)


class Tests(unittest.TestCase):
    def test_notebook_defines_run_variables_before_configure(self):
        notebook = Path(__file__).resolve().parent.parent / 'notebooks/train_and_infer_qwen35_thinking.ipynb'
        if not notebook.exists():
            self.skipTest('Notebook not included in exported runtime source')
        cells = {c.get('id'): ''.join(c['source']) for c in json.loads(notebook.read_text())['cells']}
        names = {'MODEL_PROFILE', 'RUN_DATE', 'RUN_NAME', 'EPOCHS', 'TRAIN_START', 'PREVIOUS_RUN_NAME'}
        assignments = [node for node in ast.parse(cells['cell-002']).body
                       if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
        self.assertEqual(len(assignments), len(names), 'Run settings must each be assigned exactly once')
        scope = {}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), '<notebook setup>', 'exec'), scope)
        self.assertTrue(names <= scope.keys())
        self.assertEqual(scope['TRAIN_START'], 'fresh')
        self.assertEqual(scope['PREVIOUS_RUN_NAME'], '')
        self.assertIn(scope['RUN_DATE'], scope['RUN_NAME'])

    def test_only_final_action_is_returned(self):
        self.assertEqual(action_from_tokens([4,2,3,0],Tokenizer()),'{"done":true}')

    def test_incomplete_or_malformed_completions_are_rejected(self):
        for tokens in ([3,0],[2,3],[2,2,3,0],[1,2,3,0],[2,0],[0,2,3]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                action_from_tokens(tokens,Tokenizer())

    def test_thinking_budget_reserves_room_for_action(self):
        processor = ThinkingBudget(1,Tokenizer(),2)
        scores=processor(torch.tensor([[9,4]]),torch.zeros(1,10))
        self.assertTrue(torch.isneginf(scores[0,0]))
        self.assertTrue(torch.isneginf(scores[0,1]))
        self.assertTrue(torch.isfinite(scores[0,2]))
        scores=processor(torch.tensor([[9,4,4]]),torch.zeros(1,10))
        self.assertEqual(torch.isfinite(scores).nonzero().tolist(),[[0,2]])
        scores=processor(torch.tensor([[9,4,4,2,3]]),torch.zeros(1,10))
        self.assertTrue(torch.isfinite(scores[0,0]))
        self.assertTrue(torch.isneginf(scores[0,2]))

    def test_scaffold_cannot_read_future_results(self):
        row={'target':'{"name":"app.lookup","arguments":{}}'}
        a=training_suffix(row)
        row.update(tool_results={'secret':'hidden'},label={'secret':'hidden'})
        self.assertEqual(training_suffix(row),a)
        self.assertTrue(a.endswith(row['target']+'<|im_end|>'))
        self.assertEqual(a.count('</think>'),1)

    def test_fresh_profile_and_cross_family_continuation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'runs'/'old';out.mkdir(parents=True)
            args=dict(DATA_ROOT=root,OUTPUT_DIR=out,CODE_DIR=root,CONFIG_PATH=out/'run_config.json',
                      WORK_ROOT=root,RUN_NAME='old',EPOCHS=1,TRAIN_START='fresh',PREVIOUS_RUN_NAME='',
                      RESUME_CHECKPOINT='auto',ASR_TRANSCRIPTS=root/'asr.jsonl',DRIVE_ROOT=root)
            old=workflow.configure(**args)
            self.assertIn('Qwen3-4B-Instruct',old['model_id'])
            new=root/'runs'/'new';new.mkdir()
            args.update(OUTPUT_DIR=new,CONFIG_PATH=new/'run_config.json',RUN_NAME='new',MODEL_PROFILE='qwen35_thinking')
            cfg=workflow.configure(**args)
            self.assertEqual(cfg['model_id'],'unsloth/Qwen3.5-4B')
            self.assertEqual(cfg['max_new_tokens'],cfg['thinking_tokens']+1+cfg['action_json_tokens'])
            self.assertEqual(cfg['initial_adapter'],'')
            args.update(TRAIN_START='best',PREVIOUS_RUN_NAME='old')
            with self.assertRaisesRegex(AssertionError,'incompatible model_id'):
                workflow.configure(**args)


if __name__=='__main__':
    unittest.main(verbosity=2)
