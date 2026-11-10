"""CPU checks for causal Thinking targets and deadline-safe recovery."""
import ast
import importlib.util
import json
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from agent_core import dumps, rollout, tokenize_sft
from session_runtime import SessionPaused, commit_checkpoint, export_resume_bundle, stop_if_deadline
from selftest_agent import FakeEnv, TOOLS, call
import selftest_history_router
from thinking_policy import assistant_suffix, split_completion, validate_template
from tool_router import make_router_sft_rows


class Tokenizer:
    markers = {'<think>': 300, '</think>': 301, '<|im_end|>': 302}
    eos_token_id = 302

    def convert_tokens_to_ids(self, value):
        return self.markers.get(value)

    def encode(self, value, **_):
        ids = []
        while value:
            for marker, mid in self.markers.items():
                if value.startswith(marker):
                    ids.append(mid)
                    value = value[len(marker):]
                    break
            else:
                ids.append(ord(value[0])+500)
                value = value[1:]
        return ids

    def apply_chat_template(self, messages, tokenize=True, **_):
        text = dumps(messages) + '<|im_start|>assistant\n<think>\n'
        return self.encode(text) if tokenize else text


def think_config(cfg):
    return {**cfg, 'thinking_mode': True, 'router_thinking_tokens': 160,
            'executor_thinking_tokens': 192, 'action_json_tokens': 512,
            'requested_max_new_tokens':768, 'max_new_tokens_cap':768, 'max_new_tokens':768}


class ThinkingTests(unittest.TestCase):
    def test_native_boundary_and_complete_action(self):
        tok = Tokenizer()
        validate_template(tok, True)
        with self.assertRaises(ValueError):
            validate_template(tok, False)
        self.assertEqual(split_completion([10,11,301,20,21,302],300,301,302,2,3),([20,21],2))
        for bad in ([10,301,20], [10,20,302], [10,301,20,301,302], [300,301,20,302],
                    [301,302], [1,2,3,301,20,302], [301,20,21,22,302]):
            with self.assertRaises(ValueError):
                split_completion(bad,300,301,302,2,3)

    def test_thinking_sft_masks_prefix_and_has_no_future_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj, cfg, _ = selftest_history_router.RouterTests().prepare(tmp)
            cfg = think_config(cfg)
            calls=obj['label']['tool_calls']
            obs=[{'contacts':[{'id':'person-7'}]},{'id':'FUTURE_OBSERVATION'}]
            examples=list(make_router_sft_rows(obj,TOOLS,calls,obs,Tokenizer(),cfg))
            self.assertEqual([r['task'] for r in examples],
                             ['router','tool_call','open_apps','tool_call','done'])
            self.assertNotIn('FUTURE_OBSERVATION',dumps(examples[:-1]))
            self.assertIn('FUTURE_OBSERVATION',dumps(examples[-1]['messages']))
            for row in examples:
                tok=Tokenizer()
                suffix,rationale_len=assistant_suffix(row,tok,cfg)
                encoded=tokenize_sft(row,tok,cfg)
                prefix_len=encoded['length']-len(suffix)
                self.assertGreater(rationale_len,0)
                self.assertEqual(encoded['input_ids'][prefix_len-2:prefix_len], [300,ord('\n')+500])
                self.assertEqual(encoded['labels'][:prefix_len],[-100]*prefix_len)
                self.assertEqual(encoded['labels'][prefix_len:],suffix)
                self.assertEqual(suffix[rationale_len],301)
                self.assertEqual(suffix[-1],302)

    def test_incomplete_thinking_output_never_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj,cfg,_=selftest_history_router.RouterTests().prepare(tmp)
            cfg=think_config(cfg)
            real_call=call()
            app=real_call['name'].split('.',1)[0]
            outputs=iter([dumps({'apps':[app]}),ValueError('Action exhausted its token budget without EOS'),
                          dumps(real_call),dumps({'done':True})])
            def generate(messages):
                result=next(outputs)
                generate.last_thinking={'reasoning_tokens':5, 'eos_seen':not isinstance(result,ValueError)}
                if isinstance(result,ValueError):
                    raise result
                return result
            env=FakeEnv()
            result=rollout(obj,TOOLS,env,generate,cfg)
            self.assertEqual(result['tool_calls'],[real_call])
            self.assertEqual(result['invalid_model_actions'],1)
            self.assertEqual(len(result['thinking']),4)
            self.assertIsNone(result['error'])


    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Real logits masks require PyTorch')
    def test_logits_keep_json_budget_after_closing_reasoning(self):
        import torch

        tree=ast.parse(Path(__file__).with_name('modeling_agent.py').read_text())
        klass=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='ThinkBudget')
        namespace={'torch':torch,'LogitsProcessor':object}
        exec(compile(ast.Module(body=[klass],type_ignores=[]),'<ThinkBudget>','exec'),namespace)
        cap=namespace['ThinkBudget'](2,2,3,4,5)
        early=cap(torch.tensor([[6,7,1]]),torch.zeros(1,8))
        self.assertTrue(torch.isneginf(early[0,5]))
        self.assertEqual(early[0,4].item(),0)
        forced=cap(torch.tensor([[6,7,1,2]]),torch.zeros(1,8))
        self.assertEqual(torch.isfinite(forced).nonzero().tolist(),[[0,4]])
        action=cap(torch.tensor([[6,7,1,2,4]]),torch.zeros(1,8))
        self.assertEqual(action[0,5].item(),0)
        self.assertTrue(torch.isneginf(action[0,4]))



if __name__=='__main__':
    unittest.main()
