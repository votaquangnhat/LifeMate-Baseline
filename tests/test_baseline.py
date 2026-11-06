"""Model-free tests of execution, leakage boundaries, and the supplied training SDK."""
import json
from pathlib import Path
import sys
import time
import unittest

from baseline.agent import available_tools, parse_decision, run_agent
from baseline.run import prepare_input, audio_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'train+public_test' / 'train+public_test'


class ScriptedPlanner:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.messages = []
    def fits(self, messages):
        return True
    def generate(self, messages, deadline):
        self.messages = messages.copy()
        return json.dumps(next(self.decisions))


class FakeEnv:
    def __init__(self):
        self.calls = []
    def validate(self, name, args):
        return {'ok': 'value' in args, 'error': 'missing value'}
    def call(self, name, args):
        self.calls.append({'name': name, 'arguments': args})
        return {'ok': True}
    def get_tool_calls(self):
        return self.calls.copy()
    def get_state_hash(self):
        return 'hash-from-env'


TOOLS = {'app.write': {'name': 'app.write', 'kind': 'write', 'parameters': {}},
         'sys.ask_missing_info': {'name': 'sys.ask_missing_info', 'parameters': {}}}


class AgentTests(unittest.TestCase):
    def test_repair_and_no_duplicate_write(self):
        env = FakeEnv()
        planner = ScriptedPlanner([
            {'action': 'call', 'name': 'not.installed', 'arguments': {}},
            {'action': 'call', 'name': 'app.write', 'arguments': {}},
            {'action': 'call', 'name': 'app.write', 'arguments': {'value': 1}},
            {'action': 'call', 'name': 'app.write', 'arguments': {'value': 1}},
            {'action': 'finish'}])
        result, diag = run_agent(env, planner, TOOLS, {}, [], 'request', time.monotonic() + 10)
        self.assertEqual(len(result['tool_calls']), 1)
        self.assertEqual(result['state_hash'], 'hash-from-env')
        self.assertEqual(diag['stop_reason'], 'finished')

    def test_unsupported(self):
        result, _ = run_agent(FakeEnv(), ScriptedPlanner([{'action': 'finish'}]),
            TOOLS, {}, [], 'unsupported', time.monotonic() + 10)
        self.assertEqual(result['tool_calls'], [])

    def test_deadline_prevents_calls(self):
        env = FakeEnv()
        _, diag = run_agent(env, ScriptedPlanner([]), TOOLS, {}, [], '', time.monotonic() - 1)
        self.assertEqual(diag['stop_reason'], 'timeout')
        self.assertEqual(env.calls, [])

    def test_malformed_json(self):
        for raw in ('[]', '{"action":"call","name":"a","arguments":[]}', 'blah'):
            with self.assertRaises(ValueError):
                parse_decision(raw)

    def test_history_and_label_isolation(self):
        class ASR:
            def transcribe(self, path, deadline):
                return 'heard ' + path.name
        item = {'audio': 'audio/id/latest.wav', 'audio_text': 'GOLD SECRET',
            'label': {'secret': True}, 'history': [
                {'role': 'user', 'audio': 'audio/id/old.wav', 'audio_text': 'OLD GOLD'},
                {'role': 'assistant', 'text': 'done', 'tool_calls': [{'name': 'past.call'}]}]}
        text, history = prepare_input(item, ROOT, ASR(), False, time.monotonic() + 10)
        self.assertEqual(text, 'heard latest.wav')
        self.assertEqual(history[0]['text'], 'heard old.wav')
        self.assertNotIn('GOLD', json.dumps(history))
        with self.assertRaises(ValueError):
            audio_path(ROOT, '../state.json')

    @unittest.skipUnless((DATA / 'train/train.jsonl').exists(), 'Extracted train data absent')
    def test_real_sdk_training_reference(self):
        # Integration test only: feed known train calls as scripted decisions.
        # This is NOT a model accuracy benchmark and never uses test labels.
        sys.path.insert(0, str(DATA / 'train'))
        from zaic_tools import ToolEnv
        with (DATA / 'train/train.jsonl').open(encoding='utf-8') as f:
            item = json.loads(next(f))
        with (DATA / 'train/train_tools_catalog.json').open(encoding='utf-8') as f:
            tools = available_tools(json.load(f), item['context'])
        env = ToolEnv('train', item['id'], root=DATA)
        decisions = [{'action': 'call', **c} for c in item['label']['tool_calls']] + [{'action': 'finish'}]
        result, diag = run_agent(env, ScriptedPlanner(decisions), tools, item['context'], [],
            item['audio_text'], time.monotonic() + 10)
        self.assertEqual(result['tool_calls'], item['label']['tool_calls'])
        self.assertEqual(diag['stop_reason'], 'finished')
        # Debug-only public SDK access allowed for training verification.
        state = env.get_state()
        for app, expected in item['label']['final_state'].items():
            self.assertEqual(state[app], expected)


if __name__ == '__main__':
    unittest.main()

class DiscoveryTests(unittest.TestCase):
    def test_schema_discovery_is_not_a_submitted_call(self):
        class DiscoveryPlanner(ScriptedPlanner):
            def fits(self, messages):
                # Force initial prompt fallback, then allow schema discovery.
                return 'available_tools' in messages[1]['content']
        env = FakeEnv()
        planner = DiscoveryPlanner([
            {'action': 'inspect', 'tools': ['app.write']},
            {'action': 'call', 'name': 'app.write', 'arguments': {'value': 2}},
            {'action': 'finish'}])
        result, diag = run_agent(env, planner, TOOLS, {}, [], '', time.monotonic() + 10)
        self.assertEqual(len(result['tool_calls']), 1)
        self.assertEqual(result['tool_calls'][0]['name'], 'app.write')
        self.assertEqual(diag['stop_reason'], 'finished')

    def test_late_generation_cannot_execute(self):
        from unittest.mock import patch
        env = FakeEnv()
        planner = ScriptedPlanner([{'action': 'call', 'name': 'app.write', 'arguments': {'value': 1}}])
        with patch('baseline.agent.time.monotonic', side_effect=[0, 11]):
            result, diag = run_agent(env, planner, TOOLS, {}, [], '', 10)
        self.assertEqual(result['tool_calls'], [])
        self.assertEqual(diag['stop_reason'], 'timeout')

if __name__ == '__main__':
    unittest.main()

