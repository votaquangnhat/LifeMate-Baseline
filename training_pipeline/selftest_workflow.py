"""Host-side workflow checks without network, installs or GPU allocations."""
import json
import ast
import contextlib
import importlib.metadata
import inspect
import io
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import colab_workflow as workflow


class Tests(unittest.TestCase):
    def setup_script(self, name):
        tree = ast.parse(inspect.getsource(workflow.setup_environment))
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
        return eval(compile(ast.Expression(assignment.value), '<setup script>', 'eval'),
                    {'dedent': workflow.dedent})

    def test_subprocess_payloads_compile_without_function_indentation(self):
        for name in ('probe_code', 'version_code'):
            compile(self.setup_script(name), name, 'exec')

    def test_actual_gpu_probe_script_emits_json(self):
        cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                     get_device_name=lambda i: 'Test L4',
                                     get_device_properties=lambda i: types.SimpleNamespace(total_memory=24 * 2**30))
        output = io.StringIO()
        with patch.dict('sys.modules', {'torch': types.SimpleNamespace(cuda=cuda)}), \
                patch.object(importlib.metadata, 'version', return_value='1.0'), \
                contextlib.redirect_stdout(output):
            exec(self.setup_script('probe_code'), {})
        result = json.loads(output.getvalue())
        self.assertEqual(result['gpus'], [{'name': 'Test L4', 'gib': 24}])
        self.assertEqual(result['pinned']['torch'], '1.0')

    def test_actual_package_probe_script_emits_json(self):
        output = io.StringIO()
        distribution = types.SimpleNamespace(read_text=lambda name: json.dumps({
            'url': 'https://example.invalid/repo', 'vcs_info': {'commit_id': 'abc'}}))
        with patch.dict('sys.modules', {'torch': types.SimpleNamespace()}), \
                patch.object(importlib.metadata, 'version', return_value='1.0'), \
                patch.object(importlib.metadata, 'distribution', return_value=distribution), \
                contextlib.redirect_stdout(output):
            exec(self.setup_script('version_code'), {})
        result = json.loads(output.getvalue())
        self.assertEqual(result['torch'], '1.0')
        self.assertEqual(result['source_revisions']['unsloth']['vcs_info']['commit_id'], 'abc')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'runs' / 'day1'
        self.out.mkdir(parents=True)
        self.args = dict(
            DATA_ROOT=self.root / 'data', OUTPUT_DIR=self.out,
            CODE_DIR=Path(__file__).parent.resolve(), CONFIG_PATH=self.out / 'run_config.json',
            WORK_ROOT=self.root / 'work', RUN_NAME='day1', EPOCHS=2,
            TRAIN_START='fresh', PREVIOUS_RUN_NAME='', RESUME_CHECKPOINT='auto',
            ASR_TRANSCRIPTS=self.root / 'asr.jsonl', DRIVE_ROOT=self.root,
        )

    def test_fresh_configuration_and_unchanged_rerun(self):
        first = workflow.configure(**self.args)
        self.assertEqual(first['initial_adapter'], '')
        self.assertEqual(first, workflow.configure(**self.args))
        self.assertEqual(first['epochs'], 2)

    def test_best_snapshots_previous_weights_without_overwriting_them(self):
        workflow.configure(**self.args)
        previous = self.out / 'best_adapter'
        previous.mkdir()
        (previous / 'adapter_model.safetensors').write_bytes(b'previous weights')
        (previous / 'adapter_config.json').write_text('{}')
        (self.out / 'selection.json').write_text('{}')
        next_out = self.root / 'runs' / 'day2'
        next_out.mkdir()
        self.args.update(TRAIN_START='best', PREVIOUS_RUN_NAME='day1', RUN_NAME='day2',
                         OUTPUT_DIR=next_out, CONFIG_PATH=next_out / 'run_config.json')
        config = workflow.configure(**self.args)
        snapshot = Path(config['initial_adapter']) / 'adapter_model.safetensors'
        self.assertEqual(snapshot.read_bytes(), b'previous weights')
        self.assertEqual(config, workflow.configure(**self.args))
        (previous / 'adapter_model.safetensors').write_bytes(b'changed weights')
        with self.assertRaisesRegex(AssertionError, 'Previous best changed'):
            workflow.configure(**self.args)
        self.assertEqual(snapshot.read_bytes(), b'previous weights')

    def test_changed_epochs_require_new_run(self):
        workflow.configure(**self.args)
        self.args['EPOCHS'] = 4
        with self.assertRaisesRegex(AssertionError, 'new RUN_NAME'):
            workflow.configure(**self.args)

    def test_stage_uses_selected_interpreter_repo_and_config(self):
        with patch.object(workflow.subprocess, 'run') as run:
            workflow.run_stage(Path('/agent/python'), Path('/repo/training_pipeline'),
                               Path('/drive/run.json'), 'train')
        run.assert_called_once_with([
            '/agent/python', '-u', '/repo/training_pipeline/run_pipeline.py',
            '--config', '/drive/run.json', '--stage', 'train'], check=True)

    def test_asr_manifest_preserves_only_audio_paths_and_user_order(self):
        public = self.root / 'data' / 'public_test'
        public.mkdir(parents=True)
        (self.root / 'inputs').mkdir()
        (public / 'public_test.jsonl').write_text(json.dumps({
            'id': 'public_1', 'audio': 'audio/final.wav', 'context': {'private': 1},
            'history': [{'role': 'user', 'audio': 'audio/u1.wav'},
                        {'role': 'assistant', 'text': 'response'},
                        {'role': 'user', 'audio': 'audio/u2.wav'}]}) + '\n')
        result = workflow.write_asr_manifest(self.root / 'data', self.root, self.root / 'asr.jsonl')
        self.assertEqual(json.loads(result.read_text()), {
            'id': 'public_1', 'audio': 'audio/final.wav',
            'history_audio': ['audio/u1.wav', 'audio/u2.wav']})

    def test_missing_predictions_do_not_replace_existing_submission(self):
        public = self.root / 'data' / 'public_test'
        public.mkdir(parents=True)
        (public / 'public_test.jsonl').write_text('{"id":"public_1"}\n')
        (self.out / 'public_summary.json').write_text('{"aborted_rollouts":1}')
        latest = self.root / 'submission.jsonl'
        latest.write_text('previous submission')
        with self.assertRaises(FileNotFoundError):
            workflow.publish_submission(self.out, self.root, self.root / 'data',
                                        'day1', self.out / 'best_model', self.root / 'asr.jsonl')
        self.assertEqual(latest.read_text(), 'previous submission')

    def test_complete_early_stop_exports_exact_records_and_reports_errors(self):
        public = self.root / 'data' / 'public_test'
        public.mkdir(parents=True)
        (public / 'public_test.jsonl').write_text('{"id":"public_1"}\n')
        (self.out / 'public_summary.json').write_text('{"samples":1,"aborted_rollouts":1}')
        predictions = self.out / 'public_test_best'
        predictions.mkdir()
        record = {'id':'public_1', 'tool_calls':[], 'state_hash':'recorded_hash'}
        source = predictions / 'submission.jsonl'
        source.write_text(json.dumps(record) + '\n')
        (predictions / 'predictions_with_trace.jsonl').write_text(json.dumps({**record, 'error':'Context overflow'}) + '\n')
        asr = self.root / 'asr.jsonl'
        asr.write_text('{}\n')
        with patch.object(workflow.subprocess, 'run') as validate:
            exported = workflow.publish_submission(self.out, self.root, self.root / 'data',
                                                   'day1', self.out / 'best_adapter', asr)
        self.assertEqual(exported.read_bytes(), source.read_bytes())
        self.assertIn('--validate', validate.call_args.args[0])
        report = json.loads((self.out / 'submission_quality_report.json').read_text())
        self.assertEqual(report['stopped_samples'], [{'id':'public_1', 'error':'Context overflow'}])
        exported.write_text('previous')
        with patch.object(workflow.subprocess, 'run', side_effect=RuntimeError('schema invalid')):
            with self.assertRaisesRegex(RuntimeError, 'schema invalid'):
                workflow.publish_submission(self.out, self.root, self.root / 'data',
                                            'day1', self.out / 'best_adapter', asr)
        self.assertEqual(exported.read_text(), 'previous')


if __name__ == '__main__':
    unittest.main(verbosity=2)
