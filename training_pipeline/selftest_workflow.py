"""Host-side workflow checks without network, installs or GPU allocations."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import colab_workflow as workflow


class Tests(unittest.TestCase):
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

    def test_aborted_public_run_does_not_replace_existing_submission(self):
        public = self.root / 'data' / 'public_test'
        public.mkdir(parents=True)
        (public / 'public_test.jsonl').write_text('{"id":"public_1"}\n')
        (self.out / 'public_summary.json').write_text('{"aborted_rollouts":1}')
        latest = self.root / 'submission.jsonl'
        latest.write_text('previous submission')
        with self.assertRaisesRegex(AssertionError, 'Public rollouts aborted'):
            workflow.publish_submission(self.out, self.root, self.root / 'data',
                                        'day1', self.out / 'best_model', self.root / 'asr.jsonl')
        self.assertEqual(latest.read_text(), 'previous submission')


if __name__ == '__main__':
    unittest.main(verbosity=2)
