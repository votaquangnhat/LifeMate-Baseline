"""Independent inference restoration and stage routing, without model downloads."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import infer_standalone as inference


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.training = self.root / 'runs' / 'trained'
        self.training.mkdir(parents=True)
        for name in ('run_config.json', 'selection.json', 'prepared_fingerprint.json', 'package_versions.json',
                     'best_adapter/adapter_config.json', 'tokenizer/tokenizer_config.json'):
            path = self.training / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('{}')
        (self.training / 'best_adapter/adapter_model.safetensors').write_bytes(b'selected weights')
        self.code = self.root / 'code'
        self.code.mkdir()
        (self.code / 'runtime.py').write_text('# fixture')
        self.output = self.training / 'inference' / 'fallback'

    def restore(self):
        return inference.restore_run(self.training, self.output, self.root / 'data',
                                     self.code, self.root / 'work', 'whisper', 'vinai/PhoWhisper-small',
                                     self.output / 'public_asr.jsonl')

    def test_restore_retains_training_files_and_reuses_identical_settings(self):
        before = {str(p.relative_to(self.training)): p.read_bytes()
                  for p in self.training.rglob('*') if p.is_file()}
        cfg = self.restore()
        self.assertEqual(self.restore(), cfg)
        self.assertEqual(cfg['output_dir'], str(self.output))
        self.assertEqual(cfg['eval_num_gpus'], 1)
        for name, data in before.items():
            self.assertEqual((self.training / name).read_bytes(), data)
        self.assertEqual((self.output / 'best_adapter/adapter_model.safetensors').read_bytes(), b'selected weights')

    def test_changed_model_preserves_existing_journal_and_fails(self):
        self.restore()
        journal = self.output / 'journal.jsonl'
        journal.write_text('keep')
        (self.training / 'best_adapter/adapter_model.safetensors').write_bytes(b'new weights')
        with self.assertRaisesRegex(ValueError, 'new INFERENCE_NAME'):
            self.restore()
        self.assertEqual(journal.read_text(), 'keep')

    def test_missing_selected_model_reports_selection_not_retraining(self):
        (self.training / 'selection.json').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Step 9'):
            self.restore()

    def test_full_entrypoint_only_dispatches_public_stage(self):
        workflow = Mock()
        workflow.setup_environment.return_value = self.root / 'python'
        workflow.prepare_data.return_value = self.root / 'data'
        workflow.publish_submission.return_value = self.root / 'submission.jsonl'
        with patch.object(inference.importlib, 'reload', return_value=workflow):
            result = inference.run(drive_root=self.root, trained_run_name='trained',
                                   inference_name='fallback', code_dir=self.code,
                                   work_root=self.root / 'work')
        workflow.run_stage.assert_called_once_with(self.root / 'python', self.code,
                                                  self.output / 'run_config.json', 'public')
        self.assertEqual(workflow.prepare_asr.call_args.args[-2:],
                         (self.output / 'public_asr.jsonl', 'vinai/PhoWhisper-small'))
        self.assertEqual(result, self.root / 'submission.jsonl')

    def test_missing_external_asr_fails_before_setup(self):
        workflow = Mock()
        with patch.object(inference.importlib, 'reload', return_value=workflow), \
                self.assertRaisesRegex(FileNotFoundError, 'Use whisper mode'):
            inference.run(drive_root=self.root, trained_run_name='trained', inference_name='external',
                          code_dir=self.code, asr_mode='transcripts', work_root=self.root / 'work')
        workflow.setup_environment.assert_not_called()

    def test_gipformer_mode_uses_teammate_checkpoint_and_larger_context(self):
        checkpoint = self.root / 'best.pt'
        checkpoint.write_bytes(b'ASR fine-tuned weights')
        workflow = Mock()
        workflow.setup_environment.return_value = self.root / 'python'
        workflow.prepare_data.return_value = self.root / 'data'
        with patch.object(inference.importlib, 'reload', return_value=workflow), \
                patch('gipformer_asr.transcribe') as transcribe:
            inference.run(drive_root=self.root, trained_run_name='trained', inference_name='gipformer',
                          code_dir=self.code, asr_mode='gipformer', asr_checkpoint=str(checkpoint),
                          inference_max_length=8192, work_root=self.root / 'work')
        self.assertEqual(transcribe.call_args.kwargs['checkpoint'], checkpoint)
        workflow.prepare_asr.assert_not_called()
        config = json.loads((self.training / 'inference/gipformer/run_config.json').read_text())
        self.assertEqual(config['max_length'], 8192)
        self.assertEqual(workflow.run_stage.call_args.args[-1], 'public')


if __name__ == '__main__':
    unittest.main(verbosity=2)
