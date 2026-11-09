"""CPU contracts for the new ASR handoff; no checkpoint/GPU downloads."""
import ast
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import gipformer_asr as handoff
import infer_standalone
from gipformer.infer_asr import unpack_weights
from gipformer.submission_utils import write_jsonl
from gipformer.text_processing import context_phrases


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.export = self.root / 'best_export'
        self.export.mkdir()
        self.checkpoint = self.export / 'model.pt'
        self.checkpoint.write_bytes(b'test checkpoint')

    def test_select_export_or_exact_full_checkpoint(self):
        self.assertEqual(handoff.resolve_local_checkpoint(str(self.export), self.root / 'cache'), self.checkpoint)
        self.assertEqual(handoff.resolve_local_checkpoint(str(self.checkpoint), self.root / 'cache'), self.checkpoint)

    def test_zip_export_preserves_companion_assets(self):
        archive = self.root / 'export.zip'
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('best_export/model.pt', b'fine tuned')
            stream.writestr('best_export/tokens.txt', '<blk> 0')
        checkpoint = handoff.resolve_local_checkpoint(str(archive), self.root / 'cache')
        self.assertEqual(checkpoint.read_bytes(), b'fine tuned')
        self.assertTrue(checkpoint.with_name('tokens.txt').is_file())
        self.assertEqual(handoff.resolve_local_checkpoint(str(archive), self.root / 'cache'), checkpoint)

    def test_archive_escape_rejected(self):
        archive = self.root / 'bad.zip'
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('../escape.pt', b'bad')
        with self.assertRaisesRegex(ValueError, 'escapes'):
            handoff.resolve_local_checkpoint(str(archive), self.root / 'cache')

    def test_metadata_changes_checkpoint_identity(self):
        first = handoff.checkpoint_identity(self.checkpoint)
        (self.export / 'decode_config.json').write_text('{"beam_width":4}')
        self.assertNotEqual(first, handoff.checkpoint_identity(self.checkpoint))

    def test_checkpoint_prefixes_and_collision_detection(self):
        self.assertEqual(unpack_weights({'model': {'module.asr.encoder.weight': 1}}), {'encoder.weight': 1})
        self.assertEqual(unpack_weights({'state_dict': {'model.asr.joiner.weight': 2}}), {'joiner.weight': 2})
        self.assertEqual(unpack_weights({'encoder.weight': 3}), {'encoder.weight': 3})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            unpack_weights({'asr.x': 1, 'x': 2})

    def test_context_cannot_read_hidden_state_or_gold(self):
        hidden = self.root / 'public_test_environment/public_1'
        hidden.mkdir(parents=True)
        (hidden / 'state.json').write_text('{"name":"Hidden Person"}')
        item = {'id':'public_1', 'context':{'user_profile':{'name':'Visible Person'}},
                'label':{'name':'Gold Person'}, 'audio_text':'Gold Speech',
                'history':[{'role':'assistant','tool_results':[{'name':'Prior Contact'}]}]}
        phrases = context_phrases(item, self.root, 'public_test')
        self.assertIn('Visible Person', phrases)
        self.assertIn('Prior Contact', phrases)
        self.assertNotIn('Hidden Person', phrases)
        self.assertNotIn('Gold Person', phrases)

    def test_transcript_handoff_covers_final_and_user_history(self):
        data = self.root / 'data/public_test'
        data.mkdir(parents=True)
        items = [{'id':'public_1', 'audio':'audio/main.wav','context':{},
                  'history':[{'role':'user','audio':'audio/u1.wav'}]}]
        write_jsonl(data / 'public_test.jsonl', items)
        output = self.root / 'asr_output'
        destination = self.root / 'public_asr.jsonl'
        def run_jobs(jobs, log_dir):
            self.assertEqual(jobs[0][2]['CUDA_VISIBLE_DEVICES'], '0')
            write_jsonl(output / 'asr/transcripts.rank0.jsonl', [{
                'id':'public_1','audio_text':'recognized request',
                'history_audio_texts':{'audio/u1.wav':'recognized history'}}])
        with patch('gipformer.runtime.setup_asr', return_value='/asr/python'), \
                patch('gipformer.processes.run_jobs', side_effect=run_jobs) as runner:
            handoff.transcribe(checkpoint=self.checkpoint, data_root=data.parent, output=output,
                               destination=destination, work_root=self.root / 'work')
            handoff.transcribe(checkpoint=self.checkpoint, data_root=data.parent, output=output,
                               destination=destination, work_root=self.root / 'work')
        runner.assert_called_once()
        self.assertEqual(json.loads(destination.read_text())['history_audio_texts'], ['recognized history'])

    def test_setup_probe_scripts_compile(self):
        source = Path(__file__).parent / 'gipformer/runtime.py'
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'probe' for t in node.targets):
                compile(ast.literal_eval(node.value), 'ASR setup probe', 'exec')


if __name__ == '__main__':
    unittest.main(verbosity=2)
