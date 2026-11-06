"""Regression for Colab's Python 3.13 being rejected by a local-only version pin."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from baseline import doctor


class DoctorTests(unittest.TestCase):
    def test_colab_python_313_passes_otherwise_valid_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / 'config.json').write_text('{}')
            (checkpoint / 'model.safetensors').touch()
            model = {'path': directory, 'device': 'cpu', 'adapter': None}
            cfg = {'data_root': directory, 'llm': model, 'asr': model,
                   'agent': {'fewshot': 0}}
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            output = io.StringIO()
            with patch.object(doctor.sys, 'argv', ['doctor']), \
                 patch.object(doctor.sys, 'version_info', (3, 13, 16)), \
                 patch.object(doctor, 'load_config', return_value=cfg), \
                 patch.object(doctor, 'load_split', return_value=([], {'apps': []})), \
                 patch.object(doctor.importlib.metadata, 'version', return_value='installed'), \
                 patch.dict('sys.modules', {'torch': fake_torch}), redirect_stdout(output):
                doctor.main()
            self.assertIn('Preflight passed', output.getvalue())


if __name__ == '__main__':
    unittest.main()
