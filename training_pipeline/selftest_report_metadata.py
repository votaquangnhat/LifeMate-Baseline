"""Report serialization and strict prepared-cache compatibility regression tests."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from agent_core import write_json
from report_metadata import report_json_value, refresh_prepared_report_fingerprint


DtypeStub = type("dtype", (), {"__module__":"torch", "__str__":lambda self:"torch.float16"})
DeviceStub = type("device", (), {"__module__":"torch", "__str__":lambda self:"cuda:1"})


class Tests(unittest.TestCase):
    def test_logged_dtype_error_is_reproduced_and_fixed_for_nested_report(self):
        dtype = DtypeStub()
        report = {"rank":0, "quantization":{"load_in_4bit":True, "bnb_4bit_compute_dtype":dtype,
                  "nested":[{"dtype":dtype}]}, "trainable_parameters":16515072}
        with self.assertRaisesRegex(TypeError,"dtype"):
            json.dumps(report)
        fixed = report_json_value(report)
        self.assertEqual(fixed["quantization"]["bnb_4bit_compute_dtype"],"torch.float16")
        self.assertEqual(fixed["quantization"]["nested"][0]["dtype"],"torch.float16")
        self.assertIs(report["quantization"]["bnb_4bit_compute_dtype"],dtype)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"model_report.rank0.json";write_json(path,fixed)
            self.assertEqual(json.loads(path.read_text()),fixed)

    def test_report_keeps_numeric_bool_and_device_metadata(self):
        report={"rank":1,"load_in_4bit":True,"loss":0.25,"device":DeviceStub(),"path":Path("report.json")}
        fixed=report_json_value(report)
        self.assertEqual(fixed,{"rank":1,"load_in_4bit":True,"loss":0.25,"device":"cuda:1","path":"report.json"})
        self.assertIs(type(fixed["rank"]),int)
        self.assertIs(type(fixed["load_in_4bit"]),bool)

    def test_unknown_objects_and_nonfinite_values_still_fail(self):
        with self.assertRaises(TypeError): report_json_value({"unknown":object()})
        for value in (float("nan"),float("inf")):
            with self.assertRaises(ValueError): report_json_value({"value":value})

    @unittest.skipUnless(importlib.util.find_spec("torch"),"Actual torch dtype check runs when PyTorch is installed")
    def test_actual_torch_dtypes_without_gpu_or_weights(self):
        import torch
        report={"dtype":torch.float16,"storage":torch.uint8,"nested":[torch.bfloat16],"device":torch.device("cuda:1")}
        fixed=report_json_value(report)
        self.assertEqual(fixed,{"dtype":"torch.float16","storage":"torch.uint8","nested":["torch.bfloat16"],"device":"cuda:1"})
        self.assertIs(report["dtype"],torch.float16)

    def cache_fixture(self,tmp):
        out=Path(tmp)
        old={"source":"input-sha","catalog":"catalog-sha","model_revision":"model-sha",
             "tokenizer_revision":"tokenizer-sha","retrieval":{"policy":"compact_all"},
             "lengths":{"max_length_cap":5120},"runtime_versions_sha256":"versions-sha",
             "code_sha256":{"agent_core.py":"core-sha","prepare_agent.py":"prepare-sha", "context_rerank.py":"context-sha",
                            "train_agent.py":"train-sha", "modeling_agent.py":"old-modeling", "run_pipeline.py":"old-driver"}}
        current=copy.deepcopy(old)
        current["code_sha256"].update({"modeling_agent.py":"fixed-modeling","run_pipeline.py":"fixed-driver",
                                     "report_metadata.py":"helper-sha","selftest_report_metadata.py":"test-sha"})
        write_json(out/"prepared_fingerprint.json",old)
        for part in ("train","validation","test"):
            for kind in ("items","tokens","references"):
                (out/f"{part}_{kind}.jsonl").write_text('{"id":"unchanged-data"}\n')
        write_json(out/"length_report.json",{"errors":[],"resolved":{"max_length":5120,"max_new_tokens":512}})
        (out/"tokenizer").mkdir()
        return old,current

    def test_cache_reused_for_report_only_changes_without_rewriting_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,current=self.cache_fixture(tmp)
            rows=Path(tmp)/"train_tokens.jsonl";before=rows.read_bytes()
            self.assertTrue(refresh_prepared_report_fingerprint(tmp,current))
            self.assertEqual(json.loads((Path(tmp)/"prepared_fingerprint.json").read_text()),current)
            self.assertEqual(rows.read_bytes(),before)

    def test_cache_not_reused_for_input_policy_package_or_preprocessing_changes(self):
        for field in ("source","retrieval","runtime_versions_sha256","core-code"):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                old,current=self.cache_fixture(tmp)
                if field=="core-code":current["code_sha256"]["agent_core.py"]="changed"
                else:current[field]="changed"
                self.assertFalse(refresh_prepared_report_fingerprint(tmp,current))
                self.assertEqual(json.loads((Path(tmp)/"prepared_fingerprint.json").read_text()),old)

    def test_cache_not_rewritten_after_training_started(self):
        for trained in ("training_recipe.json","training_complete.json","checkpoints/checkpoint-1"):
            with self.subTest(trained=trained),tempfile.TemporaryDirectory() as tmp:
                old,current=self.cache_fixture(tmp)
                path=Path(tmp)/trained;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("{}")
                self.assertFalse(refresh_prepared_report_fingerprint(tmp,current))
                self.assertEqual(json.loads((Path(tmp)/"prepared_fingerprint.json").read_text()),old)

    def test_incomplete_prepared_data_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,current=self.cache_fixture(tmp)
            (Path(tmp)/"test_tokens.jsonl").unlink()
            self.assertFalse(refresh_prepared_report_fingerprint(tmp,current))
            self.assertEqual(json.loads((Path(tmp)/"prepared_fingerprint.json").read_text()),old)

    def test_unchanged_cache_is_left_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,current=self.cache_fixture(tmp)
            marker=Path(tmp)/"prepared_fingerprint.json";before=marker.read_bytes()
            self.assertFalse(refresh_prepared_report_fingerprint(tmp,old))
            self.assertEqual(marker.read_bytes(),before)


if __name__ == "__main__": unittest.main(verbosity=2)
