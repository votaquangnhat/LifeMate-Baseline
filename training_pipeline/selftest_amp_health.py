"""CPU regressions for confirmed AMP overflow, fatal failures, and cache recovery."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
import math
import os
import tempfile
import types
import unittest
from pathlib import Path
from agent_core import write_json
from amp_health import AMPHealth, initialize_grad_scaler, recover_prepared_after_amp_fix


class FakeScaler:
    def __init__(self):
        self.state = {"scale":65536.0, "growth_factor":2.0, "backoff_factor":0.5,
                      "growth_interval":2000, "_growth_tracker":42}
    def is_enabled(self): return True
    def get_scale(self): return self.state["scale"]
    def state_dict(self): return self.state.copy()
    def load_state_dict(self, state): self.state=state.copy()


def actual_callback_class():
    # Execute the delivered callback class without importing Qwen/CUDA/Unsloth.
    tree=ast.parse(Path(__file__).with_name("train_agent.py").read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="HealthCallback")
    ns={"TrainerCallback":object,"Path":Path,"math":math,"os":os,"json":json,
        "AMPHealth":AMPHealth,"AMP_INIT_SCALE":1024.0,"write_json":write_json}
    exec(compile(ast.Module(body=[node],type_ignores=[]),"callback-regression","exec"),ns)
    return ns["HealthCallback"]


class Tests(unittest.TestCase):
    def test_reproduces_original_fatal_norm_and_allows_confirmed_amp_skip(self):
        logs={"loss":1.2,"grad_norm":float("inf")}
        with self.assertRaises(FloatingPointError):
            for k,v in logs.items():
                if not math.isfinite(v): raise FloatingPointError(k)
        h=AMPHealth(True);h.optimizer_step(True);h.check_logs(logs)
        h.optimizer_step(False);h.check_logs({"loss":1.1,"grad_norm":2.0})
        self.assertEqual(h.successful_steps,1);self.assertEqual(h.skipped_steps,1)

    def test_nan_inf_loss_always_fails_even_when_amp_skips(self):
        h=AMPHealth(True);h.optimizer_step(True)
        for k in ("loss","eval_loss"):
            for v in (float("nan"),float("inf")):
                with self.assertRaises(FloatingPointError):h.check_logs({k:v})

    def test_nonfinite_norm_without_confirmed_skip_is_fatal(self):
        for enabled in (True,False):
            h=AMPHealth(enabled);h.optimizer_step(False)
            with self.assertRaises(FloatingPointError):h.check_logs({"grad_norm":float("nan")})

    def test_unscaled_optimizer_skip_is_fatal(self):
        with self.assertRaises(FloatingPointError):AMPHealth(False).optimizer_step(True)

    def test_persistent_overflow_fails_instead_of_training_without_updates(self):
        h=AMPHealth(True)
        for i in range(7):h.optimizer_step(True)
        with self.assertRaisesRegex(FloatingPointError,"8 consecutive"):h.optimizer_step(True)
        self.assertEqual(h.successful_steps,0)

    def test_success_resets_consecutive_skip_budget(self):
        h=AMPHealth(True)
        for repeat in range(3):
            for i in range(7):h.optimizer_step(True)
            h.optimizer_step(False)
        self.assertEqual(h.consecutive_skips,0)
        self.assertEqual(h.skipped_steps,21)

    def test_scale_initialization_preserves_growth_and_backoff_settings(self):
        s=FakeScaler();before=s.state_dict();a=types.SimpleNamespace(mixed_precision="fp16",scaler=s)
        self.assertTrue(initialize_grad_scaler(a))
        self.assertEqual(s.state["scale"],1024.0)
        self.assertEqual(s.state["_growth_tracker"],0)
        for k in ("growth_interval","growth_factor","backoff_factor"):self.assertEqual(s.state[k],before[k])
        # Trainer's later checkpoint load must override initialization.
        resumed={**before,"scale":128.0};s.load_state_dict(resumed)
        self.assertEqual(s.state["scale"],128.0)

    def test_fp16_requires_scaler_and_other_modes_remain_untouched(self):
        with self.assertRaises(RuntimeError):initialize_grad_scaler(types.SimpleNamespace(mixed_precision="fp16",scaler=None))
        a=types.SimpleNamespace(mixed_precision="bf16",scaler=None)
        self.assertFalse(initialize_grad_scaler(a))

    def test_actual_callback_uses_accelerate_skip_signal_and_records_report(self):
        cls=actual_callback_class()
        with tempfile.TemporaryDirectory() as tmp:
            a=types.SimpleNamespace(mixed_precision="fp16",scaler=FakeScaler(),
                                    optimizer_step_was_skipped=True,is_main_process=True)
            cb=cls(a,tmp);state=types.SimpleNamespace(global_step=9)
            with contextlib.redirect_stdout(io.StringIO()):cb.on_optimizer_step(None,state,None)
            cb.on_log(None,state,None,logs={"loss":1.0,"grad_norm":float("inf")})
            a.optimizer_step_was_skipped=False;cb.on_optimizer_step(None,state,None)
            with self.assertRaises(FloatingPointError):cb.on_log(None,state,None,logs={"grad_norm":float("inf")})
            cb.on_train_end(None,state,None)
            report=json.loads(next(Path(tmp).glob("amp_health.rank*.json")).read_text())
            self.assertEqual(report["successful_steps"],1);self.assertEqual(report["skipped_steps"],1)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"Real CPU GradScaler test runs when PyTorch is installed")
    def test_real_cpu_gradscaler_skips_inf_then_recovers_without_changing_weights(self):
        import torch
        p=torch.nn.Parameter(torch.tensor([1.0]));opt=torch.optim.SGD([p],lr=0.1)
        s=torch.amp.GradScaler("cpu")
        initialize_grad_scaler(types.SimpleNamespace(mixed_precision="fp16",scaler=s))
        self.assertEqual(s.get_scale(),1024.0)
        h=AMPHealth(True)
        for bad in (True,False):
            opt.zero_grad();s.scale(p.square().sum()).backward()
            if bad:p.grad.fill_(float("inf"))
            before=p.detach().clone();scale=s.get_scale()
            s.step(opt);s.update();skipped=s.get_scale()<scale
            h.optimizer_step(skipped)
            h.check_logs({"loss":1.0,"grad_norm":float("inf") if bad else 2.0})
            self.assertEqual(torch.equal(before,p.detach()),bad)
        self.assertEqual(h.successful_steps,1)

    def cache_fixture(self,tmp):
        out=Path(tmp)
        old={"source":"data","runtime_versions_sha256":"packages","retrieval":{"policy":"compact_all"},
             "code_sha256":{"prepare_agent.py":"prep","agent_core.py":"core","context_rerank.py":"ranker",
                            "train_agent.py":"old","run_pipeline.py":"old"}}
        current=copy.deepcopy(old)
        current["code_sha256"].update({"train_agent.py":"fixed","run_pipeline.py":"fixed",
                                     "amp_health.py":"new","selftest_amp_health.py":"new"})
        write_json(out/"prepared_fingerprint.json",old)
        for p in ("train","validation","test"):
            for k in ("items","tokens","references"):(out/f"{p}_{k}.jsonl").write_text('{"token":"unchanged"}\n')
        write_json(out/"length_report.json",{})
        (out/"tokenizer").mkdir();write_json(out/"training_recipe.json",{"old_attempt":True})
        write_json(out/"preflight_report.json",{"passed":True,"old_recipe":True})
        return old,current

    def test_recovery_preserves_rows_archives_old_attempt_and_requires_new_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,current=self.cache_fixture(tmp);out=Path(tmp)
            rows={p.name:p.read_bytes() for p in out.glob("*.jsonl")}
            self.assertTrue(recover_prepared_after_amp_fix(out,current))
            self.assertEqual(rows,{p.name:p.read_bytes() for p in out.glob("*.jsonl")})
            self.assertFalse((out/"training_recipe.json").exists())
            self.assertFalse((out/"preflight_report.json").exists())
            archive=next((out/"failed_runs").iterdir())
            self.assertEqual(json.loads((archive/"prepared_fingerprint.json").read_text()),old)
            self.assertTrue((archive/"training_recipe.json").is_file())
            self.assertFalse(recover_prepared_after_amp_fix(out,current))

    def test_recovery_refuses_data_packages_or_preprocessing_changes(self):
        for key in ("source","runtime_versions_sha256","retrieval","agent_core.py","context_rerank.py","prepare_agent.py"):
            with tempfile.TemporaryDirectory() as tmp:
                old,current=self.cache_fixture(tmp)
                if key.endswith(".py"):current["code_sha256"][key]="changed"
                else:current[key]="changed"
                self.assertFalse(recover_prepared_after_amp_fix(tmp,current))
                self.assertEqual(json.loads((Path(tmp)/"prepared_fingerprint.json").read_text()),old)

    def test_recovery_refuses_existing_checkpoint_or_completed_training(self):
        for name in ("checkpoints/checkpoint-10","training_complete.json"):
            with tempfile.TemporaryDirectory() as tmp:
                old,current=self.cache_fixture(tmp);p=Path(tmp)/name;p.parent.mkdir(exist_ok=True);p.write_text("{}")
                self.assertFalse(recover_prepared_after_amp_fix(tmp,current))
                self.assertTrue((Path(tmp)/"training_recipe.json").exists())

    def test_recovery_refuses_incomplete_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,current=self.cache_fixture(tmp);(Path(tmp)/"test_tokens.jsonl").unlink()
            self.assertFalse(recover_prepared_after_amp_fix(tmp,current))


if __name__=="__main__":unittest.main(verbosity=2)
