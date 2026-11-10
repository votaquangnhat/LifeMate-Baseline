
"""Executable metric and ToolEnv interaction contract tests, no GPU needed."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from agent_core import (argument_equal, call_matches, dumps, encode_sft, fuzzy_token_f1,
                        gold_replay, initial_messages, installed_tool_names, make_sft_rows,
                        maximum_matching, metric_summary, parse_action, rollout,
                        score_sample, catalog_tools, make_splits, tokenize_sft, check_sft_limits, token_length_record,
                        resolve_token_limits, apply_token_limits)


TOOLS = catalog_tools({"apps": [{"id": "sms", "tools": [
    {"name": "sms.send", "kind": "write", "parameters": {"type": "object", "properties": {
        "contact_id": {"type": "string"}, "text": {"type": "string", "x-match": "fuzzy"}}}}]},
    {"id": "contacts", "tools": [{"name": "contacts.search", "parameters": {"properties": {"query": {"type": "string"}}}}]},
    {"id": "sys", "tools": [{"name": "sys.ask_missing_info", "parameters": {"properties": {"question": {"type": "string", "x-match": "fuzzy"}}}}]}]})
CFG = {"global_meta_tools": ["sys.ask_missing_info"], "max_agent_steps": 8, "max_repairs": 2}


def call(text="a b c d", contact="person-7"):
    return {"name": "sms.send", "arguments": {"contact_id": contact, "text": text}}


def item():
    return {"id": "train_1", "audio_text": "Nhắn cho An tôi tới muộn.", "history": [],
            "context": {"current_time": "2026-11-04T10:00:00", "timezone": "Asia/Ho_Chi_Minh",
                        "installed_apps": {"sms": {"tools": ["sms.send"]},
                                           "contacts": {"tools": ["contacts.search"]}}},
            "label": {"tool_calls": [{"name": "contacts.search", "arguments": {"query": "An"}}, call()],
                      "final_state": "SECRET_GOLD_STATE"}}


class FakeEnv:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def validate(self, name, arguments):
        return {"ok": name in TOOLS and arguments.get("contact_id") != "bad-schema"}

    def call(self, name, arguments):
        self.calls.append({"name": name, "arguments": copy.deepcopy(arguments)})
        if name == "contacts.search":
            return {"ok": True, "contacts": [{"id": "person-7", "name": "An"}]}
        return {"ok": True}

    def get_tool_calls(self):
        return copy.deepcopy(self.calls)

    def get_state_hash(self):
        return "final" if any(c["name"] == "sms.send" for c in self.calls) else "initial"

    def get_state(self):
        raise AssertionError("Direct state access is forbidden")


class FakeTokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is False
        return [9, 10, 11]

    def encode(self, text, **kwargs):
        return [12, self.eos_token_id]


class Tests(unittest.TestCase):
    def score(self, pred, gold, pred_hash="h", gold_hash="h"):
        return score_sample(pred, gold, pred_hash, gold_hash, TOOLS)["score"]

    def test_exact(self):
        self.assertEqual(self.score([call()], [call()]), 1)

    def test_extra_call(self):
        self.assertAlmostEqual(self.score([call(), call("other"), call("again")], [call(), call("other")]), 0.8)

    def test_duplicate_prediction_not_reused(self):
        self.assertAlmostEqual(self.score([call(), call()], [call()]), 2/3)

    def test_duplicate_gold_needs_two_calls(self):
        self.assertAlmostEqual(self.score([call()], [call(), call()]), 2/3)

    def test_wrong_hash_is_zero(self):
        self.assertEqual(self.score([call()], [call()], "bad", "gold"), 0)

    def test_no_matches(self):
        self.assertEqual(self.score([call(contact="wrong-id")], [call()]), 0)

    def test_empty_gold(self):
        self.assertEqual(self.score([], []), 1)
        self.assertEqual(self.score([call()], []), 0)
        self.assertEqual(self.score([], [call()]), 0)

    def test_empty_gold_special_case(self):
        self.assertEqual(self.score([], [], "different", "hash"), 1)

    def test_fuzzy_strict_threshold(self):
        self.assertEqual(fuzzy_token_f1("a b c", "a b c d e f g"), 0.6)
        self.assertFalse(call_matches(call("a b c"), call("a b c d e f g"), TOOLS))
        self.assertTrue(call_matches(call("a b c d"), call("a b c d e f g"), TOOLS))

    def test_fuzzy_lower_nfc_multiset(self):
        self.assertEqual(fuzzy_token_f1("TÔI đến", "to\u0302i đến"), 1)
        self.assertAlmostEqual(fuzzy_token_f1("a a b", "a b b"), 2/3)

    def test_no_unrequested_punctuation_normalization(self):
        self.assertEqual(fuzzy_token_f1("đến.", "đến"), 0)

    def test_exact_arguments_and_types(self):
        self.assertFalse(call_matches(call(contact="PERSON-7"), call(), TOOLS))
        extra = call()
        extra["arguments"]["extra"] = "value"
        self.assertFalse(call_matches(extra, call(), TOOLS))
        self.assertFalse(argument_equal(True, 1, {}))

    def test_order_does_not_matter(self):
        self.assertEqual(self.score([call("x"), call("y")], [call("y"), call("x")]), 1)

    def test_augmenting_path_beats_greedy(self):
        pred = [call("a b c e"), call("a b c d")]
        gold = [call("a b c d"), call("a b e f")]
        self.assertEqual(len(maximum_matching(pred, gold, TOOLS)), 2)

    def test_missing_info_is_ordinary_call(self):
        ask = {"name": "sys.ask_missing_info", "arguments": {"question": "Bạn muốn nhắn ai"}}
        self.assertEqual(self.score([ask], [ask]), 1)

    def test_macro_mean_not_micro_f1(self):
        rows = [score_sample([], [], "h", "h", TOOLS), score_sample([], [call()] * 5, "h", "h", TOOLS)]
        self.assertEqual(metric_summary(rows)["official_score_100"], 50)

    def test_labels_never_enter_prompt(self):
        messages = initial_messages(item(), TOOLS, CFG["global_meta_tools"])
        self.assertNotIn("SECRET_GOLD_STATE", dumps(messages))
        self.assertNotIn("person-7", dumps(messages))

    def test_allowed_tools_only(self):
        sample = item()
        sample["context"]["installed_apps"] = {"sms": {"tools": ["send"]}}
        names = installed_tool_names(sample, TOOLS, [])
        self.assertEqual(names, ["sms.send"])
        self.assertNotIn("contacts.search", dumps(initial_messages(sample, TOOLS, [])))

    def test_real_tool_feedback_used(self):
        sample, env = item(), FakeEnv()
        def generate(messages):
            if len(env.calls) == 0:
                return dumps(sample["label"]["tool_calls"][0])
            if len(env.calls) == 1:
                self.assertIn("person-7", messages[-1]["content"])
                return dumps(call())
            return '{"done":true}'
        result = rollout(sample, TOOLS, env, generate, CFG)
        self.assertEqual(result["state_hash"], "final")
        self.assertEqual(len(result["tool_calls"]), 2)
        self.assertIsNone(result["error"])

    def test_schema_repair_is_not_executed(self):
        outputs = iter([dumps(call(contact="bad-schema")), dumps(call()), '{"done":true}'])
        result = rollout(item(), TOOLS, FakeEnv(), lambda m: next(outputs), CFG)
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["invalid_model_actions"], 1)

    def test_rejected_actual_calls_remain_in_log(self):
        class RejectEnv(FakeEnv):
            def call(self, name, arguments):
                super().call(name, arguments)
                return {"ok": False, "error": "schema rejected by runtime"}
        outputs = iter([dumps(call("x")), '{"done":true}'])
        result = rollout(item(), TOOLS, RejectEnv(), lambda m: next(outputs), CFG)
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["trace"][0]["result"]["ok"], False)

    def test_out_of_scope_no_execution(self):
        result = rollout(item(), TOOLS, FakeEnv(), lambda m: '{"done":true}', CFG)
        self.assertEqual(result["tool_calls"], [])

    def test_no_infinite_tool_loop(self):
        result = rollout(item(), TOOLS, FakeEnv(), lambda m: dumps(call()), CFG)
        self.assertEqual(len(result["tool_calls"]), CFG["max_agent_steps"])
        self.assertTrue(result["error"])

    def test_sft_is_causal_and_teaches_done(self):
        sample = item()
        rows = list(make_sft_rows(sample, TOOLS, [{"ok": True, "id": "person-7"}, {"ok": True}], CFG["global_meta_tools"]))
        self.assertEqual(len(rows), 3)
        self.assertNotIn("person-7", dumps(rows[0]["messages"]))
        self.assertIn("person-7", dumps(rows[1]["messages"]))
        self.assertEqual(json.loads(rows[-1]["target"]), {"done": True})

    def test_response_mask_and_eos(self):
        row = {"id": "x", "step": 0, "messages": [], "target": '{"done":true}'}
        result = encode_sft(row, FakeTokenizer(), 5, 2)
        self.assertEqual(result["labels"], [-100, -100, -100, 12, 1])
        with self.assertRaises(ValueError):
            encode_sft(row, FakeTokenizer(), 4, 2)

    def test_history_transcript_required(self):
        sample = item()
        sample["history"] = [{"role": "user", "audio": "old.wav"}]
        with self.assertRaises(ValueError):
            initial_messages(sample, TOOLS, [])

    def test_parse_strict_json(self):
        for bad in ('{"done":1}', '{"done":true} trailing', '[]', '{"name":"sms.send","arguments":"{}"}'):
            with self.assertRaises((ValueError, TypeError)):
                parse_action(bad)

    def test_gold_replay_uses_hash_api_only(self):
        observations, value = gold_replay(item(), FakeEnv, ".", TOOLS, CFG["global_meta_tools"])
        self.assertEqual(value, "final")
        self.assertEqual(len(observations), 2)


    def length_cfg(self, **updates):
        return dict(max_length=4096, max_new_tokens=512, auto_token_limits=True,
                    max_length_cap=32768, max_new_tokens_cap=4096,
                    length_round_to=256, length_safety_margin=256, **updates)

    def length_rows(self):
        return [{"id": "short", "step": 0, "split": "train", "prompt_tokens": 100,
                 "action_tokens": 50, "total_tokens": 150},
                {"id": "long", "step": 1, "split": "validation", "prompt_tokens": 5000,
                 "action_tokens": 600, "total_tokens": 5600}]

    def test_auto_limits_fit_long_prompt_and_action(self):
        report = resolve_token_limits(self.length_rows(), self.length_cfg())
        self.assertFalse(report["errors"])
        self.assertEqual(report["resolved"], {"max_length": 5888, "max_new_tokens": 600})
        self.assertEqual(report["initial_overflow_actions"], 1)
        self.assertEqual(report["initial_overflow_samples"], 1)
        self.assertEqual(report["longest_rows"][0]["split"], "validation")
        self.assertEqual(report["discarded_rows"], 0)

    def test_auto_limits_keep_short_data_defaults(self):
        report = resolve_token_limits(self.length_rows()[:1], self.length_cfg())
        self.assertEqual(report["resolved"], {"max_length": 4096, "max_new_tokens": 512})
        self.assertEqual(report["initial_overflow_actions"], 0)

    def test_auto_limits_reserve_full_action_budget(self):
        rows = [{"id": "near", "step": 0, "split": "test", "prompt_tokens": 4000,
                 "action_tokens": 20, "total_tokens": 4020}]
        report = resolve_token_limits(rows, self.length_cfg())
        self.assertEqual(report["resolved"]["max_length"], 4864)
        self.assertFalse(report["errors"])

    def test_auto_limits_report_caps_without_discarding(self):
        cfg = self.length_cfg(); cfg.update(max_length_cap=4096, max_new_tokens_cap=512)
        report = resolve_token_limits(self.length_rows(), cfg)
        self.assertEqual(len(report["errors"]), 2)
        self.assertEqual(report["sft_actions"], 2)
        self.assertEqual(report["discarded_rows"], 0)
        with self.assertRaises(ValueError):
            apply_token_limits(cfg, report)

    def test_fixed_limits_fail_with_diagnostics(self):
        cfg = self.length_cfg(); cfg["auto_token_limits"] = False
        report = resolve_token_limits(self.length_rows(), cfg)
        self.assertTrue(report["errors"])
        self.assertEqual(report["resolved"]["max_length"], 4096)

    def test_auto_limits_count_actions_and_unique_samples(self):
        rows = self.length_rows() + [dict(self.length_rows()[1], step=2)]
        report = resolve_token_limits(rows, self.length_cfg())
        self.assertEqual(report["initial_overflow_actions"], 2)
        self.assertEqual(report["initial_overflow_samples"], 1)

    def test_auto_limits_empty_data_fails(self):
        with self.assertRaises(ValueError):
            resolve_token_limits([], self.length_cfg())

    def test_auto_limits_reuse_is_idempotent(self):
        cfg = self.length_cfg()
        first = resolve_token_limits(self.length_rows(), cfg)
        apply_token_limits(cfg, first)
        second = resolve_token_limits(self.length_rows(), cfg)
        self.assertEqual(first, second)
        self.assertEqual(cfg["requested_max_length"], 4096)
        self.assertEqual(cfg["max_length"], 5888)

    def test_auto_limits_persist_to_worker_config(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.length_cfg(); cfg["config_path"] = str(Path(directory) / "config.json")
            report = resolve_token_limits(self.length_rows(), cfg)
            apply_token_limits(cfg, report)
            saved = json.loads(Path(cfg["config_path"]).read_text())
            self.assertEqual(saved["max_length"], 5888)
            self.assertEqual(saved["max_new_tokens"], 600)
            self.assertEqual(saved["requested_max_length"], 4096)

    def test_long_rows_keep_every_token_and_target(self):
        class LongTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return [9] * 5000
            def encode(self, text, **kwargs):
                return [12] * 599 + [self.eos_token_id]
        row = {"id": "long", "step": 1, "messages": [], "target": "data"}
        encoded = tokenize_sft(row, LongTokenizer())
        record = token_length_record(row, encoded, "train")
        report = resolve_token_limits([record], self.length_cfg())
        check_sft_limits(encoded, **dict(max_length=report["resolved"]["max_length"],
                                        max_action_tokens=report["resolved"]["max_new_tokens"]))
        self.assertEqual(encoded["length"], 5600)
        self.assertEqual(encoded["labels"][:5000], [-100] * 5000)
        self.assertEqual(encoded["input_ids"][5000:], encoded["labels"][5000:])
        self.assertEqual(encoded["labels"][-1], 1)
        with self.assertRaises(ValueError):
            check_sft_limits(encoded, 4096, 600)
        with self.assertRaises(ValueError):
            check_sft_limits(encoded, 6000, 512)

    def test_asr_manifest_group_ids_without_single_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(dumps({"sample_ids": ["a", "b"], "split": "train"}) + "\n")
            result, _ = make_splits([{"id": "a"}, {"id": "b"}], {"asr_manifest_path": str(path)})
            self.assertEqual(result, {"a": "train", "b": "train"})

    def test_reported_4419_token_run_fits_5120_cap(self):
        # Observed in the user's run: max prompt=4413, max action=91,
        # max total=4419. Generation reserves 512 tokens plus a 128 margin.
        rows = [dict(id="longest", step=3, split="train", prompt_tokens=4413, action_tokens=6, total_tokens=4419),
                dict(id="largest-action", step=0, split="train", prompt_tokens=2000, action_tokens=91, total_tokens=2091)]
        cfg = dict(max_length=2048, max_new_tokens=512, requested_max_length=2048,
                   requested_max_new_tokens=512, auto_token_limits=True, max_length_cap=4096,
                   max_new_tokens_cap=1024, length_round_to=256, length_safety_margin=128)
        old = resolve_token_limits(rows, cfg)
        self.assertEqual(old["resolved"], {"max_length":5120, "max_new_tokens":512})
        self.assertIn("required max_length=5120 exceeds max_length_cap=4096", old["errors"])
        cfg["max_length_cap"] = 5120
        fixed = resolve_token_limits(rows, cfg)
        self.assertEqual(fixed["errors"], [])
        self.assertEqual(fixed["resolved"], old["resolved"])
        self.assertEqual(fixed["truncated_rows"], 0)
        self.assertEqual(fixed["discarded_rows"], 0)
        apply_token_limits(cfg, fixed)
        self.assertEqual(cfg["max_length"], 5120)
        self.assertEqual(cfg["requested_max_length"], 2048)


if __name__ == "__main__":
    unittest.main(verbosity=2)


