"""CPU contract tests for selection, causal history, leakage and schema constraints."""
import copy
import json
import unittest
from agent_core import catalog_tools, dumps, rollout, tokenize_sft, add_observation
from context_rerank import compile_context, compact_schema, history_blocks, CachedReranker, dependency_closure


class FakeTokenizer:
    eos_token_id = 999999
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        text = dumps(messages) + ("ASSISTANT:" if add_generation_prompt else "")
        return self.encode(text) if tokenize else text


class FakeRanker:
    def __init__(self):
        self.seen = []
    def score(self, query, documents):
        self.seen += [query] + list(documents)
        return [float(sum(term in d.lower() and term in query.lower() for term in ("calendar", "shopping", "sms", "active")))
                + (0.5 if '.update"' in d else 0) for d in documents]


def fixture():
    catalog = {"apps":[]}
    app_tools = {"calendar":["update", "search", "get"], "shopping":["add", "search"],
                 "notes":["create"], "sms":["send"], "contacts":["search"], "sys":["ask_missing_info"]}
    for app, verbs in app_tools.items():
        catalog["apps"].append({"id":app, "tools":[
            {"name":app+"."+v, "kind":"read" if v in ("search", "get") else "write",
             "description":app+" "+v, "parameters":{"type":"object", "properties":{
                 "contact_id" if app=="sms" else "id":{"type":"string", "format":"uuid"}}, "required":["contact_id" if app=="sms" else "id"]}}
            for v in verbs]})
    tools = catalog_tools(catalog)
    item = {"id":"train_1", "audio_text":"calendar update active", "history":[],
            "context":{"current_time":"2026-11-05T12:00:00", "timezone":"Asia/Ho_Chi_Minh",
                       "installed_apps":{a:{"tools":[a+"."+v for v in vs]} for a,vs in app_tools.items() if a!="sys"}},
            "label":{"tool_calls":[{"name":"notes.create", "arguments":{"id":"SECRET_GOLD_ID"}}], "final_state":"SECRET_STATE"}}
    cfg = {"global_meta_tools":["sys.ask_missing_info"], "top_k_apps":1, "top_k_tools":1,
           "top_k_history":1, "keep_recent_history":2, "schema_description_chars":120,
           "prompt_target_tokens":100000, "tool_dependencies":{}, "max_agent_steps":4, "max_repairs":1}
    return item, tools, cfg


class Tests(unittest.TestCase):
    def test_label_independence_and_no_gold_fallback(self):
        item,tools,cfg = fixture()
        ranker = FakeRanker()
        a,audit = compile_context(item,tools,ranker,FakeTokenizer(),cfg)
        changed = copy.deepcopy(item)
        changed["label"] = {"tool_calls":[{"name":"shopping.add", "arguments":{}}]}
        b,_ = compile_context(changed,tools,FakeRanker(),FakeTokenizer(),cfg)
        self.assertEqual(a,b)
        self.assertNotIn("notes.create", audit["selected_tools"])
        self.assertNotIn("SECRET_GOLD_ID", " ".join(ranker.seen) + dumps(a))
        self.assertNotIn("SECRET_STATE", dumps(a))

    def test_lookup_dependencies_and_meta_tool(self):
        item,tools,cfg = fixture()
        _,a = compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)
        self.assertTrue({"calendar.update", "calendar.search", "calendar.get", "sys.ask_missing_info"} <= set(a["selected_tools"]))

    def test_cross_app_contact_id_lookup(self):
        item,tools,cfg = fixture()
        selected = dependency_closure({"sms.send"},list(tools),tools,cfg)
        self.assertIn("contacts.search",selected)

    def test_explicit_dependency_transitive_closure(self):
        _,tools,cfg = fixture()
        cfg["tool_dependencies"] = {"notes.create":["sms.send"], "sms.send":["shopping.add"]}
        s = dependency_closure({"notes.create"},list(tools),tools,cfg)
        self.assertTrue({"sms.send","contacts.search","shopping.add","shopping.search"} <= s)

    def test_uninstalled_never_selected(self):
        item,tools,cfg = fixture()
        item["context"]["installed_apps"].pop("calendar")
        _,a = compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)
        self.assertFalse(any(n.startswith("calendar.") for n in a["selected_tools"]))

    def test_multiple_intents_keep_both_apps(self):
        item,tools,cfg = fixture()
        item["audio_text"] = "calendar update active rồi shopping add milk"
        _,a = compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)
        self.assertIn("shopping.add",a["selected_tools"])
        self.assertIn("calendar.update",a["selected_tools"])

    def history_item(self):
        item,tools,cfg = fixture()
        for i in range(5):
            item["history"] += [{"role":"user", "text":f"{'active' if i==0 else 'old'} {i}"},
                                {"role":"assistant", "text":"done", "tool_calls":[{"name":"calendar.get", "arguments":{"id":f"ID_{i}"}}]},
                                {"role":"tool", "content":{"id":f"ID_{i}"}}]
        return item,tools,cfg

    def test_history_is_chronological_and_whole(self):
        item,tools,cfg = self.history_item()
        messages,a = compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)
        self.assertEqual(a["history_blocks_kept"],[0,3,4])
        h = json.loads(messages[1]["content"].split("\nFINAL_REQUEST=")[0].removeprefix("HISTORY="))
        self.assertEqual(len(h),9)
        self.assertEqual([h[i]["text"] for i in (0,3,6)],["active 0","old 3","old 4"])
        self.assertIn("ID_0",dumps(h))

    def test_budget_removes_old_whole_blocks_only(self):
        item,tools,cfg = self.history_item()
        cfg["prompt_target_tokens"] = 1
        messages,a = compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)
        self.assertEqual(a["history_blocks_kept"],[3,4])
        self.assertIn(item["audio_text"],messages[1]["content"])
        self.assertIn(item["context"]["current_time"],messages[0]["content"])
        self.assertIn("ID_4",messages[1]["content"])
        self.assertGreater(a["compiled_prompt_tokens"],1)

    def test_schema_constraints_survive(self):
        schema = {"type":"object", "$defs":{"ID":{"type":"integer", "minimum":1}},
                  "required":["x"], "properties":{"x":{"$ref":"#/$defs/ID", "enum":[1,2],
                  "description":"a"*200}}, "additionalProperties":False, "oneOf":[{"required":["x"]}]}
        c = compact_schema(schema,10)
        self.assertEqual(c["$defs"],schema["$defs"])
        self.assertEqual(c["oneOf"],schema["oneOf"])
        self.assertFalse(c["additionalProperties"])
        self.assertEqual(c["properties"]["x"]["enum"],[1,2])

    def test_long_windows_cover_all_tokens(self):
        for length in (1,50,192,317,318,350,601,610,5000):
            ids = list(range(length))
            windows = CachedReranker.windows(ids,192)
            self.assertEqual(set(x for w in windows for x in w),set(ids))
            self.assertTrue(all(len(w)<=192 for w in windows))

    def test_target_loss_mask_and_causal_observations(self):
        item,tools,cfg = fixture()
        messages,_ = compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)
        before = copy.deepcopy(messages)
        target = {"name":"calendar.update", "arguments":{"id":"EXACT_ID"}}
        row = {"messages":messages, "target":dumps(target)}
        encoded = tokenize_sft(row,FakeTokenizer())
        prefix = FakeTokenizer().apply_chat_template(messages)
        self.assertTrue(all(x==-100 for x in encoded["labels"][:len(prefix)]))
        self.assertEqual(encoded["labels"][len(prefix):],FakeTokenizer().encode(dumps(target)+"<|im_end|>"))
        add_observation(messages,target,{"ok":True,"id":"EXACT_ID"})
        self.assertNotIn("EXACT_ID",dumps(before))
        self.assertIn("EXACT_ID",dumps(messages))

    def test_rollout_consumes_exact_compiled_prompt(self):
        item,tools,cfg = fixture()
        compiled,_ = compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)
        item["_agent_messages"] = copy.deepcopy(compiled)
        class Env:
            def get_tool_calls(self): return []
            def get_state_hash(self): return "h"
        observed = []
        def generate(messages):
            observed.append(copy.deepcopy(messages))
            return '{"done":true}'
        rollout(item,tools,Env(),generate,cfg)
        self.assertEqual(observed[0],compiled)
        self.assertEqual(item["_agent_messages"],compiled)

    def test_empty_history_and_tool_set(self):
        item,tools,cfg = fixture()
        item["context"]["installed_apps"] = {}
        cfg["global_meta_tools"] = []
        _,audit = compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)
        self.assertEqual(audit["selected_tools"],[])
        self.assertEqual(audit["history_blocks_kept"],[])


if __name__ == "__main__":
    unittest.main(verbosity=2)
