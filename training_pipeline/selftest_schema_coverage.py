"""Regression tests for visible fallback schemas and argument-name preservation."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_core import dumps, read_jsonl, write_json, write_jsonl
from context_rerank import compile_context, compact_schema
from selftest_context import fixture, FakeTokenizer, FakeRanker


def tools_in(messages):
    return json.loads(messages[0]["content"].split("\nTOOLS=",1)[1].split("\nCONTEXT=",1)[0])


class PrepareTokenizer(FakeTokenizer):
    eos_token="<|im_end|>"
    pad_token=None
    pad_token_id=None
    def convert_tokens_to_ids(self,text): return self.eos_token_id
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=True,**kwargs):
        text=dumps(messages)+("<|im_start|>assistant\n" if add_generation_prompt else "")
        return self.encode(text) if tokenize else text
    def save_pretrained(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        (path/"tokenizer_config.json").write_text("{}")


class PrepareAutoTokenizer:
    @classmethod
    def from_pretrained(cls,*args,**kwargs): return PrepareTokenizer()


class PrepareRanker(FakeRanker):
    def __init__(self,cfg): super().__init__()
    def close(self): pass


class PrepareEnv:
    def __init__(self,*args,**kwargs): self.calls=[]
    def validate(self,*args): return {"ok":True}
    def call(self,name,arguments):
        self.calls.append({"name":name,"arguments":copy.deepcopy(arguments)})
        return {"ok":True,"id":"actual-result"}
    def get_tool_calls(self): return self.calls
    def get_state_hash(self): return "expected-hash"


class Tests(unittest.TestCase):
    def test_missing_primary_tool_still_has_visible_argument_schema(self):
        item,tools,cfg=fixture()
        messages,audit=compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)
        self.assertNotIn("notes.create",audit["selected_tools"])
        self.assertIn("notes.create",audit["visible_tools"])
        spec=next(s for s in tools_in(messages) if s["name"]=="notes.create")
        self.assertEqual(spec["parameters"],tools["notes.create"]["parameters"])
        self.assertNotIn("description",spec)

    def test_all_installed_schemas_are_kept_independent_of_labels(self):
        item,tools,cfg=fixture()
        first,audit=compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)
        item["label"]={"tool_calls":[{"name":"shopping.add","arguments":{"title":"NEW_SECRET"}}]}
        second,_=compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)
        self.assertEqual(first,second)
        self.assertEqual(audit["visible_tools"],audit["available_tools"])
        self.assertNotIn("NEW_SECRET",dumps(first))

    def test_uninstalled_tool_is_not_added_by_fallback(self):
        item,tools,cfg=fixture()
        item["context"]["installed_apps"].pop("notes")
        messages,audit=compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)
        self.assertNotIn("notes.create",audit["visible_tools"])
        self.assertNotIn("notes.create",dumps(tools_in(messages)))

    def test_budget_removes_prose_but_keeps_all_schemas(self):
        item,tools,cfg=fixture();cfg["prompt_target_tokens"]=1
        for tool in tools.values(): tool["description"]="long description "*50
        messages,audit=compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)
        self.assertEqual(audit["visible_tools"],audit["available_tools"])
        self.assertEqual(audit["detailed_tools_in_prompt"],[])
        self.assertTrue(audit["descriptions_removed_for_budget"])
        self.assertTrue(all("description" not in s for s in tools_in(messages)))
        self.assertIn(item["audio_text"],messages[1]["content"])

    def test_arguments_named_like_metadata_are_preserved(self):
        schema={"type":"object","title":"Schema annotation", "description":"long prose",
                "required":["title","description","examples","type"],
                "properties":{name:{"type":"string","description":"argument prose"}
                              for name in ("title","description","examples","type")}}
        compact=compact_schema(schema,0)
        self.assertNotIn("title",compact)
        self.assertNotIn("description",compact)
        self.assertEqual(set(compact["properties"]),set(schema["properties"]))
        self.assertEqual(compact["required"],schema["required"])
        self.assertTrue(all(v=={"type":"string"} for v in compact["properties"].values()))

    def test_definition_names_and_literal_values_are_preserved(self):
        literal={"title":"value","description":"literal text","examples":[1,2]}
        schema={"$defs":{"title":{"type":"string","description":"annotation"}},
                "enum":[literal],"const":literal,"default":literal,
                "properties":{"title":{"$ref":"#/$defs/title"}},"required":["title"]}
        compact=compact_schema(schema,0)
        self.assertEqual(compact["enum"],[literal])
        self.assertEqual(compact["const"],literal)
        self.assertEqual(compact["default"],literal)
        self.assertEqual(compact["$defs"],{"title":{"type":"string"}})
        self.assertEqual(compact["properties"],schema["properties"])

    def test_hard_filter_is_explicit_and_diagnostic(self):
        item,tools,cfg=fixture();cfg["tool_schema_policy"]="hard_filter"
        messages,audit=compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)
        self.assertEqual(audit["visible_tools"],audit["selected_tools"])
        self.assertNotIn("notes.create",{s["name"] for s in tools_in(messages)})

    def test_unknown_policy_fails(self):
        item,tools,cfg=fixture();cfg["tool_schema_policy"]="typo"
        with self.assertRaises(ValueError):
            compile_context(item,tools,FakeRanker(),FakeTokenizer(),cfg)

    def test_preparation_continues_when_primary_recall_is_zero_but_schema_coverage_is_full(self):
        import prepare_agent
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);train=root/"data/train";train.mkdir(parents=True)
            out=root/"out";out.mkdir();code=root/"code";code.mkdir()
            item,tools,cfg=fixture()
            apps={}
            for tool in tools.values():
                apps.setdefault(tool["app"],{"tools":[]})["tools"].append(tool)
            write_json(train/"train_tools_catalog.json",{"apps":apps})
            items=[]
            for i in range(128):
                sample=copy.deepcopy(item)
                sample["id"]="train_"+str(i);sample["audio_text"]+=" unique_"+str(i)
                sample["label"]={"tool_calls":[{"name":"notes.create","arguments":{"id":"gold_"+str(i)}}]}
                items.append(sample)
            write_jsonl(train/"train.jsonl",items)
            write_json(out/"package_versions.json",{})
            cfg.update(data_root=str(root/"data"),output_dir=str(out),code_dir=str(code),
                config_path=str(out/"run_config.json"),model_id="m",model_revision="sha",tokenizer_id="t",tokenizer_revision="sha",
                reranker_id="r",reranker_revision="sha",seed=42,asr_manifest_path="",hash_audio_for_split=False,
                minimum_train_schema_coverage=1.0,tool_schema_policy="compact_all",
                requested_max_length=2048,requested_max_new_tokens=512,max_length=2048,max_new_tokens=512,
                auto_token_limits=True,max_length_cap=16000,max_new_tokens_cap=1024,length_round_to=256,
                length_safety_margin=128,rerank_batch_size=8,rerank_max_length=512,rerank_query_tokens=160)
            with patch.dict(sys.modules,{"transformers":types.SimpleNamespace(AutoTokenizer=PrepareAutoTokenizer)}), \
                 patch.object(prepare_agent,"CachedReranker",PrepareRanker), \
                 patch.object(prepare_agent,"import_tool_env",lambda *args:PrepareEnv), \
                 contextlib.redirect_stdout(io.StringIO()):
                prepare_agent.prepare(cfg)
            report=json.loads((out/"retrieval_report.json").read_text())
            for part in ("train","validation"):
                row=report["splits"][part]
                self.assertEqual(row["primary_gold_call_tool_recall"],0)
                self.assertEqual(row["schema_tool_coverage"],1)
                self.assertGreater(row["fallback_gold_calls"],0)
            self.assertTrue((out/"prepared_fingerprint.json").is_file())
            self.assertEqual(sum(len(read_jsonl(out/f"{part}_tokens.jsonl")) for part in ("train","validation","test")),256)


if __name__ == "__main__": unittest.main(verbosity=2)
