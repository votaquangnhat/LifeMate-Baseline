
"""No row loss or assistant-target truncation on extreme context and observations."""
import json
import unittest
from agent_core import dumps,tokenize_sft,prompt_ids
from context_window import fit_prompt
from selftest_context import fixture,FakeTokenizer,FakeRanker
from context_rerank import compile_context


class Tests(unittest.TestCase):
    def cfg(self):return {"max_length":2048,"max_length_cap":2048,"requested_max_new_tokens":256,"max_new_tokens_cap":256,"length_safety_margin":128}

    def test_long_tool_observation_is_trimmed_but_target_and_eos_are_complete(self):
        tok=FakeTokenizer();cfg=self.cfg()
        messages=[{"role":"system","content":"Return one JSON"},{"role":"user","content":"HISTORY=[]\nFINAL_REQUEST=Send to Minh"},
            {"role":"assistant","content":'{"name":"contacts.search","arguments":{}}'},
            {"role":"tool","content":dumps({"ok":True,"contacts":[{"id":str(i),"name":"Minh "+"x"*200} for i in range(300)]})}]
        target='{"name":"sms.send","arguments":{"contact_id":"c_1"}}'
        encoded=tokenize_sft({"messages":messages,"target":target},tok,cfg)
        suffix=tok.encode(target+"<|im_end|>")
        self.assertLessEqual(encoded["length"],2048)
        self.assertEqual([x for x in encoded["labels"] if x!=-100],suffix)
        self.assertTrue(encoded["context_trim"]["truncated"])

    def test_pathological_request_always_has_a_bounded_prompt(self):
        tok=FakeTokenizer();m=[{"role":"system","content":"Rules "+"s"*20000},{"role":"user","content":"Request "+"r"*50000}]
        _,ids,a=fit_prompt(tok,m,self.cfg())
        self.assertLessEqual(len(ids),1664);self.assertTrue(a["truncated"])

    def test_small_context_is_identical_in_training_and_inference(self):
        tok=FakeTokenizer();m=[{"role":"user","content":"Hello"}]
        fitted,ids,a=fit_prompt(tok,m,self.cfg())
        self.assertEqual(ids,prompt_ids(tok,m));self.assertEqual(fitted,m);self.assertFalse(a["truncated"])

    def test_many_overlong_samples_keep_one_encoded_row_each(self):
        tok=FakeTokenizer();cfg=self.cfg();target='{"done":true}'
        data=[{"messages":[{"role":"user","content":"q"*(10000+i*10)}],"target":target} for i in range(20)]
        encoded=[tokenize_sft(r,tok,cfg) for r in data]
        self.assertEqual(len(encoded),20)
        self.assertTrue(all(r["length"]<=2048 for r in encoded))
        self.assertTrue(all([x for x in r["labels"] if x!=-100]==tok.encode(target+"<|im_end|>") for r in encoded))


if __name__=="__main__":unittest.main(verbosity=2)
