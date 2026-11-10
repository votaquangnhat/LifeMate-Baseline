"""Regressions for protected context and per-sample history retrieval."""
import copy
import json
import unittest
from agent_core import dumps,prompt_ids
from context_window import fit_prompt
from history_memory import retrieve_history
from selftest_context import FakeTokenizer
from selftest_history_router import config


class Tests(unittest.TestCase):
    def cfg(self):
        c=config('/tmp');c.update(agent_variant='enhanced',max_length=2048,max_new_tokens=256,max_new_tokens_cap=256,
                                 router_prompt_tokens=1700,executor_prompt_tokens=1700,length_safety_margin=0)
        return c

    def test_schema_request_and_observations_remain_exact(self):
        cfg=self.cfg();tokenizer=FakeTokenizer()
        spec={'name':'app.send','parameters':{'type':'object','required':['text'],
              'properties':{'text':{'type':'string'},'description':{'type':'string'}},'additionalProperties':False}}
        messages=[{'role':'system','content':'MODE=EXECUTOR\nTOOLS='+dumps([spec])+'\nCONTEXT={}'},
                  {'role':'user','content':'HISTORY=[]\nFINAL_REQUEST=Gửi đúng nội dung này.'},
                  {'role':'assistant','content':'{"name":"app.lookup","arguments":{}}'},
                  {'role':'tool','content':dumps({'items':[{'id':str(i),'text':'unique '+str(i)} for i in range(8)]})}]
        original=copy.deepcopy(messages)
        fitted,ids,audit=fit_prompt(tokenizer,messages,cfg)
        self.assertEqual(fitted,original);self.assertEqual(messages,original)
        self.assertFalse(audit['schema_names_only']);self.assertFalse(audit['emergency_token_clip'])
        self.assertEqual(ids,prompt_ids(tokenizer,fitted))

    def test_overflow_never_returns_a_sliced_request_or_tool_result(self):
        messages=[{'role':'system','content':'MODE=EXECUTOR\nTOOLS=[]\nCONTEXT={}'},
                  {'role':'user','content':'HISTORY=[]\nFINAL_REQUEST='+'required '*10000}]
        before=copy.deepcopy(messages)
        with self.assertRaisesRegex(ValueError,'Protected context overflow'):fit_prompt(FakeTokenizer(),messages,self.cfg())
        self.assertEqual(messages,before)

    def test_long_recent_history_is_never_rewritten_as_a_new_user_request(self):
        cfg=self.cfg();cfg['history_token_budget']=80
        history=[{'role':'user','text':'old task'},{'role':'assistant','text':'done'},
                 {'role':'user','text':'very long original '*200},{'role':'assistant','text':'original response'}]
        class Ranker:
            def score(self,q,docs):return [1.]*len(docs)
        kept,audit=retrieve_history(history,{'segments':[]},'followup',Ranker(),FakeTokenizer(),cfg)
        self.assertEqual(kept,history[2:]);self.assertTrue(audit['oversized_recent_block'])
        self.assertFalse(audit['emergency_history_clip'])


if __name__=='__main__':unittest.main(verbosity=2)
