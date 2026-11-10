
"""History retrieval, causal routing SFT and official execution separation."""
import copy
import contextlib
import importlib.util
import inspect
import io
import json
import tempfile
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path
from agent_core import dumps,initial_messages,installed_tool_names,rollout,tokenize_sft,write_json,write_jsonl,read_jsonl
from context_rerank import compile_context,history_blocks
from context_window import fit_prompt
from history_memory import (MemoryCache,memory_key,bm25_scores,bm25_candidates,bounded_segments,
    parse_ranges,retrieve_history,segmentation_windows,history_signature)
from tool_router import (clean_context,router_messages,executor_messages,parse_apps,
    make_router_sft_rows)
from selftest_context import FakeTokenizer,FakeRanker,fixture
from selftest_agent import FakeEnv,TOOLS,item as sample_item,call


def config(tmp):
    _,_,cfg=fixture()
    cfg.update(history_router=True,output_dir=str(tmp),model_id='qwen',model_revision='fixed',
        reranker_id='bge',reranker_revision='fixed',rerank_max_length=512,rerank_query_tokens=160,rerank_batch_size=2,
        history_token_budget=700,history_bm25_candidates=3,history_bm25_k1=1.5,history_bm25_b=.75,
        history_diversity=.15,segment_window_tokens=600,segment_window_blocks=4,segment_max_tokens=300,
        segment_short_history_tokens=400,segment_max_new_tokens=128,history_compression_rate=.75,
        history_compressor_id='llmlingua2',history_compressor_revision='fixed',router_description_chars=48,
        router_prompt_tokens=4096,executor_prompt_tokens=4096,router_max_repairs=1,max_agent_steps=16,
        max_length=8192,max_length_cap=8192,requested_max_new_tokens=512,max_new_tokens=512,
        max_new_tokens_cap=512,length_safety_margin=128)
    return cfg


def cache_history(item,tools,cfg,tokenizer):
    history,_,_=clean_context(item,tools,cfg);blocks=history_blocks(history)
    ranges=[[i,min(i+1,len(blocks)-1)] for i in range(0,len(blocks),2)]
    segments=bounded_segments(blocks,ranges,tokenizer,cfg)
    for s in segments:s['index_text']=s['raw_text']
    cache=MemoryCache(Path(cfg['output_dir'])/'history_memory.sqlite')
    cache.put(memory_key(history,cfg),{'segments':segments,'compression_complete':True,
        'segmentation':{'fallback_windows':0}});cache.close()


class MemoryTests(unittest.TestCase):
    def test_unicode_bm25_retrieves_vietnamese_name_number(self):
        scores=bm25_scores('lịch chị Lan 14h',['chị Lan hẹn họp lúc 14h','mua sữa cho Minh','thời tiết hôm nay'])
        self.assertGreater(scores[0],scores[1]);self.assertGreater(scores[0],scores[2])

    def test_empty_bm25_is_well_defined(self):
        self.assertEqual(bm25_scores('x',[]),[]);self.assertEqual(bm25_scores('', ['a','b']),[0.,0.])

    def test_two_indexes_merge_to_bounded_candidate_pool(self):
        segments=[{'raw_text':f'noise {i}','index_text':f'noise {i}'} for i in range(20)]
        segments[0]['raw_text']='Lan calendar';segments[1]['index_text']='Lan calendar'
        with tempfile.TemporaryDirectory() as tmp:
            order,a=bm25_candidates('Lan calendar',segments,config(tmp))
        self.assertEqual(len(order),3);self.assertIn(0,order);self.assertIn(1,order)

    def test_segment_ranges_cover_all_blocks_exactly_once(self):
        self.assertEqual(parse_ranges('{"segments":[[2,3],[4,6]]}',2,6),[[2,3],[4,6]])
        for text in ('{"segments":[[2,3],[3,6]]}','{"segments":[[2,3],[5,6]]}','{"segments":[[2,5]]}'):
            with self.assertRaises(ValueError):parse_ranges(text,2,6)

    def test_segmenter_windowing_does_not_erase_original_history(self):
        blocks=[[{'role':'user','text':str(i)+' x'*200}] for i in range(12)];before=copy.deepcopy(blocks)
        with tempfile.TemporaryDirectory() as tmp:
            windows=segmentation_windows(blocks,FakeTokenizer(),config(tmp))
        self.assertEqual([r['block'] for w in windows for r in w],list(range(12)));self.assertEqual(blocks,before)

    def test_short_history_keeps_every_turn_without_reranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(tmp);h=[{'role':'user','text':'Nhắc lịch Lan'},{'role':'assistant','text':'Đã tạo'}]
            class NoRerank:
                def score(self,*args):raise AssertionError('Short history needs no retrieval')
            kept,a=retrieve_history(h,{'segments':[]},'Sửa lịch đó',NoRerank(),FakeTokenizer(),cfg)
            self.assertEqual(kept,h);self.assertTrue(a['full_history'])

    def test_long_history_is_bounded_and_keeps_recent_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(tmp);h=[]
            for i in range(20):h.extend([{'role':'user','text':('calendar active ' if i<2 else 'noise ')+str(i)+' detail'*6},
                                        {'role':'assistant','text':'confirmed '+str(i)}])
            blocks=history_blocks(h);segments=bounded_segments(blocks,[[i,i] for i in range(len(blocks))],FakeTokenizer(),cfg)
            kept,a=retrieve_history(h,{'segments':segments},'calendar active',FakeRanker(),FakeTokenizer(),cfg)
            self.assertLessEqual(a['history_tokens'],cfg['history_token_budget']);self.assertIn('19',dumps(kept))
            self.assertFalse(a['full_history']);self.assertLess(len(a['candidate_segments']),len(segments))

    def test_only_raw_values_reach_agent_not_compressed_rewrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(tmp);cfg['history_token_budget']=500
            h=[{'role':'user','text':'Lan ID_REAL exact phone 0912345678'},
               {'role':'assistant','text':'calendar active'},
               {'role':'user','text':'noise '*100},{'role':'assistant','text':'noise'},
               {'role':'user','text':'recent'},{'role':'assistant','text':'ok'}]
            cfg['keep_recent_history']=1;blocks=history_blocks(h)
            segments=bounded_segments(blocks,[[i,i] for i in range(len(blocks))],FakeTokenizer(),cfg)
            for s in segments:s['index_text']='calendar active INDEX_ONLY'
            kept,a=retrieve_history(h,{'segments':segments},'Lan calendar active',FakeRanker(),FakeTokenizer(),cfg)
            self.assertNotIn('INDEX_ONLY',dumps(kept));self.assertIn('0912345678',dumps(kept))


class RouterTests(unittest.TestCase):
    def prepare(self,tmp):
        cfg=config(tmp);obj=sample_item();cache_history(obj,TOOLS,cfg,FakeTokenizer())
        messages,audit=compile_context(obj,TOOLS,FakeRanker(),FakeTokenizer(),cfg);obj['_agent_messages']=messages
        return obj,cfg,audit

    def test_initial_input_is_label_independent_and_has_no_detailed_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj,cfg,audit=self.prepare(tmp);a=copy.deepcopy(obj['_agent_messages'])
            obj['label']={'tool_calls':[{'name':'sms.send','arguments':{'text':'SECRET_GOLD'}}],'final_state':'SECRET_STATE'}
            b,_=compile_context(obj,TOOLS,FakeRanker(),FakeTokenizer(),cfg)
            self.assertEqual(a,b);self.assertNotIn('SECRET',dumps(b));self.assertNotIn('"parameters"',dumps(b))
            self.assertIn('contacts',dumps(b));self.assertIn('sms',dumps(b));self.assertEqual(audit['selected_tools'],[])

    def test_parser_only_accepts_installed_apps(self):
        self.assertEqual(parse_apps('{"apps":["sms","sms"]}',['sms']),['sms'])
        for raw in ('{"apps":["unknown"]}','{"apps":"sms"}','{"apps":[],"extra":1}'):
            with self.assertRaises(ValueError):parse_apps(raw,['sms'])

    def test_sft_contains_router_open_apps_and_causal_real_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj,cfg,_=self.prepare(tmp);calls=obj['label']['tool_calls']
            observations=[{'ok':True,'contacts':[{'id':'person-7','name':'An'}]},{'ok':True}]
            rows=list(make_router_sft_rows(obj,TOOLS,calls,observations,FakeTokenizer(),cfg))
            self.assertEqual([r['task'] for r in rows],['router','tool_call','open_apps','tool_call','done'])
            self.assertEqual(json.loads(rows[0]['target']),{'apps':['contacts']})
            self.assertNotIn('person-7',dumps(rows[1]['messages']));self.assertIn('person-7',dumps(rows[3]['messages']))
            self.assertNotIn('SECRET_GOLD_STATE',dumps(rows))
            for row in rows:
                encoded=tokenize_sft(row,FakeTokenizer(),cfg)
                self.assertEqual([x for x in encoded['labels'] if x!=-100],FakeTokenizer().encode(row['target']+'<|im_end|>'))

    def test_rollout_uses_predicted_apps_and_internal_ops_never_call_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj,cfg,_=self.prepare(tmp);env=FakeEnv();seen=[]
            outputs=iter([dumps({'apps':['contacts']}),dumps(obj['label']['tool_calls'][0]),
                          dumps({'open_apps':['sms']}),dumps(call()),dumps({'done':True})])
            def generate(messages):seen.append(copy.deepcopy(messages));return next(outputs)
            result=rollout(obj,TOOLS,env,generate,cfg)
            self.assertIsNone(result['error']);self.assertEqual(len(result['tool_calls']),2)
            self.assertEqual(result['router']['initial_apps'],['contacts']);self.assertEqual(result['router']['final_apps'],['contacts','sms'])
            first_specs=json.loads(seen[1][0]['content'].split('\nTOOLS=',1)[1].split('\nCONTEXT=',1)[0])
            self.assertNotIn('sms.send',[s['name'] for s in first_specs]);self.assertIn('sms.send',seen[3][0]['content'])
            self.assertEqual(result['router']['qwen_calls'],5)

    def test_unopened_call_loads_schema_without_executing_until_regenerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj,cfg,_=self.prepare(tmp);env=FakeEnv()
            outputs=iter([dumps({'apps':['contacts']}),dumps(call()),dumps(call()),dumps({'done':True})])
            result=rollout(obj,TOOLS,env,lambda m:next(outputs),cfg)
            self.assertEqual(len(env.calls),1);self.assertEqual(result['trace'][1]['phase'],'auto_open');self.assertIsNone(result['error'])

    def test_bad_router_falls_back_without_gold_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj,cfg,_=self.prepare(tmp);env=FakeEnv();outputs=iter(['bad','{"apps":["unknown"]}',dumps({'done':True})])
            result=rollout(obj,TOOLS,env,lambda m:next(outputs),cfg)
            self.assertTrue(result['router']['fallback_all_apps']);self.assertEqual(env.calls,[])

    def test_stage_prompt_caps_keep_complete_assistant_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(tmp);cfg['router_prompt_tokens']=1200;cfg['executor_prompt_tokens']=1400
            for mode,limit in (('ROUTER',1200),('EXECUTOR',1400)):
                messages=[{'role':'system','content':'MODE='+mode+'\nRules '+('x '*5000)},
                          {'role':'user','content':'HISTORY=[]\nFINAL_REQUEST=Nhắn cho An'}]
                _,ids,_=fit_prompt(FakeTokenizer(),messages,cfg);self.assertLessEqual(len(ids),limit)
                target=dumps({'apps':['sms']}) if mode=='ROUTER' else dumps(call())
                encoded=tokenize_sft({'messages':messages,'target':target},FakeTokenizer(),cfg)
                self.assertEqual([x for x in encoded['labels'] if x!=-100],FakeTokenizer().encode(target+'<|im_end|>'))


    @unittest.skipUnless(importlib.util.find_spec('llmlingua') and importlib.util.find_spec('torch'),
                         'Installed LLMLingua-2 API requires PyTorch')
    def test_installed_llmlingua_api_supports_pinned_model_and_bounded_batches(self):
        from llmlingua import PromptCompressor
        params=inspect.signature(PromptCompressor).parameters
        for key in ('model_name','device_map','model_config','use_llmlingua2','llmlingua2_config'):self.assertIn(key,params)
        self.assertIn('rate',inspect.signature(PromptCompressor.compress_prompt_llmlingua2).parameters)


if __name__=='__main__':unittest.main(verbosity=2)
