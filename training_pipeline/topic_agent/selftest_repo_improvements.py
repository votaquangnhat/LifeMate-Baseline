"""Repeated-read guard and after-run error diagnostics contracts."""
import copy
import unittest
import tempfile
from pathlib import Path
from read_guard import ReadRepeatGuard
from agent_core import dumps,installed_tool_names,write_json,write_jsonl,read_jsonl,sha256_file
from tool_router import clean_context,router_messages,routed_rollout
from selftest_agent import TOOLS,FakeEnv,item,call
from selftest_context import FakeTokenizer
from selftest_history_router import config
from error_analysis import classify,call_differences


class Tests(unittest.TestCase):
    def test_cache_clears_on_any_non_read_and_does_not_mutate_results(self):
        g=ReadRepeatGuard();c={'name':'x.lookup','arguments':{'q':'a'}};result={'ids':['a']}
        g.record(c,result,'read');hit,seen=g.lookup(c,'read');self.assertTrue(hit)
        seen['ids'].append('b');self.assertEqual(g.lookup(c,'read')[1],result)
        self.assertFalse(g.lookup({'name':'x.write','arguments':{}},'write')[0])
        self.assertFalse(g.lookup(c,'read')[0])
        g.record(c,result,'read');g.lookup(c,'unknown');self.assertFalse(g.lookup(c,'read')[0])

    def run_calls(self,actions,enabled=True):
        cfg=config('/tmp');cfg.update(duplicate_read_guard=enabled,max_duplicate_read_blocks=3)
        tools=copy.deepcopy(TOOLS);tools['contacts.search']['kind']='read'
        obj=item();history,request,context=clean_context(obj,tools,cfg)
        ctx=dict(history=history,request=request,public_context=context,names=installed_tool_names(obj,tools,cfg['global_meta_tools']))
        obj['_router_context']=ctx;obj['_agent_messages']=router_messages(ctx,tools,cfg,FakeTokenizer())
        responses=iter([dumps({'apps':['contacts','sms']})]+[dumps(a) for a in actions])
        class Env(FakeEnv):
            hash_reads=0
            def get_state_hash(self):
                self.hash_reads+=1
                return super().get_state_hash()
        env=Env();r=routed_rollout(obj,tools,env,lambda m:next(responses),cfg)
        self.assertEqual(env.hash_reads,1,'Agent decisions must not consult hidden-state hashes')
        return r,env

    def test_only_duplicate_reads_are_skipped_writes_and_postwrite_reads_execute(self):
        read={'name':'contacts.search','arguments':{'query':'An'}}
        actions=[read,read,call(),read,call(),call(),{'done':True}]
        r,env=self.run_calls(actions)
        self.assertEqual(env.calls,[read,call(),read,call(),call()])
        self.assertEqual(sum(t['phase']=='cached_read' for t in r['trace']),1)
        self.assertIsNone(r['error'])
        _,baseline=self.run_calls(actions,False)
        self.assertEqual(baseline.calls,actions[:-1])

    def test_repeat_loop_stops_without_fabricating_executed_calls(self):
        read={'name':'contacts.search','arguments':{'query':'An'}}
        r,env=self.run_calls([read]*4)
        self.assertEqual(env.calls,[read]);self.assertIn('read loop',r['error'])
        self.assertEqual(r['tool_calls'],[read])

    def test_validation_asr_replaces_all_user_text_and_requires_exact_fingerprint(self):
        from infer import validation_source
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';train=data/'train';train.mkdir(parents=True)
            imported=root/'import';imported.mkdir();out=root/'output';out.mkdir()
            sample=item();sample['history']=[{'role':'user','audio':'h.wav','audio_text':'GOLD HISTORY'}]
            source=train/'train.jsonl';catalog=train/'train_tools_catalog.json'
            write_jsonl(source,[sample]);write_json(catalog,{'apps':[]})
            write_json(imported/'prepared_fingerprint.json',{'source':sha256_file(source),'catalog':sha256_file(catalog)})
            write_jsonl(imported/'validation_references.jsonl',[{'id':sample['id'],'tool_calls':[],'state_hash':'h'}])
            write_json(imported/'split_assignments.json',{sample['id']:'validation'})
            transcripts=root/'asr.jsonl'
            write_jsonl(transcripts,[{'id':sample['id'],'audio_text':'ASR REQUEST','history_audio_texts':['ASR HISTORY']}])
            cfg=dict(output_dir=str(out),import_dir=str(imported),data_root=str(data),
                     validation_transcripts_path=str(transcripts),validation_transcripts_sha256=sha256_file(transcripts))
            validation_source(cfg);rows=read_jsonl(out/'validation_source.jsonl')
            self.assertEqual(rows[0]['audio_text'],'ASR REQUEST')
            self.assertEqual(rows[0]['history'][0]['audio_text'],'ASR HISTORY')
            self.assertNotIn('label',rows[0]);self.assertNotIn('GOLD HISTORY',dumps(rows))
            transcripts.write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):validation_source(cfg)

    def test_diagnostics_keep_fuzzy_score_and_handle_repeated_gold_calls(self):
        a=call('hello');b=call('hello!')
        row={'tool_calls':[b],'score':1.0,'state_hash':'h','error':None}
        ref={'tool_calls':[a],'state_hash':'h'}
        self.assertEqual(classify(row,ref,TOOLS),'full_score')
        extra,missing,fields=call_differences([a],[a,a])
        self.assertEqual(missing,[a]);self.assertEqual(extra,[])
        extra,missing,fields=call_differences([b],[a])
        self.assertEqual(fields[0]['different_keys'],['text'])
        row.update(score=0.5)
        self.assertEqual(classify(row,ref,TOOLS),'correct_state_partial_call_score')


if __name__=='__main__':unittest.main(verbosity=2)
